"""Benchmark runner: one command for the whole benchmark workflow.

    python scripts/run_benchmarks.py                 # run the configured experiment
    python scripts/run_benchmarks.py --dry-run       # validate the config and show the run plan
    python scripts/run_benchmarks.py --resume        # continue an interrupted/partial experiment

For every (configuration, repetition) the runner generates ONE scenario with a
fresh generator and gives every algorithm its own copy of it. For each
algorithm it then: runs the adapter (with independent runtime measurement and a
hard timeout), validates the returned plan, evaluates valid plans with the
analytics evaluator, builds one record in the approved CSV schema and appends
it through the logger. Finally the experiment is aggregated.

Everything below the orchestration (seeds, validation, evaluation, execution
backends, logging, aggregation) is reused from src/analytics/.

ADAPTERS (analytics-owned wrappers). Greedy, GA, SVCA and the paper variants are registered through
``src.analytics.allocation_adapters``. MILP is registered too but is only run when listed
in benchmark.algorithms, because it is exact and slow. An algorithm without an adapter is reported as
unavailable and is never given a fabricated result. Adapters are registered as
``"module.path:attribute"`` references (in ``ADAPTER_REGISTRY`` or with
``--adapter ALGORITHM=module.path:attribute``) and follow OUR analytics-side contract:

    adapter(scenario: ScenarioData, seed: int, time_limit_seconds: float)
        -> mapping with the key "assignments": {robot_id: [task_id, ...]}

A real adapter translates the allocator's ``solve(scenario)`` result into this
contract. GA and SVCA receive the harness-derived algorithm seed through their
configuration objects; the harness backend owns timing and subprocess timeouts.

REWARD STAGE: the existing PreferenceAwareShapleySolver is called through an
analytics-owned adapter when enabled. Robot rewards are validated against the
scenario's agent IDs and Gini is recomputed with the approved analytics metric.

Evaluation placement follows design option P1 (after the total timer; design
decision D-H4 is still open, see docs/BENCHMARK_HARNESS_DESIGN.md Section 3.3).

Exit codes: 0 ok, 2 configuration error, 3 pre-flight error (including
unavailable algorithms), 4 persistence/data error, 5 internal harness error,
130 interrupted.
"""

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from src.analytics import seeding  # noqa: E402
from src.analytics.evaluation import (  # noqa: E402
    ENERGY_TOLERANCE_RELATIVE,
    PENALTY_STRATEGIES,
    SANITY_TOLERANCE,
    ValidationResult,
    evaluate_allocation,
    validate_assignments,
)
from src.analytics.execution import (  # noqa: E402
    ExecutionError,
    InProcessBackend,
    StageOutcome,
    Stopwatch,
    SubprocessBackend,
    resolve_adapter,
)
from src.analytics.metrics import compute_robot_reward_gini  # noqa: E402
if __name__ != "__mp_main__":
    # Spawned worker processes re-import this script as "__mp_main__". They never
    # call main(), so they skip the heavy imports (pandas) to start quickly.
    from src.analytics.logger import (  # noqa: E402
        DEFAULT_RESULTS_DIR,
        DuplicateKeyError,
        SchemaValidationError,
        append_benchmark_results,
        append_robot_rewards,
        load_benchmark_results,
        load_robot_rewards,
        summarize_experiment,
        validate_experiment_id,
        update_experiment_metadata,
        write_experiment_metadata,
    )
from src.simulation.generator import ScenarioData, ScenarioGenerator  # noqa: E402
from src.simulation.kinematics import BatteryDecayModel  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiments_config.yaml"

# Algorithm id -> analytics-owned adapter reference. MILP is registered but not in
# benchmark.algorithms by default because it is exact and slow beyond small sizes.
ADAPTER_REGISTRY: Dict[str, str] = {
    "Greedy": "src.analytics.allocation_adapters:greedy_adapter",
    "GA": "src.analytics.allocation_adapters:ga_adapter",
    "SVCA": "src.analytics.allocation_adapters:svca_adapter",
    "GA_Paper": "src.analytics.allocation_adapters:ga_paper_adapter",
    "SVCA_Paper": "src.analytics.allocation_adapters:svca_paper_adapter",
    "MILP": "src.analytics.allocation_adapters:milp_adapter",
}
REWARD_ADAPTER_REGISTRY: Dict[str, str] = {
    "default": "src.analytics.reward_adapters:preference_aware_reward_adapter",
}

EXIT_OK, EXIT_CONFIG, EXIT_PREFLIGHT, EXIT_DATA, EXIT_INTERNAL, EXIT_INTERRUPTED = 0, 2, 3, 4, 5, 130
MAX_MESSAGE_LENGTH = 500
MIN_ENTITIES, MAX_ENTITIES = 10, 80


class ConfigError(ValueError):
    """The experiment configuration is invalid."""


class PreflightError(RuntimeError):
    """A pre-flight check failed (existing data, orphans, unavailable algorithms)."""


class HarnessInternalError(RuntimeError):
    """A harness/simulator inconsistency (not an algorithm failure); aborts the experiment."""


class PersistenceError(RuntimeError):
    """The logger refused to write or read result data (duplicate key, schema violation)."""


class RunSpec(NamedTuple):
    config_id: str
    num_robots: int
    num_tasks: int
    repetition_index: int
    scenario_id: str
    scenario_seed: int
    algorithm: str
    algorithm_seed: int
    time_limit_seconds: float
    # Derived with the same --config seed templates as the other seeds. None keeps the
    # old behaviour (default config) for callers that build RunSpec by hand.
    reward_seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    """Read the YAML configuration (``yaml.safe_load``)."""
    try:
        with open(path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except FileNotFoundError:
        raise ConfigError(f"configuration file not found: {path}") from None
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read configuration {path}: {exc}") from None
    if not isinstance(config, dict):
        raise ConfigError(f"{path}: the top level must be a mapping.")
    return config


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_config(
    config: Mapping[str, Any],
    config_path: Path,
    results_dir: Path,
    check_output_paths: bool = True,
) -> None:
    """Check the whole configuration; raise one ``ConfigError`` listing every problem."""
    problems: List[str] = []

    def section(name: str) -> Mapping[str, Any]:
        value = config.get(name)
        if not isinstance(value, Mapping):
            problems.append(f"missing or invalid section '{name}'.")
            return {}
        return value

    experiment, environment = section("experiment"), section("environment")
    benchmark, time_limits = section("benchmark"), section("time_limits")
    evaluation, output = section("evaluation"), section("output")
    reward = section("reward")

    try:
        validate_experiment_id(experiment.get("id"))
    except SchemaValidationError as exc:
        problems.append(f"experiment.id: {exc}")
    base_seed = experiment.get("base_seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or not 0 <= base_seed <= seeding.SEED_MAX:
        problems.append(f"experiment.base_seed must be an int in 0..{seeding.SEED_MAX}.")

    if environment.get("dimensions") != 2:
        problems.append("environment.dimensions must be 2.")
    for key in ("width_m", "height_m"):
        if not _is_number(environment.get(key)) or environment.get(key) <= 0:
            problems.append(f"environment.{key} must be a number > 0.")

    algorithms = benchmark.get("algorithms")
    if (not isinstance(algorithms, list) or not algorithms or len(set(map(str, algorithms))) != len(algorithms)
            or not all(isinstance(a, str) and a for a in algorithms)):
        problems.append("benchmark.algorithms must be a non-empty list of unique strings.")
        algorithms = []
    repetitions = benchmark.get("repetitions_per_configuration")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        problems.append("benchmark.repetitions_per_configuration must be an int >= 1.")
    if benchmark.get("execution_mode") != "sequential":
        problems.append("benchmark.execution_mode must be 'sequential'.")
    if benchmark.get("same_scenario_for_all_algorithms") is not True:
        problems.append("benchmark.same_scenario_for_all_algorithms must be true.")
    fmt = benchmark.get("scenario_id_format")
    try:
        first = fmt.format(config_id="C1", repetition_index=0)
        if not first or first == fmt.format(config_id="C1", repetition_index=1):
            problems.append("benchmark.scenario_id_format must give a distinct id per repetition.")
    except Exception as exc:
        problems.append(f"benchmark.scenario_id_format is invalid: {exc!r}")

    configurations = config.get("configurations")
    seen_ids: Set[str] = set()
    if not isinstance(configurations, list) or not configurations:
        problems.append("configurations must be a non-empty list.")
    else:
        for entry in configurations:
            if not isinstance(entry, Mapping) or set(entry) != {"id", "num_robots", "num_tasks"}:
                problems.append(f"each configuration needs exactly the keys id, num_robots, num_tasks: {entry!r}.")
                continue
            if not isinstance(entry["id"], str) or not entry["id"] or entry["id"] in seen_ids:
                problems.append(f"configuration id {entry['id']!r} is empty or duplicated.")
            seen_ids.add(str(entry["id"]))
            counts = [entry["num_robots"], entry["num_tasks"]]
            if any(isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in counts):
                problems.append(f"configuration {entry['id']}: robot and task counts must be ints >= 1.")
            elif not MIN_ENTITIES <= sum(counts) <= MAX_ENTITIES:
                problems.append(
                    f"configuration {entry['id']}: robots + tasks = {sum(counts)} is outside "
                    f"[{MIN_ENTITIES}, {MAX_ENTITIES}]."
                )

    limits = time_limits.get("per_algorithm")
    if not isinstance(limits, Mapping):
        problems.append("time_limits.per_algorithm must be a mapping.")
        limits = {}
    for algorithm in algorithms:
        if not _is_number(limits.get(algorithm)) or not math.isfinite(limits[algorithm]) or limits[algorithm] <= 0:
            problems.append(f"time_limits.per_algorithm needs a positive number for '{algorithm}'.")
    reward_limit = time_limits.get("reward_stage_seconds")
    if not _is_number(reward_limit) or not math.isfinite(reward_limit) or reward_limit <= 0:
        problems.append("time_limits.reward_stage_seconds must be a finite number > 0.")

    if not isinstance(reward.get("enabled"), bool):
        problems.append("reward.enabled must be a boolean.")
    if reward.get("method") not in ("glauber", "exact", "stratified", "permutation"):
        problems.append("reward.method must be glauber, exact, stratified or permutation.")
    if reward.get("value_type") not in ("facility", "routing"):
        problems.append("reward.value_type must be facility or routing.")
    if reward.get("decomposition", "clusters") not in ("clusters", "global"):
        problems.append("reward.decomposition must be clusters or global.")
    alpha = reward.get("preference_sensitivity_alpha")
    if not _is_number(alpha) or not math.isfinite(alpha) or alpha < 0:
        problems.append("reward.preference_sensitivity_alpha must be a finite number >= 0.")
    for key, minimum in (("num_samples_Tw", 1), ("samples_per_stratum", 1)):
        value = reward.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            problems.append(f"reward.{key} must be an integer >= {minimum}.")
    burn_in = reward.get("burn_in")
    if isinstance(burn_in, bool) or not isinstance(burn_in, int) or burn_in < 0:
        problems.append("reward.burn_in must be an integer >= 0.")
    thin = reward.get("thin")
    if thin is not None and (isinstance(thin, bool) or not isinstance(thin, int) or thin < 1):
        problems.append("reward.thin must be null or an integer >= 1.")
    max_log_gamma = reward.get("max_log_gamma")
    if max_log_gamma is not None and (not _is_number(max_log_gamma) or not math.isfinite(max_log_gamma)
                                      or max_log_gamma <= 0):
        problems.append("reward.max_log_gamma must be null or a finite number > 0.")

    penalty = evaluation.get("energy_penalty")
    if not isinstance(penalty, Mapping) or penalty.get("strategy") not in PENALTY_STRATEGIES:
        problems.append(f"evaluation.energy_penalty.strategy must be one of {list(PENALTY_STRATEGIES)}.")
    elif (not _is_number(penalty.get("lambda_E")) or not math.isfinite(penalty["lambda_E"])
          or penalty["lambda_E"] < 0):
        problems.append("evaluation.energy_penalty.lambda_E must be a finite number >= 0.")
    stability = evaluation.get("coalition_stability")
    if not isinstance(stability, Mapping) or stability.get("enabled") is not False:
        problems.append("evaluation.coalition_stability.enabled must be false (no formal definition exists).")

    if check_output_paths:
        expected = {
            "benchmark_results": results_dir / "benchmark_results.csv",
            "robot_rewards": results_dir / "robot_rewards.csv",
            "metadata_dir": results_dir,
        }
        for key, want in expected.items():
            got = output.get(key)
            if not isinstance(got, str) or os.path.normpath(str(REPO_ROOT / got)) != os.path.normpath(str(want)):
                problems.append(f"output.{key} must be {want} (the logger fixes the file names), got {got!r}.")

    try:
        seeding.get_seed_templates(config_path)
    except (FileNotFoundError, seeding.SeedConfigError) as exc:
        problems.append(f"seeds: {exc}")

    if problems:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(problems))


# ---------------------------------------------------------------------------
# Run plan, scenarios, adapters
# ---------------------------------------------------------------------------

def build_run_plan(
    config: Mapping[str, Any],
    config_path: Path,
    only_configs: Optional[Sequence[str]] = None,
    only_algorithms: Optional[Sequence[str]] = None,
    repetitions: Optional[int] = None,
) -> List[RunSpec]:
    """Expand the configuration: configuration-major, then repetition, then algorithm."""
    benchmark = config["benchmark"]
    base_seed = config["experiment"]["base_seed"]
    reps = repetitions if repetitions is not None else benchmark["repetitions_per_configuration"]
    algorithms = [a for a in benchmark["algorithms"] if only_algorithms is None or a in only_algorithms]
    plan: List[RunSpec] = []
    for entry in config["configurations"]:
        if only_configs is not None and entry["id"] not in only_configs:
            continue
        for rep in range(reps):
            scenario_seed = seeding.scenario_seed(base_seed, entry["id"], rep, config_path=config_path)
            scenario_id = benchmark["scenario_id_format"].format(config_id=entry["id"], repetition_index=rep)
            for algorithm in algorithms:
                plan.append(RunSpec(
                    entry["id"], entry["num_robots"], entry["num_tasks"], rep, scenario_id, scenario_seed,
                    algorithm,
                    seeding.algorithm_seed(scenario_seed, algorithm, config_path=config_path),
                    float(config["time_limits"]["per_algorithm"][algorithm]),
                    seeding.reward_seed(scenario_seed, config["seeds"]["reward_stage_label"],
                                        config_path=config_path),
                ))
    return plan


def generate_scenario(config: Mapping[str, Any], spec: RunSpec) -> ScenarioData:
    """Build the scenario of ``spec`` with a FRESH generator (so ``scenario.seed`` is truthful)."""
    env = config["environment"]
    generator = ScenarioGenerator(
        map_width=float(env["width_m"]), map_height=float(env["height_m"]), seed=spec.scenario_seed
    )
    scenario = generator.generate_scenario(num_agents=spec.num_robots, num_tasks=spec.num_tasks)
    if (scenario.seed != spec.scenario_seed or len(scenario.agents) != spec.num_robots
            or len(scenario.tasks) != spec.num_tasks):
        raise HarnessInternalError(f"generated scenario does not match {spec.scenario_id}.")
    return scenario


def scenario_fingerprint(scenario: ScenarioData) -> str:
    """SHA-256 of the scenario content (used to prove the master copy never changes)."""
    text = json.dumps(scenario.to_dict(), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def algorithm_availability(
    algorithms: Sequence[str], registry: Mapping[str, str]
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Resolve every algorithm's adapter. Returns (callables, {algorithm: reason unavailable})."""
    available: Dict[str, Any] = {}
    unavailable: Dict[str, str] = {}
    for algorithm in algorithms:
        try:
            available[algorithm] = resolve_adapter(algorithm, registry)
        except ExecutionError as exc:
            unavailable[algorithm] = str(exc)
    return available, unavailable


def _reward_stage_status(config: Optional[Mapping[str, Any]] = None, available: bool = True) -> str:
    """Describe whether reward execution is configured and its analytics adapter is available."""
    if config is not None and not config.get("reward", {}).get("enabled", False):
        return "disabled by experiment configuration"
    if not available:
        return "unavailable: reward solver adapter could not be loaded; no rewards were fabricated"
    reward = (config or {}).get("reward", {})
    return ("enabled: PreferenceAwareShapleySolver "
            f"(method={reward.get('method', 'glauber')}, value_type={reward.get('value_type', 'facility')}, "
            f"decomposition={reward.get('decomposition', 'clusters')}); "
            "coalition stability remains disabled")


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def _bounded(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_MESSAGE_LENGTH else text[: MAX_MESSAGE_LENGTH - 3] + "..."


def _extract_assignments(outcome: StageOutcome, master: ScenarioData) -> ValidationResult:
    """Validate the plan inside the adapter's output (contract: mapping with 'assignments')."""
    payload = outcome.payload
    if not isinstance(payload, Mapping) or "assignments" not in payload:
        return ValidationResult(False, ["adapter output must be a mapping with an 'assignments' entry"], None)
    return validate_assignments(payload["assignments"], master)


def _extract_clusters(outcome: StageOutcome, master: ScenarioData) -> Optional[List[Dict[str, List[int]]]]:
    """Stage-1 coalitions from the adapter output, or None if absent or not about this scenario."""
    payload = outcome.payload
    clusters = payload.get("clusters") if isinstance(payload, Mapping) else None
    if not isinstance(clusters, list) or not clusters:
        return None
    agents = {a.agent_id for a in master.agents}
    tasks = {t.task_id for t in master.tasks}
    try:
        cleaned = [{"agents": [int(a) for a in c["agents"]], "tasks": [int(t) for t in c["tasks"]]}
                   for c in clusters]
    except (TypeError, KeyError, ValueError):
        return None
    if any(not set(c["agents"]) <= agents or not set(c["tasks"]) <= tasks for c in cleaned):
        return None
    return cleaned


def _execute_run_details(
    spec: RunSpec,
    master: ScenarioData,
    config: Mapping[str, Any],
    backend: Any,
    adapter: Any,
    reward_adapter: Any = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run allocation and optional reward stages, returning a result row and robot rows.

    Both adapters share the backend timing/timeout rules. The reward stage is
    attempted only after successful allocation validation; analytics evaluation
    runs after the independently measured adapter pipeline.
    """
    solver_scenario = copy.deepcopy(master)            # prep: outside every timer
    total = Stopwatch()
    total.start()
    allocation_outcome = backend.run(adapter, solver_scenario, spec.algorithm_seed, spec.time_limit_seconds)
    validation = _extract_assignments(allocation_outcome, master) if allocation_outcome.completed else None
    reward_outcome = None
    reward_rows: List[Dict[str, Any]] = []
    gini_index: Optional[float] = None
    invalid_reward_flag = False
    reward_error = ""
    if validation is not None and validation.valid and reward_adapter is not None:
        # Local import keeps spawned allocation workers light; this typed context
        # carries the validated allocation into the reward worker.
        from src.analytics.reward_adapters import (
            RewardStageInput,
            extract_robot_reward_pairs,
            robot_reward_records,
        )
        reward_seed = spec.reward_seed if spec.reward_seed is not None else seeding.reward_seed(
            spec.scenario_seed, config["seeds"]["reward_stage_label"]
        )
        reward_limit = float(config["time_limits"]["reward_stage_seconds"])
        stage_input = RewardStageInput(
            scenario=copy.deepcopy(master), assignments=validation.normalized,
            options=dict(config["reward"]), clusters=_extract_clusters(allocation_outcome, master),
        )
        reward_outcome = backend.run(reward_adapter, stage_input, reward_seed, reward_limit)

    if reward_outcome is not None:
        if reward_outcome.completed:
            try:
                pairs = extract_robot_reward_pairs(
                    reward_outcome.payload, [agent.agent_id for agent in master.agents]
                )
                gini_result = compute_robot_reward_gini(pairs)
                invalid_reward_flag = gini_result.invalid_reward_flag
                gini_index = gini_result.gini_index
                # The logger does not store NaN/Inf reward cells. Preserve the
                # all-or-none group rule: incomplete numeric data creates no rows.
                if all(math.isfinite(float(value)) for _, value in pairs):
                    reward_rows = robot_reward_records(
                        pairs, config["experiment"]["id"], spec.scenario_id, spec.algorithm
                    )
            except (TypeError, ValueError, OverflowError) as exc:
                gini_index = None
                invalid_reward_flag = False
                reward_error = f"reward validation failed: {exc}"
        else:
            reward_error = reward_outcome.error or "reward solver did not complete"

    # Independently time the complete allocation/reward pipeline, including
    # reward extraction and Gini. The allocation evaluator and persistence are
    # deliberately outside this total.
    total_seconds = total.stop()

    evaluation = None
    if validation is not None and validation.valid:    # excluded from all timers (design option P1)
        try:
            evaluation = evaluate_allocation(master, validation.normalized, config["evaluation"]["energy_penalty"])
        except Exception as exc:
            raise HarnessInternalError(
                f"evaluation of a validated plan failed for {spec.scenario_id}/{spec.algorithm}: {exc!r}"
            ) from exc

    success = evaluation is not None
    if not allocation_outcome.completed:
        message = allocation_outcome.error or ""
    elif not validation.valid:
        message = "invalid allocation: " + "; ".join(validation.errors)
    else:
        message = reward_error
    record = {
        "experiment_id": config["experiment"]["id"],
        "scenario_id": spec.scenario_id,
        "algorithm": spec.algorithm,
        "num_robots": spec.num_robots,
        "num_tasks": spec.num_tasks,
        "total_entities": spec.num_robots + spec.num_tasks,
        "random_seed": spec.scenario_seed,
        "allocation_cost": evaluation.allocation_cost if success else None,
        "allocation_runtime_seconds": (allocation_outcome.elapsed_seconds if allocation_outcome.started
                                        else float("nan")),
        "reward_runtime_seconds": (reward_outcome.elapsed_seconds
                                    if reward_outcome is not None and reward_outcome.started else float("nan")),
        "total_runtime_seconds": total_seconds,
        "allocation_stage_completed": bool(allocation_outcome.completed),
        "reward_stage_completed": bool(reward_outcome is not None and reward_outcome.completed),
        "gini_index": gini_index,
        "invalid_reward_flag": invalid_reward_flag,
        "energy_feasible": evaluation.energy_feasible if success else None,
        "coalition_stability": None,
        "success": success,
        "error_message": _bounded(message),
    }
    return record, reward_rows


def execute_run(
    spec: RunSpec,
    master: ScenarioData,
    config: Mapping[str, Any],
    backend: Any,
    adapter: Any,
) -> Dict[str, Any]:
    """Compatibility entry point for allocation-only callers and tests."""
    return _execute_run_details(spec, master, config, backend, adapter)[0]


def execute_run_with_rewards(
    spec: RunSpec,
    master: ScenarioData,
    config: Mapping[str, Any],
    backend: Any,
    adapter: Any,
    reward_adapter: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute allocation and reward stages; disabled/unavailable adapters are not called."""
    return _execute_run_details(spec, master, config, backend, adapter, reward_adapter)


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

def preflight(
    config: Mapping[str, Any],
    experiment_id: str,
    results_dir: Path,
    resume: bool,
    dry_run: bool = False,
) -> Set[Tuple[str, str, str]]:
    """Protect existing data. Returns the completed keys (experiment, scenario, algorithm).

    Aborts (``PreflightError``) if: metadata exists without ``--resume``; rows of
    this experiment exist without metadata; the stored configuration differs on
    resume; or ANY orphan reward row exists for this experiment (a reward row
    without a main record), regardless of the current plan or filters. Orphans
    are never ignored or deleted.
    """
    metadata_path = results_dir / f"metadata_{experiment_id}.yaml"
    results_path = results_dir / "benchmark_results.csv"
    rewards_path = results_dir / "robot_rewards.csv"

    if metadata_path.exists():
        if not resume and not dry_run:      # a dry run writes nothing, so existing metadata is harmless
            raise PreflightError(f"{metadata_path} already exists; use --resume or a new --experiment-id.")
        if resume:
            with open(metadata_path, encoding="utf-8") as handle:
                stored = (yaml.safe_load(handle) or {}).get("config")
            if stored != config:
                raise PreflightError("the stored configuration differs from the current one; cannot resume.")

    completed: Set[Tuple[str, str, str]] = set()
    frame = None
    rewards = None
    inconsistent: List[str] = []
    try:
        if results_path.exists():
            frame = load_benchmark_results(results_path)
            frame = frame[frame["experiment_id"] == experiment_id]
            completed = set(zip(frame["experiment_id"], frame["scenario_id"], frame["algorithm"]))
        orphans: List[Tuple[str, str, str]] = []
        if rewards_path.exists():
            rewards = load_robot_rewards(rewards_path)
            rewards = rewards[rewards["experiment_id"] == experiment_id]
            keys = set(zip(rewards["experiment_id"], rewards["scenario_id"], rewards["algorithm"]))
            orphans = sorted(keys - completed)
        if resume and frame is not None and not frame.empty:
            reward_groups = {}
            if rewards is not None:
                reward_groups = {
                    key: len(group)
                    for key, group in rewards[rewards["experiment_id"] == experiment_id].groupby(
                        ["experiment_id", "scenario_id", "algorithm"], sort=False
                    )
                }
            inconsistent = []
            for _, row in frame.iterrows():
                key = (row["experiment_id"], row["scenario_id"], row["algorithm"])
                count = reward_groups.get(key, 0)
                expected = int(row["num_robots"])
                if row["reward_stage_completed"] and count != expected:
                    inconsistent.append(f"{key}: found {count} of {expected} robot reward records")
                elif not row["reward_stage_completed"] and count:
                    inconsistent.append(f"{key}: {count} reward records exist although the reward stage did not complete")
    except (SchemaValidationError, DuplicateKeyError) as exc:
        raise PreflightError(f"existing result files are invalid: {exc}") from exc

    if completed and not metadata_path.exists():
        raise PreflightError(f"results for '{experiment_id}' exist but {metadata_path} is missing.")
    if completed and not resume and not dry_run:
        raise PreflightError(f"results for '{experiment_id}' already exist; use --resume or a new --experiment-id.")
    if orphans:
        raise PreflightError(
            f"orphan reward rows exist for experiment '{experiment_id}' (no matching main record): {orphans}. "
            "Resolve them manually or use a new --experiment-id; they are never ignored or deleted."
        )
    if resume and inconsistent:
        raise PreflightError("reward records are incomplete or inconsistent; resolve these before resuming: "
                             + "; ".join(inconsistent[:10]))
    return completed


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def _describe_plan(config: Mapping[str, Any], plan: Sequence[RunSpec]) -> List[str]:
    lines = [
        f"experiment '{config['experiment']['id']}': {len(plan)} runs "
        f"({len({s.scenario_id for s in plan})} scenarios x {len({s.algorithm for s in plan})} algorithms), "
        f"base seed {config['experiment']['base_seed']}",
        "time limits (provisional, seconds): "
        + ", ".join(f"{a}={config['time_limits']['per_algorithm'][a]}" for a in config["benchmark"]["algorithms"]),
        "energy penalty: {strategy} (lambda_E={lambda_E}), {state}".format(
            state="confirmed" if str(config["evaluation"]["energy_penalty"].get("status", "")).startswith("confirmed")
            else "provisional", **config["evaluation"]["energy_penalty"]),
        "configuration  robots tasks   N  scenario seeds",
    ]
    by_config: Dict[str, List[RunSpec]] = {}
    for spec in plan:
        by_config.setdefault(spec.config_id, []).append(spec)
    for config_id, specs in by_config.items():
        seeds = sorted({(s.repetition_index, s.scenario_seed) for s in specs})
        lines.append(
            f"{config_id:<14} {specs[0].num_robots:>6} {specs[0].num_tasks:>5} "
            f"{specs[0].num_robots + specs[0].num_tasks:>3}  " + " ".join(str(seed) for _, seed in seeds)
        )
    return lines


def _metadata_extra(
    options: Mapping[str, Any], unavailable: Mapping[str, str], registry: Mapping[str, str]
) -> Dict[str, Any]:
    return {
        "harness": {
            "experiment_id": options["experiment_id"],
            "backend": options["backend"],
            "filters": {
                "only_configs": options["only_configs"],
                "only_algorithms": options["only_algorithms"],
                "repetitions": options["repetitions"],
            },
            "skipped_unavailable_algorithms": dict(unavailable),
            "adapters": dict(registry),         # starts empty; filled with the adapters that contributed results
            "invocations": [],                  # one entry appended per invocation (start, each resume)
            "reward_stage": options.get("reward_stage_status", _reward_stage_status()),
            "reward_adapter": options.get("reward_adapter_reference"),
            "reward_seed_policy": "shared scenario seed + configured reward_stage_label",
            "evaluation_placement": "after the total timer (option P1); D-H4 unresolved",
            "simulator_energy_penalty_factor": 0.0,
            "energy_tolerance_relative": ENERGY_TOLERANCE_RELATIVE,
            "evaluation_sanity_tolerance": SANITY_TOLERANCE,
            "scenario_generation": "ScenarioGenerator defaults: uniform tasks, default agent/task ranges",
            "battery_model": dataclasses.asdict(BatteryDecayModel()),
        }
    }


def _record_invocation(
    results_dir: Path,
    experiment_id: str,
    options: Mapping[str, Any],
    resumed: bool,
    used_adapters: Mapping[str, str],
    unavailable: Mapping[str, str],
    executed: int,
    skipped: int,
    failures: int,
    status: str,
) -> None:
    """Keep the metadata accurate across --skip-unavailable and --resume (append-only history).

    ``extra.harness.adapters`` is rebuilt to list only the algorithms that have contributed result
    rows (with the adapter that produced them) and ``skipped_unavailable_algorithms`` to list the
    unavailable algorithms that still have no results. One entry per invocation is appended to
    ``extra.harness.invocations``; the original config, seeds, time limits, environment and
    creation time are never touched (see logger.update_experiment_metadata).
    """
    try:
        contributing: List[str] = []
        results_path = results_dir / "benchmark_results.csv"
        if results_path.exists():
            frame = load_benchmark_results(results_path)
            contributing = sorted(set(frame[frame["experiment_id"] == experiment_id]["algorithm"]))
        metadata_path = results_dir / f"metadata_{experiment_id}.yaml"
        with open(metadata_path, encoding="utf-8") as handle:
            harness = ((yaml.safe_load(handle) or {}).get("extra") or {}).get("harness") or {}
        adapters = {a: ref for a, ref in (harness.get("adapters") or {}).items() if a in contributing}
        adapters.update({a: ref for a, ref in used_adapters.items() if a in contributing})
        entry = {
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "resume" if resumed else "start",
            "status": status,
            "backend": options["backend"],
            "filters": {"only_configs": options["only_configs"], "only_algorithms": options["only_algorithms"],
                        "repetitions": options["repetitions"]},
            "algorithms_executed": sorted(used_adapters),
            "adapters_used": dict(used_adapters),
            "runs_executed": executed,
            "runs_skipped_already_completed": skipped,
            "failed_runs": failures,
            "unavailable_algorithms": dict(unavailable),
        }
        update_experiment_metadata(
            experiment_id,
            set_values={
                "harness.adapters": adapters,
                "harness.skipped_unavailable_algorithms": {a: r for a, r in unavailable.items()
                                                           if a not in contributing},
            },
            append_values={"harness.invocations": entry},
            results_dir=results_dir,
        )
    except Exception as exc:        # never mask the original outcome or error
        print(f"warning: could not update the experiment metadata: {exc}", file=sys.stderr)


def run_experiment(
    config: Mapping[str, Any],
    config_path: Path,
    options: Mapping[str, Any],
    registry: Mapping[str, str],
) -> Dict[str, Any]:
    """Run (or resume) the experiment; returns a small summary dictionary."""
    results_dir: Path = options["results_dir"]
    experiment_id = options["experiment_id"]
    run_config = copy.deepcopy(dict(config))
    run_config["experiment"] = dict(run_config["experiment"], id=experiment_id)

    plan = build_run_plan(config, config_path, options["only_configs"], options["only_algorithms"],
                          options["repetitions"])
    algorithms = list(dict.fromkeys(spec.algorithm for spec in plan))
    available, unavailable = algorithm_availability(algorithms, registry)
    reward_reference = REWARD_ADAPTER_REGISTRY.get("default")
    reward_callable = None
    reward_available = False
    if config["reward"]["enabled"] and reward_reference:
        try:
            reward_callable = resolve_adapter("reward", {"reward": reward_reference})
            reward_available = True
        except ExecutionError as exc:
            print(f"UNAVAILABLE reward stage: {exc}")
    reward_status = _reward_stage_status(config, reward_available or not config["reward"]["enabled"])

    for line in _describe_plan(run_config, plan):
        print(line)
    for algorithm, reason in unavailable.items():
        print(f"UNAVAILABLE {algorithm}: {reason}")
    if unavailable:
        pending = ", ".join(unavailable)
        print(f"note: unavailable algorithms ({pending}) need registered implementations.")
    elif "MILP" not in algorithms:
        print("note: MILP is registered but not in benchmark.algorithms; add it there to include the exact baseline.")
    print("reward stage: " + reward_status)

    completed = preflight(config, experiment_id, results_dir, options["resume"], options["dry_run"])

    if options["dry_run"]:
        print("dry run: nothing was executed or written.")
        return {"planned": len(plan), "executed": 0, "unavailable": unavailable}

    if unavailable and not options["skip_unavailable"]:
        raise PreflightError(
            "algorithms without an adapter: " + ", ".join(unavailable) + ". Register adapters with "
            "--adapter ALGORITHM=module.path:attribute, or pass "
            "--skip-unavailable to run only the available algorithms."
        )
    if not available:
        raise PreflightError("no algorithm is available to run.")

    plan = [spec for spec in plan if spec.algorithm in available]
    metadata_path = results_dir / f"metadata_{experiment_id}.yaml"
    resumed = metadata_path.exists()
    if not resumed:
        try:
            # "adapters" starts empty: it lists the adapters that actually contributed results and is
            # filled in by _record_invocation() after the runs (so it stays accurate across --resume).
            write_experiment_metadata(
                experiment_id,
                config=config,
                seed_info={"base_seed": config["experiment"]["base_seed"], "derivation": config["seeds"]},
                time_limits=config["time_limits"],
                extra=_metadata_extra(dict(options, reward_stage_status=reward_status,
                                           reward_adapter_reference=reward_reference if reward_available else None),
                                      unavailable, {}),
                results_dir=results_dir,
            )
        except SchemaValidationError as exc:
            raise PersistenceError(f"cannot write the experiment metadata: {exc}") from exc

    backend = SubprocessBackend() if options["backend"] == "subprocess" else InProcessBackend()
    executed = skipped = failures = 0
    used_adapters: Dict[str, str] = {}                  # algorithms that ran in THIS invocation -> adapter reference
    master: Optional[ScenarioData] = None
    master_id: Optional[str] = None
    fingerprint = ""
    outcome_status = "completed"
    try:
        for index, spec in enumerate(plan, start=1):
            key = (experiment_id, spec.scenario_id, spec.algorithm)
            if key in completed:
                skipped += 1
                continue
            if master_id != spec.scenario_id:           # new scenario: generate once, share across algorithms
                master = generate_scenario(config, spec)
                master_id, fingerprint = spec.scenario_id, scenario_fingerprint(master)
            adapter = registry[spec.algorithm] if options["backend"] == "subprocess" else available[spec.algorithm]
            if reward_available:
                reward_adapter = reward_reference if options["backend"] == "subprocess" else reward_callable
                record, reward_rows = execute_run_with_rewards(
                    spec, master, run_config, backend, adapter, reward_adapter
                )
            else:
                record, reward_rows = execute_run(spec, master, run_config, backend, adapter), []
            if scenario_fingerprint(master) != fingerprint:
                raise HarnessInternalError(f"the master scenario {spec.scenario_id} was modified.")
            try:
                # Write robot records first. If the main row then fails, preflight
                # recognizes the reward rows as orphans instead of hiding them.
                if reward_rows:
                    append_robot_rewards(reward_rows, results_dir)
                append_benchmark_results([record], results_dir)
            except (SchemaValidationError, DuplicateKeyError) as exc:
                raise PersistenceError(f"cannot record {spec.scenario_id}/{spec.algorithm}: {exc}") from exc
            executed += 1
            used_adapters[spec.algorithm] = registry[spec.algorithm]
            failures += 0 if record["success"] else 1
            status = (f"ok cost={record['allocation_cost']:.3f} feasible={record['energy_feasible']} "
                      f"gini={record['gini_index']}" if record["success"] else f"FAILED ({record['error_message']})")
            print(f"[{index}/{len(plan)}] {spec.scenario_id} {spec.algorithm}: {status} "
                  f"allocation={record['allocation_runtime_seconds']:.3f}s total={record['total_runtime_seconds']:.3f}s")
    except BaseException as exc:
        outcome_status = f"aborted: {type(exc).__name__}"
        raise
    finally:
        _record_invocation(results_dir, experiment_id, options, resumed, used_adapters, unavailable,
                           executed, skipped, failures, outcome_status)

    summary_path = None
    try:
        summary_path = summarize_experiment(experiment_id, results_dir, options["summary_dir"])
        print(f"summary written: {summary_path}")
    except SchemaValidationError as exc:
        raise PersistenceError(f"cannot aggregate the results: {exc}") from exc
    except ValueError as exc:
        print(f"no summary written: {exc}")
    return {"planned": len(plan), "executed": executed, "skipped_completed": skipped,
            "failed_runs": failures, "unavailable": unavailable, "summary": summary_path}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the benchmark experiment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="experiment configuration (YAML)")
    parser.add_argument("--results-dir", default=None,
                        help="folder for the raw CSV files and metadata (default: the configured data/results/raw)")
    parser.add_argument("--summary-dir", default=None, help="folder for the summary CSV (default: outputs/tables)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the configuration, show the run plan and check existing data; run nothing")
    parser.add_argument("--resume", action="store_true", help="continue an existing experiment (skips finished runs)")
    parser.add_argument("--backend", choices=("subprocess", "inprocess"), default="subprocess",
                        help="subprocess enforces the time limit (default); inprocess does not")
    parser.add_argument("--adapter", action="append", default=[], metavar="ALGORITHM=module.path:attribute",
                        help="register an adapter for an algorithm (repeatable)")
    parser.add_argument("--skip-unavailable", action="store_true",
                        help="run only the algorithms that have an adapter")
    parser.add_argument("--experiment-id", default=None, help="override experiment.id (needed with the options below)")
    parser.add_argument("--only-configs", default=None, help="comma-separated configuration ids")
    parser.add_argument("--only-algorithms", default=None, help="comma-separated algorithm ids")
    parser.add_argument("--repetitions", type=int, default=None, help="override repetitions per configuration")
    return parser.parse_args(argv)


def _parse_registry(entries: Sequence[str], base: Mapping[str, str]) -> Dict[str, str]:
    registry = dict(base)
    for entry in entries:
        algorithm, separator, reference = entry.partition("=")
        if not separator or not algorithm or not reference:
            raise ConfigError(f"--adapter expects ALGORITHM=module.path:attribute, got {entry!r}.")
        registry[algorithm] = reference
    return registry


def main(argv: Optional[Sequence[str]] = None, registry: Optional[Mapping[str, str]] = None) -> int:
    """Entry point. ``registry`` (optional) adds/overrides entries of ``ADAPTER_REGISTRY``."""
    args = parse_args(argv)
    try:
        config_path = Path(args.config)
        config_path = config_path if config_path.is_absolute() else REPO_ROOT / config_path
        config = load_config(config_path)
        results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR
        validate_config(config, config_path, results_dir, check_output_paths=args.results_dir is None)

        only_configs = args.only_configs.split(",") if args.only_configs else None
        only_algorithms = args.only_algorithms.split(",") if args.only_algorithms else None
        known_configs = {c["id"] for c in config["configurations"]}
        if only_configs and not set(only_configs) <= known_configs:
            raise ConfigError(f"--only-configs contains unknown ids: {sorted(set(only_configs) - known_configs)}.")
        if only_algorithms and not set(only_algorithms) <= set(config["benchmark"]["algorithms"]):
            raise ConfigError("--only-algorithms contains ids that are not in benchmark.algorithms.")
        if args.repetitions is not None and args.repetitions < 1:
            raise ConfigError("--repetitions must be >= 1.")
        experiment_id = args.experiment_id or config["experiment"]["id"]
        try:
            validate_experiment_id(experiment_id)
        except SchemaValidationError as exc:
            raise ConfigError(f"--experiment-id: {exc}") from exc
        overridden = bool(only_configs or only_algorithms or args.repetitions is not None)
        if overridden and experiment_id == config["experiment"]["id"]:
            raise ConfigError("--only-configs/--only-algorithms/--repetitions require a different --experiment-id, "
                              "so a partial run is never mistaken for the full experiment.")

        options = {
            "results_dir": results_dir,
            "summary_dir": Path(args.summary_dir) if args.summary_dir else None,
            "experiment_id": experiment_id,
            "dry_run": args.dry_run,
            "resume": args.resume,
            "backend": args.backend,
            "skip_unavailable": args.skip_unavailable,
            "only_configs": only_configs,
            "only_algorithms": only_algorithms,
            "repetitions": args.repetitions,
        }
        run_experiment(config, config_path, options, _parse_registry(args.adapter, {**ADAPTER_REGISTRY, **(registry or {})}))
        return EXIT_OK
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except PreflightError as exc:
        print(f"pre-flight error: {exc}", file=sys.stderr)
        return EXIT_PREFLIGHT
    except (PersistenceError, OSError) as exc:
        print(f"data/persistence error: {exc}", file=sys.stderr)
        return EXIT_DATA
    except (HarnessInternalError, ExecutionError) as exc:
        print(f"internal harness error: {exc}", file=sys.stderr)
        return EXIT_INTERNAL
    except KeyboardInterrupt:
        print("interrupted; the run in progress was not recorded. Use --resume to continue.", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())

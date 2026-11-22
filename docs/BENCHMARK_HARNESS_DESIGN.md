# Benchmark Harness Design (Task 6)

**Workstream:** Benchmarking & Analytics
**Status:** DESIGN ONLY. Nothing in this document is implemented. No Python, configuration, test or other repository file was created or changed for it. This is the only file written.
**Related documents:** `docs/ANALYTICS_BENCHMARKING_PLAN.md` (approved plan, not modified), `configs/experiments_config.yaml` (approved configuration, not modified).
**Repository state inspected:** `main` at `fcc5905 "Simulation module"` plus our uncommitted analytics files.

---

## 0. How to read this document

Every statement is tagged with its evidence level:

| Tag | Meaning |
| :--- | :--- |
| **[V]** | **Verified**: read from the current source files (file and line given), or checked by an in-memory experiment (Section 1.2). |
| **[A]** | **Assumption**: an engineering choice of ours, not confirmed by anyone else. |
| **[B-ALLOC]** | **Blocked** on the Allocation owner's interface or confirmation. |
| **[B-GT]** | **Blocked** on the Game Theory owner's interface or confirmation. |
| **[B-SIM]** | Needs confirmation from the Simulation owner. |

No teammate API is invented here. Where a teammate interface does not exist yet, this document defines only **our own, analytics-side contract** that a future adapter must satisfy, and says so.

---

## 1. What exists today

### 1.1 Verified interfaces

| Item | Location | Facts **[V]** |
| :--- | :--- | :--- |
| Scenario generator | `src/simulation/generator.py:99` `ScenarioGenerator` | `__init__(map_width=500.0, map_height=500.0, seed: Optional[int]=42)` (`:105-114`); private `random.Random(seed)`; `set_seed(seed)` (`:116-119`); `generate_scenario(num_agents, num_tasks, clustered_tasks=False, **kwargs) -> ScenarioData` (`:208-228`). The `**kwargs` reach only the task generators. Agent and task ids are integers starting at 1, so agent ids and task ids overlap numerically. Agents have no capability/skill attribute. |
| Scenario container | `generator.py:13` `ScenarioData` | `map_width, map_height, seed, agents: List[Agent], tasks: List[Task]`; `to_dict`, `from_dict`, `save_to_file`, `load_from_file`. `seed` is `self.seed or 0` (`:225`). |
| Environment | `src/simulation/environment.py:23` | `load_scenario(scenario)` deep-copies agents and tasks (`:45-53`); `reset()` (`:55-59`); `execute_allocation(assignments, energy_violation_penalty_factor=1000.0) -> (AllocationResult, Dict[int, SimulationTrace])` (`:77-185`). |
| Plan execution | `environment.py:94-158` | Calls `reset()` first. For each `agent_id` key in `assignments` it runs the ordered task list. `eta_t` is cumulative travel time **plus** execution time (`:141`). Energy per step = motion + execution (`:128`). Penalty (`:131-134`): if remaining energy `<` step energy, `deficit = step - remaining`, `total_penalty += deficit * factor`, remaining energy is clamped to 0, and execution **continues**. |
| Simulator outputs | `environment.py:171-184` | `total_cost = round(..., 2)` (`:176`); `runtime_seconds=0.0` (`:177`); `solver_name="EnvironmentSimulator"` (`:178`); `metadata` = `priority_completion_cost`, `travel_cost`, `energy_penalty`, each rounded to 2 decimals (`:180-182`). `agent_energy_used` is the **unclamped** sum of step energies (`:128`, `:140`, `:157`). `result.assignments` is the caller's own dict object (`:172`). |
| Core types | `src/core/types.py:19-29`, `:32-41` | `AllocationResult(assignments, task_completion_times, agent_distances, agent_energy_used, total_cost, runtime_seconds, solver_name, metadata)`. `RewardDistribution(payoffs: Dict[int, float], total_coalition_value, solver_name, preference_sensitivity_alpha, runtime_seconds, gini_index, metadata)`. There is no seed field. |
| Agent / Task | `src/core/agent.py:8-34`, `src/core/task.py:9-21` | `Agent(agent_id, position, velocity, energy_capacity, current_energy, usage_penalty, consumption_per_meter, consumption_per_sec)`; `Task(task_id, location, execution_time, priority, ...)`. Both are mutable dataclasses. |
| Allocation / game theory | `src/allocation/__init__.py:1-8`, `src/game_theory/__init__.py:1-9` | **Docstrings only.** No MILP, GA, Greedy, SVCA or reward solver exists. No function or class is defined in either package. |
| Gini | `src/analytics/metrics.py:28` | `compute_robot_reward_gini(robot_rewards: Optional[Iterable[Tuple[Hashable, float]]]) -> GiniResult(gini_index, invalid_reward_flag)`. |
| Logger | `src/analytics/logger.py` | `validate_experiment_id` (`:147`), `append_benchmark_results(records, results_dir=None)` (`:553`), `append_robot_rewards(records, results_dir=None)` (`:582`), `collect_environment_info` (`:626`), `write_experiment_metadata(experiment_id, config, seed_info, time_limits, extra=None, results_dir=None, overwrite=False)` (`:684`), `load_benchmark_results(path=None)` (`:759`), `load_robot_rewards(path=None)` (`:788`), `aggregate_benchmark_results(results, experiment_id=None)` (`:863`), `write_summary_table` (`:965`), `summarize_experiment(experiment_id, results_dir=None, output_dir=None)` (`:990`). File names are fixed to `benchmark_results.csv` and `robot_rewards.csv` inside the folder passed as `results_dir`. |
| Config | `configs/experiments_config.yaml` | Sections `experiment`, `environment`, `benchmark`, `configurations`, `seeds`, `time_limits`, `evaluation`, `output`. It has **no** aggregate output path, **no** reward-stage time limit, **no** task-distribution setting, and **no** adapter registry. |

### 1.2 Experiments performed for this design (in memory only; no files written, no benchmark)

A throw-away Python snippet (run with bytecode writing disabled) exercised the existing simulator. Results **[V]**:

1. **Generator.** Same seed gives an identical scenario. A **second call on the same generator without `set_seed()` produces a different scenario while still reporting the old seed** in `ScenarioData.seed`. Seed `0` is stored as `0`.
2. **Unrounded cost is reconstructable.** With a plan of 10 tasks over 5 robots and `energy_violation_penalty_factor=0.0`, `Σ δ_t·η_t + Σ λ_r·d_r` computed from `task_completion_times` and `agent_distances` plus the scenario's priorities and usage penalties gave `8601.062086802343`; the simulator's `total_cost` was `8601.06` (equal after rounding). `task_completion_times`, `agent_distances` and `agent_energy_used` are **not** rounded.
3. **The penalty factor does not influence the dynamics.** `task_completion_times`, `agent_distances` and `agent_energy_used` were identical for factor `0.0` and `1000.0`.
4. **Energy deficit identity.** With very small batteries and one robot overloaded, `agent_energy_used[r] - initial_current_energy` was `3634.3077`, and the simulator's `metadata["energy_penalty"]` with factor `1.0` was `3634.31`. So per-robot deficit `= max(0, used - initial)` matches the simulator's own arithmetic.
5. **Isolation.** Running two `Environment` instances loaded from one `ScenarioData` did not change the `ScenarioData` (compared via `to_dict()`).
6. **No plan validation in the simulator.** Missing tasks are silently ignored; a task given to two robots yields one completion entry; an unknown agent id or task id raises `KeyError`; robots absent from `assignments` are absent from `agent_distances`; robots with an empty list are present with `0.0`.
7. **Aliasing.** `result.assignments is` the input dict.

### 1.3 What the findings mean for the harness
- Our cost definition can be computed **exactly (unrounded) from the simulator's unrounded outputs**, so the simulator's two-decimal `total_cost` is only used as a sanity check (Section 8).
- The simulator's linear penalty must **not** be combined with our quadratic one. We call it with factor `0.0` and apply the configured penalty ourselves (Section 8).
- Because the simulator does not validate plans, **validation is entirely our job** and must run before the simulator is called (Section 6).
- A fresh generator per scenario is required to make `ScenarioData.seed` truthful (Section 4).

---

## 2. Classification of everything in this design

| Bucket | Items |
| :--- | :--- |
| **A. Implementable now with verified interfaces** | Config loading and validation; SHA-256 seed derivation; scenario generation and per-algorithm isolation; assignment validation; evaluation through `Environment`; unrounded cost reconstruction; energy feasibility and the three penalty strategies; independent timers; record building against the existing logger schema; write order, resume and orphan pre-flight; metadata; aggregation call; the full test suite using **fake** solvers and **fake** reward functions that live only in tests. |
| **B. Needs Allocation-owner interfaces** | Real adapters for `MILP`, `GA`, `Greedy`, `SVCA`: call signature, return type, seed argument, time-limit argument, how a timeout or error is signalled, whether the solver mutates its scenario, picklability (for the subprocess backend), exact `solver_name` labels, SVCA's inner allocator. |
| **C. Needs Game-Theory-owner interfaces** | The real reward solver call; what it needs as input (D6); whether `payoffs` contains only robots or also tasks; whether rewards can be negative; seed argument; any time limit. |
| **D. Provisional engineering assumptions (ours)** | Quadratic penalty with `lambda_E = 1.0`; simulator called with penalty factor `0.0`; energy-feasibility tolerance; 300 s timeouts; uniform task placement; subprocess execution backend; one record write per run; "every task assigned exactly once" as the validity rule; behaviour of an infeasible plan in the simulator. |
| **E. Cannot be implemented until a teammate confirms a definition** | Coalition stability (Game Theory, D3); whether the simulator is endorsed as the shared evaluator and its energy semantics (Simulation owner); the incumbent-solution timeout policy (Allocation owner); the reward-stage time limit (not configured); **where simulator evaluation sits relative to the reward stage and the total timer (D-H4: depends on the reward solver's required inputs, D6)**. |

---

## 3. Overall workflow

### 3.1 Experiment level

```
main()
  1. parse_args                              -> options
  2. load_config + validate_config           -> config (fail fast, nothing written)
  3. build_run_plan                          -> ordered list of RunSpec (config-major, repetition, algorithm)
  4. preflight_checks                        -> existing files, ANY orphan reward row (abort), resume compatibility,
                                                adapter resolution (execution only; skipped for --dry-run)
  5. write_experiment_metadata               -> metadata_<id>.yaml (refuses overwrite; skipped on resume after compatibility check)
  6. for each (configuration, repetition):                       # one scenario
       6a. generate_scenario_instance        -> master scenario (never passed to a solver)
       6b. for each algorithm in config order:                   # sequential
             execute_run  -> persist_run
  7. summarize_experiment                    -> outputs/tables/summary_<id>.csv
```

Loop order is **configuration-major, then repetition, then algorithm** in the order of `benchmark.algorithms`. All four algorithms therefore run on one scenario before the next scenario is generated. The fixed algorithm order is kept as configured **[A]** (no randomization of order).

### 3.2 One run (one algorithm on one scenario)

```
 prep (outside all timers)
   solver_scenario = deepcopy(master)            # independent copy per algorithm
   seeds           = algorithm_seed, reward_seed (derived, deterministic)

 ┌──────────────────────── total timer (perf_counter) ─────────────────────────┐
 │  allocation stage   (in-call timer around the solver call only)             │
 │  hand-off: validate_assignments (pure, cheap)                               │
 │  reward stage       (in-call timer around the reward call only)             │
 └────────────────────────────────────────────────────────────────────────────┘

 after the total timer (outside all timers)   <-- PROVISIONAL placement shown for
                                                  illustration only; UNRESOLVED, see 3.3 / D-H4
   evaluate_allocation   -> A(U), energy feasibility                 (simulator)
   standardize rewards   -> robot rows, Gini                         (analytics)
   build records         -> 19-column main record + reward rows
   persist_run           -> rewards first, then main record
```

The reward stage runs only if the allocation is valid (`success` will be `True`), because the logger enforces `reward_stage_completed = True ⇒ allocation_stage_completed = True ∧ success = True` (Section 5.4 rule 2 of the approved plan).

### 3.3 Where simulator evaluation sits relative to the reward stage and the timers — **UNRESOLVED (D-H4)**

The approved plan (§2.3) excludes metric computation from all three timers. Whether that is achievable depends on what the reward solver needs as input, which is **not yet known** (dependency D6, Game Theory owner). **This document does not assume that the reward solver can operate without simulator evaluation outputs** (completion times, distances, energy used, cost). Two placements are therefore kept open:

| Placement | Order | Consequences |
| :--- | :--- | :--- |
| **P1** | allocation → pure validation → reward stage → *(total timer stops)* → simulator evaluation | Literal reading of plan §2.3 (evaluation outside every timer). Only possible if the reward solver needs **no** simulator-evaluation output. A validated plan cannot make `execute_allocation` fail (no unknown ids, Section 8.1), so an evaluation failure after the reward stage is an internal harness error that aborts the experiment without writing the run (Section 11, F11); this avoids `reward_stage_completed = True` together with `success = False`. |
| **P2** | allocation → pure validation → simulator evaluation → reward stage | Required if the reward solver consumes evaluation outputs. Evaluation time then falls inside the total timer (as part of the hand-off between stages), which **amends plan §2.3** or must be documented as accepted overhead. `success` is known before the reward stage, so F11 does not arise: an evaluation failure is an ordinary failure (`success = False`, reward stage not started). |

**Status:** D-H4 stays **open** until the Game Theory owner states the reward solver's required inputs. No default is chosen. Consequences for implementation:
- Components that are **independent of D-H4** can be implemented now: assignment validation, the simulator evaluation function, cost and energy computation, seeds, timers, record building, logging.
- The **per-run pipeline order** (the position of the evaluation step relative to the reward stage and the total timer) must not be fixed in code until D-H4 is resolved. If implementation of the pipeline starts earlier, the evaluation step must be a single, isolated step that can be moved, and tests must cover both placements.
- Sections 3.2 (diagram), 10 and 11 (F11) describe P1 only as an illustration; they are conditional on D-H4.

---

## 4. Module layout (new files only; nothing existing is changed)

| File | Kind | Content |
| :--- | :--- | :--- |
| `scripts/run_benchmarks.py` | already planned (plan §8) | CLI and orchestration (Section 5). |
| `src/analytics/seeding.py` | new, small | SHA-256 seed derivation. |
| `src/analytics/evaluation.py` | new | Assignment validation, simulator-based evaluation, energy penalty strategies, feasibility, reward standardization. Pure and unit-testable. Imports `src.simulation` (read-only use). |
| `src/analytics/execution.py` | new | Timers, execution backends (in-process and subprocess with hard timeout), adapter protocol and registry loader. |
| `tests/test_seeding.py`, `tests/test_evaluation.py`, `tests/test_execution.py`, `tests/test_run_benchmarks.py`, `tests/fake_adapters.py` | new | See Section 14. |

`src/analytics/metrics.py` and `logger.py` are reused **unchanged**. No file under `src/core`, `src/simulation`, `src/allocation`, `src/game_theory` or `configs` is touched.

> **Decision D-H1.** Four small modules plus the script is the proposed minimum for testability (the existing tests import from `src`, and `scripts/` is not a package). The alternative is to put everything in the script and test through `importlib`; this makes the pure logic harder to test and review.

---

## 5. Orchestration: functions in `scripts/run_benchmarks.py`

Common data structures (plain dataclasses, defined in the script or `execution.py`):

- `RunSpec`: `config_id, num_robots, num_tasks, repetition_index, scenario_id, scenario_seed, algorithm, algorithm_seed, reward_seed, time_limit_seconds`.
- `StageOutcome`: `started: bool`, `completed: bool`, `elapsed_seconds: Optional[float]` (in-call), `error: Optional[str]`, `timed_out: bool`, `payload: Any`.
- `RunResult`: everything needed to build the records (Section 12).

**`parse_args(argv) -> Namespace`**
- Purpose: CLI. Options: `--config` (default `configs/experiments_config.yaml`), `--results-dir` (default `data/results/raw`), `--summary-dir` (default `outputs/tables`, see D-H8), `--resume`, `--dry-run`, `--no-reward`, `--backend {subprocess,inprocess}`, and development filters `--only-configs`, `--only-algorithms`, `--repetitions`.
- Inputs: `argv`. Outputs: namespace. Dependencies: `argparse`.
- Errors: invalid choices exit with usage text. Any filter or override is later recorded in the metadata (`extra`) and requires a distinct `experiment.id` to be used, since the config id is the unique key (a filtered run must not be mistaken for the full experiment).

**`load_config(path) -> dict`**
- Purpose: read the YAML with `yaml.safe_load`.
- Inputs: path. Outputs: plain dict. Dependencies: PyYAML.
- Errors: missing file, YAML error, or non-mapping root -> `ConfigError` (a harness exception derived from `ValueError`), exit code 2.

**`validate_config(config, options) -> None`**
- Purpose: reject a bad configuration before any work. Checks:
  - `experiment.id` passes `logger.validate_experiment_id`; `base_seed` is an int in `[0, seeds.seed_max]`.
  - `environment`: `dimensions == 2`, `width_m > 0`, `height_m > 0`.
  - `benchmark.algorithms` is the list `MILP, GA, Greedy, SVCA` (unique ids; the order is the execution order); `execution_mode == "sequential"`; `same_scenario_for_all_algorithms is True`; `repetitions_per_configuration >= 1`.
  - `configurations`: unique ids; each has **exactly** the keys `id, num_robots, num_tasks`; counts are ints `>= 1`; `num_robots + num_tasks` in `[10, 80]`. A stored total entity count is rejected.
  - `time_limits.per_algorithm` has a positive number for **every** configured algorithm; the `status` is printed as a warning while it starts with `provisional`.
  - `seeds`: method `sha256`, `seed_bytes == 4`, `byte_order == "big"`, templates contain exactly the documented placeholders.
  - `benchmark.scenario_id_format` renders for a sample and yields a non-empty id.
  - `evaluation.energy_penalty.strategy` in `{quadratic, linear, zero}`; `lambda_E` is finite and `>= 0`.
  - `evaluation.coalition_stability.enabled is False`. Any other value is an error (no definition exists).
  - `output` paths equal `<results-dir>/benchmark_results.csv`, `<results-dir>/robot_rewards.csv` and `<results-dir>/`, because the logger fixes the file names.
- Errors: collects all problems and raises one `ConfigError` listing them.

**`build_run_plan(config, options) -> List[RunSpec]`**
- Purpose: expand the configuration into the ordered run list with every seed and id precomputed.
- Dependencies: `seeding.py`. Outputs: `len(configurations) × repetitions × len(algorithms)` specs (320 for the approved config).
- Errors: formatting failures -> `ConfigError`.

**`preflight_checks(plan, options, results_dir) -> PreflightReport`**
- Purpose: protect existing data before running. Checks:
  1. If `metadata_<id>.yaml` exists and `--resume` is absent -> abort.
  2. If `--resume`: the stored metadata `config` must equal the current config (and filters), otherwise abort.
  3. Load existing `benchmark_results.csv` (via `load_benchmark_results`, which already validates rows and duplicate keys) and compute the set of completed keys for this experiment.
  4. Load `robot_rewards.csv` and compute **orphan reward keys** (reward keys whose `experiment_id` equals the current experiment and that have no main record). **If any orphan exists, abort and print every orphan key** — regardless of whether the orphaned run is included in the current run plan, a filtered run (`--only-configs`, `--only-algorithms`, `--repetitions`) or a resumed run. An orphan is never silently ignored, never skipped because its run is "not in this plan", and never deleted; resolving it is a human decision (plan §5.4). The same check also runs under `--dry-run` (read-only) so the problem is reported before a real run.
  5. **Execution only (not `--dry-run`):** resolve every adapter that the run plan needs and verify it is importable (Section 5.5); abort if any adapter cannot be resolved. Under `--dry-run` this step is **skipped and never causes a failure** (see the dry-run rules in Section 13).
- Outputs: `PreflightReport(completed_keys, orphan_keys, resolved_adapters)`; `resolved_adapters` is `None` for a dry run.
- Errors: `PreflightError` (exit code 3). Logger errors (`SchemaValidationError`, `DuplicateKeyError`) propagate unchanged.

**`generate_scenario_instance(config_entry, repetition_index, config) -> (ScenarioData, scenario_id, scenario_seed)`** — specified in Section 6.2.

**`execute_run(spec, master_scenario, adapters, config, options) -> RunResult`**
- Purpose: the whole per-run pipeline of Section 3.2.
- Dependencies: `execution.py`, `evaluation.py`, `metrics.compute_robot_reward_gini`.
- Errors: any solver or reward failure is **recorded**, never raised (Section 11). Only harness-internal anomalies raise `HarnessInternalError`.

**`build_records(spec, run_result, config) -> (main_record, reward_rows)`**
- Purpose: produce the 19-column main record and the robot reward rows exactly in the logger's schemas (Section 12).
- Errors: none expected; the logger's validation is the final check.

**`persist_run(main_record, reward_rows, results_dir) -> None`**
- Purpose: `append_robot_rewards(reward_rows)` first (skipped when the list is empty), then `append_benchmark_results([main_record])`.
- Errors: **never swallowed**. A logger error means a data-integrity problem; the experiment stops with exit code 4 and the orphan rule of plan §5.4 applies.

**`run_experiment(config, options) -> ExperimentSummary`**
- Purpose: steps 3-7 of Section 3.1: loop, skip completed keys when resuming, progress messages, summary generation.
- Errors: `KeyboardInterrupt` finishes cleanly after the current run is either fully persisted or not persisted at all (Section 11, F12).

**`main(argv) -> int`**
- Purpose: entry point; maps exceptions to exit codes (0 success; 2 config; 3 preflight; 4 persistence; 5 internal error; 130 interrupted).

### 5.5 Adapter resolution (in `execution.py`)
**`resolve_adapter(algorithm_id, registry) -> Callable`** imports a `"module:attribute"` string from an in-code registry. It fails with a clear message if the module or attribute is missing. **The registry content is blocked on D1 (Section 7).** Until then, tests inject fake adapters.

---

## 6. Seeding, scenario generation and isolation

### 6.1 Seed derivation (`src/analytics/seeding.py`)

**`derive_seed(message: str) -> int`**
- Purpose: `int.from_bytes(sha256(message.encode("utf-8")).digest()[:4], "big")`, range `0 … 4294967295`, exactly as `configs/experiments_config.yaml -> seeds`.
- Errors: non-`str` -> `TypeError`.

**`scenario_seed(base_seed, config_id, repetition_index) -> int`**, **`algorithm_seed(scenario_seed, algorithm_id) -> int`**, **`reward_seed(scenario_seed, reward_stage_label) -> int`**
- Purpose: fill the templates from the config (`"scenario|{base_seed}|{config_id}|{repetition_index}"`, `"algorithm|{scenario_seed}|{algorithm_id}"`, `"reward|{scenario_seed}|{reward_stage_label}"`). Integers are rendered in plain decimal. Repetition indices are zero-based.
- The templates are read from the config, not hard-coded, so the code stays in step with the documented scheme.

**Test vectors** (computed from the documented rule; they check the scheme, they are not experimental results):

| Message | Seed |
| :--- | :--- |
| `scenario\|42\|C1\|0` | 1593276916 |
| `scenario\|42\|C1\|9` | 383252898 |
| `scenario\|42\|C8\|0` | 906788627 |
| `scenario\|42\|C8\|9` | 2601196458 |
| `algorithm\|1593276916\|MILP` | 4211836502 |
| `algorithm\|1593276916\|GA` | 4089982009 |
| `algorithm\|1593276916\|Greedy` | 4190077063 |
| `algorithm\|1593276916\|SVCA` | 4235073208 |
| `reward\|1593276916\|reward_stage` | 581375934 |

Consequences (from the approved config): the scenario seed is shared by all algorithms on a scenario; each algorithm has its own seed; the reward seed depends only on the scenario seed, so it is the same for every algorithm on that scenario.

### 6.2 Scenario generation

**`generate_scenario_instance(config_entry, repetition_index, config) -> (ScenarioData, scenario_id, scenario_seed)`**
- Purpose: build the master scenario for one `(configuration, repetition)`.
- Method **[V]-based, [A] for task placement**:
  1. `seed = scenario_seed(base_seed, config_id, repetition_index)`.
  2. `generator = ScenarioGenerator(map_width=float(width_m), map_height=float(height_m), seed=seed)` — a **fresh generator per scenario** (not `set_seed` on a shared one). This keeps `ScenarioData.seed == seed`, which a reused generator would not (Section 1.2 item 1).
  3. `generator.generate_scenario(num_agents=num_robots, num_tasks=num_tasks)` — uniform task placement and default agent/task ranges. The config does not choose clustered tasks or custom ranges, so the generator defaults apply **[A]**; this is recorded in the metadata `extra`.
  4. Assert `scenario.seed == seed` and `len(agents) == R`, `len(tasks) == T`.
  5. `scenario_id = scenario_id_format.format(config_id=..., repetition_index=...)`.
- Dependencies: `ScenarioGenerator`, `seeding.py`. Errors: assertion failure -> `HarnessInternalError`.

**`scenario_fingerprint(scenario) -> str`**
- Purpose: SHA-256 of a canonical JSON dump of `scenario.to_dict()` (sorted keys). Used to prove identical scenarios across algorithms, in tests and (optionally) in the metadata/log, not in the CSV schema.

### 6.3 Isolation rules
- The **master scenario** is never given to any solver, reward function or environment.
- Each algorithm receives `copy.deepcopy(master)` made in the prep step (outside the timers).
- The simulator gets its own `Environment` instance per evaluation; `load_scenario` deep-copies again (`environment.py:49-52`).
- Plans returned by solvers are deep-copied into a normalized dict before evaluation, because the simulator echoes the input dict (`environment.py:172`).
- In-process backend: a solver that mutates its copy cannot change the master. Subprocess backend: the copy is pickled, so mutation is also impossible.
- Check: after the run, `scenario_fingerprint(master)` must equal the value taken before. A mismatch raises `HarnessInternalError` (a harness bug, not an algorithm failure).

---

## 7. Allocation solver integration **[B-ALLOC]**

### 7.1 What we define (our contract, not the Allocation owner's)

An **allocation adapter** is a callable owned by analytics:

```
adapter(scenario: ScenarioData, seed: int, time_limit_seconds: float) -> AdapterOutput
AdapterOutput = mapping with
    "assignments"          : Dict[int, List[int]]   (required)  agent_id -> ordered task_ids
    "solver_reported_cost" : float                  (optional)  informational only
    "metadata"             : dict                   (optional)
```

Only `assignments` is required, because everything else (η, distances, energy, cost) is recomputed by our evaluator from the plan. This keeps the coupling to the Allocation owner as small as possible: the harness needs the **plan**, nothing else. If their solver returns `AllocationResult` (`types.py:19-29`), the adapter simply forwards `result.assignments`.

### 7.2 What is blocked
For each of `MILP`, `GA`, `Greedy`, `SVCA` the adapter body needs from the Allocation owner (D1, D7):
- Callable, import path and signature; whether it needs `Environment`, `ScenarioData` or other inputs.
- Whether it takes a seed and a time limit, and how a timeout is signalled (exception, flagged result, or partial plan).
- Policy for a solver that reaches the limit but has an incumbent solution (config comment: **undefined**, pending the Allocation owner).
- Whether the solver mutates its input scenario/agents.
- Whether callables are picklable / importable by a worker process (needed for the subprocess backend).
- Exact algorithm labels (`MILP`, `GA`, `Greedy`, `SVCA` are the config's identifiers).
- Which allocator SVCA uses inside each cluster (D7).

Until those answers exist **no real adapter is written**. The registry loader and fake adapters (tests only) are implementable now.

### 7.3 Execution backends (`src/analytics/execution.py`)

**`InProcessBackend.run(adapter, scenario, seed, time_limit) -> StageOutcome`**
- Calls the adapter directly in the harness process. **Cannot enforce a hard timeout.** Relies on the solver honouring `time_limit_seconds`. Intended for tests and trusted fast solvers.

**`SubprocessBackend.run(adapter_ref, scenario, seed, time_limit) -> StageOutcome`** **[A]**
- Purpose: enforce the configured limit even for a solver that does not stop by itself.
- Mechanism (spawn start method, required on Windows):
  1. Start a worker process; pass the adapter reference (`"module:attr"`), the pickled scenario copy, the seed and the limit.
  2. The worker imports everything, then sets a `ready` event.
  3. The parent starts the **timeout clock when `ready` arrives**, so interpreter and import start-up does not consume the algorithm's budget.
  4. The worker times the adapter call with its own `perf_counter` (this is the `allocation_runtime_seconds` for completed runs) and sends back `(assignments, solver_reported_cost, metadata, in_call_seconds)` or a formatted exception.
  5. If the limit passes without a result, the parent kills the worker **and its child processes** and reports `timed_out=True` with the parent-measured elapsed time.
- Risk: a solver such as PuLP/CBC launches its own child process. Killing only the worker would leave the CBC process running into the next run. The design kills the **process tree** (`taskkill /T /F` on Windows, process group on POSIX) and asks the Allocation owner to also pass the time limit to the underlying solver (defence in depth).
- Errors: worker crash (non-zero exit, no result) -> failure with the exit code; unpicklable inputs -> `HarnessInternalError`.

> **Decision D-H2.** Default backend. Recommendation: `subprocess` for real runs, `inprocess` for tests. Alternative: in-process only, which is simpler but cannot stop a runaway solver and would let one hung algorithm block the whole sequential experiment.

> **Decision D-H3.** What the 300 s limit covers. This design treats it as the limit for the **allocation call** only **[A]**. The config defines no reward-stage limit (Section 9.3).

---

## 8. Assignment validation, evaluation, cost and energy

### 8.1 Assignment validation (`src/analytics/evaluation.py`)

**`validate_assignments(assignments, scenario) -> ValidationResult`**
- Purpose: decide whether a plan is a valid allocation **before** the simulator sees it (the simulator raises `KeyError` on unknown ids and silently accepts missing or duplicate tasks, Section 1.2 item 6).
- Inputs: the raw object returned by the adapter; the master scenario (for the id sets).
- Output: `ValidationResult(valid: bool, errors: List[str], normalized: Optional[Dict[int, List[int]]])`. `normalized` is a deep copy containing **every** robot id, with `[]` for idle robots.
- Rules **[A]**, derived from proposal Eqs. (5)-(6) and the simulator's input type:

| # | Rule | Rationale |
| :--- | :--- | :--- |
| V1 | `assignments` is a `Mapping` | type safety |
| V2 | every key is an `int` (not `bool`) and is the id of a scenario robot | unknown agent would raise `KeyError` in the simulator |
| V3 | every value is a `list` or `tuple` of `int` (not `bool`) | ordered task sequences, Eq. (5) |
| V4 | every task id is the id of a scenario task | unknown task would raise `KeyError` |
| V5 | **no task appears more than once**, within one list or across robots | Eq. (6): at most one robot per task; the simulator would hide a duplicate |
| V6 | **every scenario task appears** | Eq. (6): exactly one robot per task; the simulator would hide a missing task |
| V7 | robots absent from the mapping, or with an empty list, are idle and allowed | normalization |

- Errors: never raises for bad content; it returns `valid=False` with all violated rules (truncated to a bounded message length for `error_message`).
- **[A]** V6 (all tasks must be assigned) follows Eq. (6) literally. If the team accepts partial plans, V6 is the one rule to relax.
- **Not checked here:** energy (Eq. 7) is a feasibility **flag**, not a validity rule (plan §2.2): an energy-infeasible plan is still a successful run.

### 8.2 Evaluation adapter over the existing simulator

**`evaluate_allocation(scenario, normalized_assignments, energy_penalty) -> EvaluationResult`**
- Purpose: compute `A(U)`, `energy_feasible` and diagnostics from a **validated** plan.
- Steps **[V]-based**:
  1. `env = Environment(map_width=scenario.map_width, map_height=scenario.map_height)` — a fresh instance; `env.load_scenario(scenario)`.
  2. `result, _ = env.execute_allocation(copy.deepcopy(normalized_assignments), energy_violation_penalty_factor=0.0)`. Factor `0.0` removes the simulator's linear penalty from its own total, so the two penalty models are never combined.
  3. Unrounded base cost (Section 8.3).
  4. Energy (Section 8.4) and penalty `φ(U)`.
  5. `allocation_cost = base_cost + φ(U)`.
  6. Sanity check: `abs(round(base_cost, 2) - result.total_cost) <= 0.011`. A larger gap means the simulator changed and raises `HarnessInternalError`.
- Output: `EvaluationResult(allocation_cost, base_cost, energy_penalty_value, energy_feasible, remaining_energy: Dict[int, float], deficit: Dict[int, float], task_completion_times, agent_distances)`.
- Dependencies: `Environment` only. **No change to the simulator.**
- Errors: any exception here -> `HarnessInternalError` (Section 3.3), because a validated plan must evaluate.
- `AllocationResult.runtime_seconds` (always `0.0`) and `solver_name` (`"EnvironmentSimulator"`) are ignored.

### 8.3 Cost calculation and the rounding question

**Question from the task:** can the simulator's unrounded objective be reconstructed? **Answer [V]: yes, for the cost terms we use.**

- `task_completion_times`, `agent_distances` and `agent_energy_used` are returned unrounded (`environment.py:147`, `:156-157`). Rounding is applied only to `total_cost` and the three `metadata` values (`:176`, `:180-182`).
- `base_cost = Σ_t δ_t · η_t + Σ_r λ_r · d_r`, with `δ_t` and `λ_r` taken from the scenario's `Task.priority` and `Agent.usage_penalty`. Terms are summed in **sorted id order** so the value is deterministic.
- Limit to document: summing in a different order than the simulator can differ in the last bits (`~1e-12` relative). We do **not** claim bit-equality with the simulator's internal pre-rounding value. Our reconstruction is the canonical definition, and the simulator's rounded total is used only for the sanity check in 8.2.
- The simulator's own penalty term **cannot** be reconstructed unrounded (only its rounded `metadata["energy_penalty"]` is returned), and is not needed because we compute `φ(U)` ourselves.

### 8.4 Energy feasibility and the penalty `φ(U)`

**`compute_remaining_energy(scenario, agent_energy_used) -> Dict[int, float]`**
- `E_r(U) = Agent.current_energy (initial) − agent_energy_used.get(r, 0.0)` for every robot. `current_energy` (not `energy_capacity`) is the initial energy, matching the simulator's own start state (`environment.py:51`, `:113`).
- **[V]** This equals the simulator's depletion arithmetic (Section 1.2 item 4). **[B-SIM]** The Simulation owner should confirm it is the intended definition.

**`compute_energy_feasibility(remaining_energy, tolerance) -> (bool, Dict[int, float])`**
- `deficit_r = max(0, −E_r)`; a deficit smaller than `tolerance_r = 1e-9 · max(1, initial_energy_r)` counts as 0, to absorb floating-point noise at exactly-zero remaining energy. The tolerance is **[A]**.
- `energy_feasible = all(deficit_r == 0)`.

**`energy_penalty_phi(deficit, strategy, lambda_E) -> float`** (replaceable component, plan §2.2)

| Strategy | `φ(U)` |
| :--- | :--- |
| `quadratic` (config default, **provisional**) | `lambda_E · Σ_r deficit_r²` |
| `linear` | `lambda_E · Σ_r deficit_r` |
| `zero` | `0` |

- Unknown strategy -> `ValueError` (config validation already rejects it).
- **Provisional assumptions to keep visible [A]:** the quadratic form and `lambda_E = 1.0` are engineering assumptions pending team confirmation. In **energy units** the deficit can be in the thousands for an overloaded robot (the check in Section 1.2 item 4 produced a deficit of about 3634), so `lambda_E · deficit²` can be orders of magnitude larger than the base cost. This is a consequence of the provisional choice, not a defect of the harness; it must be reviewed when `lambda_E` is confirmed.
- **Simulator behaviour for infeasible plans [V]:** after depletion the simulator keeps executing with battery 0 and reduced speed (`KinematicsModel.compute_effective_velocity`), so `η_t` of later tasks reflects that. `A(U)` for an infeasible plan therefore includes this effect plus `φ(U)`. This is documented, not corrected.
- `energy_feasible` is reported as its **own** column and is independent of the penalty strategy (also correct under `zero`).

---

## 9. Reward stage and Gini **[B-GT]**

### 9.1 What exists
- `RewardDistribution.payoffs: Dict[int, float]` (`types.py:35`) keyed by `agent_id`. Whether it also contains task entities is **unknown** (the proposal defines Shapley values over robots *and* tasks). Because agent ids and task ids are both `1..n`, a dict mixing both cannot be disentangled by id alone.
- No reward solver exists, no reward seed argument is defined, and the type has no seed field.

### 9.2 What we define (our contract, not the Game Theory owner's)

A **reward adapter** is an analytics-owned callable:
```
reward_adapter(scenario: ScenarioData, validated_assignments: Dict[int, List[int]], seed: int) -> raw reward output
```
Its inputs are **placeholders**: what the real reward solver needs (assignment only? costs? completion times?) is dependency D6 and is not decided here. The design does **not** assume that the reward solver can run without simulator evaluation outputs (completion times, distances, energy used, cost); see D-H4 (Section 3.3). The harness treats the return value as opaque until `standardize_robot_rewards` (below) converts it.

**`standardize_robot_rewards(raw_payoffs, expected_robot_ids) -> RewardAdaptation`** (in `evaluation.py`)
- Purpose: the "completeness check" of plan §2.4 and §5.2, before Gini and before any row is written.
- Input: a `Mapping` from robot id to reward (the verified type of `payoffs`), plus the scenario's robot ids. **Provisional [B-GT]:** it assumes `payoffs` holds robots only, keyed by `agent_id`.
- Rules:
  - the key set must equal `expected_robot_ids` exactly (no missing robot, no extra entity, no duplicates);
  - every value must be a finite real number (not `bool`, not NaN, not infinite);
  - negative values are **kept as they are** (never shifted, clipped or rescaled).
- Output: `RewardAdaptation(ok: bool, rows: List[(robot_id, reward)], reason: Optional[str])`.
- Errors: returns `ok=False` with a reason; it never raises on bad content.

**Gini** — no new implementation. After a successful adaptation the harness calls the existing `compute_robot_reward_gini(rows)` (`metrics.py:28`). The reward solver's own `gini_index` field is ignored. Outcomes follow the approved table (plan §2.4):

| Situation | `gini_index` | `invalid_reward_flag` | reward rows written |
| :--- | :--- | :--- | :--- |
| complete, all rewards `>= 0` | formula / `0` conventions | `False` | yes |
| any negative reward | `NaN` | `True` (+ note in `error_message`) | yes |
| adaptation failed (incomplete, extra, non-finite) | `NaN` | `False` | **no** (+ reason in `error_message`) |
| reward stage did not complete | `NaN` | `False` | no |

### 9.3 Open points
- **Reward-stage time limit:** the config has none. Until one is agreed, the reward stage has **no hard limit** unless the Game Theory owner provides one. A runaway reward solver would stall a sequential experiment. **Decision D-H5.**
- **Reward seed:** derived and available, but the solver must accept it (D2).
- **Negative rewards:** possible or not (D2) decides how often `invalid_reward_flag` occurs.
- **`--no-reward` mode [A]:** an explicit flag that skips the reward stage. Records then have `reward_stage_completed=False`, `reward_runtime_seconds` empty and `gini_index` empty. This lets the allocation half be benchmarked before the reward solver exists. It is recorded in the metadata.

---

## 10. Independent runtime measurement (`src/analytics/execution.py`)

**`Stopwatch`** — thin wrapper around `time.perf_counter` (monotonic), replaceable in tests. No other clock is used.

Per run, three **independent** measurements **[A]** consistent with plan §2.3:

| Field | Measured by | Span |
| :--- | :--- | :--- |
| `allocation_runtime_seconds` | stopwatch around the adapter call, **inside the worker** for the subprocess backend (in-process: around the call) | from handing the scenario to the solver until the plan is returned |
| `reward_runtime_seconds` | stopwatch around the reward adapter call | reward-stage call only |
| `total_runtime_seconds` | a **separate** stopwatch in the parent | from just before the allocation stage starts to just after the reward stage ends |

Rules:
- `total_runtime_seconds` is **measured**, never the sum of the others. For the subprocess backend it includes process start-up and result transfer; the gap between total and the stage times is the pipeline overhead (plan §2.3).
- `AllocationResult.runtime_seconds` and `RewardDistribution.runtime_seconds` are **never used**. The simulator hard-codes `0.0` (`environment.py:177`).
- Excluded from all three timers: scenario generation and deep copies (prep), cost/energy/Gini computation, record building and file I/O. **Simulator evaluation is excluded only under placement P1; under P2 it sits between the stages inside the total timer. This is unresolved (D-H4, Section 3.3).**
- Failure rules follow plan §2.3 exactly: a stage that never starts has runtime `NaN`; a stage that starts and then fails or times out records its measured elapsed time; the total is measured up to the end of the pipeline, including failures, and is `NaN` only if the pipeline never started.
- Timing integrity: runs are sequential and the harness does not run other CPU-heavy work during a run. Machine details are in the metadata.

---

## 11. Failure and timeout handling

Principle: **an algorithm failure is data, a harness failure is an error.** Solver and reward failures are recorded and the experiment continues. Harness/persistence failures stop the experiment.

Columns not listed in a row follow the plan: `coalition_stability` is always empty; `random_seed` is the scenario seed; `energy_feasible` is empty whenever `success` is `False`.

| ID | Situation | `allocation_stage_completed` | `reward_stage_completed` | `success` | `allocation_cost` | `energy_feasible` | Runtimes | Gini / flag | Reward rows | `error_message` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| F0 | Normal run | True | True | True | value | bool | all measured | value / False | yes | empty |
| F1 | Allocation timeout (worker killed) | False | False | False | empty | empty | allocation = elapsed (about the limit); reward empty; total = elapsed | empty / False | no | `timeout after <limit>s` |
| F2 | Allocation raised / worker crashed | False | False | False | empty | empty | allocation = elapsed; reward empty; total = elapsed | empty / False | no | exception type and message (single bounded line) |
| F3 | Adapter returned something that is not a plan | True | False | False | empty | empty | allocation measured; reward empty; total measured | empty / False | no | `invalid allocation: <rules violated>` |
| F4 | Plan violates V2-V6 | True | False | False | empty | empty | as F3 | empty / False | no | `invalid allocation: <rules violated>` |
| F5 | Solver returns an incumbent after reaching the limit | **undefined [B-ALLOC]** | — | — | — | — | — | — | — | treated as **F1** until the Allocation owner defines the policy |
| F6 | Reward solver raised | True | False | True | value | bool | reward = elapsed; total measured | empty / False | no | `reward failed: <message>` |
| F7 | Reward solver timed out (only if a limit is agreed, D-H5) | True | False | True | value | bool | reward = elapsed | empty / False | no | `reward timeout after <limit>s` |
| F8 | Reward returned but adaptation failed (missing/extra robots, non-finite) | True | **True** | True | value | bool | all measured | empty / False | **no** | `reward adaptation failed: <reason>` |
| F9 | Reward has a negative value | True | True | True | value | bool | all measured | `NaN` → empty / **True** | yes | `invalid rewards: negative reward` |
| F10 | `--no-reward` mode | True | False | True | value | bool | reward empty; total = allocation hand-off | empty / False | no | empty |
| F11 | Evaluation of a validated plan raised, or sanity check failed — **placement P1 only (D-H4 unresolved); under P2 this is an ordinary failure: `success` False, reward stage not started** | — | — | — | — | — | — | — | — | **no record written**; `HarnessInternalError`, experiment aborts (exit 5), nothing is orphaned because rewards are written only after evaluation |
| F12 | `KeyboardInterrupt` | — | — | — | — | — | — | — | — | the run in progress is not recorded; a later `--resume` redoes it |
| F13 | Logger raised (`DuplicateKeyError`, `SchemaValidationError`, I/O) | — | — | — | — | — | — | — | — | experiment stops (exit 4); never swallowed; plan §5.4 orphan rules apply |

Every F0-F10 record is **built to satisfy the five cross-field rules of plan §5.4** (checked by tests, enforced again by `append_benchmark_results`). Messages are bounded in length and single-source: the harness never puts a traceback with file paths of other machines into the CSV beyond the exception type and message.

**Timeout clock:** the limit starts when the worker is ready (Section 7.3). In-process backend runs are subject to the solver's own limit only (no F1 via kill).

**Never done:** repairing an invalid plan, accepting a partial plan, retrying a failed run silently, deleting orphan rows.

---

## 12. Records and integration with the existing logger and metadata

**`build_records(...)`** produces, for one run:

- **Main record** with exactly the 19 columns of `BENCHMARK_RESULT_COLUMNS` (`logger.py:42`):
  - `experiment_id` (config), `scenario_id` (Section 6.2), `algorithm`, `num_robots`, `num_tasks`, `total_entities = num_robots + num_tasks` (**calculated**, never read from the config), `random_seed` (scenario seed);
  - `allocation_cost` (`EvaluationResult.allocation_cost`, **unrounded**), the three runtime fields, the two stage-completion flags, `gini_index`, `invalid_reward_flag`, `energy_feasible`, `coalition_stability = None`, `success`, `error_message`.
- **Reward rows** (`experiment_id, scenario_id, algorithm, robot_id, reward`): one per robot from `RewardAdaptation.rows`, with `robot_id = agent_id` (int).

**Persistence order (plan §5.4):** `append_robot_rewards` first, `append_benchmark_results` second, one run at a time. There is no cross-file transaction and no automatic orphan cleanup; both writers are all-or-nothing and atomic. A crash between the two writes can leave orphan reward rows, which `preflight_checks` detects on the next start.

**Metadata** — `write_experiment_metadata(experiment_id, config, seed_info, time_limits, extra, results_dir, overwrite=False)`:
- `config`: the parsed YAML dict (energy-penalty strategy and `lambda_E` included).
- `seed_info`: `{"base_seed": ..., "derivation": <the config's seeds section>}`.
- `time_limits`: the config's `time_limits` section.
- `extra` **[A]**: `harness` details — CLI overrides/filters, backend, `--no-reward`, generator defaults used (task placement mode, agent and task ranges), `BatteryDecayModel` field values, energy-feasibility tolerance, validation rule set version, simulator penalty factor used for evaluation (`0.0`), adapter references. These make the evaluator configuration reproducible.
- The logger already records platform, library versions, `git_commit` and `git_dirty`. A dirty tree is printed as a warning before the first run.
- Written **before** the first run (fail fast on an existing file). On `--resume` it is **not** rewritten; it is compared instead (Section 5, `preflight_checks`).

**Reuse, not duplication:** the harness calls the logger's writers, loaders, metadata writer, aggregation and `metrics.compute_robot_reward_gini`. It adds no second validation of the CSV schema.

---

## 13. Experiment-level execution and aggregation

- **Size:** `8 configurations × 10 repetitions × 4 algorithms = 320` runs on `80` scenarios (approved config). The worst-case wall time is bounded by the timeouts: for MILP alone `80 × 300 s = 24,000 s` (about 6.7 hours) if every run timed out; the same bound applies to each algorithm. This is arithmetic on the configured limits, not a performance claim.
- **Resume (`--resume`)** **[A]:** completed keys `(experiment_id, scenario_id, algorithm)` come from the existing CSV and are skipped. The scenario is still regenerated deterministically, because later algorithms on the same scenario still need it. Resume is refused if the stored configuration differs, and **any** orphan reward row for the experiment aborts the run (Section 5, `preflight_checks` step 4).
- **Dry run (`--dry-run`):** validates the configuration, builds the run plan and displays it (scenario ids, seeds, algorithms, time limits). It **does not require real allocation or reward adapters**: adapter resolution is skipped, so a dry run works while the Allocation and Game Theory interfaces do not yet exist. It may print each adapter as "not resolved (dry run)" for information only, and that never fails the dry run. It calls no solver, generates no scenario that is persisted, and writes nothing. It still performs the read-only pre-flight data checks (existing files, orphan reward rows, resume compatibility). **Real adapter resolution remains mandatory before any actual execution** (Section 5, `preflight_checks` step 5).
- **Progress:** one line per run (scenario id, algorithm, outcome, runtimes). No dependency beyond the standard library is required.
- **Aggregation:** at the end the harness calls `summarize_experiment(experiment_id, results_dir, summary_dir)`; the aggregation rules (eligibility, counts, `ddof=1`, one experiment per call) are already implemented and unchanged. It raises `ValueError` if no records exist, which the harness reports. **Decision D-H8:** the config deliberately has no aggregate output path; the harness uses the logger's default `outputs/tables` and the `--summary-dir` option until the team decides.
- **Not in scope here:** plots and report assets (separate task).

---

## 14. Testing strategy

Principles: no teammate solver or reward solver is needed; all fakes live in `tests/`; every test uses temporary directories; the existing suite must keep passing.

**`tests/fake_adapters.py`** (importable by worker processes): deterministic fakes — round-robin plan, nearest-task plan, invalid plans (duplicate task, missing task, unknown robot/task, wrong type), a solver that raises, a solver that sleeps past a small limit, a solver that mutates its scenario copy, a crashing solver; fake reward functions — equal split, unequal split, negative values, incomplete robot set, extra entities, non-finite values, raising.

| Test group | Checks |
| :--- | :--- |
| `test_seeding` | The nine vectors of Section 6.1; determinism; range; different inputs differ; templates read from config. |
| Scenario | Same `(base, config, repetition)` gives an identical scenario (`scenario_fingerprint`); `ScenarioData.seed == scenario_seed`; fresh generator per scenario; counts match; different repetitions differ. |
| Isolation | A mutating solver does not change the master or another algorithm's copy; plans are deep-copied; environments are independent. |
| `validate_assignments` | Each rule V1-V7 with passing and failing cases; idle robots normalized; result never raises on garbage. |
| `evaluate_allocation` | Unrounded base cost against a hand-computed small scenario; agreement with simulator `total_cost` after rounding; factor `0.0` used (no simulator penalty in the total); deficit identity against `metadata["energy_penalty"]` with a small battery; tolerance behaviour at exactly zero remaining energy; the three penalty strategies against hand calculations; `energy_feasible` independent of strategy; scenario not mutated. |
| `standardize_robot_rewards` | Complete set accepted; missing/extra/duplicate keys, bool, NaN, inf rejected; negative kept as is; ids int. |
| Gini integration | Negative reward -> `NaN` + flag; all-zero -> `0`; uses `compute_robot_reward_gini` (no reimplementation). |
| Timers | Monkeypatched stopwatch: allocation, reward and total are independent; total is not the sum; failure cases leave `NaN` where the plan says. |
| Backends | In-process success/exception; subprocess success, exception, crash, **timeout** (very small limit), process-tree kill, clock starts at `ready` (a slow start does not count). Marked slower; skipped-with-reason if spawning is unavailable. |
| Failure matrix | Each of F0-F10 produces a record that **passes the logger's validation** (including all five cross-field rules) with the expected flags, runtimes and messages; F11 raises and writes nothing; F13 propagates. |
| Persistence | Rewards written before main; orphan after a simulated main-write failure; retry raises `DuplicateKeyError`; nothing is deleted. |
| Orphan pre-flight | Pre-flight aborts on **any** orphan reward row of the current `experiment_id`: for a run in the plan, for a run **outside** a filtered plan, and under `--resume`; every orphan key is printed; orphans of other experiments are not reported as errors; nothing is deleted or ignored. |
| Dry run | `--dry-run` validates the config, builds and displays the plan, and succeeds with **no adapters available**; writes no file; still reports orphans; a real run with an unresolvable adapter aborts before the first solver call. |
| Resume | Completed keys skipped; changed configuration refused; metadata not rewritten. |
| Config validation | Each rule of `validate_config` (bad total outside 10-80, missing timeout, stored total key, wrong algorithm list, coalition enabled, output path mismatch, bad strategy, unsafe id). |
| End-to-end (tiny) | Mini config (e.g. 2 configurations, 2 repetitions) with fake solvers and fake rewards: correct number of rows, identical scenario fingerprints across algorithms, `random_seed` equal across algorithms of a scenario, summary file produced, counts match. |
| Determinism | Two runs with the same config give identical cost, Gini, seeds and ids (runtimes excluded). |
| Existing suite | `python -m unittest discover tests` and `python -m pytest -q` both still pass. |

Python 3.9-compatible syntax; no new dependencies (standard library, numpy/pandas/PyYAML already declared).

---

## 15. Function index (where each lives, readiness)

| Function | File | Bucket |
| :--- | :--- | :--- |
| `parse_args`, `load_config`, `validate_config`, `build_run_plan`, `preflight_checks`, `run_experiment`, `main`, `persist_run`, `build_records` | `scripts/run_benchmarks.py` | A |
| `generate_scenario_instance`, `scenario_fingerprint` | script or `evaluation.py` | A |
| `derive_seed`, `scenario_seed`, `algorithm_seed`, `reward_seed` | `src/analytics/seeding.py` | A |
| `validate_assignments`, `evaluate_allocation`, `compute_remaining_energy`, `compute_energy_feasibility`, `energy_penalty_phi` | `src/analytics/evaluation.py` | A (D: penalty form and tolerance) |
| `standardize_robot_rewards` | `src/analytics/evaluation.py` | A for the logic, **B-GT** for its input assumption |
| `Stopwatch`, `InProcessBackend`, `SubprocessBackend`, `resolve_adapter` | `src/analytics/execution.py` | A (mechanism); registry content **B-ALLOC** |
| Real `MILP`/`GA`/`Greedy`/`SVCA` adapters | `src/analytics/execution.py` registry targets | **B-ALLOC** |
| Real reward adapter | analytics-side | **B-GT** |
| Coalition stability metric | — | **E** (not implemented; column stays empty) |

---

## 16. Decisions requiring your approval

| # | Decision | Recommendation |
| :--- | :--- | :--- |
| D-H1 | New helper modules `seeding.py`, `evaluation.py`, `execution.py` (+ tests) inside `src/analytics/`. | Approve. |
| D-H2 | Default execution backend: subprocess (hard timeout, process-tree kill) vs in-process. | Subprocess for real runs, in-process for tests. |
| D-H3 | The 300 s limit applies to the allocation call only. | Approve. |
| D-H4 | Simulator evaluation after the total timer (P1, literal plan §2.3) vs between the stages inside it (P2). | **UNRESOLVED. No recommendation until the Game Theory owner states the reward solver's required inputs (D6).** Do not assume the reward solver can run without simulator evaluation outputs. |
| D-H5 | Reward stage has no configured limit; add one to the config later. | Ask the Game Theory owner for a limit. |
| D-H6 | Validation rule V6: every task must be assigned. | Approve. |
| D-H7 | Energy-feasibility tolerance `1e-9 · max(1, initial_energy)`. | Approve. |
| D-H8 | Aggregate output path: harness default `outputs/tables` + `--summary-dir`. | Approve until the team decides. |
| D-H9 | Resume/dry-run/no-reward/filter options and writing one record pair per run. | Approve. |
| D-H10 | Solver-reported cost vs our cost: mismatches go to console/log only, not to `error_message`. | Approve (a solver may legitimately optimize a different penalty). |
| D-H11 | Tasks are generated uniformly with generator default ranges (the config is silent). | Confirm with the Simulation owner. |
| D-H12 | Use the simulator as the shared evaluator with factor `0.0` plus our penalty. | Confirm with the Simulation owner (also asked in the earlier interface questions). |

---

## 17. Questions for teammates

**Allocation owner** — callable names and signatures; seed and time-limit arguments; timeout/incumbent policy; whether solvers mutate the scenario; picklability for worker processes; exact algorithm labels; whether the time limit is also passed to the underlying solver; SVCA's inner allocator (D7).
**Game Theory owner** — the reward solver call and its exact required inputs (D6), in particular whether it needs simulator-evaluation outputs (completion times, distances, energy used, cost), which decides D-H4; whether `payoffs` contains only robots; whether rewards can be negative; seed argument; a time limit; the coalition-stability definition (D3).
**Simulation owner** — whether `execute_allocation()` is the intended shared evaluator; that `initial current_energy − agent_energy_used` is the intended feasibility definition; why `Agent.consumption_per_meter/consumption_per_sec` are not used by `Environment`; whether continuing execution at reduced speed after depletion is intended; the seed behaviour of `ScenarioData.seed` when the generator is reused.

---

## 18. Effects on the approved plan (not edited here)

1. Plan §10's "repository observation" about `src/core/types.py` duplication and `BenchmarkRecord` is **stale**: `fcc5905` rewrote `types.py` as a single definition without `BenchmarkRecord`.
2. Plan §2.3 excludes metric computation from the timers; whether it can be kept depends on the reward solver's inputs, so D-H4 is left **unresolved** (Section 3.3).
3. Plan D4 is partly answered by code: `η_t` includes travel plus execution time (`environment.py:141`); `E_r(U)` is derivable as initial energy minus `agent_energy_used`; the simulator's own penalty differs from the provisional quadratic one.
4. Plan D5 is answered: `ScenarioGenerator(seed=...).generate_scenario(R, T)`, with the generator-reuse caveat of Section 1.2.
5. The config has no aggregate output path, no reward-stage time limit, no task-distribution setting and no adapter registry (D-H3, D-H5, D-H8, D-H11).

---

## 19. Suggested implementation order (after approval)

1. `seeding.py` + tests; `evaluation.py` (validation, evaluation, energy, reward standardization) + tests — all against the verified simulator.
2. `execution.py` (stopwatch, in-process backend, then subprocess backend with timeout) + `tests/fake_adapters.py` + tests.
3. `scripts/run_benchmarks.py` skeleton: config loading/validation, run plan, pre-flight, metadata, dry-run; then the per-run pipeline and persistence with fake adapters (the evaluation step kept isolated until D-H4 is resolved); resume; end-to-end tiny experiment test. `--dry-run` must work without any adapters from the start.
4. Real allocation adapters once D1/D7 are answered.
5. Real reward adapter once D2/D6 are answered; remove the provisional robot-only assumption.
6. Coalition stability only after D3.
7. Review `lambda_E` and penalty scale together with the team, then freeze the configuration for the full run.

"""Analytics-owned adapters for Stage-2 reward results.

The current game-theory API returns ``RewardDistribution.payoffs`` keyed by
``CharacteristicFunction.player_ids``. The actual characteristic functions set
those player IDs from ``scenario.agents`` only; tasks contribute value but are
not payoff recipients, even when task IDs overlap robot IDs.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from src.core.types import RewardDistribution
from src.game_theory.solver import PreferenceAwareShapleySolver
from src.simulation.generator import ScenarioData


@dataclass(frozen=True)
class RewardStageInput:
    """Picklable context passed to the reward adapter in a worker process."""

    scenario: ScenarioData
    assignments: Dict[int, List[int]]
    options: Dict[str, Any]


@dataclass(frozen=True)
class _AssignmentView:
    assignments: Dict[int, List[int]]


def extract_robot_reward_pairs(
    reward_output: Any, expected_robot_ids: Sequence[Any]
) -> List[Tuple[Any, Any]]:
    """Extract exactly one payoff for every expected robot, rejecting extras or duplicates.

    Mapping is the current solver contract. An iterable of pairs is also accepted
    by this analytics boundary so duplicate IDs from a future adapter are detected
    before they can be collapsed into a dictionary.
    """
    expected = list(expected_robot_ids)
    if len(set(expected)) != len(expected):
        raise ValueError("scenario contains duplicate robot identifiers")
    payoffs = getattr(reward_output, "payoffs", None)
    if isinstance(payoffs, Mapping):
        pairs = list(payoffs.items())
    elif payoffs is not None:
        try:
            pairs = list(payoffs)
        except TypeError:
            raise ValueError("reward output payoffs must be a mapping or an iterable of pairs") from None
    else:
        raise ValueError("reward output must expose a payoffs mapping")
    ids = []
    normalized = []
    for item in pairs:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"reward payoff must be an (agent_id, value) pair, got {item!r}")
        robot_id, value = item
        ids.append(robot_id)
        normalized.append((robot_id, value))
    if len(set(ids)) != len(ids):
        raise ValueError("reward output contains duplicate robot identifiers")
    expected_set, actual_set = set(expected), set(ids)
    missing, extra = expected_set - actual_set, actual_set - expected_set
    if missing or extra:
        raise ValueError(f"reward robot identifiers mismatch; missing={sorted(map(str, missing))}, "
                         f"extra={sorted(map(str, extra))}")
    by_id = dict(normalized)
    return [(robot_id, by_id[robot_id]) for robot_id in expected]


def robot_reward_records(
    reward_pairs: Iterable[Tuple[Any, Any]], experiment_id: str, scenario_id: str, algorithm: str
) -> List[Dict[str, Any]]:
    """Build records in the logger's existing robot-reward schema."""
    return [{"experiment_id": experiment_id, "scenario_id": scenario_id,
             "algorithm": algorithm, "robot_id": robot_id, "reward": reward}
            for robot_id, reward in reward_pairs]


def preference_aware_reward_adapter(
    stage_input: RewardStageInput, seed: int, time_limit_seconds: float
) -> RewardDistribution:
    """Invoke the existing preference-aware Shapley solver with configured parameters.

    The hard time limit is enforced by ``SubprocessBackend``. The solver's own
    reported runtime is ignored by the harness in favor of backend wall time.
    """
    del time_limit_seconds
    options = dict(stage_input.options)
    solver = PreferenceAwareShapleySolver.from_scenario(
        stage_input.scenario,
        allocation=_AssignmentView(stage_input.assignments),
        value_type=options.get("value_type", "facility"),
        alpha=options.get("preference_sensitivity_alpha", 1.5),
        seed=seed,
        max_log_gamma=options.get("max_log_gamma", 3.0),
        num_samples=options.get("num_samples_Tw", 1000),
        burn_in=options.get("burn_in", 200),
        thin=options.get("thin"),
        samples_per_stratum=options.get("samples_per_stratum", 10),
    )
    distribution = solver.solve(options.get("method", "glauber"), compute_stability=False)
    if not isinstance(distribution, RewardDistribution):
        raise TypeError(f"reward solver returned {type(distribution).__name__}, expected RewardDistribution")
    return distribution

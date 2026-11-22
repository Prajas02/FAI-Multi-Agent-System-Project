"""Evaluation metrics for the benchmarking harness.

Currently implemented:
- Gini index of robot rewards (see docs/ANALYTICS_BENCHMARKING_PLAN.md, Section 2.4).

Owner: Yash Yeole (Benchmarking & Analytics).
"""

import math
from typing import Hashable, Iterable, NamedTuple, Optional, Tuple

import numpy as np


class GiniResult(NamedTuple):
    """Result of :func:`compute_robot_reward_gini`.

    Attributes:
        gini_index: Gini index G of the robot rewards, or ``NaN`` when it is
            not available or not valid.
        invalid_reward_flag: ``True`` only if at least one robot reward is
            negative (Gini is then ``NaN`` by rule).
    """
    gini_index: float
    invalid_reward_flag: bool


def compute_robot_reward_gini(
    robot_rewards: Optional[Iterable[Tuple[Hashable, float]]],
) -> GiniResult:
    """Compute the Gini index of the robot-level rewards of ONE run.

    Input format (standardized robot-level data, one run):
        An iterable of ``(robot_id, reward)`` pairs, i.e. the ``robot_id`` and
        ``reward`` columns of one (experiment_id, scenario_id, algorithm)
        group of ``robot_rewards.csv``. Only robots must be included; task
        entities are not handled here. ``None`` or an empty iterable means
        that rewards are missing.

    Rules (Section 2.4 of the plan):
        * Two or more rewards, all >= 0, total > 0:
          ``G = sum_i sum_j |x_i - x_j| / (2 * R * sum_i x_i)``.
        * All rewards zero: ``(0.0, False)``.
        * Exactly one robot: ``(0.0, False)`` (reporting convention).
        * Any negative reward: ``(NaN, True)``. Rewards are never shifted,
          clipped or rescaled. This is checked before NaN rewards.
        * Otherwise, missing rewards (``None``, empty, or a reward that is
          ``NaN``): ``(NaN, False)``.

    Args:
        robot_rewards: ``(robot_id, reward)`` pairs for one run, or ``None``.

    Returns:
        ``GiniResult(gini_index, invalid_reward_flag)``.

    Raises:
        ValueError: if ``robot_id`` values are duplicated within the run, if a
            pair is malformed, or if a reward is not a real number
            (including +/- infinity). Duplicates are never deduplicated.
    """
    if robot_rewards is None:
        return GiniResult(float("nan"), False)

    ids = []
    values = []
    for item in robot_rewards:
        try:
            robot_id, reward = item
        except (TypeError, ValueError):
            raise ValueError(
                f"Each entry must be a (robot_id, reward) pair, got {item!r}."
            ) from None
        if isinstance(reward, bool) or not isinstance(reward, (int, float, np.integer, np.floating)):
            raise ValueError(
                f"Reward for robot {robot_id!r} must be a real number, got {reward!r}."
            )
        reward = float(reward)
        if math.isinf(reward):
            raise ValueError(f"Reward for robot {robot_id!r} must be finite, got {reward!r}.")
        ids.append(robot_id)
        values.append(reward)

    if len(set(ids)) != len(ids):
        duplicates = sorted({str(i) for i in ids if ids.count(i) > 1})
        raise ValueError(
            "Duplicate robot_id values within one run are not allowed: "
            + ", ".join(duplicates)
        )

    if not values:
        return GiniResult(float("nan"), False)

    x = np.asarray(values, dtype=np.float64)

    # Precedence: a negative reward is checked before a NaN reward.
    if (x < 0).any():
        return GiniResult(float("nan"), True)

    if np.isnan(x).any():
        # An unavailable reward means the run's rewards are incomplete.
        return GiniResult(float("nan"), False)

    if x.size == 1:
        return GiniResult(0.0, False)

    total = float(x.sum())
    if total == 0.0:
        return GiniResult(0.0, False)

    abs_diff_sum = float(np.abs(x[:, None] - x[None, :]).sum())
    gini = abs_diff_sum / (2.0 * x.size * total)
    return GiniResult(gini, False)

"""Reward-fairness and coalition-stability metrics for a payoff vector.

* :func:`gini` - Gini index G in [0, 1] of a payoff vector (0 = perfectly equal).
* :func:`core_stability` - how far a payoff vector is from the core:
  for each coalition S the *excess* is e(S, x) = V(S) - x(S). A payoff vector is in the
  core iff max_S e(S, x) <= 0 (nobody has an incentive to break away).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.game_theory.characteristic import CharacteristicFunction


def gini(payoffs) -> float:
    """Gini index of non-negative payoffs (negative entries are shifted to 0)."""
    x = np.asarray(list(payoffs), dtype=float)
    if x.size == 0:
        return 0.0
    if x.min() < 0:
        x = x - x.min()
    total = x.sum()
    if total <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * x)) / (n * total) - (n + 1.0) / n)


def core_stability(
    value: CharacteristicFunction,
    payoffs,
    num_samples: int = 5000,
    exhaustive_up_to: int = 16,
    tol: float = 1e-9,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Core-excess statistics of a payoff vector (indexed by player index).

    Exhaustive over all 2^n - 2 proper coalitions when n <= ``exhaustive_up_to``,
    otherwise over singletons, all (n-1)-coalitions and ``num_samples`` random
    coalitions of uniformly random size.

    Returns:
        max_excess      - largest V(S) - x(S) (<= 0 means in the core; also the
                          epsilon needed for the epsilon-core)
        relative_excess - max_excess / V(N): scale-free "how far from the core"
        violation_rate  - fraction of checked coalitions with positive excess
        stability_score - 1 - violation_rate  (1.0 = no checked coalition blocks)
        num_checked     - number of coalitions evaluated
        exhaustive      - 1.0 if all coalitions were enumerated
    """
    x = np.asarray(payoffs, dtype=float)
    n = value.n_players
    if x.size != n:
        raise ValueError("payoffs must have one entry per player")
    full = (1 << n) - 1

    def excess(mask: int) -> float:
        members = [i for i in range(n) if (mask >> i) & 1]
        return value.value_of_mask(mask) - x[members].sum()

    if n <= exhaustive_up_to:
        masks = range(1, full)
        exhaustive = True
    else:
        rng = np.random.default_rng(seed)
        picked = {1 << i for i in range(n)} | {full ^ (1 << i) for i in range(n)}
        for _ in range(num_samples):
            size = int(rng.integers(1, n))
            picked.add(sum(1 << int(i) for i in rng.choice(n, size=size, replace=False)))
        masks = sorted(picked)
        exhaustive = False

    ex = np.array([excess(m) for m in masks])
    grand = abs(value.grand_coalition_value()) or 1.0
    if ex.size == 0:
        return {"max_excess": 0.0, "relative_excess": 0.0, "violation_rate": 0.0, "stability_score": 1.0,
                "num_checked": 0.0, "exhaustive": float(exhaustive)}
    rate = float(np.mean(ex > tol))
    return {
        "max_excess": float(ex.max()),
        "relative_excess": float(ex.max()) / grand,
        "violation_rate": rate,
        "stability_score": 1.0 - rate,
        "num_checked": float(ex.size),
        "exhaustive": float(exhaustive),
    }

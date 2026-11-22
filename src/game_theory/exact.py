"""Exact Shapley values, used to validate the approximate solvers.

* :func:`exact_shapley`            - classical Shapley value (Eq. 1) by subset enumeration, n <= 20.
* :func:`exact_preference_shapley` - preference-aware value (Eq. 9) by enumerating all n!
  orderings weighted by mu(sigma), n <= 9.
"""

from __future__ import annotations

from math import factorial
from typing import Optional

import numpy as np

from src.game_theory.characteristic import CharacteristicFunction
from src.game_theory.preference import PreferenceGraph


def _value_table(value: CharacteristicFunction) -> np.ndarray:
    n = value.n_players
    return np.array([value.value_of_mask(m) for m in range(1 << n)])


def exact_shapley(value: CharacteristicFunction, max_players: int = 20) -> np.ndarray:
    """phi_i = sum_S |S|!(n-|S|-1)!/n! [V(S+i) - V(S)]  (classical, uniform mu)."""
    n = value.n_players
    if n > max_players:
        raise ValueError(f"exact_shapley limited to n <= {max_players} (got {n})")
    table = _value_table(value)
    sizes = np.array([bin(m).count("1") for m in range(1 << n)])
    weight = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)])
    masks = np.arange(1 << n)
    phi = np.zeros(n)
    for i in range(n):
        without = masks[(masks >> i) & 1 == 0]
        phi[i] = np.sum(weight[sizes[without]] * (table[without | (1 << i)] - table[without]))
    return phi


def exact_preference_shapley(
    value: CharacteristicFunction,
    pref: PreferenceGraph,
    alpha: float,
    max_players: int = 9,
) -> np.ndarray:
    """phi_i(V) = sum_sigma mu(sigma) [V(S_i^sigma + i) - V(S_i^sigma)]  (Eq. 9)."""
    n = value.n_players
    if n != pref.n:
        raise ValueError("value function and preference graph disagree on n")
    if n > max_players:
        raise ValueError(f"exact_preference_shapley limited to n <= {max_players} (got {n})")
    table = _value_table(value)
    perms, probs = pref.exact_distribution(alpha)                    # (n!, n), (n!,)
    masks = np.cumsum(np.left_shift(1, perms), axis=1)               # prefix masks (disjoint bits)
    prev = np.concatenate([np.zeros((perms.shape[0], 1), dtype=np.int64), masks[:, :-1]], axis=1)
    marg = table[masks] - table[prev]                                # (n!, n) by position
    phi = np.zeros(n)
    np.add.at(phi, perms.ravel(), (marg * probs[:, None]).ravel())
    return phi

"""Monte Carlo estimators for the *classical* (uniform-permutation) Shapley value.

These serve two purposes:
1. the alpha = 0 reference/baseline for the MCMC solver, and
2. a generic, reusable estimator for any cost game J(S) (e.g. the SVCA
   coalition cost of Eq. 11 / the q-permutation estimate of Eq. 12).

:func:`stratified_shapley` stratifies on the *coalition size* seen by player i
(the position of i in a random ordering). Each stratum s in {0..n-1} has equal
weight 1/n in the Shapley formula, so averaging per-stratum means is unbiased and
has lower variance than plain permutation sampling.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import comb
from typing import Optional

import numpy as np

from src.game_theory.characteristic import CharacteristicFunction


@dataclass
class SamplingResult:
    phi: np.ndarray            # (n,) estimates (efficiency-corrected if requested)
    std_error: np.ndarray      # (n,) estimated standard error
    raw_sum_gap: float         # V(N) - sum(phi_raw) before efficiency correction
    num_samples: int


def _mask(indices) -> int:
    m = 0
    for i in indices:
        m |= 1 << int(i)
    return m


def stratified_shapley(
    value: CharacteristicFunction,
    samples_per_stratum: int = 10,
    allocation: str = "uniform",
    enforce_efficiency: bool = True,
    seed: Optional[int] = None,
) -> SamplingResult:
    """Stratified Monte Carlo estimate of the classical Shapley value.

    Args:
        samples_per_stratum: average number of sampled coalitions per (player, size).
            Strata with <= that many subsets are enumerated exactly.
        allocation: ``"uniform"`` (equal samples per stratum) or ``"neyman"``
            (same total budget, but a pilot pass reallocates samples toward
            high-variance strata).
        enforce_efficiency: add the (V(N) - sum phi)/n correction so payoffs sum
            to V(N) exactly (the standard fix for non-permutation estimators).
    """
    if allocation not in ("uniform", "neyman"):
        raise ValueError("allocation must be 'uniform' or 'neyman'")
    rng = np.random.default_rng(seed)
    n = value.n_players
    m = max(int(samples_per_stratum), 2)
    phi = np.zeros(n)
    var_of_mean = np.zeros(n)
    total_samples = 0

    for i in range(n):
        others = np.array([j for j in range(n) if j != i])
        bit_i = 1 << i

        def draw(size: int, k: int) -> np.ndarray:
            out = np.empty(k)
            for q in range(k):
                sub = rng.choice(others, size=size, replace=False) if size else []
                mk = _mask(sub)
                out[q] = value.value_of_mask(mk | bit_i) - value.value_of_mask(mk)
            return out

        strata = []                        # per size: list of marginal samples or ("exact", mean)
        exact_means = {}
        for s in range(n):
            if comb(n - 1, s) <= m:
                vals = [
                    value.value_of_mask(_mask(c) | bit_i) - value.value_of_mask(_mask(c))
                    for c in itertools.combinations(others.tolist(), s)
                ]
                exact_means[s] = float(np.mean(vals))
                strata.append(None)
            else:
                strata.append(draw(s, m if allocation == "uniform" else max(2, m // 2)))
        sampled = [s for s in range(n) if s not in exact_means]

        if allocation == "neyman" and sampled:
            budget = (m - max(2, m // 2)) * len(sampled)
            sd = np.array([strata[s].std(ddof=1) for s in sampled])
            share = sd / sd.sum() if sd.sum() > 0 else np.full(len(sampled), 1 / len(sampled))
            extra = np.floor(share * budget).astype(int)
            for s, k in zip(sampled, extra):
                if k > 0:
                    strata[s] = np.concatenate([strata[s], draw(s, int(k))])

        mean_i, var_i = 0.0, 0.0
        for s in range(n):
            if s in exact_means:
                mean_i += exact_means[s] / n
            else:
                arr = strata[s]
                mean_i += arr.mean() / n
                var_i += arr.var(ddof=1) / len(arr) / n**2
                total_samples += len(arr)
        phi[i] = mean_i
        var_of_mean[i] = var_i

    gap = value.grand_coalition_value() - phi.sum()
    if enforce_efficiency:
        phi = phi + gap / n
    return SamplingResult(phi, np.sqrt(var_of_mean), float(gap), total_samples)


def permutation_shapley(
    value: CharacteristicFunction,
    num_permutations: int = 200,
    antithetic: bool = True,
    seed: Optional[int] = None,
) -> SamplingResult:
    """Plain uniform permutation sampling (Castro et al.), optionally antithetic.

    Each permutation yields one marginal contribution per player and always sums
    to V(N), so the estimator is exactly efficient.
    """
    rng = np.random.default_rng(seed)
    n = value.n_players
    draws = []
    for _ in range(num_permutations):
        perm = rng.permutation(n)
        for p in ((perm, perm[::-1]) if antithetic else (perm,)):
            contrib = np.zeros(n)
            mask, prev = 0, 0.0
            for i in p:
                mask |= 1 << int(i)
                cur = value.value_of_mask(mask)
                contrib[i] = cur - prev
                prev = cur
            draws.append(contrib)
        if antithetic:                       # average the pair -> one independent draw
            draws[-2:] = [(draws[-2] + draws[-1]) / 2.0]
    D = np.array(draws)
    se = D.std(axis=0, ddof=1) / np.sqrt(len(D)) if len(D) > 1 else np.zeros(n)
    return SamplingResult(D.mean(axis=0), se, 0.0, len(D))

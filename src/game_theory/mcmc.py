"""Glauber-dynamics MCMC over the permutation space for preference-aware Shapley values.

Target distribution (Eq. 3):  mu(sigma) ~ exp(alpha * Upsilon(sigma)).

Chain
-----
State: an ordering sigma of the N players. Neighbourhood F(sigma): orderings obtained by
swapping the players at two positions (i, j) - all C(N,2) transpositions by default, or only
pairs at most ``max_gap`` apart (``max_gap=1`` = adjacent swaps). One step picks a position
pair from a state-independent distribution, forms sigma~ and moves to it with the heat-bath
probability of Eq. (13)

    P(sigma -> sigma~) = exp(a*U~) / (exp(a*U) + exp(a*U~)) = sigmoid(a * (U~ - U)),

otherwise stays. The proposal is symmetric (a swap is an involution and the pair law does not
depend on the state), so mu is stationary; transpositions generate S_N so the chain is
irreducible, and it is lazy, hence ergodic. Unrestricted transpositions mix in ~ N log N steps
(adjacent swaps need ~ N^3 log N), which is why they are the default.

Cost
----
Swapping positions i < j only changes the prefix sets of sizes i+1..j, and their log-Gamma
values follow from two cumulative sums, so Delta-Upsilon costs O(N) numpy per step.
The chain's law does not depend on V at all, so V is only evaluated on the *collected*
orderings (via ``CharacteristicFunction.prefix_values``), never inside the chain.

Estimator (Eq. 14):  phi_hat_i = (1/T_w) sum_t Delta(sigma^(t), i). Every ordering telescopes to
V(N) - V(empty), so the estimate sums to V(N) exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import expit

from src.game_theory.characteristic import CharacteristicFunction
from src.game_theory.preference import PreferenceGraph


@dataclass
class MCMCResult:
    phi: np.ndarray                     # (n,) preference-aware Shapley estimate (by player index)
    std_error: np.ndarray               # (n,) batch-means standard error
    acceptance_rate: float
    num_samples: int
    burn_in: int
    thin: int
    upsilon_trace: np.ndarray = field(default_factory=lambda: np.zeros(0))   # Upsilon(sigma^(t))
    marginals: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # (T_w, n) raw draws


class GlauberSampler:
    """Glauber dynamics on orderings with incremental Upsilon updates."""

    def __init__(
        self,
        value: CharacteristicFunction,
        pref: PreferenceGraph,
        alpha: float,
        seed: Optional[int] = None,
        max_gap: Optional[int] = None,
    ):
        if value.n_players != pref.n:
            raise ValueError("value function and preference graph disagree on n")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if max_gap is not None and max_gap < 1:
            raise ValueError("max_gap must be >= 1")
        self.value, self.pref, self.alpha = value, pref, float(alpha)
        self.n = pref.n
        self.max_gap = self.n - 1 if max_gap is None else min(int(max_gap), self.n - 1)
        self.rng = np.random.default_rng(seed)
        self.perm = self.rng.permutation(self.n).astype(np.int64)
        self.s = pref.prefix_log_gamma(self.perm)       # s[k]: log Gamma of first k+1 players
        self.ups = float(np.exp(self.s).sum())
        self.steps = 0
        self.accepted = 0

    # ------------------------------------------------------------------ step
    def _propose_pair(self):
        """State-independent pair (i < j) with j - i <= max_gap."""
        i = int(self.rng.integers(0, self.n - 1))
        j = i + 1 + int(self.rng.integers(0, min(self.max_gap, self.n - 1 - i)))
        return i, j

    def step(self) -> bool:
        """One Glauber update. Returns True if the chain moved."""
        self.steps += 1
        if self.n < 2:
            return False
        i, j = self._propose_pair()
        perm, Wsym = self.perm, self.pref.Wsym
        a, b = int(perm[i]), int(perm[j])
        head = perm[:j]
        ca = np.cumsum(Wsym[a, head])[i:j]               # pair weight of a with each prefix (k=i..j-1)
        cb = np.cumsum(Wsym[b, head])[i:j]
        s_old = self.s[i:j]
        s_new = s_old - ca + cb - Wsym[b, a]             # prefix k holds b instead of a
        d_ups = float((np.exp(s_new) - np.exp(s_old)).sum())
        if self.rng.random() >= expit(self.alpha * d_ups):   # heat-bath acceptance (Eq. 13)
            return False
        perm[i], perm[j] = b, a
        self.s[i:j] = s_new
        self.ups += d_ups
        self.accepted += 1
        return True

    # ------------------------------------------------------------- marginals
    def marginal_contributions(self) -> np.ndarray:
        """Delta(sigma, i) = V(S_i + i) - V(S_i) for every player under the current ordering."""
        pv = self.value.prefix_values(self.perm)
        out = np.empty(self.n)
        out[self.perm] = np.diff(pv, prepend=0.0)        # V(empty) = 0
        return out

    # ------------------------------------------------------------------- run
    def run(
        self,
        num_samples: int = 1000,
        burn_in: int = 200,
        thin: Optional[int] = None,
        num_batches: int = 20,
    ) -> MCMCResult:
        """Burn in, then collect ``num_samples`` orderings spaced ``thin`` steps apart.

        ``burn_in`` and ``thin`` are counted in single Glauber steps. The default thinning is
        N steps (about one transposition per player) to reduce autocorrelation.
        """
        thin = max(self.n, 1) if thin is None else max(int(thin), 1)
        for _ in range(burn_in):
            self.step()
        start_steps, start_acc = self.steps, self.accepted

        draws = np.empty((num_samples, self.n))
        ups_trace = np.empty(num_samples)
        for t in range(num_samples):
            for _ in range(thin):
                self.step()
            draws[t] = self.marginal_contributions()
            ups_trace[t] = self.ups

        phi = draws.mean(axis=0)
        se = _batch_means_se(draws, num_batches)
        acc = (self.accepted - start_acc) / max(self.steps - start_steps, 1)
        return MCMCResult(phi, se, acc, num_samples, burn_in, thin, ups_trace, draws)


def _batch_means_se(draws: np.ndarray, num_batches: int) -> np.ndarray:
    """Standard error of the mean that accounts for autocorrelation (batch means)."""
    T = draws.shape[0]
    B = min(num_batches, T // 2)
    if B < 2:
        return np.full(draws.shape[1], np.nan)
    size = T // B
    means = draws[: B * size].reshape(B, size, -1).mean(axis=1)
    return means.std(axis=0, ddof=1) / np.sqrt(B)


def glauber_preference_shapley(
    value: CharacteristicFunction,
    pref: PreferenceGraph,
    alpha: float,
    num_samples: int = 1000,
    burn_in: int = 200,
    thin: Optional[int] = None,
    seed: Optional[int] = None,
    max_gap: Optional[int] = None,
) -> MCMCResult:
    """Convenience wrapper: build a sampler and run it."""
    return GlauberSampler(value, pref, alpha, seed, max_gap).run(num_samples, burn_in, thin)

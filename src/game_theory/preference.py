"""Pairwise preference graph G = (N, W) and the permutation distribution mu.

Implements Definitions 3 and 4 of the proposal:

    Gamma(W_C)  = exp( gamma_scale * sum_{m != n in C} w_mn )          (Eq. 2)
    Upsilon(s)  = sum_{n=1..N} Gamma(W_{ {s_1..s_n} })                 (Eq. 3)
    mu(s)       ~ exp( alpha * Upsilon(s) )

Notes
-----
* The diagonal of W is ignored (an agent's "preference" for itself is not
  meaningful); w_mn is player m's willingness to cooperate with n.
* ``gamma_scale`` multiplies the exponent. With the default of 1.0 and
  ``max_log_gamma=None`` this is the paper's definition exactly. Because
  Gamma(C) = exp(sum of weights) grows like exp(N), for larger teams
  mu(sigma) ~ exp(alpha * Upsilon) becomes essentially a point mass (the chain
  freezes on the Upsilon-maximising ordering). ``max_log_gamma`` tempers this by
  rescaling so that log Gamma(N) <= max_log_gamma, which keeps alpha a meaningful
  sensitivity knob. If the exponent would overflow float64 the scale is lowered
  automatically. The scale actually used is ``PreferenceGraph.gamma_scale``.
"""

from __future__ import annotations

import itertools
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

_MAX_EXPONENT = 600.0   # exp(600) ~ 4e260, safely inside float64


class PreferenceGraph:
    def __init__(
        self,
        weights: np.ndarray,
        player_ids: Optional[Sequence[int]] = None,
        gamma_scale: Optional[float] = 1.0,
        max_log_gamma: Optional[float] = None,
    ):
        W = np.array(weights, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("weights must be a square matrix")
        if W.min() < 0.0 or W.max() > 1.0:
            raise ValueError("preference weights must lie in [0, 1]")
        np.fill_diagonal(W, 0.0)
        self.W = W
        self.n = W.shape[0]
        self.player_ids = list(range(self.n)) if player_ids is None else list(player_ids)
        if len(self.player_ids) != self.n:
            raise ValueError("player_ids length must match weight matrix")

        total = float(W.sum())          # = sum of all ordered pairs = log Gamma(N) at scale 1
        if gamma_scale is None:
            gamma_scale = 1.0
        if max_log_gamma is not None and total > 0 and gamma_scale * total > max_log_gamma:
            # tempering: cap log Gamma(N) so exp(alpha*Upsilon) is not degenerate
            gamma_scale = max_log_gamma / total
        if gamma_scale * total > _MAX_EXPONENT:
            gamma_scale = _MAX_EXPONENT / total
        self.gamma_scale = float(gamma_scale)
        # symmetrised, scaled pair weight: adding player a to a coalition C raises
        # the exponent by sum_{c in C} Wsym[a, c]
        self.Wsym = self.gamma_scale * (W + W.T)

    # ------------------------------------------------------------ evaluation
    def log_gamma(self, members: Sequence[int]) -> float:
        """log Gamma(W_C) for coalition C given as player indices."""
        idx = np.asarray(list(members), dtype=int)
        if idx.size < 2:
            return 0.0
        return float(self.gamma_scale * self.W[np.ix_(idx, idx)].sum())

    def prefix_log_gamma(self, perm: np.ndarray) -> np.ndarray:
        """Vector s with s[k] = log Gamma of the first k+1 players of ``perm``."""
        perm = np.asarray(perm)
        sub = self.Wsym[np.ix_(perm, perm)]
        return np.tril(sub, k=-1).sum(axis=1).cumsum()

    def upsilon(self, perm: np.ndarray) -> float:
        """Upsilon(sigma) = sum of Gamma over all prefixes of the ordering."""
        return float(np.exp(self.prefix_log_gamma(perm)).sum())

    # ---------------------------------------------------- exact distribution
    def exact_distribution(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """All n! orderings and their probabilities mu (only for tiny n)."""
        if self.n > 9:
            raise ValueError("exact_distribution is limited to n <= 9 (n! orderings)")
        perms = np.array(list(itertools.permutations(range(self.n))), dtype=np.int64)
        ups = self._upsilon_batch(perms)
        logw = alpha * ups
        probs = np.exp(logw - logsumexp(logw))
        return perms, probs

    def _upsilon_batch(self, perms: np.ndarray) -> np.ndarray:
        """Vectorised Upsilon for a (M, n) array of orderings."""
        M, n = perms.shape
        s = np.zeros((M, n))
        for k in range(1, n):
            acc = np.zeros(M)
            for j in range(k):
                acc += self.Wsym[perms[:, k], perms[:, j]]
            s[:, k] = s[:, k - 1] + acc
        return np.exp(s).sum(axis=1)

    # ---------------------------------------------------------- constructors
    @classmethod
    def uniform(cls, n: int, weight: float = 0.0, **kw) -> "PreferenceGraph":
        """Constant preference. weight = 0 makes mu uniform for every alpha."""
        return cls(np.full((n, n), weight), **kw)

    @classmethod
    def random(
        cls, n: int, mean_weight: Optional[float] = None, seed: Optional[int] = None, **kw
    ) -> "PreferenceGraph":
        """Random directed preferences with the given mean (default ~ 1/(n-1)
        so every agent's total out-weight is ~1 regardless of team size)."""
        rng = np.random.default_rng(seed)
        mean = (1.0 / max(n - 1, 1)) if mean_weight is None else mean_weight
        mean = float(np.clip(mean, 1e-6, 0.999))
        # Beta(a, b) with the requested mean
        a = 2.0
        b = a * (1.0 - mean) / mean
        return cls(rng.beta(a, b, size=(n, n)), **kw)

    @classmethod
    def from_agents(
        cls,
        agents: Sequence,
        spatial_weight: float = 0.5,
        skill_weight: float = 0.3,
        noise_weight: float = 0.2,
        mean_weight: Optional[float] = None,
        seed: Optional[int] = None,
        **kw,
    ) -> "PreferenceGraph":
        """Operational affinities derived from the simulation agents.

        An agent prefers partners that (i) start nearby, (ii) have a similar speed
        (compatible operational tempo) and (iii) a personal random component.
        The result is rescaled to the requested mean weight and kept in [0, 1].
        """
        rng = np.random.default_rng(seed)
        n = len(agents)
        pos = np.array([[a.position.x, a.position.y] for a in agents])
        vel = np.array([a.velocity for a in agents])
        d = np.linalg.norm(pos[:, None] - pos[None, :], axis=2)
        spatial = np.exp(-d / max(d.mean(), 1e-9))
        vspan = max(np.ptp(vel), 1e-9)
        skill = 1.0 - np.abs(vel[:, None] - vel[None, :]) / vspan
        noise = rng.random((n, n))
        raw = spatial_weight * spatial + skill_weight * skill + noise_weight * noise
        raw = raw / raw.max()
        mean = (1.0 / max(n - 1, 1)) if mean_weight is None else mean_weight
        off = ~np.eye(n, dtype=bool)
        W = np.clip(raw * (mean / raw[off].mean()), 0.0, 1.0)
        return cls(W, player_ids=[a.agent_id for a in agents], **kw)

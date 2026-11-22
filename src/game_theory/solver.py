"""High-level reward solver: the public entry point for Stage 2.

    solver = PreferenceAwareShapleySolver.from_scenario(scenario, alpha=1.5, seed=0)
    dist = solver.solve("glauber")          # -> src.core.types.RewardDistribution

Methods (``solver_name`` in the returned RewardDistribution):
    "glauber"      GlauberMCMC   preference-aware, any alpha >= 0 (the proposed method)
    "exact"        Exact         preference-aware, brute force over n! orderings (n <= 9);
                                 falls back to the 2^n classical formula when alpha == 0
    "stratified"   Stratified    classical Shapley (alpha ignored), stratified MC
    "permutation"  MonteCarlo    classical Shapley (alpha ignored), antithetic permutation MC
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import numpy as np

from src.core.types import RewardDistribution
from src.game_theory.characteristic import (
    CharacteristicFunction,
    FacilityLocationValue,
    RoutingValue,
    tasks_completed_in,
)
from src.game_theory.exact import exact_preference_shapley, exact_shapley
from src.game_theory.fairness import core_stability, gini
from src.game_theory.mcmc import glauber_preference_shapley
from src.game_theory.preference import PreferenceGraph
from src.game_theory.sampling import permutation_shapley, stratified_shapley

METHODS = ("glauber", "exact", "stratified", "permutation")


class PreferenceAwareShapleySolver:
    def __init__(
        self,
        value: CharacteristicFunction,
        preference: PreferenceGraph,
        alpha: float = 1.5,
        num_samples: int = 1000,
        burn_in: int = 200,
        thin: Optional[int] = None,
        samples_per_stratum: int = 10,
        seed: Optional[int] = 42,
    ):
        if value.n_players != preference.n:
            raise ValueError("value function and preference graph disagree on n")
        self.value, self.pref = value, preference
        self.alpha = float(alpha)
        self.num_samples, self.burn_in, self.thin = num_samples, burn_in, thin
        self.samples_per_stratum = samples_per_stratum
        self.seed = seed

    # ---------------------------------------------------------- constructors
    @classmethod
    def from_scenario(
        cls,
        scenario,
        allocation=None,
        value_type: str = "facility",
        preference: Optional[PreferenceGraph] = None,
        alpha: float = 1.5,
        seed: Optional[int] = 42,
        max_log_gamma: Optional[float] = 3.0,
        **kwargs,
    ) -> "PreferenceAwareShapleySolver":
        """Build the game from a ``ScenarioData``.

        Args:
            allocation: optional ``AllocationResult`` from Stage 1. If given, only
                tasks that allocation actually assigned generate reward.
            value_type: ``"facility"`` (fast, default) or ``"routing"`` (richer, slower).
            preference: preference graph; default is derived from agent geometry/speed.
            max_log_gamma: tempering cap on log Gamma(N) for the *default* preference graph
                (see ``PreferenceGraph``); ``None`` uses the paper's literal Gamma = exp(sum w),
                which becomes degenerate (frozen chain) as the team grows.
        """
        task_ids = tasks_completed_in(allocation) if allocation is not None else None
        games = {"facility": FacilityLocationValue, "routing": RoutingValue}
        if value_type not in games:
            raise ValueError(f"value_type must be one of {sorted(games)}")
        value = games[value_type].from_scenario(scenario, task_ids=task_ids)
        pref = preference or PreferenceGraph.from_agents(
            scenario.agents, seed=seed, max_log_gamma=max_log_gamma)
        return cls(value, pref, alpha=alpha, seed=seed, **kwargs)

    @classmethod
    def from_config(cls, scenario, config: Mapping[str, Any], **kwargs):
        """Use the ``mcmc_reward`` block of ``configs/default_config.yaml``."""
        c = config.get("mcmc_reward", {})
        return cls.from_scenario(
            scenario,
            alpha=c.get("preference_sensitivity_alpha", 1.5),
            num_samples=c.get("num_samples_Tw", 1000),
            burn_in=c.get("burn_in", 200),
            thin=c.get("thin"),
            samples_per_stratum=c.get("samples_per_stratum", 10),
            max_log_gamma=c.get("max_log_gamma", 3.0),
            seed=config.get("seed", 42),
            **kwargs,
        )

    # ----------------------------------------------------------------- solve
    def solve(self, method: str = "glauber", compute_stability: bool = False) -> RewardDistribution:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        v = self.value
        evals0, calls0 = v.num_evaluations, v.num_calls
        meta: Dict[str, Any] = {"num_players": v.n_players, "seed": self.seed,
                                "gamma_scale": self.pref.gamma_scale}
        t0 = time.perf_counter()

        if method == "glauber":
            res = glauber_preference_shapley(
                v, self.pref, self.alpha, self.num_samples, self.burn_in, self.thin, self.seed)
            phi, name = res.phi, "GlauberMCMC"
            meta.update(std_error=res.std_error.tolist(), acceptance_rate=res.acceptance_rate,
                        num_samples=res.num_samples, burn_in=res.burn_in, thin=res.thin,
                        mean_upsilon=float(res.upsilon_trace.mean()))
        elif method == "exact":
            if self.alpha == 0.0 and v.n_players <= 20:
                phi = exact_shapley(v)
            else:
                phi = exact_preference_shapley(v, self.pref, self.alpha)
            name = "Exact"
        elif method == "stratified":
            res = stratified_shapley(v, self.samples_per_stratum, seed=self.seed)
            phi, name = res.phi, "Stratified"
            meta.update(std_error=res.std_error.tolist(), num_samples=res.num_samples,
                        efficiency_gap_before_correction=res.raw_sum_gap)
        else:
            res = permutation_shapley(v, self.num_samples, seed=self.seed)
            phi, name = res.phi, "MonteCarlo"
            meta.update(std_error=res.std_error.tolist(), num_samples=res.num_samples)

        runtime = time.perf_counter() - t0
        meta.update(value_evaluations=v.num_evaluations - evals0,
                    value_calls=v.num_calls - calls0, cache_hit_rate=v.cache_hit_rate)
        if compute_stability:
            meta["stability"] = core_stability(v, phi, seed=self.seed)

        payoffs = {pid: float(p) for pid, p in zip(v.player_ids, phi)}
        return RewardDistribution(
            payoffs=payoffs,
            total_coalition_value=float(v.grand_coalition_value()),
            solver_name=name,
            preference_sensitivity_alpha=self.alpha,
            runtime_seconds=runtime,
            gini_index=gini(phi),
            metadata=meta,
        )

    def compare(self, methods: Iterable[str] = METHODS, **kw) -> Dict[str, RewardDistribution]:
        """Run several methods on the same game (skipping 'exact' when too large)."""
        out = {}
        for m in methods:
            if m == "exact" and self.value.n_players > (20 if self.alpha == 0 else 9):
                continue
            out[m] = self.solve(m, **kw)
        return out


def equal_split(value: CharacteristicFunction) -> RewardDistribution:
    """Baseline for fairness experiments: split V(N) equally."""
    total = value.grand_coalition_value()
    share = total / max(value.n_players, 1)
    return RewardDistribution(
        payoffs={pid: share for pid in value.player_ids}, total_coalition_value=total,
        solver_name="EqualSplit", preference_sensitivity_alpha=0.0, runtime_seconds=0.0,
        gini_index=0.0)

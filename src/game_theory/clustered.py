"""Stage-1 -> Stage-2 link: reward sharing *inside* Stage-1 clusters.

Stage 1 (SVCA / GA clustering) splits the team into clusters ``{"agents": [...], "tasks": [...]}``
(``AllocationResult.metadata["clusters"]``). Here every cluster becomes its own reward game:
only the cluster's agents can earn from the cluster's tasks, and preferences only act inside
the cluster. Merging the cluster payoffs gives one payoff per agent, directly comparable with
the single global game, which answers

    "Does Stage-1 clustering change Stage-2 fairness (Gini) or coalition stability?"

Comparability rules (so differences are due to clustering, not to bookkeeping):
* one common time horizon H (taken from the full scenario) for every game,
* the same preference matrix W and the same tempering scale (sub-matrices of the global graph),
* same method / alpha / sample budget / seed.

Allocators that produce no clusters (Greedy, MILP) are treated as a single cluster.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.core.types import RewardDistribution
from src.game_theory.characteristic import (
    CharacteristicFunction,
    FacilityLocationValue,
    RoutingValue,
    natural_horizon,
    tasks_completed_in,
)
from src.game_theory.fairness import core_stability, gini
from src.game_theory.preference import PreferenceGraph
from src.game_theory.solver import PreferenceAwareShapleySolver

_GAMES = {"facility": (FacilityLocationValue, 1.25), "routing": (RoutingValue, 2.0)}


def _clusters_from(clusters, allocation, scenario) -> List[Dict[str, List[int]]]:
    if clusters is None and allocation is not None:
        clusters = allocation.metadata.get("clusters")
    if not clusters:       # no clustering -> one big cluster (Greedy / MILP / global game)
        clusters = [{"agents": [a.agent_id for a in scenario.agents],
                     "tasks": [t.task_id for t in scenario.tasks]}]
    return [{"agents": list(c["agents"]), "tasks": list(c["tasks"])} for c in clusters]


def _setup(scenario, value_type, preference, seed, max_log_gamma):
    if value_type not in _GAMES:
        raise ValueError(f"value_type must be one of {sorted(_GAMES)}")
    cls, factor = _GAMES[value_type]
    horizon = natural_horizon(scenario.agents, scenario.tasks, factor)
    pref = preference or PreferenceGraph.from_agents(
        scenario.agents, seed=seed, max_log_gamma=max_log_gamma)
    return cls, horizon, pref


def _solver_kwargs(alpha, num_samples, burn_in, thin, samples_per_stratum, seed):
    return dict(alpha=alpha, num_samples=num_samples, burn_in=burn_in, thin=thin,
                samples_per_stratum=samples_per_stratum, seed=seed)


def solve_by_clusters(
    scenario,
    clusters: Optional[Sequence[Mapping[str, Sequence[int]]]] = None,
    allocation=None,
    method: str = "glauber",
    value_type: str = "facility",
    alpha: float = 1.5,
    preference: Optional[PreferenceGraph] = None,
    max_log_gamma: Optional[float] = 3.0,
    num_samples: int = 1000,
    burn_in: int = 200,
    thin: Optional[int] = None,
    samples_per_stratum: int = 10,
    seed: Optional[int] = 42,
    compute_stability: bool = False,
) -> RewardDistribution:
    """Solve one reward game per Stage-1 cluster and merge into a single RewardDistribution.

    Args:
        clusters: list of ``{"agents": [ids], "tasks": [ids]}``. Default: read
            ``allocation.metadata["clusters"]``; if absent, one cluster with everything.
        allocation: optional ``AllocationResult``; if given only tasks it assigned earn reward.
        compute_stability: also report core stability, per cluster and of the *merged* payoff
            vector against the global game (can coalitions spanning clusters block it?).

    Metadata keys: ``clusters`` (per-cluster value / gini / runtime / size), ``within_cluster_gini``
    (size-weighted mean), ``between_cluster_gini`` (Gini after replacing each agent's payoff by its
    cluster mean), ``num_clusters``, ``uncovered_agents`` and, when requested, stability results.
    """
    cls, horizon, pref = _setup(scenario, value_type, preference, seed, max_log_gamma)
    cl_list = _clusters_from(clusters, allocation, scenario)
    allowed = set(tasks_completed_in(allocation)) if allocation is not None else None
    agents_by_id = {a.agent_id: a for a in scenario.agents}
    tasks_by_id = {t.task_id: t for t in scenario.tasks}
    idx_of = {a.agent_id: i for i, a in enumerate(scenario.agents)}
    kw = _solver_kwargs(alpha, num_samples, burn_in, thin, samples_per_stratum, seed)

    payoffs: Dict[int, float] = {a.agent_id: 0.0 for a in scenario.agents}
    cluster_info: List[Dict[str, Any]] = []
    total_value, total_runtime = 0.0, 0.0
    seen: set = set()

    for c_idx, c in enumerate(cl_list):
        agent_ids = [a for a in c["agents"] if a in agents_by_id and a not in seen]
        seen.update(agent_ids)
        task_ids = [t for t in c["tasks"] if t in tasks_by_id and (allowed is None or t in allowed)]
        info = {"index": c_idx, "agents": agent_ids, "tasks": task_ids, "value": 0.0,
                "gini": 0.0, "runtime_seconds": 0.0, "solver": None}
        if agent_ids and task_ids:
            agents = [agents_by_id[a] for a in agent_ids]
            tasks = [tasks_by_id[t] for t in task_ids]
            value = cls(agents, tasks, horizon=horizon)
            sel = [idx_of[a] for a in agent_ids]
            sub_pref = PreferenceGraph(pref.W[np.ix_(sel, sel)], player_ids=agent_ids,
                                       gamma_scale=pref.gamma_scale)
            dist = PreferenceAwareShapleySolver(value, sub_pref, **kw).solve(
                method, compute_stability=compute_stability)
            payoffs.update(dist.payoffs)
            info.update(value=dist.total_coalition_value, gini=dist.gini_index,
                        runtime_seconds=dist.runtime_seconds, solver=dist.solver_name)
            if compute_stability:
                info["stability"] = dist.metadata["stability"]
            total_value += dist.total_coalition_value
            total_runtime += dist.runtime_seconds
        cluster_info.append(info)

    order = [a.agent_id for a in scenario.agents]
    x = np.array([payoffs[a] for a in order])
    sizes = np.array([len(c["agents"]) for c in cluster_info], dtype=float)
    within = (float(np.average([c["gini"] for c in cluster_info], weights=sizes))
              if sizes.sum() > 0 else 0.0)
    mean_by_agent = x.copy()
    for c in cluster_info:
        if c["agents"]:
            ids = [idx_of[a] for a in c["agents"]]
            mean_by_agent[ids] = x[ids].mean()

    meta: Dict[str, Any] = {
        "num_clusters": len(cluster_info), "clusters": cluster_info,
        "within_cluster_gini": within, "between_cluster_gini": gini(mean_by_agent),
        "uncovered_agents": [a for a in order if a not in seen],
        "gamma_scale": pref.gamma_scale, "horizon": horizon, "seed": seed,
    }
    if compute_stability:
        global_value = cls(scenario.agents, scenario.tasks if allowed is None else
                           [t for t in scenario.tasks if t.task_id in allowed], horizon=horizon)
        meta["global_stability_of_merged_payoffs"] = core_stability(global_value, x, seed=seed)

    return RewardDistribution(
        payoffs=payoffs, total_coalition_value=float(total_value),
        solver_name=f"{method}+clusters", preference_sensitivity_alpha=float(alpha),
        runtime_seconds=total_runtime, gini_index=gini(x), metadata=meta)


def compare_global_vs_clustered(scenario, clusters=None, allocation=None, **kwargs) -> Dict[str, Any]:
    """Same game solved globally vs per Stage-1 cluster, with the deltas the report needs.

    Returns a dict with ``global`` and ``clustered`` (both ``RewardDistribution``) plus
    ``gini_delta`` (clustered - global), ``value_retained`` (sum_k V_k(N) / V(N), < 1 means
    decomposition destroys cross-cluster synergy), ``runtime_ratio`` (clustered / global),
    ``payoff_shift`` (mean |x_clustered - x_global| / V(N) per agent) and ``rank_correlation``
    (Spearman of the two payoff vectors).
    """
    cl = solve_by_clusters(scenario, clusters=clusters, allocation=allocation, **kwargs)
    one = [{"agents": [a.agent_id for a in scenario.agents],
            "tasks": [t.task_id for t in scenario.tasks]}]
    gl = solve_by_clusters(scenario, clusters=one, allocation=allocation, **kwargs)
    order = [a.agent_id for a in scenario.agents]
    xg = np.array([gl.payoffs[a] for a in order])
    xc = np.array([cl.payoffs[a] for a in order])
    vg = gl.total_coalition_value
    from scipy.stats import spearmanr
    rho = float(spearmanr(xg, xc).statistic) if len(order) > 2 and xg.std() > 0 and xc.std() > 0 else float("nan")
    return {
        "global": gl, "clustered": cl,
        "gini_delta": cl.gini_index - gl.gini_index,
        "value_retained": cl.total_coalition_value / vg if vg > 0 else float("nan"),
        "runtime_ratio": cl.runtime_seconds / gl.runtime_seconds if gl.runtime_seconds > 0 else float("nan"),
        "payoff_shift": float(np.abs(xc - xg).mean() / vg) if vg > 0 else float("nan"),
        "rank_correlation": rho,
    }

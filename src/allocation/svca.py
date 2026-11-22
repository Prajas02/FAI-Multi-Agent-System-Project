"""EXTENDED SVCA variant ("SVCA-ext"): my own cluster construction, NOT the paper's algorithm.

The paper-faithful version is in svca_paper.py (solver name "SVCA"). This variant keeps the
Shapley-ranking idea but uses a cost-aware coalition model (coalition.py) and a refinement step.

Pipeline
    1. Estimate each entity's Shapley value of the coalition cost J from q sampled
       permutations (Eq. 12). phi_i is the average INCREMENTAL COST of adding entity i,
       so an agent that makes coalitions much cheaper (it "unlocks" nearby tasks) has a
       large negative phi, and a heavy task has a large positive phi.
    2. Rank entities by contribution and build clusters. The proposal only says
       "ranked ... and recursively merged"; this is the concrete reading used here (TO BE
       CHECKED against paper [1], which has not been read yet):
         a. K = ceil(N / target_cluster_size) clusters (capped at R: each needs an agent).
         b. Seeds: K agents chosen by high contribution AND spatial spread.
         c. Remaining agents, best contribution first, join the spatially closest seed cluster
            (count-balanced).
         d. Tasks, heaviest (largest phi) first, each join the cluster where adding it raises J
            least (near an agent that is not already overloaded).
         e. Refinement: single-entity moves between clusters while sum_k J(C_k) decreases.
       Merging order = Shapley ranking. Merge target = marginal coalition cost.
    3. Each cluster is a smaller allocation problem, solved by ``solve_subproblem``
       (identical sub-solver to the GA baseline).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.allocation.base import Allocator
from src.allocation.coalition import CoalitionModel, CoalitionParams, CoalitionState
from src.allocation.evaluator import AllocationEvaluator, Routes
from src.allocation.insertion import solve_subproblem
from src.core.types import AllocationResult
from src.simulation.generator import ScenarioData


@dataclass
class SVCAConfig:
    # --- Eq. 11 parameters (names/defaults follow configs/default_config.yaml: svca) ---
    alpha1: float = 0.05
    alpha2: float = 1.5
    rho: float = 10.0
    num_permutations_q: int = 50
    # --- assumptions / extensions (see coalition.py) ---
    psi_penalty: float = 10.0
    load_weight: float = 1.0
    # --- clustering ---
    target_cluster_size: int = 20  # desired entities (agents + tasks) per sub-problem -> K = ceil(N / target)
    size_slack: float = 0.3        # a cluster may grow to ceil(target * (1 + slack)) entities
    refine_passes: int = 10        # coordinate-descent passes on sum_k J(C_k) after construction (0 = off)
    seed_spread: float = 1.0       # weight of spatial spread vs. Shapley rank when picking seeds
    antithetic: bool = True        # also use each sampled permutation reversed (variance reduction)
    # --- per-cluster solver ---
    improve: bool = True
    local_search_passes: int = 20
    seed: Optional[int] = 42

    @classmethod
    def from_yaml(cls, path: str, **overrides: Any) -> "SVCAConfig":
        """Load the ``svca:`` section (and top-level ``seed``) of configs/default_config.yaml."""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        section = raw.get("svca", {}) or {}
        kwargs: Dict[str, Any] = {k: v for k, v in section.items() if k in cls.__dataclass_fields__}
        if "seed" in raw:
            kwargs["seed"] = raw["seed"]
        kwargs.update(overrides)
        return cls(**kwargs)

    def coalition_params(self) -> CoalitionParams:
        return CoalitionParams(
            alpha1=self.alpha1, alpha2=self.alpha2, rho=self.rho,
            psi_penalty=self.psi_penalty, load_weight=self.load_weight,
        )


@dataclass
class Cluster:
    agent_ids: List[int]
    task_ids: List[int]

    @property
    def size(self) -> int:
        return len(self.agent_ids) + len(self.task_ids)


# ----------------------------------------------------------------------------- Shapley
def estimate_shapley(
    model: CoalitionModel,
    q: int,
    rng: random.Random,
    antithetic: bool = True,
) -> List[float]:
    """Monte-Carlo Shapley estimate  phi_i ~ (1/q) sum_pi [J(S_i^pi + i) - J(S_i^pi)]  (Eq. 12).

    With antithetic sampling each random permutation is also evaluated reversed, so
    q permutations in total. Per permutation the marginals telescope to J(N), hence
    sum_i phi_i == J(N) exactly (efficiency); the unit tests check this.
    """
    n = model.N
    phi = [0.0] * n
    perms: List[List[int]] = []
    base = list(range(n))
    while len(perms) < q:
        p = base[:]
        rng.shuffle(p)
        perms.append(p)
        if antithetic and len(perms) < q:
            perms.append(p[::-1])

    for perm in perms:
        st = model.new_state()
        prev = 0.0
        for e in perm:
            st.add(e)
            c = st.cost()
            phi[e] += c - prev
            prev = c
    return [x / len(perms) for x in phi]


# ----------------------------------------------------------------------------- clustering
def _pick_seeds(model: CoalitionModel, contribution: List[float], k: int, spread: float) -> List[int]:
    """K agents: first the best-contributing one, then trade off contribution vs. distance to chosen seeds."""
    R = model.R
    lo, hi = min(contribution), max(contribution)

    def norm(c: float) -> float:
        return (c - lo) / (hi - lo) if hi > lo else 0.5

    seeds = [max(range(R), key=lambda i: (contribution[i], -i))]
    while len(seeds) < k:
        remaining = [i for i in range(R) if i not in seeds]
        near = {
            i: min(model.agents[i].position.distance_to(model.agents[s].position) for s in seeds)
            for i in remaining
        }
        dmax = max(near.values()) or 1.0
        best = max(remaining, key=lambda i: (norm(contribution[i]) + spread * near[i] / dmax, -i))
        seeds.append(best)
    return seeds


def refine_clusters(
    model: CoalitionModel,
    members: List[List[int]],
    limit: int,
    max_passes: int,
) -> List[List[int]]:
    """Move one entity at a time to the cluster that lowers sum_k J(C_k) most (first sweep order: index).

    A move is rejected if it would leave a cluster that has tasks without an agent, or push a
    cluster past ``limit`` entities. Strictly decreasing, so it terminates.
    """
    R = model.R
    k = len(members)
    members = [list(m) for m in members]
    costs = [model.cost_of(m) for m in members]

    for _ in range(max_passes):
        improved = False
        for e in range(model.N):
            c0 = next(c for c in range(k) if e in members[c])
            rest = [x for x in members[c0] if x != e]
            if e < R and not any(x < R for x in rest) and any(x >= R for x in rest):
                continue                      # would strand tasks without an agent
            cost_rest = model.cost_of(rest)
            best = None
            for c in range(k):
                if c == c0 or len(members[c]) >= limit:
                    continue
                cost_new = model.cost_of(members[c] + [e])
                delta = (cost_rest - costs[c0]) + (cost_new - costs[c])
                if delta < -1e-9 and (best is None or delta < best[0]):
                    best = (delta, c, cost_new)
            if best is not None:
                _, c, cost_new = best
                members[c0], costs[c0] = rest, cost_rest
                members[c].append(e)
                costs[c] = cost_new
                improved = True
        if not improved:
            break
    return members


def build_clusters(model: CoalitionModel, phi: List[float], cfg: SVCAConfig) -> List[Cluster]:
    R, N = model.R, model.N
    target = max(2, cfg.target_cluster_size)
    k = max(1, min(R, math.ceil(N / target)))
    limit = max(math.ceil(N / k), math.ceil(target * (1.0 + cfg.size_slack)))  # hard size limit per cluster

    contribution = [-phi[i] for i in range(R)]       # cost reduction caused by the agent
    seeds = _pick_seeds(model, contribution, k, cfg.seed_spread)

    def pos(e: int):
        return model.agents[e].position if e < R else model.tasks[e - R].location

    states: List[CoalitionState] = [model.new_state() for _ in range(k)]
    cur = [0.0] * k
    n_tasks = [0] * k
    members: List[List[int]] = [[] for _ in range(k)]
    for c, s in enumerate(seeds):
        states[c].add(s)
        members[c].append(s)
        cur[c] = states[c].cost()

    def place(e: int, full: List[bool]) -> int:
        """Join the non-full cluster with the smallest cost increase; ties -> spatially closest."""
        best_c, best_key = 0, None
        for c in range(k):
            trial = states[c].clone()
            trial.add(e)
            near = min(pos(e).distance_to(pos(x)) for x in members[c])
            key = (full[c], round(trial.cost() - cur[c], 9), near, c)
            if best_key is None or key < best_key:
                best_c, best_key = c, key
        states[best_c].add(e)
        members[best_c].append(e)
        cur[best_c] = states[best_c].cost()
        return best_c

    # (1) Agents join the seed they are spatially closest to, best contributor first, with a
    #     balanced-count cap. (With no tasks present yet the marginal cost is identical
    #     everywhere, so the proximity tie-break in place() decides.)
    agent_cap = max(1, math.ceil(math.ceil(R / k) * (1.0 + cfg.size_slack)))
    n_agents = [1] * k
    for e in sorted((i for i in range(R) if i not in seeds), key=lambda i: (-contribution[i], i)):
        c = place(e, [n_agents[c] >= agent_cap for c in range(k)])
        n_agents[c] += 1

    # (2) Tasks, heaviest (largest Shapley cost) first, join the cluster where they raise J
    #     least, i.e. near an agent that is not already overloaded.
    for e in sorted(range(R, N), key=lambda e: (-phi[e], e)):
        place(e, [len(members[c]) >= limit for c in range(k)])

    if cfg.refine_passes > 0 and k > 1:
        members = refine_clusters(model, members, limit, cfg.refine_passes)

    clusters = []
    for c in range(k):
        agent_ids = sorted(model.agents[e].agent_id for e in members[c] if e < R)
        task_ids = sorted(model.tasks[e - R].task_id for e in members[c] if e >= R)
        clusters.append(Cluster(agent_ids, task_ids))
    return clusters


# ----------------------------------------------------------------------------- allocator
class SVCAAllocator(Allocator):
    name = "SVCA-ext"

    def __init__(self, config: Optional[SVCAConfig] = None):
        self.config = config or SVCAConfig()

    def partition(self, scenario: ScenarioData):
        """Stage 1a only: returns (clusters, shapley dict, model). Useful for plots and tests."""
        cfg = self.config
        model = CoalitionModel(scenario, cfg.coalition_params())
        rng = random.Random(cfg.seed)
        phi = estimate_shapley(model, cfg.num_permutations_q, rng, cfg.antithetic)
        clusters = build_clusters(model, phi, cfg)
        return clusters, {model.entity_label(e): phi[e] for e in range(model.N)}, model

    def solve(self, scenario: ScenarioData) -> AllocationResult:
        cfg = self.config
        t0 = time.perf_counter()
        ev = AllocationEvaluator(scenario)

        model = CoalitionModel(scenario, cfg.coalition_params())
        rng = random.Random(cfg.seed)
        phi = estimate_shapley(model, cfg.num_permutations_q, rng, cfg.antithetic)
        t1 = time.perf_counter()

        clusters = build_clusters(model, phi, cfg)
        t2 = time.perf_counter()

        routes: Routes = {}
        for cl in clusters:
            routes.update(
                solve_subproblem(ev, cl.agent_ids, cl.task_ids,
                                 improve=cfg.improve, max_passes=cfg.local_search_passes)
            )
        t3 = time.perf_counter()

        return ev.evaluate(
            routes,
            solver_name=self.name,
            runtime_seconds=t3 - t0,
            metadata={
                "num_clusters": len(clusters),
                "clusters": [{"agents": c.agent_ids, "tasks": c.task_ids} for c in clusters],
                "shapley": {model.entity_label(e): phi[e] for e in range(model.N)},
                "runtime_shapley": t1 - t0,
                "runtime_clustering": t2 - t1,
                "runtime_subsolve": t3 - t2,
                "config": asdict(cfg),
            },
        )

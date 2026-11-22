"""SVCA as specified by Martin et al. (2023), Sec. 3.1 + Sec. 4 (Algorithms 1 and 2).

Algorithm 1 (adaptive Shapley estimate, Eq. 13, 16-18)
    q_ini = ceil(beta * Z_{theta/2}^2), beta = 1/xi^2. With theta=0.1 and xi=0.08 this gives
    q_ini = 423, the value reported in the paper. Then keep adding one random permutation at a
    time and stop once the ranking of the players has not changed for l_max consecutive steps.

Algorithm 2 (clustering)
    Every robot starts as its own cluster ("player"). For Nc = R, R-1, ..., 1:
      1. distribute ALL tasks one by one: take a remaining task and add it to the player with the
         highest |phi|; the merged player's phi is the SUM of the two (so tasks, whose phi is
         negative, level the robots' positive phi);
      2. record the resulting clustering C^s;
      3. merge the two robot-players with the lowest |phi| (phis add), and repeat with Nc-1.
    Finally V(C^s) (Eq. 8) is computed for every s and the best clustering is kept.

Each cluster is then a small allocation problem, solved here by the shared sub-solver
(greedy insertion + local search). The paper uses an exact branch-and-bound solver instead;
the MILP module will provide that.

Ambiguity in the paper: the pseudo-code takes "the task with the LOWEST |phi|" first, while the
prose says the MOST DEMANDING tasks (lowest, i.e. most negative, phi) go to the most capable robots.
``task_order="lowest_abs"`` follows the pseudo-code (default); ``"most_demanding"`` follows the prose.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.allocation.base import Allocator
from src.allocation.evaluator import AllocationEvaluator, Routes
from src.allocation.insertion import solve_subproblem
from src.allocation.paper_model import PaperCoalitionModel, PaperParams
from src.core.types import AllocationResult
from src.simulation.generator import ScenarioData


@dataclass
class PaperSVCAConfig:
    params: PaperParams = field(default_factory=PaperParams)
    # Algorithm 1
    theta: float = 0.1
    xi: float = 0.08
    l_max: int = 10
    q_max: int = 5000                 # safety cap on the adaptive loop
    adaptive: bool = True
    q_fixed: Optional[int] = None     # if adaptive=False: use exactly this many permutations
    # Algorithm 2
    task_order: str = "lowest_abs"    # "lowest_abs" (pseudo-code) | "most_demanding" (prose)
    # sub-solver
    improve: bool = True
    local_search_passes: int = 20
    seed: Optional[int] = 42

    @classmethod
    def from_yaml(cls, path: str, **overrides: Any) -> "PaperSVCAConfig":
        """Read the ``svca_paper:`` section (and top-level ``seed``) of a YAML file."""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        section = dict(raw.get("svca_paper", {}) or {})
        param_keys = set(PaperParams.__dataclass_fields__)
        params = PaperParams(**{k: v for k, v in section.items() if k in param_keys})
        kwargs: Dict[str, Any] = {k: v for k, v in section.items()
                                  if k in cls.__dataclass_fields__ and k != "params"}
        if "seed" in raw:
            kwargs["seed"] = raw["seed"]
        kwargs.update(overrides)
        return cls(params=params, **kwargs)


# ----------------------------------------------------------------------------- Algorithm 1
def initial_q(theta: float, xi: float) -> int:
    """q_ini = ceil(beta * Z^2_{theta/2}) with beta = 1 / xi^2   (Eq. 18)."""
    z = NormalDist().inv_cdf(1.0 - theta / 2.0)
    return math.ceil(z * z / (xi * xi))


def _add_permutation(model: PaperCoalitionModel, rng: random.Random, phi_sum: List[float]) -> None:
    perm = list(range(model.N))
    rng.shuffle(perm)
    state = model.new_state()
    prev = 0.0
    for e in perm:
        state.add(e)
        c = state.cost()
        phi_sum[e] += c - prev              # marginal contribution gamma_i^pi (Eq. 12)
        prev = c


def _ranking(phi: Sequence[float]) -> Tuple[int, ...]:
    return tuple(sorted(range(len(phi)), key=lambda i: (-phi[i], i)))


def estimate_shapley(
    model: PaperCoalitionModel,
    cfg: PaperSVCAConfig,
    rng: random.Random,
) -> Tuple[List[float], int, int]:
    """Algorithm 1. Returns (phi_tilde, q_used, q_initial)."""
    n = model.N
    phi_sum = [0.0] * n

    if not cfg.adaptive:
        q = cfg.q_fixed if cfg.q_fixed else initial_q(cfg.theta, cfg.xi)
        for _ in range(q):
            _add_permutation(model, rng, phi_sum)
        return [x / q for x in phi_sum], q, q

    q_ini = cfg.q_fixed if cfg.q_fixed else initial_q(cfg.theta, cfg.xi)
    q = q_ini
    for _ in range(q_ini):
        _add_permutation(model, rng, phi_sum)
    order = _ranking([x / q for x in phi_sum])
    stable = 0
    while stable < cfg.l_max and q < cfg.q_max:
        q += 1
        _add_permutation(model, rng, phi_sum)
        new_order = _ranking([x / q for x in phi_sum])
        stable = stable + 1 if new_order == order else 0
        order = new_order
    return [x / q for x in phi_sum], q, q_ini


# ----------------------------------------------------------------------------- Algorithm 2
def svca_clusterings(
    num_robots: int,
    num_tasks: int,
    phi: Sequence[float],
    task_order: str = "lowest_abs",
) -> List[List[List[int]]]:
    """All R clusterings C^1..C^R of Algorithm 2 (Nc = R, R-1, ..., 1).

    ``phi`` is indexed by entity: robots 0..R-1, tasks R..R+T-1. Each clustering is a list of
    clusters, each cluster a list of entity indices; every cluster contains at least one robot.
    """
    R, T = num_robots, num_tasks
    if task_order == "lowest_abs":
        tasks = sorted(range(R, R + T), key=lambda e: (abs(phi[e]), e))
    elif task_order == "most_demanding":
        tasks = sorted(range(R, R + T), key=lambda e: (phi[e], e))
    else:
        raise ValueError(f"unknown task_order {task_order!r}")

    base: List[Tuple[List[int], float]] = [([i], phi[i]) for i in range(R)]   # robot players
    clusterings: List[List[List[int]]] = []

    while base:
        players = [(list(m), v) for m, v in base]
        for t in tasks:
            j = max(range(len(players)), key=lambda k: (abs(players[k][1]), -k))
            members, value = players[j]
            members.append(t)
            players[j] = (members, value + phi[t])
        clusterings.append([m for m, _ in players])

        if len(base) == 1:
            break
        order = sorted(range(len(base)), key=lambda k: (abs(base[k][1]), k))
        i, j = order[0], order[1]                      # lowest and second lowest |phi|
        merged = (base[j][0] + base[i][0], base[i][1] + base[j][1])
        base = [p for k, p in enumerate(base) if k not in (i, j)] + [merged]
    return clusterings


# ----------------------------------------------------------------------------- allocator
class PaperSVCAAllocator(Allocator):
    name = "SVCA"

    def __init__(self, config: Optional[PaperSVCAConfig] = None):
        self.config = config or PaperSVCAConfig()

    def partition(self, scenario: ScenarioData):
        """Stage 1a only. Returns (best clustering, info dict, model)."""
        cfg = self.config
        model = PaperCoalitionModel(scenario, cfg.params)
        rng = random.Random(cfg.seed)
        phi, q, q_ini = estimate_shapley(model, cfg, rng)
        clusterings = svca_clusterings(model.R, model.T, phi, cfg.task_order)
        values = [model.value_of_clustering(c) for c in clusterings]
        best = max(range(len(values)), key=lambda s: (values[s], -s))
        info = {
            "phi": phi, "q": q, "q_initial": q_ini, "values": values,
            "num_clusters_per_round": [len(c) for c in clusterings], "best_index": best,
        }
        return clusterings[best], info, model

    def solve(self, scenario: ScenarioData) -> AllocationResult:
        cfg = self.config
        t0 = time.perf_counter()
        ev = AllocationEvaluator(scenario)
        clusters, info, model = self.partition(scenario)
        t1 = time.perf_counter()

        routes: Routes = {a: [] for a in ev.agent_ids}
        cluster_ids = []
        for members in clusters:
            agents = sorted(model.agents[e].agent_id for e in members if e < model.R)
            tasks = sorted(model.tasks[e - model.R].task_id for e in members if e >= model.R)
            cluster_ids.append({"agents": agents, "tasks": tasks})
            if tasks:
                routes.update(solve_subproblem(ev, agents, tasks, improve=cfg.improve,
                                               max_passes=cfg.local_search_passes))
        t2 = time.perf_counter()

        values = info["values"]
        return ev.evaluate(
            routes,
            solver_name=self.name,
            runtime_seconds=t2 - t0,
            metadata={
                "t_clu": t1 - t0,                    # clustering time (paper's t_clu)
                "t_all": t2 - t1,                    # sequential allocation time (paper's t_all)
                "V": values[info["best_index"]],
                "num_clusters": len(clusters),
                "clusters": cluster_ids,
                "q_used": info["q"],
                "q_initial": info["q_initial"],
                "v_curve": [{"num_clusters": n, "V": v}
                            for n, v in zip(info["num_clusters_per_round"], values)],
                "shapley": {model.entity_label(e): info["phi"][e] for e in range(model.N)},
                "rho": model.rho,
                # Algorithm 2 assumes robots have positive and tasks negative phi (paper Remark 6).
                # If many tasks have phi > 0 the "highest |phi|" rule lets a few clusters hoard tasks.
                "frac_tasks_positive_phi": (
                    sum(1 for e in range(model.R, model.N) if info["phi"][e] > 0) / model.T if model.T else 0.0
                ),
                "frac_robots_negative_phi": sum(1 for e in range(model.R) if info["phi"][e] < 0) / model.R,
                "config": asdict(cfg),
            },
        )

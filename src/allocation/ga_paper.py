"""GA clustering baseline as described in Martin et al. (2023), Sec. 5.3.

Genome   g = [g_i], g_i in [0, N): the cluster label of player i (robots first, then tasks); the
         number of distinct labels is the number of clusters (paper Table 2).
Fitness  V(C) of Eq. 8, MAXIMISED, i.e. the same clustering criterion SVCA selects on.
         (The cost-driven GA in ``ga_clustering.py`` instead minimises the true allocation cost A(U);
          it is a much stronger baseline and is kept as "GA-cost".)

The paper ran MATLAB ``ga`` with default settings, which cannot be reproduced exactly, so this is a
standard generational GA with the same ingredients (elitism, crossover/mutation children, scattered
crossover). One necessary addition: a random genome almost surely contains a task-only cluster,
which makes Omega(C)=0 and V(C)=0 for the whole population. A REPAIR step therefore moves the tasks
of any task-only cluster into the cluster of a random robot, so Omega(C)=1 for every individual.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.allocation.base import Allocator
from src.allocation.evaluator import AllocationEvaluator, Routes
from src.allocation.insertion import solve_subproblem
from src.allocation.paper_model import PaperCoalitionModel, PaperParams
from src.core.types import AllocationResult
from src.simulation.generator import ScenarioData


@dataclass
class PaperGAConfig:
    params: PaperParams = field(default_factory=PaperParams)
    population_size: int = 200          # MATLAB ga default for > 5 variables
    max_generations: int = 100
    stall_generations: int = 50         # stop if the best V has not improved for this many generations
    elite_fraction: float = 0.05        # MATLAB default
    crossover_fraction: float = 0.8     # MATLAB default (share of non-elite children made by crossover)
    mutation_rate: float = 0.05         # per-gene rate for mutation children
    tournament_size: int = 3
    improve: bool = True                # sub-solver local search (same as SVCA)
    local_search_passes: int = 20
    seed: Optional[int] = 42


class PaperGAAllocator(Allocator):
    name = "GA"

    def __init__(self, config: Optional[PaperGAConfig] = None):
        self.config = config or PaperGAConfig()

    def solve(self, scenario: ScenarioData) -> AllocationResult:
        cfg = self.config
        t0 = time.perf_counter()
        ev = AllocationEvaluator(scenario)
        model = PaperCoalitionModel(scenario, cfg.params)
        R, N = model.R, model.N
        rng = random.Random(cfg.seed)
        cache: Dict[Tuple[int, ...], float] = {}

        def groups(g: List[int]) -> Dict[int, List[int]]:
            out: Dict[int, List[int]] = {}
            for e, lab in enumerate(g):
                out.setdefault(lab, []).append(e)
            return out

        def repair(g: List[int]) -> List[int]:
            robot_labels = [g[i] for i in range(R)]
            ok = set(robot_labels)
            for e in range(R, N):
                if g[e] not in ok:
                    g[e] = robot_labels[rng.randrange(R)]
            return g

        def fitness(g: List[int]) -> float:
            total = 0.0
            for members in groups(g).values():
                key = tuple(members)
                val = cache.get(key)
                if val is None:
                    val = model.cost_of(members)
                    cache[key] = val
                total += val
            return total

        def tournament(pop: List[List[int]], fit: List[float]) -> List[int]:
            idx = [rng.randrange(len(pop)) for _ in range(cfg.tournament_size)]
            return pop[max(idx, key=lambda i: fit[i])]

        def mutate(g: List[int]) -> List[int]:
            g = g[:]
            for e in range(N):
                if rng.random() < cfg.mutation_rate:
                    # half: brand-new label, half: join the cluster of another player
                    g[e] = rng.randrange(N) if rng.random() < 0.5 else g[rng.randrange(N)]
            return repair(g)

        pop = [repair([rng.randrange(N) for _ in range(N)]) for _ in range(cfg.population_size)]
        fit = [fitness(g) for g in pop]
        n_elite = max(1, math.ceil(cfg.elite_fraction * cfg.population_size))
        n_cross = int(round(cfg.crossover_fraction * (cfg.population_size - n_elite)))

        history = [max(fit)]
        best_val, stall = history[0], 0
        for _ in range(cfg.max_generations):
            order = sorted(range(len(pop)), key=lambda i: -fit[i])
            new_pop = [pop[i][:] for i in order[:n_elite]]
            for _ in range(n_cross):
                p1, p2 = tournament(pop, fit), tournament(pop, fit)
                child = [a if rng.random() < 0.5 else b for a, b in zip(p1, p2)]   # scattered
                new_pop.append(repair(child))
            while len(new_pop) < cfg.population_size:
                new_pop.append(mutate(tournament(pop, fit)))
            pop = new_pop
            fit = [fitness(g) for g in pop]
            history.append(max(fit))
            if history[-1] > best_val + 1e-9:
                best_val, stall = history[-1], 0
            else:
                stall += 1
                if stall >= cfg.stall_generations:
                    break

        best = pop[max(range(len(pop)), key=lambda i: fit[i])]
        clusters = list(groups(best).values())
        t1 = time.perf_counter()

        routes: Routes = {a: [] for a in ev.agent_ids}
        cluster_ids = []
        for members in clusters:
            agents = sorted(model.agents[e].agent_id for e in members if e < R)
            tasks = sorted(model.tasks[e - R].task_id for e in members if e >= R)
            cluster_ids.append({"agents": agents, "tasks": tasks})
            if tasks:
                routes.update(solve_subproblem(ev, agents, tasks, improve=cfg.improve,
                                               max_passes=cfg.local_search_passes))
        t2 = time.perf_counter()

        return ev.evaluate(
            routes,
            solver_name=self.name,
            runtime_seconds=t2 - t0,
            metadata={
                "t_clu": t1 - t0,
                "t_all": t2 - t1,
                "V": model.value_of_clustering(clusters),
                "num_clusters": len(clusters),
                "clusters": cluster_ids,
                "generations": len(history) - 1,
                "v_history": history,
                "rho": model.rho,
                "config": asdict(cfg),
            },
        )

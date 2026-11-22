"""Genetic-Algorithm clustering baseline (proposal Sec. 4.3).

Chromosome : one cluster label in [0, K) for every entity (agents first, then tasks).
Fitness    : total A(U) after solving every cluster with the SAME sub-solver SVCA uses.
Validity   : constraint (6) needs every cluster that contains tasks to contain an agent.
             A repair step moves the nearest spare agent into any agent-less cluster, so
             every chromosome is feasible and the GA never wastes evaluations on invalid ones.
Fairness   : K defaults to the K that SVCA would use, so both methods solve the same
             number of sub-problems.

Speed: fitness is cached on the cluster signature (agents, tasks). Children share most
clusters with their parents, so most evaluations are cache hits.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.allocation.base import Allocator
from src.allocation.evaluator import AllocationEvaluator, Routes
from src.allocation.insertion import solve_subproblem
from src.core.types import AllocationResult
from src.simulation.generator import ScenarioData


@dataclass
class GAConfig:
    num_clusters: Optional[int] = None   # None -> ceil(N / target_cluster_size), clamped to [1, R]
    target_cluster_size: int = 20        # only used to derive K when num_clusters is None
    population_size: int = 30
    generations: int = 40
    tournament_size: int = 3
    crossover_prob: float = 0.9
    mutation_prob: Optional[float] = None  # per gene; None -> 2 / N
    elite: int = 2
    spatial_init_fraction: float = 0.5   # share of the initial population seeded spatially (rest random)
    fitness_improve: bool = False        # local search inside fitness (slower, tighter)
    improve: bool = True                 # local search for the FINAL solve (same as SVCA)
    local_search_passes: int = 20
    seed: Optional[int] = 42


Chromosome = List[int]


class GAClusteringAllocator(Allocator):
    name = "GA"

    def __init__(self, config: Optional[GAConfig] = None):
        self.config = config or GAConfig()

    # ------------------------------------------------------------------ helpers
    def _setup(self, ev: AllocationEvaluator):
        self.ev = ev
        self.agent_ids = ev.agent_ids
        self.task_ids = ev.task_ids
        self.R = len(self.agent_ids)
        self.T = len(self.task_ids)
        self.N = self.R + self.T
        cfg = self.config
        k = cfg.num_clusters or math.ceil(self.N / max(2, cfg.target_cluster_size))
        self.K = max(1, min(self.R, k))
        self.pos = [ev.agents[a].position for a in self.agent_ids] + [ev.tasks[t].location for t in self.task_ids]
        self._fit_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
        self.evaluations = 0

    def _clusters(self, genes: Chromosome) -> List[Tuple[List[int], List[int]]]:
        out = [([], []) for _ in range(self.K)]
        for e, c in enumerate(genes):
            if e < self.R:
                out[c][0].append(self.agent_ids[e])
            else:
                out[c][1].append(self.task_ids[e - self.R])
        return out

    def _repair(self, genes: Chromosome) -> Chromosome:
        """Give every task-bearing cluster at least one agent (always possible since K <= R)."""
        g = genes[:]
        while True:
            n_ag = [0] * self.K
            n_tk = [0] * self.K
            for e, c in enumerate(g):
                if e < self.R:
                    n_ag[c] += 1
                else:
                    n_tk[c] += 1
            needy = [c for c in range(self.K) if n_tk[c] > 0 and n_ag[c] == 0]
            if not needy:
                return g
            c = needy[0]
            tx = [self.pos[e] for e in range(self.R, self.N) if g[e] == c]
            cx = sum(p.x for p in tx) / len(tx)
            cy = sum(p.y for p in tx) / len(tx)
            # donors: agents in clusters that keep a required agent after giving one away
            donors = [
                e for e in range(self.R)
                if n_ag[g[e]] >= 2 or (n_ag[g[e]] == 1 and n_tk[g[e]] == 0)
            ]
            e_best = min(donors, key=lambda e: ((self.pos[e].x - cx) ** 2 + (self.pos[e].y - cy) ** 2, e))
            g[e_best] = c

    def _fitness(self, genes: Chromosome, improve: bool) -> float:
        total = 0.0
        for agents, tasks in self._clusters(genes):
            if not tasks:
                continue
            key = (tuple(agents), tuple(tasks))
            cost = self._fit_cache.get(key)
            if cost is None:
                self.evaluations += 1
                routes = solve_subproblem(self.ev, agents, tasks, improve=improve,
                                          max_passes=self.config.local_search_passes)
                cost = self.ev.total_cost(routes)
                self._fit_cache[key] = cost
            total += cost
        return total

    def _random_individual(self, rng: random.Random) -> Chromosome:
        return self._repair([rng.randrange(self.K) for _ in range(self.N)])

    def _spatial_individual(self, rng: random.Random) -> Chromosome:
        """Nearest-of-K-random-agents partition (a randomised k-medoids style start)."""
        seeds = rng.sample(range(self.R), self.K)
        genes = []
        for e in range(self.N):
            genes.append(min(range(self.K), key=lambda c: self.pos[e].distance_to(self.pos[seeds[c]])))
        return self._repair(genes)

    # --------------------------------------------------------------------- solve
    def solve(self, scenario: ScenarioData) -> AllocationResult:
        cfg = self.config
        t0 = time.perf_counter()
        ev = AllocationEvaluator(scenario)
        self._setup(ev)
        rng = random.Random(cfg.seed)
        pm = cfg.mutation_prob if cfg.mutation_prob is not None else 2.0 / self.N

        n_spatial = int(round(cfg.population_size * cfg.spatial_init_fraction))
        pop: List[Chromosome] = [self._spatial_individual(rng) for _ in range(n_spatial)]
        pop += [self._random_individual(rng) for _ in range(cfg.population_size - n_spatial)]
        fit = [self._fitness(g, cfg.fitness_improve) for g in pop]

        history: List[float] = [min(fit)]
        for _ in range(cfg.generations):
            order = sorted(range(len(pop)), key=lambda i: fit[i])
            new_pop = [pop[i][:] for i in order[: cfg.elite]]
            new_fit = [fit[i] for i in order[: cfg.elite]]
            while len(new_pop) < cfg.population_size:
                p1 = pop[self._tournament(fit, rng)]
                p2 = pop[self._tournament(fit, rng)]
                child = p1[:]
                if rng.random() < cfg.crossover_prob:
                    child = [a if rng.random() < 0.5 else b for a, b in zip(p1, p2)]
                for e in range(self.N):
                    if rng.random() < pm:
                        child[e] = rng.randrange(self.K)
                child = self._repair(child)
                new_pop.append(child)
                new_fit.append(self._fitness(child, cfg.fitness_improve))
            pop, fit = new_pop, new_fit
            history.append(min(fit))

        best = pop[min(range(len(pop)), key=lambda i: fit[i])]
        t_ga = time.perf_counter()

        routes: Routes = {}
        clusters = self._clusters(best)
        for agents, tasks in clusters:
            routes.update(solve_subproblem(ev, agents, tasks, improve=cfg.improve,
                                           max_passes=cfg.local_search_passes))
        runtime = time.perf_counter() - t0

        return ev.evaluate(
            routes,
            solver_name=self.name,
            runtime_seconds=runtime,
            metadata={
                "num_clusters": self.K,
                "clusters": [{"agents": a, "tasks": t} for a, t in clusters],
                "fitness_history": history,
                "subproblem_solves": self.evaluations,
                "runtime_search": t_ga - t0,
                "config": asdict(cfg),
            },
        )

    def _tournament(self, fit: List[float], rng: random.Random) -> int:
        cand = [rng.randrange(len(fit)) for _ in range(self.config.tournament_size)]
        return min(cand, key=lambda i: fit[i])

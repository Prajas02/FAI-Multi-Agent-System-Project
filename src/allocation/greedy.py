"""Greedy allocation baseline."""

from __future__ import annotations

import time

from src.allocation.base import Allocator
from src.allocation.evaluator import AllocationEvaluator
from src.allocation.insertion import greedy_insertion, local_search
from src.core.types import AllocationResult
from src.simulation.generator import ScenarioData


class GreedyAllocator(Allocator):
    """Global priority-ordered cheapest insertion over ALL agents and ALL tasks.

    ``use_local_search=False`` (default) is the plain Greedy baseline from the proposal.
    ``use_local_search=True`` adds relocate/swap improvement over the whole problem. It is
    the like-for-like "no clustering" reference for SVCA/GA, which use the same
    local search inside each cluster.
    """

    def __init__(self, use_local_search: bool = False, max_passes: int = 20):
        self.use_local_search = use_local_search
        self.max_passes = max_passes
        self.name = "Greedy+LS" if use_local_search else "Greedy"

    def solve(self, scenario: ScenarioData) -> AllocationResult:
        t0 = time.perf_counter()
        ev = AllocationEvaluator(scenario)
        routes = greedy_insertion(ev, ev.agent_ids, ev.task_ids)
        if self.use_local_search:
            routes = local_search(ev, routes, max_passes=self.max_passes)
        runtime = time.perf_counter() - t0
        return ev.evaluate(routes, solver_name=self.name, runtime_seconds=runtime)

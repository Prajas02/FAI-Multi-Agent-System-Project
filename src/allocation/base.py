"""Common interface for every Stage-1 allocator (Greedy, SVCA, GA, MILP).

Yash's harness only needs:  result = allocator.solve(scenario)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.types import AllocationResult
from src.simulation.generator import ScenarioData


class Allocator(ABC):
    name: str = "Allocator"

    @abstractmethod
    def solve(self, scenario: ScenarioData) -> AllocationResult:
        """Return a complete allocation for ``scenario``.

        Implementations must:
          * assign every task to exactly one agent (Eq. 5-6),
          * time the WHOLE solve (including building their evaluator) in ``runtime_seconds``,
          * build a fresh AllocationEvaluator, so that route caches are never shared
            between solvers (a shared cache would make later solvers look faster).
        """

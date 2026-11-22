"""Task Allocation and SVCA Clustering Module.

Owner: Dhairya Shah (dhairya.shah@iitgn.ac.in)
Responsibilities:
- Shapley Value Clustering Algorithm (SVCA) partitioning
- Baselines: Greedy Allocation, Genetic Algorithm (GA) clustering, MILP
- Optimization Problem 1 solver interface

Every allocator implements ``Allocator.solve(scenario) -> AllocationResult`` and is scored
by ``AllocationEvaluator`` (identical to ``Environment.execute_allocation``).
"""

from src.allocation.base import Allocator
from src.allocation.evaluator import AllocationEvaluator
from src.allocation.ga_clustering import GAClusteringAllocator, GAConfig
from src.allocation.greedy import GreedyAllocator
from src.allocation.svca import SVCAAllocator, SVCAConfig

__all__ = [
    "Allocator",
    "AllocationEvaluator",
    "GreedyAllocator",
    "SVCAAllocator",
    "SVCAConfig",
    "GAClusteringAllocator",
    "GAConfig",
]

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
from src.allocation.ga_paper import PaperGAAllocator, PaperGAConfig
from src.allocation.greedy import GreedyAllocator
from src.allocation.paper_model import PaperParams
from src.allocation.svca import SVCAAllocator, SVCAConfig
from src.allocation.svca_paper import PaperSVCAAllocator, PaperSVCAConfig

__all__ = [
    "Allocator",
    "AllocationEvaluator",
    "GreedyAllocator",
    "PaperParams",
    "PaperSVCAAllocator",     # solver name "SVCA"     (Martin et al. 2023, Algorithms 1-2)
    "PaperSVCAConfig",
    "PaperGAAllocator",       # solver name "GA"       (fitness V(C), as in the paper)
    "PaperGAConfig",
    "SVCAAllocator",          # solver name "SVCA-ext" (own extended variant)
    "SVCAConfig",
    "GAClusteringAllocator",  # solver name "GA-cost"  (minimises true cost A(U))
    "GAConfig",
]

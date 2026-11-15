"""Core types and data contracts for simulation, allocation, and game theory."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass(frozen=True)
class Position:
    """2D spatial coordinate (x, y) in meters."""
    x: float
    y: float

    def distance_to(self, other: "Position") -> float:
        """Euclidean distance to another position."""
        return float(math.hypot(self.x - other.x, self.y - other.y))


@dataclass
class AllocationResult:
    """Result of an allocation algorithm (Stage 1)."""
    assignments: Dict[int, List[int]]       # agent_id -> [task_id_1, task_id_2, ...]
    task_completion_times: Dict[int, float] # task_id -> completion time (seconds)
    agent_distances: Dict[int, float]       # agent_id -> total distance traveled (m)
    agent_energy_used: Dict[int, float]     # agent_id -> energy consumed
    total_cost: float                       # Total objective cost A(U)
    runtime_seconds: float                  # Solver runtime in seconds
    solver_name: str                        # e.g., 'Greedy', 'SVCA', 'GA', 'MILP'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardDistribution:
    """Result of a preference-aware Shapley solver (Stage 2)."""
    payoffs: Dict[int, float]               # agent_id -> reward share phi_i
    total_coalition_value: float            # Total coalition value V(N)
    solver_name: str                        # e.g., 'GlauberMCMC', 'Exact', 'MonteCarlo'
    preference_sensitivity_alpha: float     # alpha parameter
    runtime_seconds: float                  # Solver runtime in seconds
    gini_index: float                       # Fairness inequality index G in [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)

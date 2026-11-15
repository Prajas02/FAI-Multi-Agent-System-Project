"""Core data structures and type definitions for MAS MRTA and Reward Sharing."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


@dataclass(frozen=True)
class Position:
    """2D spatial coordinate (x, y) in meters."""
    x: float
    y: float

    def distance_to(self, other: "Position") -> float:
        """Euclidean distance to another position."""
        return float(np.hypot(self.x - other.x, self.y - other.y))

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass
class AllocationResult:
    """Output produced by an MRTA allocation solver.
    
    Attributes:
        assignments: Dict mapping agent_id -> list of assigned task_ids in sequence order.
        task_completion_times: Dict mapping task_id -> completion timestamp (eta_t).
        agent_distances: Dict mapping agent_id -> total distance traveled (d_r).
        agent_energy_used: Dict mapping agent_id -> total energy consumed.
        total_cost: Evaluated allocation cost A(U).
        runtime_seconds: Execution time taken by the allocation solver.
        solver_name: Identifier of the algorithm (e.g., 'Greedy', 'SVCA', 'GA', 'MILP').
        metadata: Optional solver-specific diagnostics.
    """
    assignments: Dict[int, List[int]]
    task_completion_times: Dict[int, float]
    agent_distances: Dict[int, float]
    agent_energy_used: Dict[int, float]
    total_cost: float
    runtime_seconds: float
    solver_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardDistribution:
    """Output produced by a Shapley value reward solver.
    
    Attributes:
        payoffs: Dict mapping agent_id -> allocated reward / utility share (phi_i).
        total_coalition_value: Sum of distributed values V(N).
        solver_name: Identifier of the reward solver (e.g., 'GlauberMCMC', 'MonteCarlo', 'Exact').
        preference_sensitivity_alpha: Sensitivity parameter used in Softmax distribution.
        runtime_seconds: Time taken to estimate/solve rewards.
        gini_index: Fairness inequality metric G in [0, 1].
    """
    payoffs: Dict[int, float]
    total_coalition_value: float
    solver_name: str
    preference_sensitivity_alpha: float
    runtime_seconds: float
    gini_index: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkRecord:
    """A single execution record within the benchmarking harness."""
    run_id: str
    problem_size_n: int
    num_agents: int
    num_tasks: int
    algorithm: str
    allocation_cost: float
    allocation_runtime_sec: float
    reward_solver: Optional[str] = None
    reward_runtime_sec: Optional[float] = None
    alpha: Optional[float] = None
    gini_index: Optional[float] = None
    is_feasible: bool = True
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

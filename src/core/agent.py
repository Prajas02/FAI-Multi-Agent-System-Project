"""Agent representation with kinematics and energy parameters."""

from dataclasses import dataclass
from src.core.types import Position


@dataclass
class Agent:
    """Represents a heterogeneous robot agent r in R.
    
    Mathematical specification:
        x_r = (p_r, v_r, E_r, lambda_r)
    """
    agent_id: int
    position: Position
    velocity: float                  # v_r (m/s)
    energy_capacity: float           # Max battery capacity
    current_energy: float            # E_r (current energy)
    usage_penalty: float             # lambda_r (usage/travel cost penalty weight)
    consumption_per_meter: float = 1.0
    consumption_per_sec: float = 0.1

    def compute_travel_time(self, target: Position) -> float:
        """Compute time needed to reach a target position."""
        dist = self.position.distance_to(target)
        return dist / self.velocity if self.velocity > 0 else float("inf")

    def compute_energy_required(self, distance: float, execution_time: float = 0.0) -> float:
        """Compute energy needed for distance traveled and execution duration."""
        return (distance * self.consumption_per_meter) + (execution_time * self.consumption_per_sec)

    def can_undertake(self, distance: float, execution_time: float = 0.0) -> bool:
        """Check if remaining energy is sufficient."""
        return self.compute_energy_required(distance, execution_time) <= self.current_energy

"""Task representation in the MRTA environment."""

from dataclasses import dataclass
from typing import Optional
from src.core.types import Position


@dataclass
class Task:
    """Represents a spatially distributed task t in T.
    
    Mathematical specification:
        y_t = (q_t, tau_t, delta_t)
    """
    task_id: int
    location: Position               # q_t
    execution_time: float            # tau_t (processing duration in seconds)
    priority: float                  # delta_t (priority weight, higher = more urgent)
    completed: bool = False
    completion_time: Optional[float] = None
    assigned_agent_id: Optional[int] = None

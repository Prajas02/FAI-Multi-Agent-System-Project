"""Kinematics and battery/velocity decay models for heterogeneous agents."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
from src.core.types import Position
from src.core.agent import Agent


@dataclass
class BatteryDecayModel:
    """Calculates energy consumption for agent movement and task execution.
    
    Attributes:
        base_rate_per_meter: Energy consumed per meter of movement at nominal velocity.
        rate_per_second_exec: Energy consumed per second during task execution.
        rate_per_second_idle: Standby energy consumed per second when idle.
        payload_factor: Coefficient increasing energy consumption under load (default 1.0).
        non_linear_velocity_penalty: Exponent for higher-speed aerodynamic/drag energy losses (default 1.0 = linear).
    """
    base_rate_per_meter: float = 1.0
    rate_per_second_exec: float = 0.1
    rate_per_second_idle: float = 0.01
    payload_factor: float = 1.0
    non_linear_velocity_penalty: float = 1.0

    def compute_motion_energy(self, distance: float, velocity: float) -> float:
        """Compute energy required to traverse a given distance at a given velocity."""
        # Non-linear speed penalty: energy ~ distance * (velocity / 10.0)^penalty
        speed_multiplier = (velocity / 10.0) ** (self.non_linear_velocity_penalty - 1.0) if velocity > 0 else 1.0
        return distance * self.base_rate_per_meter * self.payload_factor * max(1.0, speed_multiplier)

    def compute_execution_energy(self, duration: float) -> float:
        """Compute energy required for active task execution duration."""
        return duration * self.rate_per_second_exec

    def compute_idle_energy(self, duration: float) -> float:
        """Compute energy consumed while waiting/idle."""
        return duration * self.rate_per_second_idle

    def compute_total_energy(self, distance: float, velocity: float, execution_time: float, idle_time: float = 0.0) -> float:
        """Compute complete energy expenditure for a task mission leg."""
        return (
            self.compute_motion_energy(distance, velocity)
            + self.compute_execution_energy(execution_time)
            + self.compute_idle_energy(idle_time)
        )


class KinematicsModel:
    """Kinematics calculations: velocity decay, travel times, and discrete waypoint interpolation."""

    @staticmethod
    def compute_effective_velocity(agent: Agent, battery_ratio: Optional[float] = None) -> float:
        """Computes effective agent velocity, with optional decay when battery drops below 15%.
        
        Args:
            agent: The Agent entity.
            battery_ratio: Fraction of remaining battery (in [0.0, 1.0]). If None, uses current_energy / capacity.
        """
        if battery_ratio is None:
            battery_ratio = agent.current_energy / agent.energy_capacity if agent.energy_capacity > 0 else 1.0
        
        # When battery is critically low (< 15%), velocity drops linearly down to 50% max speed to conserve power
        if battery_ratio < 0.15:
            scaling = 0.5 + 0.5 * (max(0.0, battery_ratio) / 0.15)
            return agent.velocity * scaling
        return agent.velocity

    @staticmethod
    def compute_travel_time(start: Position, target: Position, velocity: float) -> float:
        """Compute travel duration between two spatial positions at given velocity."""
        if velocity <= 0:
            return float("inf")
        distance = start.distance_to(target)
        return distance / velocity

    @staticmethod
    def interpolate_trajectory(start: Position, end: Position, num_steps: int = 10) -> list[Position]:
        """Generate a series of discrete intermediate waypoints between start and end."""
        if num_steps <= 1:
            return [start, end]
        
        waypoints = []
        for step in range(num_steps + 1):
            fraction = step / float(num_steps)
            x = start.x + fraction * (end.x - start.x)
            y = start.y + fraction * (end.y - start.y)
            waypoints.append(Position(round(x, 3), round(y, 3)))
        return waypoints

"""2D Continuous Spatial Simulation Environment for Multi-Agent Task Allocation."""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from src.core.types import Position, AllocationResult
from src.core.agent import Agent
from src.core.task import Task
from src.simulation.kinematics import BatteryDecayModel, KinematicsModel
from src.simulation.generator import ScenarioData


@dataclass
class SimulationTrace:
    """Historical telemetry record of an agent's execution."""
    agent_id: int
    waypoints: List[Position] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    energy_history: List[float] = field(default_factory=list)
    tasks_visited: List[int] = field(default_factory=list)


class Environment:
    """2D spatial simulation world managing spatial geometry, physics, and plan execution.
    
    Default coordinates are [0, map_width] x [0, map_height] (500m x 500m).
    """

    def __init__(
        self,
        map_width: float = 500.0,
        map_height: float = 500.0,
        battery_model: Optional[BatteryDecayModel] = None,
    ):
        self.map_width = map_width
        self.map_height = map_height
        self.battery_model = battery_model or BatteryDecayModel()
        
        self.agents: Dict[int, Agent] = {}
        self.tasks: Dict[int, Task] = {}
        self.initial_agents: Dict[int, Agent] = {}
        self.initial_tasks: Dict[int, Task] = {}
        self.time: float = 0.0

    def load_scenario(self, scenario: ScenarioData) -> None:
        """Load agents and tasks into the environment from a ScenarioData container."""
        self.map_width = scenario.map_width
        self.map_height = scenario.map_height
        self.agents = {a.agent_id: copy.deepcopy(a) for a in scenario.agents}
        self.tasks = {t.task_id: copy.deepcopy(t) for t in scenario.tasks}
        self.initial_agents = copy.deepcopy(self.agents)
        self.initial_tasks = copy.deepcopy(self.tasks)
        self.time = 0.0

    def reset(self) -> None:
        """Reset environment to the initial scenario state."""
        self.agents = copy.deepcopy(self.initial_agents)
        self.tasks = copy.deepcopy(self.initial_tasks)
        self.time = 0.0

    def get_agent_task_distance_matrix(self) -> Dict[Tuple[int, int], float]:
        """Compute Euclidean distance between each agent and each task."""
        distances = {}
        for r_id, agent in self.agents.items():
            for t_id, task in self.tasks.items():
                distances[(r_id, t_id)] = agent.position.distance_to(task.location)
        return distances

    def get_task_task_distance_matrix(self) -> Dict[Tuple[int, int], float]:
        """Compute Euclidean distance between all pairs of tasks."""
        distances = {}
        for t1_id, t1 in self.tasks.items():
            for t2_id, t2 in self.tasks.items():
                distances[(t1_id, t2_id)] = t1.location.distance_to(t2.location)
        return distances

    def execute_allocation(
        self,
        assignments: Dict[int, List[int]],
        energy_violation_penalty_factor: float = 1000.0,
    ) -> Tuple[AllocationResult, Dict[int, SimulationTrace]]:
        """Simulate the execution of an allocation plan across all agents.
        
        Evaluates the objective function A(U) from Optimization Problem 1 (Eq. 4-7):
            min A(U) = sum(delta_t * eta_t) + sum(lambda_r * d_r) + phi(U)
        
        Args:
            assignments: Mapping agent_id -> ordered list of assigned task_ids.
            energy_violation_penalty_factor: Penalty constant M for energy deficit.
            
        Returns:
            Tuple of (AllocationResult, Dict[agent_id, SimulationTrace]).
        """
        self.reset()
        task_completion_times: Dict[int, float] = {}
        agent_distances: Dict[int, float] = {}
        agent_energy_used: Dict[int, float] = {}
        telemetry: Dict[int, SimulationTrace] = {}

        total_penalty = 0.0

        for agent_id, task_ids in assignments.items():
            agent = self.agents[agent_id]
            current_pos = agent.position
            current_time = 0.0
            distance_traveled = 0.0
            energy_spent = 0.0

            trace = SimulationTrace(
                agent_id=agent_id,
                waypoints=[current_pos],
                timestamps=[0.0],
                energy_history=[agent.current_energy],
                tasks_visited=[],
            )

            for task_id in task_ids:
                task = self.tasks[task_id]
                dist_to_task = current_pos.distance_to(task.location)
                
                # Check effective velocity considering battery decay
                eff_velocity = KinematicsModel.compute_effective_velocity(agent)
                travel_time = dist_to_task / eff_velocity if eff_velocity > 0 else 0.0
                
                # Compute energy for transit + execution
                transit_energy = self.battery_model.compute_motion_energy(dist_to_task, eff_velocity)
                exec_energy = self.battery_model.compute_execution_energy(task.execution_time)
                step_energy = transit_energy + exec_energy

                # Check for battery depletion / penalty
                if agent.current_energy < step_energy:
                    deficit = step_energy - agent.current_energy
                    total_penalty += deficit * energy_violation_penalty_factor
                    agent.current_energy = 0.0
                else:
                    agent.current_energy -= step_energy

                # Update states
                distance_traveled += dist_to_task
                energy_spent += step_energy
                current_time += travel_time + task.execution_time
                current_pos = task.location

                task.completed = True
                task.completion_time = current_time
                task.assigned_agent_id = agent_id
                task_completion_times[task_id] = current_time

                # Append to trace
                trace.waypoints.append(current_pos)
                trace.timestamps.append(current_time)
                trace.energy_history.append(agent.current_energy)
                trace.tasks_visited.append(task_id)

            agent.position = current_pos
            agent_distances[agent_id] = distance_traveled
            agent_energy_used[agent_id] = energy_spent
            telemetry[agent_id] = trace

        # Calculate Total Cost A(U) = sum(delta_t * eta_t) + sum(lambda_r * d_r) + phi(U)
        priority_completion_cost = sum(
            self.tasks[t_id].priority * task_completion_times[t_id]
            for t_id in task_completion_times
        )
        travel_cost = sum(
            self.agents[r_id].usage_penalty * agent_distances[r_id]
            for r_id in agent_distances
        )
        total_cost = priority_completion_cost + travel_cost + total_penalty

        result = AllocationResult(
            assignments=assignments,
            task_completion_times=task_completion_times,
            agent_distances=agent_distances,
            agent_energy_used=agent_energy_used,
            total_cost=round(total_cost, 2),
            runtime_seconds=0.0,
            solver_name="EnvironmentSimulator",
            metadata={
                "priority_completion_cost": round(priority_completion_cost, 2),
                "travel_cost": round(travel_cost, 2),
                "energy_penalty": round(total_penalty, 2),
            },
        )
        return result, telemetry

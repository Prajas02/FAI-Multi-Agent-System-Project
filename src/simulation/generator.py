"""Scenario generator for synthetic multi-agent and task allocation environments."""

import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from src.core.types import Position
from src.core.agent import Agent
from src.core.task import Task


@dataclass
class ScenarioData:
    """Serializable container for generated simulation scenarios."""
    map_width: float
    map_height: float
    seed: int
    agents: List[Agent]
    tasks: List[Task]

    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario into a JSON-compatible dictionary."""
        return {
            "map_width": self.map_width,
            "map_height": self.map_height,
            "seed": self.seed,
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "x": a.position.x,
                    "y": a.position.y,
                    "velocity": a.velocity,
                    "energy_capacity": a.energy_capacity,
                    "current_energy": a.current_energy,
                    "usage_penalty": a.usage_penalty,
                    "consumption_per_meter": a.consumption_per_meter,
                    "consumption_per_sec": a.consumption_per_sec,
                }
                for a in self.agents
            ],
            "tasks": [
                {
                    "task_id": t.task_id,
                    "x": t.location.x,
                    "y": t.location.y,
                    "execution_time": t.execution_time,
                    "priority": t.priority,
                }
                for t in self.tasks
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioData":
        """Reconstruct scenario from a JSON dictionary."""
        agents = [
            Agent(
                agent_id=d["agent_id"],
                position=Position(d["x"], d["y"]),
                velocity=d["velocity"],
                energy_capacity=d["energy_capacity"],
                current_energy=d["current_energy"],
                usage_penalty=d["usage_penalty"],
                consumption_per_meter=d.get("consumption_per_meter", 1.0),
                consumption_per_sec=d.get("consumption_per_sec", 0.1),
            )
            for d in data["agents"]
        ]
        tasks = [
            Task(
                task_id=d["task_id"],
                location=Position(d["x"], d["y"]),
                execution_time=d["execution_time"],
                priority=d["priority"],
            )
            for d in data["tasks"]
        ]
        return cls(
            map_width=data["map_width"],
            map_height=data["map_height"],
            seed=data["seed"],
            agents=agents,
            tasks=tasks,
        )

    def save_to_file(self, file_path: str) -> None:
        """Export scenario to a JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str) -> "ScenarioData":
        """Load scenario from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class ScenarioGenerator:
    """Generates random and clustered multi-agent and task scenarios.
    
    Default parameters conform to the FAI proposal (500m x 500m environment).
    """

    def __init__(
        self,
        map_width: float = 500.0,
        map_height: float = 500.0,
        seed: Optional[int] = 42,
    ):
        self.map_width = map_width
        self.map_height = map_height
        self.seed = seed
        self.rng = random.Random(seed)

    def set_seed(self, seed: int) -> None:
        """Reset the random number generator seed for deterministic generation."""
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_agents(
        self,
        num_agents: int,
        velocity_range: Tuple[float, float] = (5.0, 15.0),
        energy_range: Tuple[float, float] = (500.0, 2000.0),
        penalty_range: Tuple[float, float] = (1.0, 3.0),
    ) -> List[Agent]:
        """Generate a fleet of heterogeneous agents."""
        agents = []
        for i in range(1, num_agents + 1):
            x = round(self.rng.uniform(0.0, self.map_width), 2)
            y = round(self.rng.uniform(0.0, self.map_height), 2)
            velocity = round(self.rng.uniform(velocity_range[0], velocity_range[1]), 2)
            energy = round(self.rng.uniform(energy_range[0], energy_range[1]), 1)
            penalty = round(self.rng.uniform(penalty_range[0], penalty_range[1]), 2)

            agent = Agent(
                agent_id=i,
                position=Position(x, y),
                velocity=velocity,
                energy_capacity=energy,
                current_energy=energy,
                usage_penalty=penalty,
                consumption_per_meter=1.0,
                consumption_per_sec=0.1,
            )
            agents.append(agent)
        return agents

    def generate_tasks_uniform(
        self,
        num_tasks: int,
        execution_time_range: Tuple[float, float] = (10.0, 60.0),
        priority_range: Tuple[float, float] = (1.0, 5.0),
    ) -> List[Task]:
        """Generate tasks uniformly distributed across the 2D space."""
        tasks = []
        for i in range(1, num_tasks + 1):
            x = round(self.rng.uniform(0.0, self.map_width), 2)
            y = round(self.rng.uniform(0.0, self.map_height), 2)
            exec_time = round(self.rng.uniform(execution_time_range[0], execution_time_range[1]), 1)
            priority = round(self.rng.uniform(priority_range[0], priority_range[1]), 2)

            task = Task(
                task_id=i,
                location=Position(x, y),
                execution_time=exec_time,
                priority=priority,
            )
            tasks.append(task)
        return tasks

    def generate_tasks_clustered(
        self,
        num_tasks: int,
        num_clusters: int = 3,
        cluster_std: float = 40.0,
        execution_time_range: Tuple[float, float] = (10.0, 60.0),
        priority_range: Tuple[float, float] = (1.0, 5.0),
    ) -> List[Task]:
        """Generate tasks centered around dense hotspots (e.g. disaster zones or depots)."""
        cluster_centers = [
            (
                self.rng.uniform(self.map_width * 0.15, self.map_width * 0.85),
                self.rng.uniform(self.map_height * 0.15, self.map_height * 0.85),
            )
            for _ in range(num_clusters)
        ]

        tasks = []
        for i in range(1, num_tasks + 1):
            center = self.rng.choice(cluster_centers)
            # Sample using Gaussian distribution around cluster center
            x = max(0.0, min(self.map_width, self.rng.gauss(center[0], cluster_std)))
            y = max(0.0, min(self.map_height, self.rng.gauss(center[1], cluster_std)))
            exec_time = round(self.rng.uniform(execution_time_range[0], execution_time_range[1]), 1)
            priority = round(self.rng.uniform(priority_range[0], priority_range[1]), 2)

            task = Task(
                task_id=i,
                location=Position(round(x, 2), round(y, 2)),
                execution_time=exec_time,
                priority=priority,
            )
            tasks.append(task)
        return tasks

    def generate_scenario(
        self,
        num_agents: int,
        num_tasks: int,
        clustered_tasks: bool = False,
        **kwargs
    ) -> ScenarioData:
        """Create a full simulation scenario with heterogeneous agents and tasks."""
        agents = self.generate_agents(num_agents)
        if clustered_tasks:
            tasks = self.generate_tasks_clustered(num_tasks, **kwargs)
        else:
            tasks = self.generate_tasks_uniform(num_tasks, **kwargs)

        return ScenarioData(
            map_width=self.map_width,
            map_height=self.map_height,
            seed=self.seed or 0,
            agents=agents,
            tasks=tasks,
        )

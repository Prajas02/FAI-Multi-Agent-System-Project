"""Unit test verifying core data structures using standard unittest."""

import unittest
from src.core.types import Position, AllocationResult, RewardDistribution
from src.core.agent import Agent
from src.core.task import Task


class TestCoreDataStructures(unittest.TestCase):
    def test_position_distance(self):
        p1 = Position(0.0, 0.0)
        p2 = Position(3.0, 4.0)
        self.assertEqual(p1.distance_to(p2), 5.0)

    def test_agent_energy_and_travel(self):
        p = Position(0.0, 0.0)
        target = Position(10.0, 0.0)
        agent = Agent(
            agent_id=1,
            position=p,
            velocity=5.0,
            energy_capacity=100.0,
            current_energy=100.0,
            usage_penalty=1.0,
            consumption_per_meter=1.0,
            consumption_per_sec=0.1
        )
        self.assertEqual(agent.compute_travel_time(target), 2.0)
        self.assertEqual(agent.compute_energy_required(distance=10.0, execution_time=5.0), 10.5)
        self.assertTrue(agent.can_undertake(distance=10.0, execution_time=5.0))

    def test_task_creation(self):
        task = Task(
            task_id=1,
            location=Position(100.0, 200.0),
            execution_time=30.0,
            priority=3.0
        )
        self.assertEqual(task.task_id, 1)
        self.assertFalse(task.completed)


if __name__ == "__main__":
    unittest.main()

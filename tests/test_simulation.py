"""Unit tests for the Simulation & Environment Engine (Prajas Kulkarni)."""

import os
import tempfile
import unittest
from src.core.types import Position
from src.core.agent import Agent
from src.core.task import Task
from src.simulation.environment import Environment
from src.simulation.generator import ScenarioGenerator, ScenarioData
from src.simulation.kinematics import BatteryDecayModel, KinematicsModel


class TestSimulationModule(unittest.TestCase):
    def setUp(self):
        self.generator = ScenarioGenerator(map_width=500.0, map_height=500.0, seed=42)

    def test_scenario_generation(self):
        scenario = self.generator.generate_scenario(num_agents=4, num_tasks=10)
        self.assertEqual(len(scenario.agents), 4)
        self.assertEqual(len(scenario.tasks), 10)
        
        # Verify agents bounds & heterogeneity
        for agent in scenario.agents:
            self.assertTrue(0.0 <= agent.position.x <= 500.0)
            self.assertTrue(0.0 <= agent.position.y <= 500.0)
            self.assertTrue(5.0 <= agent.velocity <= 15.0)
            self.assertTrue(500.0 <= agent.energy_capacity <= 2000.0)

        # Verify tasks bounds
        for task in scenario.tasks:
            self.assertTrue(0.0 <= task.location.x <= 500.0)
            self.assertTrue(0.0 <= task.location.y <= 500.0)
            self.assertTrue(10.0 <= task.execution_time <= 60.0)
            self.assertTrue(1.0 <= task.priority <= 5.0)

    def test_reproducibility_with_seed(self):
        gen1 = ScenarioGenerator(seed=123)
        s1 = gen1.generate_scenario(num_agents=3, num_tasks=5)

        gen2 = ScenarioGenerator(seed=123)
        s2 = gen2.generate_scenario(num_agents=3, num_tasks=5)

        self.assertEqual(s1.agents[0].position.x, s2.agents[0].position.x)
        self.assertEqual(s1.tasks[0].location.y, s2.tasks[0].location.y)

    def test_scenario_json_serialization(self):
        scenario = self.generator.generate_scenario(num_agents=2, num_tasks=3)
        
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            tmp_path = tmp.name

        try:
            scenario.save_to_file(tmp_path)
            loaded = ScenarioData.load_from_file(tmp_path)
            self.assertEqual(len(loaded.agents), 2)
            self.assertEqual(len(loaded.tasks), 3)
            self.assertEqual(loaded.agents[0].position.x, scenario.agents[0].position.x)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_kinematics_and_battery_decay(self):
        agent = Agent(
            agent_id=1,
            position=Position(0.0, 0.0),
            velocity=10.0,
            energy_capacity=100.0,
            current_energy=10.0,  # 10% battery (< 15% threshold)
            usage_penalty=1.0,
        )
        # Should reduce speed when battery is critically low (<15%)
        eff_vel = KinematicsModel.compute_effective_velocity(agent)
        self.assertLess(eff_vel, 10.0)

        battery_model = BatteryDecayModel(base_rate_per_meter=1.0, rate_per_second_exec=0.5)
        energy = battery_model.compute_total_energy(distance=20.0, velocity=10.0, execution_time=10.0)
        # 20m * 1.0 + 10s * 0.5 = 25.0
        self.assertEqual(energy, 25.0)

    def test_environment_execution(self):
        env = Environment(map_width=500.0, map_height=500.0)
        scenario = self.generator.generate_scenario(num_agents=2, num_tasks=4)
        env.load_scenario(scenario)

        # Assign tasks: agent 1 gets task 1 & 2, agent 2 gets task 3 & 4
        assignments = {1: [1, 2], 2: [3, 4]}
        result, telemetry = env.execute_allocation(assignments)

        self.assertIn(1, result.assignments)
        self.assertIn(2, result.assignments)
        self.assertEqual(len(result.task_completion_times), 4)
        self.assertGreater(result.total_cost, 0.0)
        self.assertIn(1, telemetry)
        self.assertEqual(len(telemetry[1].tasks_visited), 2)


if __name__ == "__main__":
    unittest.main()

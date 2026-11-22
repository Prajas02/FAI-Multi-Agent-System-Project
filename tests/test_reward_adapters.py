"""Reward adapter and benchmark integration tests; all writes stay in temp folders."""

import math
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.analytics import allocation_adapters, reward_adapters
from src.analytics.evaluation import validate_assignments
from src.analytics.execution import InProcessBackend, StageOutcome
from src.analytics.logger import (
    aggregate_benchmark_results,
    append_benchmark_results,
    append_robot_rewards,
    load_benchmark_results,
    load_robot_rewards,
)
from src.analytics.metrics import compute_robot_reward_gini
from src.simulation.generator import ScenarioGenerator
from scripts.run_benchmarks import RunSpec, execute_run_with_rewards


def reward_config(robot_count):
    return {
        "experiment": {"id": "reward_test"},
        "evaluation": {"energy_penalty": {"strategy": "quadratic", "lambda_E": 1.0}},
        "seeds": {"reward_stage_label": "reward_stage"},
        "time_limits": {"reward_stage_seconds": 5.0},
        "reward": {
            "enabled": True, "method": "exact", "value_type": "facility",
            "preference_sensitivity_alpha": 0.0, "num_samples_Tw": 8,
            "burn_in": 0, "thin": None, "samples_per_stratum": 2,
            "max_log_gamma": 3.0,
        },
    }


class TestRewardAdapter(unittest.TestCase):
    def test_extracts_exactly_expected_robot_payoffs_in_scenario_order(self):
        output = SimpleNamespace(payoffs={2: 3.0, 0: 1.0, 1: 2.0})
        self.assertEqual(reward_adapters.extract_robot_reward_pairs(output, [0, 1, 2]),
                         [(0, 1.0), (1, 2.0), (2, 3.0)])

    def test_missing_extra_and_duplicate_robot_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing=.*2"):
            reward_adapters.extract_robot_reward_pairs(SimpleNamespace(payoffs={0: 1.0, 1: 2.0}), [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "extra=.*task"):
            reward_adapters.extract_robot_reward_pairs(SimpleNamespace(payoffs={0: 1.0, 1: 2.0, "task": 4.0}), [0, 1])
        duplicate = SimpleNamespace(payoffs=[(0, 1.0), (0, 2.0), (1, 3.0)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            reward_adapters.extract_robot_reward_pairs(duplicate, [0, 1])

    def test_all_zero_single_robot_and_unequal_gini_conventions(self):
        self.assertEqual(compute_robot_reward_gini([(0, 0.0), (1, 0.0)]).gini_index, 0.0)
        self.assertEqual(compute_robot_reward_gini([(0, 9.0)]).gini_index, 0.0)
        self.assertAlmostEqual(compute_robot_reward_gini([(0, 1.0), (1, 2.0), (2, 3.0)]).gini_index,
                               2.0 / 9.0)
        result = compute_robot_reward_gini([(0, -1.0), (1, 2.0)])
        self.assertTrue(math.isnan(result.gini_index))
        self.assertTrue(result.invalid_reward_flag)


class TestRewardBenchmarkIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix="reward_integration_"))
        self.addCleanup(shutil.rmtree, str(self.temp), True)
        self.scenario = ScenarioGenerator(seed=2026).generate_scenario(3, 4)
        # Simulator robot IDs and task IDs intentionally overlap numerically.
        self.assertTrue(set(a.agent_id for a in self.scenario.agents)
                        & set(t.task_id for t in self.scenario.tasks))
        self.config = reward_config(len(self.scenario.agents))
        self.spec = RunSpec("C1", 3, 4, 0, "C1_run00", self.scenario.seed, "Greedy", 19, 5.0)

    def run_with_payoffs(self, payoffs):
        def adapter(stage_input, seed, limit):
            self.assertEqual(stage_input.scenario.seed, self.scenario.seed)
            return SimpleNamespace(payoffs=payoffs)
        return execute_run_with_rewards(self.spec, self.scenario, self.config, InProcessBackend(),
                                        allocation_adapters.greedy_adapter, adapter)

    def test_reward_run_logs_robot_records_and_aggregates_per_run_gini(self):
        robot_ids = [agent.agent_id for agent in self.scenario.agents]
        record, rows = self.run_with_payoffs(dict(zip(robot_ids, [1.0, 2.0, 3.0])))
        expected = compute_robot_reward_gini(list(zip(robot_ids, [1.0, 2.0, 3.0])))
        self.assertTrue(record["success"])
        self.assertTrue(record["reward_stage_completed"])
        self.assertGreaterEqual(record["reward_runtime_seconds"], 0.0)
        self.assertAlmostEqual(record["gini_index"], expected.gini_index)
        self.assertFalse(record["invalid_reward_flag"])
        self.assertEqual(len(rows), len(robot_ids))
        rewards_dir = self.temp / "raw"
        append_robot_rewards(rows, rewards_dir)
        append_benchmark_results([record], rewards_dir)
        stored = load_robot_rewards(rewards_dir / "robot_rewards.csv")
        self.assertEqual(len(stored), len(robot_ids))
        summary = aggregate_benchmark_results(load_benchmark_results(rewards_dir / "benchmark_results.csv"))
        self.assertAlmostEqual(summary.loc[0, "gini_index_mean"], expected.gini_index)
        self.assertEqual(summary.loc[0, "n_gini"], 1)

    def test_negative_rewards_do_not_fail_the_allocation(self):
        robot_ids = [agent.agent_id for agent in self.scenario.agents]
        record, rows = self.run_with_payoffs(dict(zip(robot_ids, [-1.0, 2.0, 3.0])))
        self.assertTrue(record["success"])
        self.assertTrue(record["reward_stage_completed"])
        self.assertTrue(record["invalid_reward_flag"])
        self.assertTrue(math.isnan(record["gini_index"]))
        self.assertEqual([row["reward"] for row in rows], [-1.0, 2.0, 3.0])  # never shifted or clipped

    def test_missing_reward_value_is_not_persisted_as_a_partial_group(self):
        robot_ids = [agent.agent_id for agent in self.scenario.agents]
        record, rows = self.run_with_payoffs({robot_ids[0]: 1.0, robot_ids[1]: float("nan"), robot_ids[2]: 2.0})
        self.assertTrue(record["reward_stage_completed"])
        self.assertFalse(record["invalid_reward_flag"])
        self.assertTrue(math.isnan(record["gini_index"]))
        self.assertEqual(rows, [])

    def test_actual_solver_returns_robot_only_payoffs_even_when_ids_overlap(self):
        assignments = allocation_adapters.greedy_adapter(self.scenario, 19, 5.0)["assignments"]
        stage_input = reward_adapters.RewardStageInput(self.scenario, assignments, self.config["reward"])
        first = reward_adapters.preference_aware_reward_adapter(stage_input, 812, 5.0)
        second = reward_adapters.preference_aware_reward_adapter(stage_input, 812, 5.0)
        robot_ids = [agent.agent_id for agent in self.scenario.agents]
        self.assertEqual(set(first.payoffs), set(robot_ids))
        self.assertEqual(first.payoffs, second.payoffs)
        pairs = reward_adapters.extract_robot_reward_pairs(first, robot_ids)
        self.assertEqual({robot_id for robot_id, _ in pairs}, set(robot_ids))

    def test_reward_solver_failure_keeps_runtime_and_does_not_fail_allocation(self):
        def broken(stage_input, seed, limit):
            raise RuntimeError("reward solver failed")
        record, rows = execute_run_with_rewards(self.spec, self.scenario, self.config, InProcessBackend(),
                                                allocation_adapters.greedy_adapter, broken)
        self.assertTrue(record["success"])
        self.assertFalse(record["reward_stage_completed"])
        self.assertIsNotNone(record["reward_runtime_seconds"])
        self.assertIn("RuntimeError: reward solver failed", record["error_message"])
        self.assertEqual(rows, [])

    def test_reward_timeout_keeps_elapsed_runtime_and_marks_stage_incomplete(self):
        class Backend:
            def __init__(self): self.calls = 0
            def run(self, adapter, stage_input, seed, limit):
                self.calls += 1
                if self.calls == 1:
                    assignments = allocation_adapters.greedy_adapter(stage_input, seed, limit)["assignments"]
                    return StageOutcome(True, True, 0.01, None, False, {"assignments": assignments})
                return StageOutcome(True, False, 2.5, "TimeoutError: limit", True, None)
        backend = Backend()
        record, rows = execute_run_with_rewards(self.spec, self.scenario, self.config, backend,
                                                allocation_adapters.greedy_adapter, lambda *_: None)
        self.assertTrue(record["success"])
        self.assertFalse(record["reward_stage_completed"])
        self.assertEqual(record["reward_runtime_seconds"], 2.5)
        self.assertEqual(rows, [])

    def test_allocation_failure_prevents_reward_stage(self):
        calls = []
        def reward(stage_input, seed, limit):
            calls.append(1)
            return SimpleNamespace(payoffs={})
        def invalid_allocation(scenario, seed, limit):
            return {"assignments": {}}
        record, rows = execute_run_with_rewards(self.spec, self.scenario, self.config, InProcessBackend(),
                                                invalid_allocation, reward)
        self.assertFalse(record["success"])
        self.assertFalse(record["reward_stage_completed"])
        self.assertTrue(math.isnan(record["reward_runtime_seconds"]))
        self.assertEqual(calls, [])
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()

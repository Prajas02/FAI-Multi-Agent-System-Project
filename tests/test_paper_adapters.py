"""Analytics adapters for the supplied paper allocation implementations."""

import dataclasses
import math
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from src.allocation import ga_paper, svca_paper
from src.analytics import allocation_adapters
from src.analytics.execution import InProcessBackend
from src.core.types import AllocationResult
from src.simulation.generator import ScenarioGenerator
from scripts import run_benchmarks as runner


ROOT = Path(__file__).resolve().parents[1]
PAPER_SVCA_CONFIG = ROOT / "configs" / "svca_paper.yaml"


def allocation_result(assignments):
    return AllocationResult(assignments, {}, {}, {}, 0.0, 0.0, "paper-test")


def runner_config():
    return {
        "experiment": {"id": "paper_adapter_test"},
        "evaluation": {"energy_penalty": {"strategy": "quadratic", "lambda_E": 1.0}},
        "seeds": {"reward_stage_label": "reward_stage"},
        "time_limits": {"reward_stage_seconds": 5.0},
        "reward": {
            "enabled": True, "method": "exact", "value_type": "facility",
            "preference_sensitivity_alpha": 0.0, "num_samples_Tw": 4,
            "burn_in": 0, "thin": None, "samples_per_stratum": 1,
            "max_log_gamma": 3.0,
        },
    }


class TestPaperAllocationAdapters(unittest.TestCase):
    def setUp(self):
        self.scenario = ScenarioGenerator(seed=702).generate_scenario(3, 4)
        self.assignments = {
            self.scenario.agents[0].agent_id: [self.scenario.tasks[2].task_id, self.scenario.tasks[0].task_id],
            self.scenario.agents[1].agent_id: [self.scenario.tasks[3].task_id],
            self.scenario.agents[2].agent_id: [self.scenario.tasks[1].task_id],
        }

    def test_paper_ga_uses_existing_defaults_with_only_seed_overridden(self):
        seed = 102938
        result = allocation_result(self.assignments)
        with mock.patch.object(allocation_adapters, "PaperGAAllocator") as factory:
            factory.return_value.solve.return_value = result
            payload = allocation_adapters.ga_paper_adapter(self.scenario, seed, 30.0)
        config = factory.call_args.args[0]
        self.assertEqual(dataclasses.asdict(config), dataclasses.asdict(ga_paper.PaperGAConfig(seed=seed)))
        self.assertEqual(payload["assignments"], self.assignments)
        self.assertEqual(list(payload["assignments"]), list(self.assignments))
        self.assertEqual(payload["assignments"][self.scenario.agents[0].agent_id],
                         [self.scenario.tasks[2].task_id, self.scenario.tasks[0].task_id])

    def test_paper_svca_loads_supplied_yaml_and_overrides_only_seed(self):
        seed = 564738
        supplied = svca_paper.PaperSVCAConfig.from_yaml(str(PAPER_SVCA_CONFIG))
        result = allocation_result(self.assignments)
        with mock.patch.object(allocation_adapters, "PaperSVCAAllocator") as factory:
            factory.return_value.solve.return_value = result
            payload = allocation_adapters.svca_paper_adapter(self.scenario, seed, 30.0)
        config = factory.call_args.args[0]
        supplied_values = dataclasses.asdict(supplied)
        actual_values = dataclasses.asdict(config)
        self.assertEqual(actual_values.pop("seed"), seed)
        supplied_values.pop("seed")
        self.assertEqual(actual_values, supplied_values)
        self.assertEqual(payload["assignments"], self.assignments)

    def test_both_paper_adapters_propagate_solver_failures(self):
        for class_name, adapter in (("PaperGAAllocator", allocation_adapters.ga_paper_adapter),
                                    ("PaperSVCAAllocator", allocation_adapters.svca_paper_adapter)):
            with self.subTest(adapter=class_name), mock.patch.object(
                allocation_adapters, class_name
            ) as factory:
                factory.return_value.solve.side_effect = RuntimeError("paper allocation failed")
                with self.assertRaisesRegex(RuntimeError, "paper allocation failed"):
                    adapter(self.scenario, 12, 2.0)

    def test_benchmark_config_registers_both_paper_variants_alongside_baselines(self):
        config_path = ROOT / "configs" / "experiments_config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        algorithms = config["benchmark"]["algorithms"]
        self.assertEqual(algorithms, ["Greedy", "GA", "SVCA", "GA_Paper", "SVCA_Paper"])
        self.assertTrue(config["benchmark"]["same_scenario_for_all_algorithms"])
        self.assertEqual(config["benchmark"]["repetitions_per_configuration"], 10)
        self.assertEqual(set(runner.ADAPTER_REGISTRY) - set(algorithms), set())
        self.assertIn("GA", algorithms)
        self.assertIn("SVCA", algorithms)

    def test_both_paper_variants_run_through_reward_and_gini_pipeline(self):
        result = allocation_result(self.assignments)
        robot_ids = [agent.agent_id for agent in self.scenario.agents]

        def reward_adapter(stage_input, seed, limit):
            self.assertEqual(stage_input.assignments, self.assignments)
            return SimpleNamespace(payoffs={robot_ids[0]: 1.0, robot_ids[1]: 2.0, robot_ids[2]: 3.0})

        for algorithm, factory_name, adapter in (
            ("GA_Paper", "PaperGAAllocator", allocation_adapters.ga_paper_adapter),
            ("SVCA_Paper", "PaperSVCAAllocator", allocation_adapters.svca_paper_adapter),
        ):
            with self.subTest(algorithm=algorithm), mock.patch.object(
                allocation_adapters, factory_name
            ) as factory:
                factory.return_value.solve.return_value = result
                spec = runner.RunSpec("C1", 3, 4, 0, "C1_run00", self.scenario.seed, algorithm, 442, 30.0)
                record, reward_rows = runner.execute_run_with_rewards(
                    spec, self.scenario, runner_config(), InProcessBackend(), adapter, reward_adapter,
                )
            self.assertTrue(record["success"])
            self.assertTrue(record["reward_stage_completed"])
            self.assertEqual(len(reward_rows), len(robot_ids))
            self.assertTrue(math.isfinite(record["gini_index"]))
            self.assertEqual({row["algorithm"] for row in reward_rows}, {algorithm})

    def test_paper_solver_failure_keeps_allocation_runtime_and_does_not_start_rewards(self):
        calls = []

        def reward_adapter(*args):
            calls.append(args)
            return SimpleNamespace(payoffs={})

        spec = runner.RunSpec("C1", 3, 4, 0, "C1_run00", self.scenario.seed, "SVCA_Paper", 443, 30.0)
        with mock.patch.object(allocation_adapters, "PaperSVCAAllocator") as factory:
            factory.return_value.solve.side_effect = RuntimeError("paper solver failed")
            record, reward_rows = runner.execute_run_with_rewards(
                spec, self.scenario, runner_config(), InProcessBackend(),
                allocation_adapters.svca_paper_adapter, reward_adapter,
            )
        self.assertFalse(record["success"])
        self.assertFalse(record["allocation_stage_completed"])
        self.assertTrue(math.isfinite(record["allocation_runtime_seconds"]))
        self.assertTrue(math.isnan(record["reward_runtime_seconds"]))
        self.assertFalse(record["reward_stage_completed"])
        self.assertEqual(calls, [])
        self.assertEqual(reward_rows, [])

    def test_total_runtime_is_independent_and_excludes_analytics_evaluation(self):
        robot_ids = [agent.agent_id for agent in self.scenario.agents]
        config = runner_config()
        config["reward"]["enabled"] = True

        def allocation_adapter(scenario, seed, limit):
            time.sleep(0.02)
            return {"assignments": self.assignments}

        def reward_adapter(stage_input, seed, limit):
            time.sleep(0.02)
            return SimpleNamespace(payoffs={robot_id: 1.0 for robot_id in robot_ids})

        original_evaluate = runner.evaluate_allocation

        def delayed_evaluation(*args, **kwargs):
            time.sleep(0.15)
            return original_evaluate(*args, **kwargs)

        spec = runner.RunSpec("C1", 3, 4, 0, "C1_run00", self.scenario.seed, "GA_Paper", 444, 30.0)
        wall_start = time.perf_counter()
        with mock.patch.object(runner, "evaluate_allocation", side_effect=delayed_evaluation):
            record, _ = runner.execute_run_with_rewards(
                spec, self.scenario, config, InProcessBackend(), allocation_adapter, reward_adapter
            )
        wall_elapsed = time.perf_counter() - wall_start
        stage_sum = record["allocation_runtime_seconds"] + record["reward_runtime_seconds"]
        self.assertGreater(record["total_runtime_seconds"], stage_sum)
        self.assertGreater(wall_elapsed - record["total_runtime_seconds"], 0.12)


if __name__ == "__main__":
    unittest.main()

"""Integration checks for real solver adapters using temporary result folders only."""

import contextlib
import copy
import importlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from src.analytics import allocation_adapters as adapters
from src.analytics.evaluation import validate_assignments
from src.analytics.execution import InProcessBackend
from src.analytics.logger import load_benchmark_results, load_robot_rewards
from src.core.types import AllocationResult
from src.simulation.generator import ScenarioGenerator

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments_config.yaml"
rb = importlib.import_module("scripts.run_benchmarks")
REAL_RESULTS = ROOT / "data" / "results" / "raw" / "benchmark_results.csv"


def _snapshot(path: Path):
    """(exists, size, mtime) of a file: unchanged snapshot = the file was not written."""
    if not path.exists():
        return (False, None, None)
    stat = path.stat()
    return (True, stat.st_size, stat.st_mtime_ns)


class TestRealAllocationAdapters(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="allocation_adapter_test_"))
        self.addCleanup(shutil.rmtree, str(self.tmp), True)

    def test_all_real_adapters_return_valid_assignments_for_a_small_scenario(self):
        scenario = ScenarioGenerator(seed=103).generate_scenario(3, 5)
        before = rb.scenario_fingerprint(scenario)
        for algorithm, adapter in (("Greedy", adapters.greedy_adapter), ("GA", adapters.ga_adapter),
                                   ("SVCA", adapters.svca_adapter)):
            with self.subTest(algorithm=algorithm):
                result = adapter(scenario, 2468, 30.0)
                validation = validate_assignments(result["assignments"], scenario)
                self.assertTrue(validation.valid, validation.errors)
        for adapter in (adapters.ga_adapter, adapters.svca_adapter):
            first = adapter(scenario, 2468, 30.0)["assignments"]
            second = adapter(scenario, 2468, 30.0)["assignments"]
            self.assertEqual(first, second)
        self.assertEqual(rb.scenario_fingerprint(scenario), before)

    def test_stochastic_adapters_receive_the_harness_derived_seed(self):
        dummy = AllocationResult({}, {}, {}, {}, 0.0, 0.0, "stub")
        with mock.patch.object(adapters, "GAClusteringAllocator") as factory:
            factory.return_value.solve.return_value = dummy
            adapters.ga_adapter(None, 123456, 10.0)
            self.assertEqual(factory.call_args.args[0].seed, 123456)
        with mock.patch.object(adapters, "SVCAAllocator") as factory:
            factory.return_value.solve.return_value = dummy
            adapters.svca_adapter(None, 654321, 10.0)
            self.assertEqual(factory.call_args.args[0].seed, 654321)

    def test_real_adapters_run_through_harness_and_write_schema_records(self):
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        config["experiment"]["id"] = "adapter_smoke"
        config["benchmark"]["repetitions_per_configuration"] = 1
        config["benchmark"]["algorithms"] = ["Greedy", "GA", "SVCA"]
        config["configurations"] = [{"id": "C1", "num_robots": 5, "num_tasks": 5}]
        config_path = self.tmp / "mini.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        real_before = _snapshot(REAL_RESULTS)
        seen, seen_scenarios = [], []
        originals = {name: getattr(adapters, name) for name in
                     ("greedy_adapter", "ga_adapter", "svca_adapter")}
        for name, adapter in originals.items():
            def tracking(scenario, seed, limit, _adapter=adapter):
                payload = _adapter(scenario, seed, limit)
                seen.append((rb.scenario_fingerprint(scenario), seed, limit, copy.deepcopy(payload["assignments"])))
                seen_scenarios.append(copy.deepcopy(scenario))
                return payload
            setattr(adapters, name, tracking)
            self.addCleanup(setattr, adapters, name, adapter)
        results_dir, summary_dir = self.tmp / "raw", self.tmp / "summary"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = rb.main(["--config", str(config_path), "--results-dir", str(results_dir),
                            "--summary-dir", str(summary_dir), "--backend", "inprocess"])
        self.assertEqual(code, rb.EXIT_OK, output.getvalue())
        self.assertEqual(len(seen), 3)
        self.assertEqual(len({entry[0] for entry in seen}), 1)
        self.assertEqual(len({entry[1] for entry in seen}), 3)  # distinct algorithm-specific seeds
        frame = load_benchmark_results(results_dir / "benchmark_results.csv")
        self.assertEqual(list(frame["algorithm"]), ["Greedy", "GA", "SVCA"])
        self.assertTrue(frame["success"].all())
        self.assertTrue(frame["allocation_stage_completed"].all())
        self.assertTrue((frame["allocation_runtime_seconds"] >= 0).all())
        self.assertTrue(frame["allocation_cost"].notna().all())
        self.assertTrue(frame["energy_feasible"].notna().all())
        self.assertTrue(frame["reward_stage_completed"].all())
        self.assertTrue(frame["reward_runtime_seconds"].notna().all())
        rewards = load_robot_rewards(results_dir / "robot_rewards.csv")
        self.assertEqual(len(rewards), 15)
        self.assertTrue(frame["gini_index"].notna().all())
        for index, (_, row) in enumerate(frame.iterrows()):
            expected = rb.evaluate_allocation(seen_scenarios[index], seen[index][3], config["evaluation"]["energy_penalty"])
            self.assertAlmostEqual(row["allocation_cost"], expected.allocation_cost, places=9)
            self.assertEqual(bool(row["energy_feasible"]), expected.energy_feasible)
        # The run must not touch the project's real results (which may exist after a real benchmark).
        self.assertEqual(_snapshot(REAL_RESULTS), real_before)

    def test_solver_exception_becomes_a_failed_logger_record(self):
        scenario = ScenarioGenerator(seed=104).generate_scenario(3, 4)
        config = {"experiment": {"id": "adapter_failure"},
                  "evaluation": {"energy_penalty": {"strategy": "quadratic", "lambda_E": 1.0}}}
        spec = rb.RunSpec("C1", 3, 4, 0, "C1_run00", scenario.seed, "GA", 77, 2.0)
        with mock.patch.object(adapters.GAClusteringAllocator, "solve", side_effect=RuntimeError("solver failed")):
            record = rb.execute_run(spec, scenario, config, InProcessBackend(), adapters.ga_adapter)
        self.assertFalse(record["success"])
        self.assertFalse(record["allocation_stage_completed"])
        self.assertIn("RuntimeError: solver failed", record["error_message"])
        from src.analytics.logger import append_benchmark_results
        append_benchmark_results([record], self.tmp / "failed")
        persisted = load_benchmark_results(self.tmp / "failed" / "benchmark_results.csv")
        self.assertEqual(len(persisted), 1)
        self.assertFalse(bool(persisted.loc[0, "success"]))


if __name__ == "__main__":
    unittest.main()

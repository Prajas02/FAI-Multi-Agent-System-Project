"""Tests for scripts/run_benchmarks.py (the benchmark runner).

Most orchestration cases use controlled fake adapters; dedicated integration
tests exercise the actual Greedy, GA and SVCA adapters. Nothing writes to the
project's benchmark result folders.
"""

import contextlib
import copy
import importlib.util
import io
import math
import os
import shutil
import sys
import tempfile
import textwrap
import unittest
import uuid
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml

from src.analytics import seeding
from src.analytics.evaluation import EvaluationError, evaluate_allocation
from src.analytics.logger import (
    DEFAULT_RESULTS_DIR,
    DuplicateKeyError,
    append_robot_rewards,
    load_benchmark_results,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_benchmarks.py"
REAL_CONFIG = REPO_ROOT / "configs" / "experiments_config.yaml"

_spec = importlib.util.spec_from_file_location("run_benchmarks", str(SCRIPT))
rb = importlib.util.module_from_spec(_spec)
sys.modules["run_benchmarks"] = rb
_spec.loader.exec_module(rb)

MODULE = __name__
RUNTIME_COLUMNS = ["allocation_runtime_seconds", "reward_runtime_seconds", "total_runtime_seconds"]


# ---------------------------------------------------------------------------
# Fake adapters (the runner's analytics-side contract:
#   adapter(scenario, seed, time_limit) -> {"assignments": {robot_id: [task_id, ...]}})
# ---------------------------------------------------------------------------

SEEN = []          # (adapter name, seed, time_limit, scenario fingerprint) of every in-process call


def _round_robin(scenario):
    robots = [a.agent_id for a in scenario.agents]
    plan = {r: [] for r in robots}
    for index, task in enumerate(scenario.tasks):
        plan[robots[index % len(robots)]].append(task.task_id)
    return plan


def adapter_round_robin(scenario, seed, time_limit):
    SEEN.append(("round_robin", seed, time_limit, rb.scenario_fingerprint(scenario)))
    return {"assignments": _round_robin(scenario)}


def adapter_nearest(scenario, seed, time_limit):
    SEEN.append(("nearest", seed, time_limit, rb.scenario_fingerprint(scenario)))
    positions = {a.agent_id: a.position for a in scenario.agents}
    plan = {r: [] for r in positions}
    for task in scenario.tasks:
        robot = min(positions, key=lambda r: (positions[r].distance_to(task.location), r))
        plan[robot].append(task.task_id)
        positions[robot] = task.location
    return {"assignments": plan}


def adapter_missing_task(scenario, seed, time_limit):
    plan = _round_robin(scenario)
    first = next(iter(plan))
    plan[first] = plan[first][1:]
    return {"assignments": plan}


def adapter_duplicate_task(scenario, seed, time_limit):
    plan = _round_robin(scenario)
    robots = list(plan)
    plan[robots[1]].append(plan[robots[0]][0])
    return {"assignments": plan}


def adapter_unknown_robot(scenario, seed, time_limit):
    plan = _round_robin(scenario)
    plan[999] = []
    return {"assignments": plan}


def adapter_no_assignments(scenario, seed, time_limit):
    return {"cost": 12.0}


def adapter_not_a_mapping(scenario, seed, time_limit):
    return [1, 2, 3]


def adapter_raises(scenario, seed, time_limit):
    raise RuntimeError("solver blew up")


def adapter_interrupt(scenario, seed, time_limit):
    raise KeyboardInterrupt()


def adapter_mutator(scenario, seed, time_limit):
    plan = _round_robin(scenario)
    scenario.agents[0].current_energy = -1.0        # tries to corrupt "its" scenario
    scenario.tasks.clear()
    return {"assignments": plan}


def ref(name):
    return "%s:%s" % (MODULE, name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(directory, algorithms=("Greedy", "GA"), repetitions=2, limits=None, mutate=None):
    """A small experiment config derived from the real one (same seeds section)."""
    config = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    config["experiment"]["id"] = "mini"
    config["benchmark"]["repetitions_per_configuration"] = repetitions
    config["benchmark"]["algorithms"] = list(algorithms)
    config["reward"]["enabled"] = False  # orchestration tests isolate the allocation stage
    config["configurations"] = [{"id": "C1", "num_robots": 5, "num_tasks": 5}]
    config["time_limits"]["per_algorithm"] = {a: float((limits or {}).get(a, 5.0)) for a in algorithms}
    if mutate:
        mutate(config)
    path = Path(directory) / "mini_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


class RunnerTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="runner_test_"))
        self.addCleanup(shutil.rmtree, str(self.dir), True)
        self.results = self.dir / "raw"
        self.summary = self.dir / "tables"
        SEEN.clear()

    def cli(self, config_path, *extra, registry=None, backend="inprocess"):
        argv = ["--config", str(config_path), "--results-dir", str(self.results),
                "--summary-dir", str(self.summary), "--backend", backend] + list(extra)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rb.main(argv, registry=registry)
        return code, out.getvalue(), err.getvalue()

    def results_frame(self):
        return load_benchmark_results(self.results / "benchmark_results.csv")


GREEDY_AND_GA = {"Greedy": ref("adapter_round_robin"), "GA": ref("adapter_nearest")}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfiguration(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))

    def check(self, mutate, pattern):
        config = copy.deepcopy(self.config)
        mutate(config)
        with self.assertRaisesRegex(rb.ConfigError, pattern):
            rb.validate_config(config, REAL_CONFIG, DEFAULT_RESULTS_DIR)

    def test_real_configuration_is_valid(self):
        rb.validate_config(self.config, REAL_CONFIG, DEFAULT_RESULTS_DIR)

    def test_missing_file_and_bad_yaml(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(directory), True)
        with self.assertRaisesRegex(rb.ConfigError, "not found"):
            rb.load_config(directory / "nope.yaml")
        (directory / "bad.yaml").write_text("a: [unclosed", encoding="utf-8")
        with self.assertRaises(rb.ConfigError):
            rb.load_config(directory / "bad.yaml")
        (directory / "list.yaml").write_text("- a\n", encoding="utf-8")
        with self.assertRaisesRegex(rb.ConfigError, "mapping"):
            rb.load_config(directory / "list.yaml")

    def test_entity_count_outside_the_range(self):
        self.check(lambda c: c["configurations"][0].update(num_robots=2, num_tasks=3), "outside")
        self.check(lambda c: c["configurations"][-1].update(num_tasks=100), "outside")

    def test_missing_timeout_for_an_algorithm(self):
        self.check(lambda c: c["time_limits"]["per_algorithm"].pop("SVCA"), "time_limits.per_algorithm.*SVCA")
        self.check(lambda c: c["time_limits"]["per_algorithm"].update(GA=0), "positive number for 'GA'")

    def test_stored_total_entity_count_is_rejected(self):
        self.check(lambda c: c["configurations"][0].update(total_entities=10), "exactly the keys")

    def test_duplicate_configuration_ids(self):
        self.check(lambda c: c["configurations"][1].update(id="C1"), "duplicated")

    def test_coalition_stability_must_stay_disabled(self):
        self.check(lambda c: c["evaluation"]["coalition_stability"].update(enabled=True), "coalition_stability")

    def test_energy_penalty_settings(self):
        self.check(lambda c: c["evaluation"]["energy_penalty"].update(strategy="cubic"), "strategy")
        self.check(lambda c: c["evaluation"]["energy_penalty"].update(lambda_E=-1.0), "lambda_E")
        self.check(lambda c: c["evaluation"]["energy_penalty"].update(lambda_E=float("nan")), "lambda_E")

    def test_other_invalid_settings(self):
        self.check(lambda c: c["experiment"].update(id="../x"), "experiment.id")
        self.check(lambda c: c["experiment"].update(base_seed=-1), "base_seed")
        self.check(lambda c: c["environment"].update(dimensions=3), "dimensions")
        self.check(lambda c: c["environment"].update(width_m=0), "width_m")
        self.check(lambda c: c["benchmark"].update(repetitions_per_configuration=0), "repetitions")
        self.check(lambda c: c["benchmark"].update(execution_mode="parallel"), "sequential")
        self.check(lambda c: c["benchmark"].update(algorithms=["MILP", "MILP"]), "unique")
        self.check(lambda c: c["benchmark"].update(scenario_id_format="C1"), "distinct id")

    def test_invalid_seed_templates_in_the_configuration_file(self):
        # the seed functions read the file itself, so the invalid templates must be in a file
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(directory), True)
        config = copy.deepcopy(self.config)
        config["seeds"]["messages"]["scenario"] = "scenario|{base_seed}"
        path = directory / "bad_seeds.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with self.assertRaisesRegex(rb.ConfigError, "seeds"):
            rb.validate_config(config, path, DEFAULT_RESULTS_DIR)

    def test_output_paths_must_match_the_logger_layout(self):
        self.check(lambda c: c["output"].update(benchmark_results="elsewhere/results.csv"), "output.benchmark_results")
        config = copy.deepcopy(self.config)
        config["output"]["robot_rewards"] = "elsewhere/rewards.csv"
        rb.validate_config(config, REAL_CONFIG, Path("/tmp/anything"), check_output_paths=False)   # overridden dir

    def test_all_problems_are_listed_together(self):
        config = copy.deepcopy(self.config)
        config["experiment"]["base_seed"] = -5
        config["environment"]["dimensions"] = 7
        with self.assertRaises(rb.ConfigError) as caught:
            rb.validate_config(config, REAL_CONFIG, DEFAULT_RESULTS_DIR)
        self.assertIn("base_seed", str(caught.exception))
        self.assertIn("dimensions", str(caught.exception))


# ---------------------------------------------------------------------------
# Plan and scenarios
# ---------------------------------------------------------------------------

class TestPlanAndScenarios(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))

    def test_real_plan_has_400_runs_in_configuration_major_order(self):
        plan = rb.build_run_plan(self.config, REAL_CONFIG)
        self.assertEqual(len(plan), 8 * 10 * 5)
        self.assertEqual([s.algorithm for s in plan[:5]], ["Greedy", "GA", "SVCA", "GA_Paper", "SVCA_Paper"])
        self.assertEqual({s.scenario_id for s in plan[:5]}, {"C1_run00"})
        self.assertEqual(plan[5].scenario_id, "C1_run01")
        self.assertEqual(plan[-1].scenario_id, "C8_run09")
        self.assertEqual(len({s.scenario_id for s in plan}), 80)
        self.assertEqual(max(s.num_robots + s.num_tasks for s in plan), 80)

    def test_seeds_follow_the_documented_derivation(self):
        plan = rb.build_run_plan(self.config, REAL_CONFIG)
        first = plan[0]
        self.assertEqual(first.scenario_seed, 1593276916)               # design test vector
        self.assertEqual(first.algorithm, "Greedy")
        self.assertEqual(first.algorithm_seed, seeding.algorithm_seed(1593276916, "Greedy"))
        for spec in plan[:3]:
            self.assertEqual(spec.scenario_seed, 1593276916)            # shared by all algorithms
        self.assertEqual(len({s.algorithm_seed for s in plan[:3]}), 3)  # separate per algorithm
        self.assertEqual([s.time_limit_seconds for s in plan[:3]], [300.0] * 3)

    def test_filters(self):
        plan = rb.build_run_plan(self.config, REAL_CONFIG, ["C2"], ["GA", "SVCA"], 3)
        self.assertEqual(len(plan), 1 * 3 * 2)
        self.assertEqual({s.config_id for s in plan}, {"C2"})
        self.assertEqual([s.algorithm for s in plan[:2]], ["GA", "SVCA"])     # configured order

    def test_scenarios_are_reproducible_fresh_and_distinct(self):
        plan = rb.build_run_plan(self.config, REAL_CONFIG)
        a, b, other = plan[0], plan[5], plan[50]
        first = rb.generate_scenario(self.config, a)
        again = rb.generate_scenario(self.config, a)
        self.assertEqual(rb.scenario_fingerprint(first), rb.scenario_fingerprint(again))
        self.assertEqual(first.seed, a.scenario_seed)                   # truthful seed (fresh generator)
        self.assertEqual((len(first.agents), len(first.tasks)), (5, 5))
        self.assertEqual((first.map_width, first.map_height), (500.0, 500.0))
        self.assertNotEqual(rb.scenario_fingerprint(first), rb.scenario_fingerprint(rb.generate_scenario(self.config, b)))
        self.assertEqual(len(rb.generate_scenario(self.config, other).agents), other.num_robots)


# ---------------------------------------------------------------------------
# Dry run and unavailable algorithms
# ---------------------------------------------------------------------------

class TestDryRunAndAvailability(RunnerTestCase):
    def test_dry_run_with_the_real_configuration(self):
        code, out, err = self.cli(REAL_CONFIG, "--dry-run")
        self.assertEqual(code, 0, err)
        self.assertIn("400 runs", out)
        self.assertIn("1593276916", out)
        self.assertNotIn("UNAVAILABLE", out)
        self.assertIn("MILP is registered but not in benchmark.algorithms", out)
        self.assertIn("reward stage: enabled: PreferenceAwareShapleySolver", out)
        self.assertFalse(self.results.exists())                        # nothing written
        self.assertFalse(self.summary.exists())

    def test_dry_run_needs_no_adapters_and_ignores_broken_ones(self):
        code, out, _ = self.cli(make_config(self.dir), "--dry-run", "--adapter", "Greedy=no_such_module_xyz:f")
        self.assertEqual(code, 0)
        self.assertIn("UNAVAILABLE Greedy", out)

    def test_real_run_without_adapters_is_refused_and_writes_nothing(self):
        missing = {name: "missing:test_adapter" for name in ("Greedy", "GA", "SVCA")}
        code, out, err = self.cli(REAL_CONFIG, registry=missing)
        self.assertEqual(code, 3)
        self.assertIn("algorithms without an adapter: Greedy, GA, SVCA", err)
        self.assertFalse(self.results.exists())

    def test_adapter_that_cannot_be_imported_is_unavailable_not_fabricated(self):
        code, out, err = self.cli(make_config(self.dir), "--adapter", "Greedy=no_such_module_xyz:f",
                                  "--adapter", "GA=no_such_module_xyz:g")
        self.assertEqual(code, 3)
        self.assertIn("without an adapter", err)
        self.assertFalse(self.results.exists())

    def test_nothing_runs_if_no_algorithm_is_available_even_with_skip_unavailable(self):
        code, out, err = self.cli(make_config(self.dir), "--skip-unavailable",
                                  "--adapter", "Greedy=no_such_module_xyz:f",
                                  "--adapter", "GA=no_such_module_xyz:g")
        self.assertEqual(code, 3)
        self.assertIn("no algorithm is available", err)
        self.assertFalse(self.results.exists())

    def test_skip_unavailable_runs_only_available_algorithms(self):
        code, out, err = self.cli(make_config(self.dir), "--skip-unavailable",
                                  "--adapter", "Greedy=" + ref("adapter_round_robin"),
                                  "--adapter", "GA=no_such_module_xyz:g")
        self.assertEqual(code, 0, err)
        frame = self.results_frame()
        self.assertEqual(set(frame["algorithm"]), {"Greedy"})
        self.assertEqual(len(frame), 2)
        self.assertIn("UNAVAILABLE GA", out)
        metadata = yaml.safe_load((self.results / "metadata_mini.yaml").read_text(encoding="utf-8"))
        self.assertIn("GA", metadata["extra"]["harness"]["skipped_unavailable_algorithms"])

    def test_registry_can_be_injected_programmatically(self):
        code, _, err = self.cli(make_config(self.dir), registry=GREEDY_AND_GA)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.results_frame()), 4)


# ---------------------------------------------------------------------------
# End to end (in-process fake adapters, real simulator/evaluation/logger)
# ---------------------------------------------------------------------------

class TestEndToEnd(RunnerTestCase):
    def run_mini(self, **kwargs):
        config_path = make_config(self.dir, **kwargs)
        code, out, err = self.cli(config_path, registry=GREEDY_AND_GA)
        self.assertEqual(code, 0, err)
        return config_path, out

    def test_records_scenarios_seeds_and_costs(self):
        config_path, out = self.run_mini()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        frame = self.results_frame()
        self.assertEqual(len(frame), 4)
        self.assertEqual(list(frame["scenario_id"]), ["C1_run00", "C1_run00", "C1_run01", "C1_run01"])
        self.assertEqual(list(frame["algorithm"]), ["Greedy", "GA", "Greedy", "GA"])
        self.assertTrue(frame["success"].all())
        self.assertTrue(frame["allocation_stage_completed"].all())
        self.assertFalse(frame["reward_stage_completed"].any())             # reward stage blocked, never faked
        self.assertTrue(frame["reward_runtime_seconds"].isna().all())
        self.assertTrue(frame["gini_index"].isna().all())
        self.assertFalse(frame["invalid_reward_flag"].any())
        self.assertTrue(frame["coalition_stability"].isna().all())
        self.assertFalse((self.results / "robot_rewards.csv").exists())     # no fabricated reward records
        self.assertTrue((frame["total_entities"] == 10).all())
        for rep in (0, 1):
            expected_seed = seeding.scenario_seed(42, "C1", rep)
            self.assertTrue((frame[frame["scenario_id"] == "C1_run0%d" % rep]["random_seed"] == expected_seed).all())
        self.assertTrue((frame["total_runtime_seconds"] >= frame["allocation_runtime_seconds"]).all())
        self.assertTrue((frame["allocation_runtime_seconds"] >= 0).all())

        # identical scenario for every algorithm of a repetition, a different one per repetition
        self.assertEqual(len(SEEN), 4)
        (n0, s0, t0, f0), (n1, s1, t1, f1), (n2, s2, t2, f2), (n3, s3, t3, f3) = SEEN
        self.assertEqual(f0, f1)
        self.assertEqual(f2, f3)
        self.assertNotEqual(f0, f2)
        scenario_seed0 = seeding.scenario_seed(42, "C1", 0)
        self.assertEqual(s0, seeding.algorithm_seed(scenario_seed0, "Greedy"))
        self.assertEqual(s1, seeding.algorithm_seed(scenario_seed0, "GA"))
        self.assertNotEqual(s0, s1)
        self.assertEqual((t0, t1), (5.0, 5.0))                              # time limit from the config

        # allocation cost and energy feasibility equal an independent evaluation
        for _, row in frame.iterrows():
            spec = [s for s in rb.build_run_plan(config, config_path)
                    if s.scenario_id == row["scenario_id"] and s.algorithm == row["algorithm"]][0]
            scenario = rb.generate_scenario(config, spec)
            plan = (adapter_round_robin if row["algorithm"] == "Greedy" else adapter_nearest)(scenario, 0, 1.0)
            SEEN.pop()
            evaluation = evaluate_allocation(scenario, plan["assignments"], config["evaluation"]["energy_penalty"])
            self.assertAlmostEqual(row["allocation_cost"], evaluation.allocation_cost, places=9)
            self.assertEqual(bool(row["energy_feasible"]), evaluation.energy_feasible)

    def test_metadata_summary_and_console_output(self):
        config_path, out = self.run_mini()
        metadata = yaml.safe_load((self.results / "metadata_mini.yaml").read_text(encoding="utf-8"))
        self.assertEqual(metadata["experiment_id"], "mini")
        self.assertEqual(metadata["config"], yaml.safe_load(config_path.read_text(encoding="utf-8")))
        self.assertEqual(metadata["seed_info"]["base_seed"], 42)
        self.assertEqual(metadata["time_limits"]["per_algorithm"], {"Greedy": 5.0, "GA": 5.0})
        harness = metadata["extra"]["harness"]
        self.assertEqual(harness["simulator_energy_penalty_factor"], 0.0)
        self.assertIn("disabled", harness["reward_stage"])
        self.assertIn("git_commit", metadata["environment"])
        summary = pd.read_csv(self.summary / "summary_mini.csv")
        self.assertEqual(sorted(summary["algorithm"]), ["GA", "Greedy"])
        self.assertTrue((summary["n_runs"] == 2).all())
        self.assertTrue((summary["n_success"] == 2).all())
        self.assertTrue((summary["n_reward_runtime"] == 0).all())
        self.assertIn("[4/4]", out)
        self.assertIn("summary written", out)

    def test_runs_are_reproducible(self):
        self.run_mini()
        first = self.results_frame().drop(columns=RUNTIME_COLUMNS)
        shutil.rmtree(str(self.results))
        shutil.rmtree(str(self.summary))
        SEEN.clear()
        self.run_mini()
        second = self.results_frame().drop(columns=RUNTIME_COLUMNS)
        pd.testing.assert_frame_equal(first, second)

    def test_solver_cannot_change_the_scenario_other_algorithms_receive(self):
        config_path = make_config(self.dir)
        registry = {"Greedy": ref("adapter_mutator"), "GA": ref("adapter_round_robin")}
        code, _, err = self.cli(config_path, registry=registry)
        self.assertEqual(code, 0, err)
        frame = self.results_frame()
        self.assertTrue(frame["success"].all())
        self.assertEqual(len(SEEN), 2)                      # only GA records fingerprints (rep 0 and 1)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for rep, seen in enumerate(SEEN):
            spec = rb.build_run_plan(config, config_path)[rep * 2]
            self.assertEqual(seen[3], rb.scenario_fingerprint(rb.generate_scenario(config, spec)))


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

class TestFailureHandling(RunnerTestCase):
    def run_with_bad_first_algorithm(self, bad):
        config_path = make_config(self.dir, algorithms=("GA", "Greedy"))
        code, out, err = self.cli(config_path, registry={"GA": ref(bad), "Greedy": ref("adapter_round_robin")})
        return code, out, err, self.results_frame()

    def assert_failed_but_others_ran(self, frame, completed, pattern):
        bad, good = frame[frame["algorithm"] == "GA"], frame[frame["algorithm"] == "Greedy"]
        self.assertEqual(len(bad), 2)
        self.assertEqual(len(good), 2)                       # the other algorithm still ran on the same scenarios
        self.assertTrue(good["success"].all())
        self.assertFalse(bad["success"].any())
        self.assertTrue(bad["allocation_cost"].isna().all())
        self.assertTrue(bad["energy_feasible"].isna().all())
        self.assertEqual(bool(bad["allocation_stage_completed"].iloc[0]), completed)
        self.assertFalse(bad["reward_stage_completed"].any())
        self.assertTrue(bad["error_message"].str.contains(pattern).all(), list(bad["error_message"]))
        self.assertTrue(bad["allocation_runtime_seconds"].notna().all())      # elapsed time is kept

    def test_missing_task(self):
        code, out, err, frame = self.run_with_bad_first_algorithm("adapter_missing_task")
        self.assertEqual(code, 0, err)
        self.assert_failed_but_others_ran(frame, True, "invalid allocation: tasks not assigned")

    def test_duplicate_task(self):
        code, out, err, frame = self.run_with_bad_first_algorithm("adapter_duplicate_task")
        self.assertEqual(code, 0, err)
        self.assert_failed_but_others_ran(frame, True, "invalid allocation: task .* is assigned 2 times")

    def test_unknown_robot(self):
        code, out, err, frame = self.run_with_bad_first_algorithm("adapter_unknown_robot")
        self.assertEqual(code, 0, err)
        self.assert_failed_but_others_ran(frame, True, "unknown robot id 999")

    def test_output_without_assignments(self):
        code, out, err, frame = self.run_with_bad_first_algorithm("adapter_no_assignments")
        self.assertEqual(code, 0, err)
        self.assert_failed_but_others_ran(frame, True, "'assignments' entry")

    def test_output_that_is_not_a_mapping(self):
        code, out, err, frame = self.run_with_bad_first_algorithm("adapter_not_a_mapping")
        self.assertEqual(code, 0, err)
        self.assert_failed_but_others_ran(frame, True, "'assignments' entry")

    def test_adapter_exception_is_recorded_and_not_retried(self):
        with mock.patch.object(sys.modules[MODULE], "adapter_raises", wraps=adapter_raises) as spy:
            code, out, err, frame = self.run_with_bad_first_algorithm("adapter_raises")
        self.assertEqual(code, 0, err)
        self.assert_failed_but_others_ran(frame, False, "RuntimeError: solver blew up")
        self.assertEqual(spy.call_count, 2)                    # once per scenario: no silent retry

    def test_energy_infeasible_plan_is_still_a_success(self):
        def poor_robots(config):
            config["environment"].update(width_m=500, height_m=500)

        config_path = make_config(self.dir, algorithms=("Greedy",), mutate=poor_robots)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        # use the real evaluator on a plan that overloads one robot (batteries of this seed are not
        # large enough for all tasks), through the runner's own execute_run
        spec = rb.build_run_plan(config, config_path)[0]
        master = rb.generate_scenario(config, spec)
        for agent in master.agents:
            agent.current_energy = agent.energy_capacity = 1.0
        outcome = rb.StageOutcome(True, True, 0.01, None, False, {"assignments": _round_robin(master)})
        backend = mock.Mock()
        backend.run.return_value = outcome
        record = rb.execute_run(spec, master, config, backend, "ignored")
        self.assertTrue(record["success"])
        self.assertIs(record["energy_feasible"], False)
        self.assertGreater(record["allocation_cost"], 0)
        self.assertFalse(record["reward_stage_completed"])

    def test_evaluation_failure_aborts_the_experiment_and_keeps_earlier_runs(self):
        calls = []

        def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise EvaluationError("simulator disagreement")
            return evaluate_allocation(*args, **kwargs)

        with mock.patch.object(rb, "evaluate_allocation", flaky):
            code, out, err = self.cli(make_config(self.dir), registry=GREEDY_AND_GA)
        self.assertEqual(code, 5)
        self.assertIn("evaluation of a validated plan failed", err)
        self.assertEqual(len(self.results_frame()), 1)         # the first run stays, the failing one is not written

    def test_logger_errors_stop_the_experiment(self):
        with mock.patch.object(rb, "append_benchmark_results", side_effect=DuplicateKeyError("dup")):
            code, out, err = self.cli(make_config(self.dir), registry=GREEDY_AND_GA)
        self.assertEqual(code, 4)
        self.assertIn("cannot record", err)

    def test_interruption_does_not_record_the_run_in_progress(self):
        code, out, err = self.cli(make_config(self.dir, algorithms=("Greedy", "GA")),
                                  registry={"Greedy": ref("adapter_round_robin"), "GA": ref("adapter_interrupt")})
        self.assertEqual(code, 130)
        self.assertIn("--resume", err)
        self.assertEqual(list(self.results_frame()["algorithm"]), ["Greedy"])

    def test_every_failure_record_is_a_valid_logger_record(self):
        # append_benchmark_results validates all five cross-field rules; reaching here means they hold
        for bad in ("adapter_missing_task", "adapter_raises", "adapter_no_assignments"):
            with self.subTest(bad=bad):
                shutil.rmtree(str(self.results), ignore_errors=True)
                code, _, err, frame = self.run_with_bad_first_algorithm(bad)
                self.assertEqual(code, 0, err)
                self.assertEqual(len(frame), 4)


# ---------------------------------------------------------------------------
# Resume, existing data, orphans, overrides
# ---------------------------------------------------------------------------

class TestResumeAndExistingData(RunnerTestCase):
    def test_second_run_without_resume_is_refused(self):
        config_path = make_config(self.dir)
        self.assertEqual(self.cli(config_path, registry=GREEDY_AND_GA)[0], 0)
        before = (self.results / "benchmark_results.csv").read_bytes()
        code, _, err = self.cli(config_path, registry=GREEDY_AND_GA)
        self.assertEqual(code, 3)
        self.assertIn("already exists", err)
        self.assertEqual((self.results / "benchmark_results.csv").read_bytes(), before)

    def test_resume_with_nothing_pending_adds_nothing(self):
        config_path = make_config(self.dir)
        self.cli(config_path, registry=GREEDY_AND_GA)
        before = (self.results / "benchmark_results.csv").read_bytes()
        SEEN.clear()
        code, out, err = self.cli(config_path, "--resume", registry=GREEDY_AND_GA)
        self.assertEqual(code, 0, err)
        self.assertEqual(SEEN, [])                              # no adapter was called
        self.assertEqual((self.results / "benchmark_results.csv").read_bytes(), before)

    def test_resume_adds_an_algorithm_that_became_available(self):
        config_path = make_config(self.dir)
        self.cli(config_path, "--skip-unavailable", "--adapter", "GA=missing:test_adapter",
                 registry={"Greedy": ref("adapter_round_robin")})
        self.assertEqual(len(self.results_frame()), 2)
        code, out, err = self.cli(config_path, "--resume", registry=GREEDY_AND_GA)
        self.assertEqual(code, 0, err)
        frame = self.results_frame()
        self.assertEqual(len(frame), 4)                         # only the GA runs were added; no duplicates
        self.assertEqual(sorted(frame["algorithm"]), ["GA", "GA", "Greedy", "Greedy"])
        self.assertEqual(len(pd.read_csv(self.summary / "summary_mini.csv")), 2)

    def test_resume_after_an_interrupted_run(self):
        config_path = make_config(self.dir)
        self.cli(config_path, registry={"Greedy": ref("adapter_round_robin"), "GA": ref("adapter_interrupt")})
        self.assertEqual(len(self.results_frame()), 1)
        code, out, err = self.cli(config_path, "--resume", registry=GREEDY_AND_GA)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.results_frame()), 4)

    def test_resume_refuses_a_changed_configuration(self):
        self.cli(make_config(self.dir), registry=GREEDY_AND_GA)
        changed = make_config(self.dir, mutate=lambda c: c["experiment"].update(base_seed=43))
        code, _, err = self.cli(changed, "--resume", registry=GREEDY_AND_GA)
        self.assertEqual(code, 3)
        self.assertIn("differs", err)

    def test_resume_without_a_previous_run_starts_fresh(self):
        code, _, err = self.cli(make_config(self.dir), "--resume", registry=GREEDY_AND_GA)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.results_frame()), 4)

    def test_any_orphan_reward_row_aborts_even_outside_the_plan(self):
        self.results.mkdir(parents=True)
        append_robot_rewards([{"experiment_id": "mini", "scenario_id": "C1_run07", "algorithm": "SVCA",
                               "robot_id": 1, "reward": 1.0}], self.results)
        append_robot_rewards([{"experiment_id": "mini_partial", "scenario_id": "C1_run07", "algorithm": "SVCA",
                               "robot_id": 1, "reward": 1.0}], self.results)
        config_path = make_config(self.dir)
        cases = [
            [], ["--resume"], ["--dry-run"],                                  # orphan of 'mini', not in the plan
            ["--only-algorithms", "Greedy", "--repetitions", "1", "--experiment-id", "mini_partial"],   # filtered run
        ]
        for extra in cases:
            with self.subTest(extra=extra):
                code, _, err = self.cli(config_path, *extra, registry=GREEDY_AND_GA)
                self.assertEqual(code, 3, err)
                self.assertIn("orphan reward rows", err)
        self.assertFalse((self.results / "benchmark_results.csv").exists())
        self.assertFalse((self.results / "metadata_mini.yaml").exists())
        self.assertFalse((self.results / "metadata_mini_partial.yaml").exists())

    def test_orphans_of_another_experiment_are_not_an_error(self):
        self.results.mkdir(parents=True)
        append_robot_rewards([{"experiment_id": "other", "scenario_id": "C1_run00", "algorithm": "GA",
                               "robot_id": 1, "reward": 1.0}], self.results)
        code, _, err = self.cli(make_config(self.dir), registry=GREEDY_AND_GA)
        self.assertEqual(code, 0, err)

    def test_partial_runs_need_their_own_experiment_id(self):
        config_path = make_config(self.dir)
        for flags in (["--only-algorithms", "Greedy"], ["--only-configs", "C1"], ["--repetitions", "1"]):
            code, _, err = self.cli(config_path, *flags, registry=GREEDY_AND_GA)
            self.assertEqual(code, 2)
            self.assertIn("--experiment-id", err)
        code, _, err = self.cli(config_path, "--only-algorithms", "Greedy", "--repetitions", "1",
                                "--experiment-id", "mini_partial", registry=GREEDY_AND_GA)
        self.assertEqual(code, 0, err)
        frame = self.results_frame()
        self.assertEqual(list(frame["experiment_id"]), ["mini_partial"])
        self.assertEqual(list(frame["algorithm"]), ["Greedy"])

    def test_invalid_command_line_values(self):
        config_path = make_config(self.dir)
        self.assertEqual(self.cli(config_path, "--only-configs", "C99", "--experiment-id", "x")[0], 2)
        self.assertEqual(self.cli(config_path, "--only-algorithms", "NOPE", "--experiment-id", "x")[0], 2)
        self.assertEqual(self.cli(config_path, "--repetitions", "0", "--experiment-id", "x")[0], 2)
        self.assertEqual(self.cli(config_path, "--experiment-id", "../bad")[0], 2)
        self.assertEqual(self.cli(config_path, "--adapter", "nonsense")[0], 2)
        self.assertEqual(self.cli(self.dir / "missing.yaml")[0], 2)


# ---------------------------------------------------------------------------
# Metadata accuracy across --skip-unavailable and --resume
# ---------------------------------------------------------------------------

class TestMetadataAcrossResume(RunnerTestCase):
    def metadata(self):
        return yaml.safe_load((self.results / "metadata_mini.yaml").read_text(encoding="utf-8"))

    def test_skip_unavailable_then_resume_reflects_the_adapters_that_contributed(self):
        config_path = make_config(self.dir)
        self.cli(config_path, "--skip-unavailable", "--adapter", "GA=missing:test_adapter",
                 registry={"Greedy": ref("adapter_round_robin")})
        first = self.metadata()
        harness = first["extra"]["harness"]
        self.assertEqual(harness["adapters"], {"Greedy": ref("adapter_round_robin")})
        self.assertIn("GA", harness["skipped_unavailable_algorithms"])
        self.assertEqual(len(harness["invocations"]), 1)
        self.assertEqual(harness["invocations"][0]["mode"], "start")
        self.assertEqual(harness["invocations"][0]["algorithms_executed"], ["Greedy"])
        self.assertEqual(harness["invocations"][0]["runs_executed"], 2)
        self.assertEqual(harness["invocations"][0]["status"], "completed")

        code, _, err = self.cli(config_path, "--resume", registry=GREEDY_AND_GA)
        self.assertEqual(code, 0, err)
        second = self.metadata()
        harness = second["extra"]["harness"]
        self.assertEqual(harness["adapters"], GREEDY_AND_GA)                 # both algorithms contributed now
        self.assertEqual(harness["skipped_unavailable_algorithms"], {})      # GA is no longer skipped
        self.assertEqual([i["mode"] for i in harness["invocations"]], ["start", "resume"])
        self.assertEqual(harness["invocations"][1]["algorithms_executed"], ["GA"])
        self.assertEqual(harness["invocations"][1]["adapters_used"], {"GA": ref("adapter_nearest")})
        self.assertEqual(harness["invocations"][1]["runs_executed"], 2)
        self.assertEqual(harness["invocations"][1]["runs_skipped_already_completed"], 2)
        for key in ("experiment_id", "created_utc", "config", "seed_info", "time_limits", "environment"):
            self.assertEqual(second[key], first[key], key)                   # original information preserved
        self.assertIn("updated_utc", second)
        self.assertEqual(harness["reward_stage"], first["extra"]["harness"]["reward_stage"])

    def test_adapter_of_an_algorithm_that_did_not_run_again_is_kept(self):
        config_path = make_config(self.dir)
        self.cli(config_path, registry=GREEDY_AND_GA)
        changed = {"Greedy": ref("adapter_nearest"), "GA": ref("adapter_round_robin")}   # references changed
        self.cli(config_path, "--resume", registry=changed)                              # nothing is pending
        harness = self.metadata()["extra"]["harness"]
        self.assertEqual(harness["adapters"], GREEDY_AND_GA)             # the adapters that produced the rows
        self.assertEqual(harness["invocations"][1]["algorithms_executed"], [])
        self.assertEqual(harness["invocations"][1]["runs_executed"], 0)

    def test_aborted_invocation_is_recorded_with_what_actually_contributed(self):
        config_path = make_config(self.dir)
        code, _, _ = self.cli(config_path, registry={"Greedy": ref("adapter_round_robin"),
                                                     "GA": ref("adapter_interrupt")})
        self.assertEqual(code, 130)
        harness = self.metadata()["extra"]["harness"]
        self.assertEqual(harness["adapters"], {"Greedy": ref("adapter_round_robin")})   # GA produced no rows
        self.assertEqual(harness["invocations"][0]["status"], "aborted: KeyboardInterrupt")
        self.assertEqual(harness["invocations"][0]["runs_executed"], 1)

    def test_no_contribution_means_no_adapters_listed(self):
        config_path = make_config(self.dir, algorithms=("GA",))
        code, _, _ = self.cli(config_path, registry={"GA": ref("adapter_interrupt")})
        self.assertEqual(code, 130)
        harness = self.metadata()["extra"]["harness"]
        self.assertEqual(harness["adapters"], {})
        self.assertEqual(harness["invocations"][0]["algorithms_executed"], [])

    def test_a_metadata_update_failure_never_masks_the_run(self):
        with mock.patch.object(rb, "update_experiment_metadata", side_effect=OSError("disk full")):
            code, out, err = self.cli(make_config(self.dir), registry=GREEDY_AND_GA)
        self.assertEqual(code, 0)
        self.assertIn("could not update the experiment metadata", err)
        self.assertEqual(len(self.results_frame()), 4)


# ---------------------------------------------------------------------------
# Subprocess backend (hard timeout) with temporary adapter modules
# ---------------------------------------------------------------------------

class TestSubprocessBackend(RunnerTestCase):
    def make_module(self, source):
        name = "bench_adapter_%s" % uuid.uuid4().hex[:8]
        directory = Path(tempfile.mkdtemp(prefix="bench_mod_"))
        (directory / (name + ".py")).write_text(textwrap.dedent(source), encoding="utf-8")
        sys.path.insert(0, str(directory))

        def cleanup():
            sys.path.remove(str(directory))
            sys.modules.pop(name, None)
            shutil.rmtree(str(directory), ignore_errors=True)

        self.addCleanup(cleanup)
        return name

    def test_timeout_is_recorded_and_the_next_algorithm_still_runs(self):
        fast = self.make_module("""
            def adapter(scenario, seed, time_limit):
                plan = {a.agent_id: [] for a in scenario.agents}
                robots = list(plan)
                for i, task in enumerate(scenario.tasks):
                    plan[robots[i % len(robots)]].append(task.task_id)
                return {"assignments": plan}
        """)
        slow = self.make_module("""
            import time
            def adapter(scenario, seed, time_limit):
                time.sleep(120)
                return {"assignments": {}}
        """)
        config_path = make_config(self.dir, algorithms=("GA", "Greedy"), repetitions=1, limits={"GA": 1.0})
        registry = {"GA": slow + ":adapter", "Greedy": fast + ":adapter"}
        code, out, err = self.cli(config_path, registry=registry, backend="subprocess")
        self.assertEqual(code, 0, err)
        frame = self.results_frame()
        ga, greedy = frame[frame["algorithm"] == "GA"].iloc[0], frame[frame["algorithm"] == "Greedy"].iloc[0]
        self.assertFalse(ga["success"])
        self.assertFalse(ga["allocation_stage_completed"])
        self.assertFalse(ga["reward_stage_completed"])
        self.assertIn("TimeoutError", ga["error_message"])
        self.assertTrue(math.isnan(ga["allocation_cost"]))
        self.assertGreaterEqual(ga["allocation_runtime_seconds"], 0.9)       # elapsed time of the stopped stage
        self.assertTrue(greedy["success"])
        self.assertTrue(greedy["allocation_stage_completed"])
        self.assertGreater(greedy["total_runtime_seconds"], greedy["allocation_runtime_seconds"])   # includes overhead


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------

class TestBoundaries(unittest.TestCase):
    def test_real_adapter_registry_is_modular_and_includes_milp(self):
        self.assertEqual(set(rb.ADAPTER_REGISTRY), {"Greedy", "GA", "SVCA", "GA_Paper", "SVCA_Paper", "MILP"})
        self.assertTrue(all(value.startswith("src.analytics.allocation_adapters:")
                            for value in rb.ADAPTER_REGISTRY.values()))
        source = SCRIPT.read_text(encoding="utf-8")
        for forbidden in ("def milp", "def greedy", "def genetic", "def svca", "RewardDistribution"):
            self.assertNotIn(forbidden, source)

    def test_reward_stage_is_reported_as_blocked(self):
        self.assertIn("enabled", rb._reward_stage_status())


if __name__ == "__main__":
    unittest.main()

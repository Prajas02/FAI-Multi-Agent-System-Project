"""Unit tests for src/analytics/logger.py (logging, metadata, aggregation).

All tests use temporary directories and synthetic dictionaries. No simulation,
allocation or game-theory module is needed.
"""

import csv
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import yaml

from src.analytics import logger
from src.analytics.logger import (
    BENCHMARK_RESULT_COLUMNS,
    ROBOT_REWARD_COLUMNS,
    SUMMARY_COLUMNS,
    DuplicateKeyError,
    SchemaValidationError,
    aggregate_benchmark_results,
    append_benchmark_results,
    append_robot_rewards,
    load_benchmark_results,
    load_robot_rewards,
    summarize_experiment,
    validate_experiment_id,
    write_experiment_metadata,
    write_summary_table,
)

NAN = float("nan")


def make_record(**overrides):
    """A valid main-results record (fully completed, successful run)."""
    record = dict(
        experiment_id="exp1",
        scenario_id="C1_r00",
        algorithm="Greedy",
        num_robots=5,
        num_tasks=5,
        total_entities=10,
        random_seed=1,
        allocation_cost=100.0,
        allocation_runtime_seconds=1.0,
        reward_runtime_seconds=2.0,
        total_runtime_seconds=3.5,
        allocation_stage_completed=True,
        reward_stage_completed=True,
        gini_index=0.25,
        invalid_reward_flag=False,
        energy_feasible=True,
        coalition_stability=None,
        success=True,
        error_message="",
    )
    record.update(overrides)
    return record


def make_reward(**overrides):
    record = dict(
        experiment_id="exp1",
        scenario_id="C1_r00",
        algorithm="Greedy",
        robot_id=0,
        reward=1.5,
    )
    record.update(overrides)
    return record


def frame(records):
    return pd.DataFrame(records, columns=list(BENCHMARK_RESULT_COLUMNS))


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.dir), True)

    def names(self):
        return sorted(p.name for p in self.dir.iterdir())

    @property
    def results_csv(self):
        return self.dir / "benchmark_results.csv"

    @property
    def rewards_csv(self):
        return self.dir / "robot_rewards.csv"


class TestSchemas(unittest.TestCase):
    def test_main_schema_columns_and_order(self):
        self.assertEqual(
            list(BENCHMARK_RESULT_COLUMNS),
            [
                "experiment_id", "scenario_id", "algorithm", "num_robots", "num_tasks",
                "total_entities", "random_seed", "allocation_cost",
                "allocation_runtime_seconds", "reward_runtime_seconds",
                "total_runtime_seconds", "allocation_stage_completed",
                "reward_stage_completed", "gini_index", "invalid_reward_flag",
                "energy_feasible", "coalition_stability", "success", "error_message",
            ],
        )
        self.assertEqual(len(BENCHMARK_RESULT_COLUMNS), 19)

    def test_robot_reward_schema_columns_and_order(self):
        self.assertEqual(
            list(ROBOT_REWARD_COLUMNS),
            ["experiment_id", "scenario_id", "algorithm", "robot_id", "reward"],
        )

    def test_summary_schema_matches_design(self):
        self.assertEqual(len(SUMMARY_COLUMNS), 25)
        for name in (
            "n_runs", "n_success", "n_allocation_cost", "n_allocation_runtime",
            "n_reward_runtime", "n_total_runtime", "n_gini", "energy_feasible_rate",
            "n_invalid_reward", "coalition_stability_mean", "coalition_stability_std",
        ):
            self.assertIn(name, SUMMARY_COLUMNS)

    def test_default_results_dir_is_data_results_raw(self):
        self.assertEqual(logger.DEFAULT_RESULTS_DIR.parts[-3:], ("data", "results", "raw"))


class TestExperimentId(unittest.TestCase):
    def test_valid_ids(self):
        for value in ("exp1", "run-2026.11_a", "A", "x" * 128, "a.b"):
            self.assertEqual(validate_experiment_id(value), value)

    def test_invalid_ids(self):
        for value in ("", ".", "..", "../x", "a/b", "a\\b", "a b", "x" * 129, "é", None, 5):
            with self.subTest(value=value):
                with self.assertRaises(SchemaValidationError):
                    validate_experiment_id(value)

    def test_unsafe_id_rejected_in_records(self):
        with self.assertRaises(SchemaValidationError):
            append_benchmark_results([make_record(experiment_id="../x")], self.dir_for_test())
        with self.assertRaises(SchemaValidationError):
            append_robot_rewards([make_reward(experiment_id="a b")], self.dir_for_test())

    def dir_for_test(self):
        path = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(path), True)
        return path


class TestBenchmarkWriter(TempDirTestCase):
    def test_creates_file_with_exact_header_and_serialization(self):
        written = append_benchmark_results([make_record()], self.dir)
        self.assertEqual(written, 1)
        lines = self.results_csv.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], ",".join(BENCHMARK_RESULT_COLUMNS))
        self.assertEqual(
            lines[1],
            "exp1,C1_r00,Greedy,5,5,10,1,100.0,1.0,2.0,3.5,True,True,0.25,False,True,,True,",
        )
        self.assertEqual(len(lines), 2)

    def test_missing_values_are_empty_cells(self):
        record = make_record(
            success=False, allocation_cost=None, reward_stage_completed=False,
            gini_index=NAN, reward_runtime_seconds=None, energy_feasible=None,
            error_message="invalid allocation",
        )
        append_benchmark_results([record], self.dir)
        row = self.results_csv.read_text(encoding="utf-8").splitlines()[1].split(",")
        by_name = dict(zip(BENCHMARK_RESULT_COLUMNS, row))
        for name in ("allocation_cost", "reward_runtime_seconds", "gini_index",
                     "energy_feasible", "coalition_stability"):
            self.assertEqual(by_name[name], "", name)
        self.assertEqual(by_name["success"], "False")
        self.assertEqual(by_name["allocation_stage_completed"], "True")

    def test_header_written_once_and_multiple_records_appended(self):
        append_benchmark_results([make_record(scenario_id="a"), make_record(scenario_id="b")], self.dir)
        append_benchmark_results([make_record(scenario_id="c")], self.dir)
        lines = self.results_csv.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 4)
        self.assertEqual(sum(1 for l in lines if l.startswith("experiment_id,")), 1)

    def test_numpy_scalars_are_accepted(self):
        record = make_record(
            num_robots=np.int64(5), num_tasks=np.int32(5), total_entities=np.int64(10),
            allocation_cost=np.float64(7.5), success=np.bool_(True),
        )
        append_benchmark_results([record], self.dir)
        self.assertIn("7.5", self.results_csv.read_text(encoding="utf-8"))

    def test_error_message_with_comma_and_newline_round_trips(self):
        message = 'bad, "quoted"\nsecond line'
        append_benchmark_results([make_record(error_message=message)], self.dir)
        loaded = load_benchmark_results(self.results_csv)
        self.assertEqual(loaded.loc[0, "error_message"], message)

    def test_empty_batch_is_a_no_op(self):
        self.assertEqual(append_benchmark_results([], self.dir), 0)
        self.assertEqual(self.names(), [])

    def test_missing_directory_is_created_on_write(self):
        target = self.dir / "nested" / "raw"
        append_benchmark_results([make_record()], target)
        self.assertTrue((target / "benchmark_results.csv").exists())

    def test_existing_file_without_trailing_newline(self):
        append_benchmark_results([make_record(scenario_id="a")], self.dir)
        text = self.results_csv.read_text(encoding="utf-8").rstrip("\n")
        self.results_csv.write_bytes(text.encode("utf-8"))
        append_benchmark_results([make_record(scenario_id="b")], self.dir)
        self.assertEqual(len(load_benchmark_results(self.results_csv)), 2)

    def test_existing_empty_file_gets_a_header(self):
        self.results_csv.write_bytes(b"")
        append_benchmark_results([make_record()], self.dir)
        first = self.results_csv.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(first, ",".join(BENCHMARK_RESULT_COLUMNS))


class TestBenchmarkRecordValidation(TempDirTestCase):
    def assert_rejected(self, record, pattern=None):
        with self.assertRaisesRegex(SchemaValidationError, pattern or ""):
            append_benchmark_results([record], self.dir)
        self.assertEqual(self.names(), [])

    def test_missing_and_unknown_fields(self):
        record = make_record()
        del record["success"]
        self.assert_rejected(record, "missing")
        self.assert_rejected(dict(make_record(), extra_field=1), "unknown")

    def test_record_must_be_a_mapping(self):
        with self.assertRaises(SchemaValidationError):
            append_benchmark_results([["not", "a", "mapping"]], self.dir)

    def test_type_errors(self):
        cases = {
            "bool as string": dict(success="True"),
            "bool as int": dict(success=1),
            "stage flag missing": dict(allocation_stage_completed=None),
            "invalid flag missing": dict(invalid_reward_flag=None),
            "robots string": dict(num_robots="5"),
            "robots bool": dict(num_robots=True),
            "robots negative": dict(num_robots=-1, total_entities=4),
            "seed float": dict(random_seed=1.5),
            "cost string": dict(allocation_cost="abc"),
            "cost inf": dict(allocation_cost=float("inf")),
            "runtime negative": dict(allocation_runtime_seconds=-0.1),
            "energy not bool": dict(energy_feasible="yes"),
            "algorithm empty": dict(algorithm=""),
            "scenario not str": dict(scenario_id=3),
            "error not str": dict(error_message=5),
        }
        for label, override in cases.items():
            with self.subTest(label):
                self.assert_rejected(make_record(**override))

    def test_total_entities_must_equal_robots_plus_tasks(self):
        self.assert_rejected(make_record(total_entities=11), "total_entities")

    def test_coalition_stability_non_missing_rejected(self):
        for value in (0.0, 0.5, 1, "stable", True):
            with self.subTest(value=value):
                self.assert_rejected(make_record(coalition_stability=value), "coalition_stability")

    def test_coalition_stability_missing_accepted(self):
        for value in (None, NAN):
            with self.subTest(value=value):
                path = Path(tempfile.mkdtemp())
                self.addCleanup(shutil.rmtree, str(path), True)
                append_benchmark_results([make_record(coalition_stability=value)], path)
                loaded = load_benchmark_results(path / "benchmark_results.csv")
                self.assertTrue(math.isnan(loaded.loc[0, "coalition_stability"]))

    # --- the five cross-field rules of Section 5.4 -------------------------

    def test_rule1_success_requires_allocation_stage_completed(self):
        self.assert_rejected(
            make_record(allocation_stage_completed=False, reward_stage_completed=False,
                        gini_index=None),
            "Rule 1",
        )

    def test_rule2_reward_stage_requires_allocation_stage_and_success(self):
        # allocation stage not completed
        self.assert_rejected(
            make_record(success=False, allocation_cost=None, allocation_stage_completed=False,
                        gini_index=None),
            "Rule 2",
        )
        # allocation completed but not successful
        self.assert_rejected(
            make_record(success=False, allocation_cost=None, gini_index=None),
            "Rule 2",
        )

    def test_rule3_invalid_flag_requires_reward_stage_completed(self):
        self.assert_rejected(
            make_record(invalid_reward_flag=True, reward_stage_completed=False, gini_index=None),
            "Rule 3",
        )

    def test_rule4_gini_requires_reward_stage_and_valid_rewards(self):
        self.assert_rejected(make_record(reward_stage_completed=False), "Rule 4")
        self.assert_rejected(make_record(invalid_reward_flag=True), "Rule 4")

    def test_rule5_failed_run_requires_missing_cost(self):
        self.assert_rejected(
            make_record(success=False, reward_stage_completed=False, gini_index=None,
                        allocation_cost=5.0),
            "Rule 5",
        )

    def test_valid_allocation_completed_but_not_successful(self):
        record = make_record(
            success=False, allocation_cost=None, reward_stage_completed=False,
            gini_index=None, reward_runtime_seconds=None, error_message="violates Eq. 6",
        )
        self.assertEqual(append_benchmark_results([record], self.dir), 1)

    def test_valid_invalid_reward_run(self):
        record = make_record(invalid_reward_flag=True, gini_index=None,
                             error_message="negative reward")
        self.assertEqual(append_benchmark_results([record], self.dir), 1)

    def test_valid_timeout_run(self):
        record = make_record(
            success=False, allocation_cost=None, allocation_stage_completed=False,
            reward_stage_completed=False, gini_index=None, reward_runtime_seconds=None,
            energy_feasible=None, error_message="timeout",
        )
        self.assertEqual(append_benchmark_results([record], self.dir), 1)


class TestDuplicatesAndAtomicity(TempDirTestCase):
    def bytes_and_names(self):
        return self.results_csv.read_bytes(), self.names()

    def test_duplicate_keys_within_batch(self):
        with self.assertRaises(DuplicateKeyError):
            append_benchmark_results([make_record(), make_record(random_seed=2)], self.dir)
        self.assertEqual(self.names(), [])

    def test_same_scenario_different_algorithm_is_not_a_duplicate(self):
        append_benchmark_results([make_record(algorithm="Greedy"), make_record(algorithm="GA")], self.dir)
        self.assertEqual(len(load_benchmark_results(self.results_csv)), 2)

    def test_same_key_in_another_experiment_is_not_a_duplicate(self):
        append_benchmark_results([make_record(experiment_id="e1")], self.dir)
        append_benchmark_results([make_record(experiment_id="e2")], self.dir)
        self.assertEqual(len(load_benchmark_results(self.results_csv)), 2)

    def test_duplicate_against_existing_file(self):
        append_benchmark_results([make_record()], self.dir)
        before = self.bytes_and_names()
        with self.assertRaises(DuplicateKeyError):
            append_benchmark_results([make_record(random_seed=9)], self.dir)
        self.assertEqual(self.bytes_and_names(), before)

    def test_one_valid_and_one_duplicate_rejects_whole_batch(self):
        append_benchmark_results([make_record(scenario_id="a")], self.dir)
        before = self.bytes_and_names()
        with self.assertRaises(DuplicateKeyError):
            append_benchmark_results(
                [make_record(scenario_id="new"), make_record(scenario_id="a")], self.dir
            )
        self.assertEqual(self.bytes_and_names(), before)

    def test_one_valid_and_one_invalid_rejects_whole_batch(self):
        append_benchmark_results([make_record(scenario_id="a")], self.dir)
        before = self.bytes_and_names()
        with self.assertRaisesRegex(SchemaValidationError, "record 1"):
            append_benchmark_results(
                [make_record(scenario_id="new"), make_record(scenario_id="bad", success="x")],
                self.dir,
            )
        self.assertEqual(self.bytes_and_names(), before)

    def test_invalid_batch_creates_no_file(self):
        with self.assertRaises(SchemaValidationError):
            append_benchmark_results(
                [make_record(scenario_id="ok"), make_record(scenario_id="bad", num_robots="x")],
                self.dir,
            )
        self.assertEqual(self.names(), [])

    def test_existing_file_with_duplicate_keys_is_refused(self):
        append_benchmark_results([make_record()], self.dir)
        lines = self.results_csv.read_text(encoding="utf-8").splitlines()
        self.results_csv.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")
        before = self.bytes_and_names()
        with self.assertRaises(DuplicateKeyError):
            append_benchmark_results([make_record(scenario_id="other")], self.dir)
        self.assertEqual(self.bytes_and_names(), before)

    def test_existing_header_is_validated(self):
        self.results_csv.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        before = self.bytes_and_names()
        with self.assertRaises(SchemaValidationError):
            append_benchmark_results([make_record()], self.dir)
        self.assertEqual(self.bytes_and_names(), before)

    def test_existing_header_with_wrong_order_is_rejected(self):
        columns = list(BENCHMARK_RESULT_COLUMNS)
        columns[0], columns[1] = columns[1], columns[0]
        self.results_csv.write_text(",".join(columns) + "\n", encoding="utf-8")
        with self.assertRaises(SchemaValidationError):
            append_benchmark_results([make_record()], self.dir)

    def test_existing_row_with_wrong_cell_count_is_rejected(self):
        header = ",".join(BENCHMARK_RESULT_COLUMNS)
        self.results_csv.write_text(header + "\nonly,two\n", encoding="utf-8")
        with self.assertRaises(SchemaValidationError):
            append_benchmark_results([make_record()], self.dir)

    def test_write_uses_temp_file_fsync_and_replace(self):
        real_fsync = os.fsync
        with mock.patch("src.analytics.logger.os.fsync", side_effect=real_fsync) as fsync, \
                mock.patch("src.analytics.logger.os.replace", wraps=os.replace) as replace:
            append_benchmark_results([make_record()], self.dir)
        self.assertTrue(fsync.called)
        self.assertEqual(replace.call_count, 1)
        source, target = replace.call_args[0]
        self.assertEqual(Path(source).parent, self.dir)  # temp file in the same directory
        self.assertEqual(Path(target), self.results_csv)
        self.assertEqual(self.names(), ["benchmark_results.csv"])  # temp file gone

    def test_failed_replace_leaves_existing_file_untouched(self):
        append_benchmark_results([make_record(scenario_id="a")], self.dir)
        before = self.bytes_and_names()
        with mock.patch("src.analytics.logger.os.replace", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                append_benchmark_results([make_record(scenario_id="b")], self.dir)
        self.assertEqual(self.bytes_and_names(), before)

    def test_failed_replace_on_new_file_leaves_nothing_behind(self):
        with mock.patch("src.analytics.logger.os.replace", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                append_benchmark_results([make_record()], self.dir)
        self.assertEqual(self.names(), [])

    def test_failed_fsync_leaves_no_partial_file(self):
        with mock.patch("src.analytics.logger.os.fsync", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                append_benchmark_results([make_record()], self.dir)
        self.assertEqual(self.names(), [])


class TestRobotRewardWriter(TempDirTestCase):
    def test_creates_file_with_exact_header(self):
        append_robot_rewards([make_reward(robot_id=0, reward=1.5), make_reward(robot_id="r2", reward=0.0)], self.dir)
        lines = self.rewards_csv.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], ",".join(ROBOT_REWARD_COLUMNS))
        self.assertEqual(lines[1], "exp1,C1_r00,Greedy,0,1.5")
        self.assertEqual(lines[2], "exp1,C1_r00,Greedy,r2,0.0")

    def test_negative_reward_is_stored_unchanged(self):
        append_robot_rewards([make_reward(robot_id=0, reward=-2.25), make_reward(robot_id=1, reward=3.0)], self.dir)
        loaded = load_robot_rewards(self.rewards_csv)
        self.assertEqual(loaded["reward"].tolist(), [-2.25, 3.0])

    def test_non_finite_or_non_numeric_rewards_are_rejected(self):
        for value in (NAN, float("inf"), -float("inf"), None, "1.0", True):
            with self.subTest(value=value):
                with self.assertRaises(SchemaValidationError):
                    append_robot_rewards([make_reward(reward=value)], self.dir)
        self.assertEqual(self.names(), [])

    def test_invalid_robot_ids_are_rejected(self):
        for value in (None, "", True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(SchemaValidationError):
                    append_robot_rewards([make_reward(robot_id=value)], self.dir)

    def test_uniqueness_key_includes_robot_id(self):
        append_robot_rewards([make_reward(robot_id=0), make_reward(robot_id=1)], self.dir)
        append_robot_rewards([make_reward(robot_id=0, algorithm="GA")], self.dir)
        self.assertEqual(len(load_robot_rewards(self.rewards_csv)), 3)

    def test_duplicate_robot_id_within_batch(self):
        with self.assertRaises(DuplicateKeyError):
            append_robot_rewards([make_reward(robot_id=3), make_reward(robot_id=3, reward=9.0)], self.dir)
        self.assertEqual(self.names(), [])

    def test_duplicate_against_existing_rows_rejects_whole_batch(self):
        append_robot_rewards([make_reward(robot_id=0)], self.dir)
        before = self.rewards_csv.read_bytes()
        with self.assertRaises(DuplicateKeyError):
            append_robot_rewards([make_reward(robot_id=1), make_reward(robot_id=0)], self.dir)
        self.assertEqual(self.rewards_csv.read_bytes(), before)

    def test_int_and_string_ids_with_same_text_collide(self):
        append_robot_rewards([make_reward(robot_id=1)], self.dir)
        with self.assertRaises(DuplicateKeyError):
            append_robot_rewards([make_reward(robot_id="1")], self.dir)

    def test_header_validation(self):
        self.rewards_csv.write_text("x,y\n", encoding="utf-8")
        with self.assertRaises(SchemaValidationError):
            append_robot_rewards([make_reward()], self.dir)


class TestIndependentWriters(TempDirTestCase):
    def test_orphan_reward_rows_are_kept_and_never_cleaned_up(self):
        append_robot_rewards([make_reward(robot_id=0), make_reward(robot_id=1)], self.dir)
        real_replace = os.replace

        def fail_for_results(src, dst):
            if Path(dst).name == "benchmark_results.csv":
                raise OSError("results write failed")
            return real_replace(src, dst)

        with mock.patch("src.analytics.logger.os.replace", side_effect=fail_for_results):
            with self.assertRaises(OSError):
                append_benchmark_results([make_record()], self.dir)

        self.assertFalse(self.results_csv.exists())
        self.assertEqual(len(load_robot_rewards(self.rewards_csv)), 2)  # orphans remain
        # Retrying the reward write for the same key raises; nothing is overwritten.
        before = self.rewards_csv.read_bytes()
        with self.assertRaises(DuplicateKeyError):
            append_robot_rewards([make_reward(robot_id=0), make_reward(robot_id=1)], self.dir)
        self.assertEqual(self.rewards_csv.read_bytes(), before)
        # The main writer still works on its own afterwards.
        append_benchmark_results([make_record()], self.dir)
        self.assertEqual(len(load_benchmark_results(self.results_csv)), 1)

    def test_writers_do_not_touch_each_others_files(self):
        append_robot_rewards([make_reward()], self.dir)
        before = self.rewards_csv.read_bytes()
        append_benchmark_results([make_record()], self.dir)
        self.assertEqual(self.rewards_csv.read_bytes(), before)

    def test_main_writer_does_not_check_reward_rows(self):
        # No cross-file transaction: a main record without reward rows is accepted.
        self.assertEqual(append_benchmark_results([make_record()], self.dir), 1)
        self.assertFalse(self.rewards_csv.exists())


class TestMetadata(TempDirTestCase):
    def write(self, **kwargs):
        defaults = dict(
            experiment_id="exp1",
            config={"energy_penalty": {"strategy": "quadratic", "lambda_E": 1.5}, "runs": 10},
            seed_info={"base_seed": 42, "rule": "base + 1000*config + repetition"},
            time_limits={"MILP": 60.0},
            results_dir=self.dir,
        )
        defaults.update(kwargs)
        return write_experiment_metadata(**defaults)

    def fake_git(self, args, **kwargs):
        command = args[1:]
        if command[0] == "rev-parse":
            return mock.Mock(stdout="abc123def\n")
        return mock.Mock(stdout=" M some_file.py\n")

    def test_creates_yaml_with_expected_content(self):
        with mock.patch("src.analytics.logger.subprocess.run", side_effect=self.fake_git):
            path = self.write(extra={"solver_versions": {"cbc": "2.10"}})
        self.assertEqual(path, self.dir / "metadata_exp1.yaml")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(data["experiment_id"], "exp1")
        self.assertEqual(data["config"]["energy_penalty"]["lambda_E"], 1.5)
        self.assertEqual(data["seed_info"]["base_seed"], 42)
        self.assertEqual(data["time_limits"]["MILP"], 60.0)
        self.assertEqual(data["extra"]["solver_versions"]["cbc"], "2.10")
        self.assertIn("created_utc", data)
        env = data["environment"]
        for key in ("python_version", "platform", "machine", "cpu_count", "libraries",
                    "git_commit", "git_dirty"):
            self.assertIn(key, env)
        self.assertEqual(env["git_commit"], "abc123def")
        self.assertTrue(env["git_dirty"])
        self.assertIn("numpy", env["libraries"])
        self.assertIn("pandas", env["libraries"])
        self.assertEqual(self.names(), ["metadata_exp1.yaml"])  # no temp file left

    def test_hostname_and_user_are_not_recorded(self):
        path = self.write()
        env = yaml.safe_load(path.read_text(encoding="utf-8"))["environment"]
        for key in ("hostname", "node", "user", "username"):
            self.assertNotIn(key, env)

    def test_git_unavailable_gives_null_values(self):
        with mock.patch("src.analytics.logger.subprocess.run", side_effect=FileNotFoundError):
            path = self.write()
        env = yaml.safe_load(path.read_text(encoding="utf-8"))["environment"]
        self.assertIsNone(env["git_commit"])
        self.assertIsNone(env["git_dirty"])

    def test_clean_working_tree(self):
        def clean(args, **kwargs):
            return mock.Mock(stdout="abc\n" if args[1] == "rev-parse" else "")

        with mock.patch("src.analytics.logger.subprocess.run", side_effect=clean):
            path = self.write()
        env = yaml.safe_load(path.read_text(encoding="utf-8"))["environment"]
        self.assertFalse(env["git_dirty"])

    def test_existing_metadata_is_not_overwritten_by_default(self):
        path = self.write()
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.write(time_limits={"MILP": 1.0})
        self.assertEqual(path.read_bytes(), before)

    def test_overwrite_when_requested(self):
        self.write()
        path = self.write(time_limits={"MILP": 1.0}, overwrite=True)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(data["time_limits"]["MILP"], 1.0)

    def test_unsafe_experiment_ids_are_rejected(self):
        for value in ("../evil", "a/b", "", "..", "a b"):
            with self.subTest(value=value):
                with self.assertRaises(SchemaValidationError):
                    self.write(experiment_id=value)
        self.assertEqual(self.names(), [])

    def test_numpy_values_and_tuples_are_normalized(self):
        path = self.write(config={"lambda_E": np.float64(2.5), "sizes": (10, 20), "n": np.int64(3)})
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(data["config"], {"lambda_E": 2.5, "sizes": [10, 20], "n": 3})

    def test_unsupported_values_are_rejected_without_writing(self):
        with self.assertRaises(SchemaValidationError):
            self.write(config={"bad": object()})
        with self.assertRaises(SchemaValidationError):
            self.write(config={1: "non-string key"})
        self.assertEqual(self.names(), [])


class TestLoading(TempDirTestCase):
    def test_typed_round_trip(self):
        append_benchmark_results(
            [make_record(scenario_id="a"),
             make_record(scenario_id="b", energy_feasible=None, gini_index=None,
                         invalid_reward_flag=True)],
            self.dir,
        )
        loaded = load_benchmark_results(self.results_csv)
        self.assertEqual(list(loaded.columns), list(BENCHMARK_RESULT_COLUMNS))
        self.assertEqual(str(loaded["num_robots"].dtype), "int64")
        self.assertEqual(str(loaded["allocation_cost"].dtype), "float64")
        self.assertEqual(str(loaded["success"].dtype), "bool")
        self.assertEqual(str(loaded["energy_feasible"].dtype), "boolean")
        self.assertTrue(pd.isna(loaded.loc[1, "energy_feasible"]))
        self.assertTrue(math.isnan(loaded.loc[1, "gini_index"]))
        self.assertTrue(loaded.loc[1, "invalid_reward_flag"])

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_benchmark_results(self.results_csv)

    def test_header_only_file_loads_as_empty_frame(self):
        self.results_csv.write_text(",".join(BENCHMARK_RESULT_COLUMNS) + "\n", encoding="utf-8")
        loaded = load_benchmark_results(self.results_csv)
        self.assertEqual(len(loaded), 0)
        self.assertEqual(list(loaded.columns), list(BENCHMARK_RESULT_COLUMNS))

    def test_bad_boolean_cell_is_rejected(self):
        append_benchmark_results([make_record()], self.dir)
        text = self.results_csv.read_text(encoding="utf-8").replace(",True,True,", ",maybe,True,", 1)
        self.results_csv.write_text(text, encoding="utf-8")
        with self.assertRaises(SchemaValidationError):
            load_benchmark_results(self.results_csv)


def stage_scenario_records():
    """Five runs that exercise every eligibility rule (hand-calculated below)."""
    return [
        # r1: fully completed
        make_record(scenario_id="s1", allocation_cost=10.0, allocation_runtime_seconds=1.0,
                    reward_runtime_seconds=2.0, total_runtime_seconds=4.0, gini_index=0.1,
                    energy_feasible=True),
        # r2: allocation timed out (elapsed time recorded, stages not completed)
        make_record(scenario_id="s2", success=False, allocation_cost=None,
                    allocation_stage_completed=False, reward_stage_completed=False,
                    allocation_runtime_seconds=5.0, reward_runtime_seconds=None,
                    total_runtime_seconds=5.5, gini_index=None, energy_feasible=None,
                    error_message="timeout"),
        # r3: allocation returned but failed validation; reward stage never started
        make_record(scenario_id="s3", success=False, allocation_cost=None,
                    reward_stage_completed=False, allocation_runtime_seconds=3.0,
                    reward_runtime_seconds=None, total_runtime_seconds=3.5, gini_index=None,
                    energy_feasible=None, error_message="violates Eq. 6"),
        # r4: reward stage started and then failed (elapsed 7.0 recorded)
        make_record(scenario_id="s4", allocation_cost=20.0, allocation_runtime_seconds=2.0,
                    reward_stage_completed=False, reward_runtime_seconds=7.0,
                    total_runtime_seconds=8.0, gini_index=None, energy_feasible=False,
                    error_message="reward solver failed"),
        # r5: both stages completed but reward runtime is NaN
        make_record(scenario_id="s5", allocation_cost=30.0, allocation_runtime_seconds=2.0,
                    reward_runtime_seconds=None, total_runtime_seconds=6.0, gini_index=0.3,
                    energy_feasible=True),
    ]


class TestAggregation(TempDirTestCase):
    def row(self, summary, algorithm="Greedy", total_entities=10):
        match = summary[(summary["algorithm"] == algorithm)
                        & (summary["total_entities"] == total_entities)]
        self.assertEqual(len(match), 1)
        return match.iloc[0]

    def test_empty_input_returns_empty_frame_with_columns(self):
        summary = aggregate_benchmark_results(frame([]))
        self.assertEqual(len(summary), 0)
        self.assertEqual(list(summary.columns), list(SUMMARY_COLUMNS))

    def test_missing_columns_are_rejected(self):
        with self.assertRaises(SchemaValidationError):
            aggregate_benchmark_results(frame([make_record()]).drop(columns=["success"]))

    def test_grouping_and_ordering(self):
        records = []
        for algorithm in ("SVCA", "Greedy"):
            records.append(make_record(algorithm=algorithm, scenario_id="a"))
            records.append(make_record(algorithm=algorithm, scenario_id="b"))
            records.append(make_record(algorithm=algorithm, scenario_id="c", num_tasks=15,
                                       total_entities=20))
        summary = aggregate_benchmark_results(frame(records))
        self.assertEqual(len(summary), 4)
        self.assertEqual(
            list(zip(summary["algorithm"], summary["total_entities"])),
            [("Greedy", 10), ("SVCA", 10), ("Greedy", 20), ("SVCA", 20)],
        )
        self.assertEqual(self.row(summary, "Greedy", 10)["n_runs"], 2)
        self.assertEqual(self.row(summary, "SVCA", 20)["n_runs"], 1)
        self.assertEqual(list(summary.columns), list(SUMMARY_COLUMNS))

    def test_filters_by_experiment_id(self):
        records = [
            make_record(experiment_id="A", scenario_id="a", allocation_cost=10.0),
            make_record(experiment_id="A", scenario_id="b", allocation_cost=20.0),
            make_record(experiment_id="B", scenario_id="a", allocation_cost=1000.0),
        ]
        summary = aggregate_benchmark_results(frame(records), experiment_id="A")
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["n_runs"], 2)
        self.assertAlmostEqual(summary.iloc[0]["allocation_cost_mean"], 15.0)

    def test_several_experiments_without_id_is_an_error(self):
        records = [make_record(experiment_id="A"), make_record(experiment_id="B")]
        with self.assertRaises(ValueError):
            aggregate_benchmark_results(frame(records))

    def test_single_experiment_without_id_is_allowed(self):
        summary = aggregate_benchmark_results(frame([make_record()]))
        self.assertEqual(len(summary), 1)

    def test_unknown_experiment_gives_empty_summary(self):
        summary = aggregate_benchmark_results(frame([make_record()]), experiment_id="other")
        self.assertEqual(len(summary), 0)
        self.assertEqual(list(summary.columns), list(SUMMARY_COLUMNS))

    def test_sample_standard_deviation_hand_calculated(self):
        records = [make_record(scenario_id=str(i), allocation_cost=cost)
                   for i, cost in enumerate([10.0, 20.0, 30.0])]
        row = self.row(aggregate_benchmark_results(frame(records)))
        self.assertEqual(row["n_allocation_cost"], 3)
        self.assertAlmostEqual(row["allocation_cost_mean"], 20.0)
        self.assertAlmostEqual(row["allocation_cost_std"], 10.0)  # ddof=1 (ddof=0 would give 8.165)

    def test_std_is_nan_with_fewer_than_two_values_and_mean_nan_with_none(self):
        records = [
            make_record(scenario_id="a", allocation_cost=12.0),
            make_record(scenario_id="b", success=False, allocation_cost=None,
                        reward_stage_completed=False, gini_index=None),
        ]
        row = self.row(aggregate_benchmark_results(frame(records)))
        self.assertEqual(row["n_allocation_cost"], 1)
        self.assertAlmostEqual(row["allocation_cost_mean"], 12.0)
        self.assertTrue(math.isnan(row["allocation_cost_std"]))

        none_valid = frame([make_record(coalition_stability=None)])
        row = self.row(aggregate_benchmark_results(none_valid))
        self.assertTrue(math.isnan(row["coalition_stability_mean"]))
        self.assertTrue(math.isnan(row["coalition_stability_std"]))

    def test_every_metric_eligibility_condition_and_count(self):
        write = stage_scenario_records()
        append_benchmark_results(write, self.dir)  # also validates the five scenarios
        summary = aggregate_benchmark_results(load_benchmark_results(self.results_csv))
        row = self.row(summary)

        self.assertEqual(row["n_runs"], 5)
        self.assertEqual(row["n_success"], 3)  # r1, r4, r5

        # cost: success=True and non-NaN -> r1, r4, r5
        self.assertEqual(row["n_allocation_cost"], 3)
        self.assertAlmostEqual(row["allocation_cost_mean"], 20.0)
        self.assertAlmostEqual(row["allocation_cost_std"], 10.0)

        # allocation runtime: allocation_stage_completed=True -> r1, r3, r4, r5
        # (r2's 5.0 s elapsed time of the timed-out stage is excluded)
        self.assertEqual(row["n_allocation_runtime"], 4)
        self.assertAlmostEqual(row["allocation_runtime_seconds_mean"], 2.0)
        self.assertAlmostEqual(row["allocation_runtime_seconds_std"], math.sqrt(2.0 / 3.0))

        # reward runtime: reward_stage_completed=True and non-NaN -> r1 only
        # (r4's failed-stage 7.0 s excluded, r5's NaN excluded)
        self.assertEqual(row["n_reward_runtime"], 1)
        self.assertAlmostEqual(row["reward_runtime_seconds_mean"], 2.0)
        self.assertTrue(math.isnan(row["reward_runtime_seconds_std"]))

        # total runtime: both stages completed -> r1, r5
        self.assertEqual(row["n_total_runtime"], 2)
        self.assertAlmostEqual(row["total_runtime_seconds_mean"], 5.0)
        self.assertAlmostEqual(row["total_runtime_seconds_std"], math.sqrt(2.0))

        # gini: non-NaN -> r1, r5
        self.assertEqual(row["n_gini"], 2)
        self.assertAlmostEqual(row["gini_index_mean"], 0.2)
        self.assertAlmostEqual(row["gini_index_std"], math.sqrt(0.02))

        # energy: successful runs with non-missing flag -> r1 True, r4 False, r5 True
        self.assertAlmostEqual(row["energy_feasible_rate"], 2.0 / 3.0)
        self.assertEqual(row["n_invalid_reward"], 0)

    def test_completion_is_not_derived_from_success_or_error_message(self):
        # allocation completed but success=False: still counts for allocation runtime
        records = [
            make_record(scenario_id="a", success=False, allocation_cost=None,
                        reward_stage_completed=False, gini_index=None,
                        allocation_runtime_seconds=4.0, error_message=""),
            make_record(scenario_id="b", allocation_runtime_seconds=2.0,
                        error_message="some note"),
        ]
        row = self.row(aggregate_benchmark_results(frame(records)))
        self.assertEqual(row["n_allocation_runtime"], 2)
        self.assertAlmostEqual(row["allocation_runtime_seconds_mean"], 3.0)
        self.assertEqual(row["n_allocation_cost"], 1)

    def test_aggregation_does_not_need_reward_rows(self):
        # No robot_rewards.csv exists anywhere; reward runtimes still aggregate.
        row = self.row(aggregate_benchmark_results(frame([make_record()])))
        self.assertEqual(row["n_reward_runtime"], 1)

    def test_energy_feasible_rate_denominator(self):
        records = [
            make_record(scenario_id="a", energy_feasible=True),
            make_record(scenario_id="b", energy_feasible=False),
            make_record(scenario_id="c", energy_feasible=None),  # successful, missing
            make_record(scenario_id="d", success=False, allocation_cost=None,
                        reward_stage_completed=False, gini_index=None,
                        energy_feasible=True),                   # not successful
        ]
        row = self.row(aggregate_benchmark_results(frame(records)))
        self.assertAlmostEqual(row["energy_feasible_rate"], 0.5)  # 1 / 2

    def test_energy_feasible_rate_is_nan_when_denominator_is_zero(self):
        records = [
            make_record(scenario_id="a", energy_feasible=None),
            make_record(scenario_id="b", success=False, allocation_cost=None,
                        reward_stage_completed=False, gini_index=None, energy_feasible=True),
        ]
        row = self.row(aggregate_benchmark_results(frame(records)))
        self.assertTrue(math.isnan(row["energy_feasible_rate"]))

    def test_energy_feasible_rate_when_all_infeasible(self):
        records = [make_record(scenario_id=str(i), energy_feasible=False) for i in range(3)]
        row = self.row(aggregate_benchmark_results(frame(records)))
        self.assertEqual(row["energy_feasible_rate"], 0.0)

    def test_invalid_reward_count_and_gini_exclusion(self):
        records = [
            make_record(scenario_id="a", gini_index=0.4),
            make_record(scenario_id="b", invalid_reward_flag=True, gini_index=None,
                        error_message="negative reward"),
        ]
        row = self.row(aggregate_benchmark_results(frame(records)))
        self.assertEqual(row["n_invalid_reward"], 1)
        self.assertEqual(row["n_gini"], 1)
        self.assertAlmostEqual(row["gini_index_mean"], 0.4)

    def test_coalition_stability_stays_nan(self):
        records = [make_record(scenario_id=str(i)) for i in range(3)]
        row = self.row(aggregate_benchmark_results(frame(records)))
        self.assertTrue(math.isnan(row["coalition_stability_mean"]))
        self.assertTrue(math.isnan(row["coalition_stability_std"]))

    def test_counts_are_integers(self):
        summary = aggregate_benchmark_results(frame([make_record()]))
        for name in ("n_runs", "n_success", "n_allocation_cost", "n_allocation_runtime",
                     "n_reward_runtime", "n_total_runtime", "n_gini", "n_invalid_reward"):
            self.assertTrue(pd.api.types.is_integer_dtype(summary[name]), name)


class TestSummaryOutput(TempDirTestCase):
    def test_write_summary_table_serialization(self):
        summary = aggregate_benchmark_results(frame([make_record()]))
        path = write_summary_table(summary, "exp1", self.dir)
        self.assertEqual(path, self.dir / "summary_exp1.csv")
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], ",".join(SUMMARY_COLUMNS))
        cells = dict(zip(SUMMARY_COLUMNS, lines[1].split(",")))
        self.assertEqual(cells["algorithm"], "Greedy")
        self.assertEqual(cells["n_runs"], "1")
        self.assertEqual(cells["allocation_cost_mean"], "100.0")
        self.assertEqual(cells["allocation_cost_std"], "")          # NaN -> empty cell
        self.assertEqual(cells["coalition_stability_mean"], "")
        self.assertEqual(self.names(), ["summary_exp1.csv"])

    def test_summary_is_regenerated_atomically(self):
        summary = aggregate_benchmark_results(frame([make_record()]))
        write_summary_table(summary, "exp1", self.dir)
        write_summary_table(summary, "exp1", self.dir)  # derived data: replaced, no error
        self.assertEqual(self.names(), ["summary_exp1.csv"])

    def test_write_summary_table_validates_inputs(self):
        summary = aggregate_benchmark_results(frame([make_record()]))
        with self.assertRaises(SchemaValidationError):
            write_summary_table(summary.drop(columns=["n_runs"]), "exp1", self.dir)
        with self.assertRaises(SchemaValidationError):
            write_summary_table(summary, "../exp1", self.dir)
        self.assertEqual(self.names(), [])

    def test_summarize_experiment_end_to_end(self):
        raw = self.dir / "raw"
        out = self.dir / "tables"
        append_benchmark_results(
            [make_record(experiment_id="A", scenario_id="a", allocation_cost=10.0),
             make_record(experiment_id="A", scenario_id="b", allocation_cost=20.0),
             make_record(experiment_id="B", scenario_id="a", allocation_cost=999.0)],
            raw,
        )
        path = summarize_experiment("A", results_dir=raw, output_dir=out)
        self.assertEqual(path, out / "summary_A.csv")
        summary = pd.read_csv(path)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.loc[0, "n_runs"], 2)
        self.assertAlmostEqual(summary.loc[0, "allocation_cost_mean"], 15.0)

    def test_summarize_unknown_experiment_raises(self):
        append_benchmark_results([make_record()], self.dir)
        with self.assertRaises(ValueError):
            summarize_experiment("nope", results_dir=self.dir, output_dir=self.dir / "out")
        self.assertFalse((self.dir / "out").exists())


def edit_csv(path, row, edits):
    """Manually edit cells of data row ``row`` (0-based) of a CSV file."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    for column, value in edits.items():
        rows[row + 1][header.index(column)] = value
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)


class TestStrictLoadingOfMainCsv(TempDirTestCase):
    def setUp(self):
        super().setUp()
        append_benchmark_results(
            [make_record(scenario_id="a"), make_record(scenario_id="b")], self.dir
        )

    def assert_load_and_append_rejected(self, edits, pattern=None, row=0):
        edit_csv(self.results_csv, row, edits)
        before = self.results_csv.read_bytes()
        with self.assertRaisesRegex(SchemaValidationError, pattern or ""):
            load_benchmark_results(self.results_csv)
        with self.assertRaisesRegex(SchemaValidationError, pattern or ""):
            append_benchmark_results([make_record(scenario_id="new")], self.dir)
        self.assertEqual(self.results_csv.read_bytes(), before)  # untouched

    # 1. fractional values in integer columns
    def test_fractional_values_in_integer_columns_are_rejected(self):
        for column in ("num_robots", "num_tasks", "total_entities", "random_seed"):
            for value in ("5.7", "0.5", "-1.2"):
                with self.subTest(column=column, value=value):
                    edit_csv(self.results_csv, 0, {column: value})
                    with self.assertRaisesRegex(SchemaValidationError, "invalid integer"):
                        load_benchmark_results(self.results_csv)
                    edit_csv(self.results_csv, 0, {column: "5" if column != "random_seed" else "1"})

    def test_fractional_value_is_not_truncated(self):
        edit_csv(self.results_csv, 0, {"random_seed": "7.9"})
        with self.assertRaises(SchemaValidationError):
            load_benchmark_results(self.results_csv)

    def test_integral_decimal_and_signed_integers_are_accepted(self):
        edit_csv(self.results_csv, 0, {"random_seed": "7.0"})
        edit_csv(self.results_csv, 1, {"random_seed": "-3"})
        loaded = load_benchmark_results(self.results_csv)
        self.assertEqual(loaded["random_seed"].tolist(), [7, -3])
        self.assertEqual(str(loaded["random_seed"].dtype), "int64")

    # 2. invalid integer strings and infinities
    def test_invalid_integer_strings_and_infinities_are_rejected(self):
        for value in ("abc", "", "inf", "-inf", "Infinity", "nan", "1e3", "0x10", " 5",
                      "5 ", "1_0", "--5", "5.", ".5", "١٢", "9" * 20):
            with self.subTest(value=value):
                edit_csv(self.results_csv, 0, {"random_seed": value})
                with self.assertRaises(SchemaValidationError):
                    load_benchmark_results(self.results_csv)
        edit_csv(self.results_csv, 0, {"random_seed": "1"})
        self.assertEqual(len(load_benchmark_results(self.results_csv)), 2)

    def test_error_message_names_the_line(self):
        edit_csv(self.results_csv, 1, {"num_tasks": "x"})
        with self.assertRaisesRegex(SchemaValidationError, "line 3"):
            load_benchmark_results(self.results_csv)

    # 3 (loading side). malformed boolean and numeric cells
    def test_malformed_boolean_cells_are_rejected(self):
        for column in ("success", "allocation_stage_completed", "reward_stage_completed",
                       "invalid_reward_flag"):
            for value in ("true", "1", "0", "yes", "", "TRUE"):
                with self.subTest(column=column, value=value):
                    edit_csv(self.results_csv, 0, {column: value})
                    with self.assertRaises(SchemaValidationError):
                        load_benchmark_results(self.results_csv)
                    edit_csv(self.results_csv, 0, {column: "True" if column != "invalid_reward_flag" else "False"})
        for value in ("true", "1", "yes"):
            with self.subTest(column="energy_feasible", value=value):
                edit_csv(self.results_csv, 0, {"energy_feasible": value})
                with self.assertRaises(SchemaValidationError):
                    load_benchmark_results(self.results_csv)

    def test_malformed_numeric_cells_are_rejected(self):
        for column in ("allocation_cost", "allocation_runtime_seconds",
                       "reward_runtime_seconds", "total_runtime_seconds", "gini_index"):
            for value in ("abc", "nan", "NaN", "inf", "-inf", "1e999", "1,5", " 1.0"):
                with self.subTest(column=column, value=value):
                    edit_csv(self.results_csv, 0, {column: value})
                    with self.assertRaises(SchemaValidationError):
                        load_benchmark_results(self.results_csv)
                    edit_csv(self.results_csv, 0, {column: "1.0"})

    def test_empty_cells_remain_the_missing_value_convention(self):
        edit_csv(self.results_csv, 0, {"energy_feasible": "", "reward_runtime_seconds": ""})
        loaded = load_benchmark_results(self.results_csv)
        self.assertTrue(pd.isna(loaded.loc[0, "energy_feasible"]))
        self.assertTrue(math.isnan(loaded.loc[0, "reward_runtime_seconds"]))
        self.assertTrue(math.isnan(loaded.loc[0, "coalition_stability"]))

    def test_schema_violations_in_existing_rows_are_rejected(self):
        cases = [
            ({"total_entities": "11"}, "total_entities"),
            ({"coalition_stability": "0.5"}, "coalition_stability"),
            ({"allocation_runtime_seconds": "-1.0"}, ">= 0"),
            ({"scenario_id": ""}, "scenario_id"),
            ({"experiment_id": "../x"}, "experiment_id"),
        ]
        for edits, pattern in cases:
            with self.subTest(edits=edits):
                self.setUp()  # fresh valid file for every case
                self.assert_load_and_append_rejected(edits, pattern)

    # 5. cross-field combinations introduced by manually editing the CSV
    def test_cross_field_violations_in_existing_rows_are_rejected(self):
        cases = [
            ("Rule 1", {"allocation_stage_completed": "False"}),
            ("Rule 2", {"success": "False"}),
            ("Rule 2", {"success": "False", "allocation_cost": "", "gini_index": "",
                        "allocation_stage_completed": "False"}),
            ("Rule 3", {"invalid_reward_flag": "True", "reward_stage_completed": "False",
                        "gini_index": ""}),
            ("Rule 4", {"reward_stage_completed": "False"}),
            ("Rule 4", {"invalid_reward_flag": "True"}),
            ("Rule 5", {"success": "False", "reward_stage_completed": "False", "gini_index": ""}),
        ]
        for rule, edits in cases:
            with self.subTest(rule=rule, edits=edits):
                self.setUp()
                self.assert_load_and_append_rejected(edits, rule)

    def test_valid_manual_edit_is_still_accepted(self):
        edit_csv(self.results_csv, 0, {"success": "False", "allocation_cost": "",
                                        "reward_stage_completed": "False", "gini_index": ""})
        loaded = load_benchmark_results(self.results_csv)
        self.assertFalse(loaded.loc[0, "success"])
        self.assertTrue(loaded.loc[0, "allocation_stage_completed"])

    # 7. appending to a file containing an invalid existing record
    def test_append_rejected_when_an_existing_record_is_invalid(self):
        edit_csv(self.results_csv, 1, {"num_robots": "5.7"})  # second data row, file line 3
        before = self.results_csv.read_bytes()
        with self.assertRaisesRegex(SchemaValidationError, "line 3"):
            append_benchmark_results([make_record(scenario_id="new")], self.dir)
        self.assertEqual(self.results_csv.read_bytes(), before)
        self.assertEqual(self.names(), ["benchmark_results.csv"])

    def test_existing_validation_runs_before_duplicate_check(self):
        edit_csv(self.results_csv, 0, {"success": "maybe"})
        with self.assertRaises(SchemaValidationError):
            append_benchmark_results([make_record(scenario_id="b")], self.dir)

    def test_empty_batch_does_not_read_or_validate_the_file(self):
        edit_csv(self.results_csv, 0, {"success": "maybe"})
        self.assertEqual(append_benchmark_results([], self.dir), 0)


class TestStrictLoadingOfRewardCsv(TempDirTestCase):
    def setUp(self):
        super().setUp()
        append_robot_rewards(
            [make_reward(robot_id=0, reward=1.0), make_reward(robot_id=1, reward=-2.0)],
            self.dir,
        )

    # 6. NaN and infinity manually introduced into the reward CSV
    def test_nan_and_infinity_rewards_are_rejected(self):
        for value in ("nan", "NaN", "inf", "-inf", "Infinity", "1e999", "", "abc", "1,5"):
            with self.subTest(value=value):
                edit_csv(self.rewards_csv, 0, {"reward": value})
                before = self.rewards_csv.read_bytes()
                with self.assertRaises(SchemaValidationError):
                    load_robot_rewards(self.rewards_csv)
                with self.assertRaises(SchemaValidationError):
                    append_robot_rewards([make_reward(robot_id=9)], self.dir)
                self.assertEqual(self.rewards_csv.read_bytes(), before)

    def test_valid_rewards_including_negative_load_as_floats(self):
        loaded = load_robot_rewards(self.rewards_csv)
        self.assertEqual(loaded["reward"].tolist(), [1.0, -2.0])
        self.assertEqual(str(loaded["reward"].dtype), "float64")
        self.assertEqual(loaded["robot_id"].tolist(), ["0", "1"])

    def test_scientific_notation_finite_rewards_are_accepted(self):
        edit_csv(self.rewards_csv, 0, {"reward": "1.5e-3"})
        self.assertAlmostEqual(load_robot_rewards(self.rewards_csv).loc[0, "reward"], 0.0015)

    def test_other_malformed_reward_rows_are_rejected(self):
        for edits in ({"robot_id": ""}, {"scenario_id": ""}, {"experiment_id": "a b"}):
            with self.subTest(edits=edits):
                edit_csv(self.rewards_csv, 0, edits)
                with self.assertRaises(SchemaValidationError):
                    load_robot_rewards(self.rewards_csv)
                with self.assertRaises(SchemaValidationError):
                    append_robot_rewards([make_reward(robot_id=9)], self.dir)
                self.setUp()
                os.remove(str(self.rewards_csv))
                append_robot_rewards(
                    [make_reward(robot_id=0, reward=1.0), make_reward(robot_id=1, reward=-2.0)],
                    self.dir,
                )


def duplicate_data_row(path, row=0):
    """Manually append a copy of data row ``row`` (0-based) to a CSV file."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    rows.append(list(rows[row + 1]))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)


class TestLoadersRejectDuplicateKeys(TempDirTestCase):
    def test_main_results_duplicate_key_is_rejected(self):
        append_benchmark_results(
            [make_record(scenario_id="a"), make_record(scenario_id="b")], self.dir
        )
        duplicate_data_row(self.results_csv, row=1)
        with self.assertRaises(DuplicateKeyError) as caught:
            load_benchmark_results(self.results_csv)
        message = str(caught.exception)
        self.assertIn(str(self.results_csv), message)
        self.assertIn("exp1", message)
        self.assertIn("'b'", message)
        self.assertIn("Greedy", message)

    def test_main_results_duplicate_with_different_values_is_still_a_duplicate(self):
        append_benchmark_results([make_record(allocation_cost=10.0)], self.dir)
        duplicate_data_row(self.results_csv)
        edit_csv(self.results_csv, 1, {"allocation_cost": "999.0"})
        with self.assertRaises(DuplicateKeyError):
            load_benchmark_results(self.results_csv)

    def test_duplicate_main_results_cannot_be_double_counted_in_aggregation(self):
        append_benchmark_results([make_record(allocation_cost=10.0)], self.dir)
        duplicate_data_row(self.results_csv)
        with self.assertRaises(DuplicateKeyError):
            aggregate_benchmark_results(load_benchmark_results(self.results_csv))
        with self.assertRaises(DuplicateKeyError):
            summarize_experiment("exp1", results_dir=self.dir, output_dir=self.dir / "out")

    def test_robot_rewards_duplicate_key_is_rejected(self):
        append_robot_rewards(
            [make_reward(robot_id=0), make_reward(robot_id=1), make_reward(robot_id="r7")],
            self.dir,
        )
        duplicate_data_row(self.rewards_csv, row=2)
        with self.assertRaises(DuplicateKeyError) as caught:
            load_robot_rewards(self.rewards_csv)
        message = str(caught.exception)
        self.assertIn(str(self.rewards_csv), message)
        self.assertIn("r7", message)
        self.assertIn("exp1", message)

    def test_valid_files_with_distinct_keys_still_load(self):
        append_benchmark_results(
            [make_record(scenario_id="a"),
             make_record(scenario_id="a", algorithm="GA"),
             make_record(scenario_id="b"),
             make_record(experiment_id="exp2", scenario_id="a")],
            self.dir,
        )
        append_robot_rewards(
            [make_reward(robot_id=0), make_reward(robot_id=1),
             make_reward(robot_id=0, algorithm="GA"),
             make_reward(robot_id=0, scenario_id="other"),
             make_reward(robot_id=0, experiment_id="exp2")],
            self.dir,
        )
        self.assertEqual(len(load_benchmark_results(self.results_csv)), 4)
        self.assertEqual(len(load_robot_rewards(self.rewards_csv)), 5)

    def test_empty_and_header_only_files_still_load(self):
        self.results_csv.write_bytes(b"")
        self.rewards_csv.write_text(",".join(ROBOT_REWARD_COLUMNS) + "\n", encoding="utf-8")
        self.assertEqual(len(load_benchmark_results(self.results_csv)), 0)
        self.assertEqual(len(load_robot_rewards(self.rewards_csv)), 0)

    def test_invalid_row_is_still_reported_as_schema_error(self):
        append_benchmark_results([make_record(scenario_id="a")], self.dir)
        duplicate_data_row(self.results_csv)
        edit_csv(self.results_csv, 1, {"num_robots": "5.7"})
        with self.assertRaises(SchemaValidationError):
            load_benchmark_results(self.results_csv)


class TestStrictBooleansInAggregation(unittest.TestCase):
    def aggregate(self, **column_overrides):
        data = frame([make_record(scenario_id=str(i)) for i in range(3)])
        for name, values in column_overrides.items():
            data[name] = values
        return aggregate_benchmark_results(data)

    BOOL_COLUMNS = ("success", "allocation_stage_completed", "reward_stage_completed",
                    "invalid_reward_flag")

    # 3. string booleans
    def test_string_booleans_are_rejected_and_false_is_never_true(self):
        for name in self.BOOL_COLUMNS:
            for strings in (["False"] * 3, ["True", "False", "True"], ["false"] * 3):
                with self.subTest(column=name, values=strings):
                    with self.assertRaises(SchemaValidationError):
                        self.aggregate(**{name: strings})
        with self.assertRaises(SchemaValidationError):
            self.aggregate(energy_feasible=["False", "False", "False"])

    def test_false_string_does_not_change_counts(self):
        # Regression: bool("False") is True; this must raise, not count as success.
        with self.assertRaises(SchemaValidationError):
            self.aggregate(success=["False", "False", "False"])

    # 4. integers 0 and 1 instead of booleans
    def test_integer_flags_are_rejected(self):
        for name in self.BOOL_COLUMNS:
            for values in ([1, 1, 1], [0, 1, 0], np.array([1, 0, 1])):
                with self.subTest(column=name, values=list(values)):
                    with self.assertRaises(SchemaValidationError):
                        self.aggregate(**{name: values})
        with self.assertRaises(SchemaValidationError):
            self.aggregate(energy_feasible=[1, 0, 1])
        with self.assertRaises(SchemaValidationError):
            self.aggregate(energy_feasible=np.array([1.0, 0.0, 1.0]))

    def test_missing_values_in_required_boolean_columns_are_rejected(self):
        for missing in (None, float("nan"), pd.NA):
            with self.subTest(missing=missing):
                with self.assertRaises(SchemaValidationError):
                    self.aggregate(success=[True, missing, True])

    def test_real_booleans_are_accepted(self):
        for values in ([True, False, True], np.array([True, False, True]),
                       pd.array([True, False, True], dtype="boolean"),
                       pd.Series([True, False, True], dtype=object)):
            with self.subTest(values=type(values).__name__):
                data = frame([make_record(scenario_id=str(i)) for i in range(3)])
                data["success"] = values
                data["allocation_cost"] = [1.0, None, 3.0]  # keep cost consistent
                summary = aggregate_benchmark_results(data)
                self.assertEqual(summary.iloc[0]["n_success"], 2)

    def test_energy_feasible_accepts_booleans_and_missing_values(self):
        for values in ([True, False, None], [True, False, pd.NA], [True, False, float("nan")],
                       pd.array([True, False, None], dtype="boolean")):
            with self.subTest(values=type(values).__name__):
                summary = self.aggregate(energy_feasible=values)
                self.assertAlmostEqual(summary.iloc[0]["energy_feasible_rate"], 0.5)

    def test_energy_feasible_object_column_with_unexpected_values_is_rejected(self):
        for bad in ("False", "yes", 1, 0, 2.5):
            with self.subTest(bad=bad):
                with self.assertRaises(SchemaValidationError):
                    self.aggregate(energy_feasible=[True, bad, None])

    def test_loaded_data_still_aggregates(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(directory), True)
        append_benchmark_results([make_record(scenario_id="a", energy_feasible=None),
                                  make_record(scenario_id="b", energy_feasible=False)], directory)
        summary = aggregate_benchmark_results(
            load_benchmark_results(directory / "benchmark_results.csv")
        )
        self.assertEqual(summary.iloc[0]["n_success"], 2)
        self.assertEqual(summary.iloc[0]["energy_feasible_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

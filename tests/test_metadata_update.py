"""Tests for logger.update_experiment_metadata (safe metadata updates, e.g. after --resume)."""

import copy
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from src.analytics.logger import (
    SchemaValidationError,
    update_experiment_metadata,
    write_experiment_metadata,
)


class TestUpdateExperimentMetadata(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="meta_test_"))
        self.addCleanup(shutil.rmtree, str(self.dir), True)
        self.path = write_experiment_metadata(
            "exp1", {"energy": {"strategy": "quadratic"}}, {"base_seed": 42}, {"MILP": 300},
            extra={"harness": {"adapters": {}, "invocations": [], "kept": "yes"}, "other": 1},
            results_dir=self.dir)

    def load(self):
        return yaml.safe_load(self.path.read_text(encoding="utf-8"))

    def test_set_and_append_update_only_the_extra_section(self):
        before = self.load()
        update_experiment_metadata(
            "exp1",
            set_values={"harness.adapters": {"Greedy": "pkg.mod:fn"}, "harness.new.nested": 5},
            append_values={"harness.invocations": {"mode": "start"}},
            results_dir=self.dir)
        after = self.load()
        self.assertEqual(after["extra"]["harness"]["adapters"], {"Greedy": "pkg.mod:fn"})
        self.assertEqual(after["extra"]["harness"]["new"], {"nested": 5})
        self.assertEqual(after["extra"]["harness"]["invocations"], [{"mode": "start"}])
        self.assertEqual(after["extra"]["harness"]["kept"], "yes")          # untouched siblings survive
        self.assertEqual(after["extra"]["other"], 1)
        for key in ("experiment_id", "created_utc", "config", "seed_info", "time_limits", "environment"):
            self.assertEqual(after[key], before[key], key)                   # original run information is kept
        self.assertIn("updated_utc", after)

    def test_append_accumulates_history(self):
        for mode in ("start", "resume", "resume"):
            update_experiment_metadata("exp1", append_values={"harness.invocations": {"mode": mode}},
                                       results_dir=self.dir)
        self.assertEqual([e["mode"] for e in self.load()["extra"]["harness"]["invocations"]],
                         ["start", "resume", "resume"])

    def test_append_creates_a_missing_list_and_extra_section(self):
        path = write_experiment_metadata("bare", {}, {}, {}, results_dir=self.dir)      # no extra at all
        update_experiment_metadata("bare", append_values={"history.entries": 1}, results_dir=self.dir)
        self.assertEqual(yaml.safe_load(path.read_text(encoding="utf-8"))["extra"], {"history": {"entries": [1]}})

    def test_paths_cannot_reach_the_protected_top_level_entries(self):
        before = self.load()
        update_experiment_metadata("exp1", set_values={"config.energy": "hijacked", "created_utc": "x"},
                                   results_dir=self.dir)
        after = self.load()
        self.assertEqual(after["config"], before["config"])                  # the top-level config is untouched
        self.assertEqual(after["created_utc"], before["created_utc"])
        self.assertEqual(after["extra"]["config"], {"energy": "hijacked"})   # the path lived inside "extra"

    def test_write_experiment_metadata_still_refuses_to_overwrite(self):
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            write_experiment_metadata("exp1", {}, {}, {}, results_dir=self.dir)
        self.assertEqual(self.path.read_bytes(), before)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            update_experiment_metadata("nope", set_values={"a": 1}, results_dir=self.dir)

    def test_unsafe_ids_and_paths_and_values(self):
        for bad in ("../x", "a b", ""):
            with self.assertRaises(SchemaValidationError):
                update_experiment_metadata(bad, results_dir=self.dir)
        for bad_path in ("", "a..b", "a b", "../x", "a.", ".a"):
            with self.subTest(path=bad_path):
                with self.assertRaises(SchemaValidationError):
                    update_experiment_metadata("exp1", set_values={bad_path: 1}, results_dir=self.dir)
        with self.assertRaises(SchemaValidationError):
            update_experiment_metadata("exp1", set_values={"a": object()}, results_dir=self.dir)

    def test_path_crossing_a_non_mapping_and_append_to_a_non_list(self):
        with self.assertRaisesRegex(SchemaValidationError, "non-mapping"):
            update_experiment_metadata("exp1", set_values={"other.deeper": 1}, results_dir=self.dir)
        with self.assertRaisesRegex(SchemaValidationError, "not a list"):
            update_experiment_metadata("exp1", append_values={"other": 1}, results_dir=self.dir)

    def test_failed_update_leaves_the_file_untouched(self):
        before = self.path.read_bytes()
        with self.assertRaises(SchemaValidationError):
            update_experiment_metadata("exp1", set_values={"harness.ok": 1, "other.deeper": 1},
                                       results_dir=self.dir)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["metadata_exp1.yaml"])

    def test_malformed_or_foreign_files_are_rejected(self):
        self.path.write_text("a: [unclosed", encoding="utf-8")
        with self.assertRaises(SchemaValidationError):
            update_experiment_metadata("exp1", set_values={"a": 1}, results_dir=self.dir)
        self.path.write_text(yaml.safe_dump({"experiment_id": "someone_else"}), encoding="utf-8")
        with self.assertRaisesRegex(SchemaValidationError, "not the metadata file"):
            update_experiment_metadata("exp1", set_values={"a": 1}, results_dir=self.dir)

    def test_no_temp_files_are_left_and_inputs_are_not_mutated(self):
        values = {"harness.adapters": {"A": "m:f"}}
        snapshot = copy.deepcopy(values)
        update_experiment_metadata("exp1", set_values=values, results_dir=self.dir)
        self.assertEqual(values, snapshot)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["metadata_exp1.yaml"])


if __name__ == "__main__":
    unittest.main()

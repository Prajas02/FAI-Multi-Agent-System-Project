"""Unit tests for src/analytics/seeding.py (seed management, design Section 6.1)."""

import copy
import hashlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from src.analytics import seeding
from src.analytics.seeding import (
    DEFAULT_CONFIG_PATH,
    SEED_MAX,
    SeedConfigError,
    algorithm_seed,
    clear_seed_template_cache,
    derive_seed,
    get_seed_templates,
    resolve_config_path,
    reward_seed,
    scenario_seed,
)

ALGORITHMS = ("MILP", "GA", "Greedy", "SVCA")
CONFIG_IDS = ["C%d" % i for i in range(1, 9)]

# The nine test vectors of docs/BENCHMARK_HARNESS_DESIGN.md, Section 6.1.
SCENARIO_VECTORS = [
    ("scenario|42|C1|0", (42, "C1", 0), 1593276916),
    ("scenario|42|C1|9", (42, "C1", 9), 383252898),
    ("scenario|42|C8|0", (42, "C8", 0), 906788627),
    ("scenario|42|C8|9", (42, "C8", 9), 2601196458),
]
ALGORITHM_VECTORS = [
    ("algorithm|1593276916|MILP", "MILP", 4211836502),
    ("algorithm|1593276916|GA", "GA", 4089982009),
    ("algorithm|1593276916|Greedy", "Greedy", 4190077063),
    ("algorithm|1593276916|SVCA", "SVCA", 4235073208),
]
REWARD_VECTOR = ("reward|1593276916|reward_stage", "reward_stage", 581375934)
C1_REP0 = 1593276916


def valid_document():
    """A complete, valid configuration document (independent of the real file)."""
    return {
        "seeds": {
            "derivation_method": "sha256",
            "hash_function": "sha256",
            "seed_bytes": 4,
            "byte_order": "big",
            "signed": False,
            "seed_min": 0,
            "seed_max": 4294967295,
            "message_encoding": "utf-8",
            "messages": {
                "scenario": "scenario|{base_seed}|{config_id}|{repetition_index}",
                "algorithm": "algorithm|{scenario_seed}|{algorithm_id}",
                "reward": "reward|{scenario_seed}|{reward_stage_label}",
            },
        }
    }


class ConfigFileTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.dir), True)
        clear_seed_template_cache()
        self.addCleanup(clear_seed_template_cache)

    def write_config(self, document, name="config.yaml"):
        path = self.dir / name
        text = document if isinstance(document, str) else yaml.safe_dump(document)
        path.write_text(text, encoding="utf-8")
        return path


class TestDeriveSeed(unittest.TestCase):
    def test_exact_rule(self):
        message = "scenario|42|C1|0"
        expected = int.from_bytes(
            hashlib.sha256(message.encode("utf-8")).digest()[:4], byteorder="big", signed=False
        )
        self.assertEqual(derive_seed(message), expected)

    def test_design_document_vectors_via_derive_seed(self):
        for message, _args, expected in SCENARIO_VECTORS:
            self.assertEqual(derive_seed(message), expected, message)
        for message, _alg, expected in ALGORITHM_VECTORS:
            self.assertEqual(derive_seed(message), expected, message)
        self.assertEqual(derive_seed(REWARD_VECTOR[0]), REWARD_VECTOR[2])

    def test_deterministic(self):
        self.assertEqual(derive_seed("abc"), derive_seed("abc"))
        self.assertEqual(derive_seed(""), derive_seed(""))

    def test_range(self):
        for message in ("", "a", "scenario|0|C1|0", "x" * 1000, "é中"):
            value = derive_seed(message)
            self.assertIs(type(value), int)
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 4294967295)
        self.assertEqual(SEED_MAX, 4294967295)

    def test_utf8_encoding_is_used(self):
        message = "algorithm|1|É"
        expected = int.from_bytes(hashlib.sha256(message.encode("utf-8")).digest()[:4], "big")
        self.assertEqual(derive_seed(message), expected)

    def test_non_string_input_raises_type_error(self):
        for value in (None, 1, 1.5, b"abc", ["a"], ("a",), object()):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    derive_seed(value)

    def test_does_not_use_builtin_hash(self):
        with mock.patch("builtins.hash", side_effect=AssertionError("hash() must not be used")):
            self.assertEqual(derive_seed("scenario|42|C1|0"), C1_REP0)
            self.assertEqual(scenario_seed(42, "C1", 0), C1_REP0)


class TestScenarioSeed(unittest.TestCase):
    def test_design_document_vectors(self):
        for _message, (base, config_id, rep), expected in SCENARIO_VECTORS:
            self.assertEqual(scenario_seed(base, config_id, rep), expected)

    def test_deterministic(self):
        self.assertEqual(scenario_seed(42, "C3", 4), scenario_seed(42, "C3", 4))

    def test_range(self):
        for config_id in CONFIG_IDS:
            for rep in range(10):
                value = scenario_seed(42, config_id, rep)
                self.assertTrue(0 <= value <= 4294967295)

    def test_repetition_index_is_zero_based(self):
        self.assertEqual(scenario_seed(42, "C1", 0), derive_seed("scenario|42|C1|0"))
        self.assertEqual(scenario_seed(42, "C1", 9), derive_seed("scenario|42|C1|9"))
        self.assertNotEqual(scenario_seed(42, "C1", 0), derive_seed("scenario|42|C1|1"))
        self.assertNotEqual(scenario_seed(42, "C1", 0), scenario_seed(42, "C1", 1))

    def test_negative_or_non_integer_repetition_is_rejected(self):
        with self.assertRaises(ValueError):
            scenario_seed(42, "C1", -1)
        for value in (1.0, "0", None, True):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    scenario_seed(42, "C1", value)

    def test_different_configurations_give_distinct_seeds(self):
        seeds = {(c, r): scenario_seed(42, c, r) for c in CONFIG_IDS for r in range(10)}
        self.assertEqual(len(seeds), 80)
        self.assertEqual(len(set(seeds.values())), 80)  # all 80 scenario seeds differ
        for rep in range(10):
            per_config = {scenario_seed(42, c, rep) for c in CONFIG_IDS}
            self.assertEqual(len(per_config), 8)

    def test_different_base_seeds_give_different_seeds(self):
        self.assertNotEqual(scenario_seed(42, "C1", 0), scenario_seed(43, "C1", 0))
        self.assertNotEqual(scenario_seed(0, "C1", 0), scenario_seed(42, "C1", 0))

    def test_config_id_is_case_sensitive_and_exact(self):
        self.assertNotEqual(scenario_seed(42, "C1", 0), scenario_seed(42, "c1", 0))
        self.assertNotEqual(scenario_seed(42, "C1", 0), scenario_seed(42, "C1 ", 0))

    def test_invalid_arguments(self):
        for value in (-1, 4294967296):
            with self.subTest(base_seed=value):
                with self.assertRaises(ValueError):
                    scenario_seed(value, "C1", 0)
        for value in (1.5, "42", None, True):
            with self.subTest(base_seed=value):
                with self.assertRaises(TypeError):
                    scenario_seed(value, "C1", 0)
        with self.assertRaises(ValueError):
            scenario_seed(42, "", 0)
        for value in (None, 1):
            with self.assertRaises(TypeError):
                scenario_seed(42, value, 0)


class TestAlgorithmAndRewardSeeds(unittest.TestCase):
    def test_algorithm_vectors(self):
        for _message, algorithm_id, expected in ALGORITHM_VECTORS:
            self.assertEqual(algorithm_seed(C1_REP0, algorithm_id), expected)

    def test_algorithm_seeds_are_distinct_for_supplied_vectors(self):
        values = [algorithm_seed(C1_REP0, a) for a in ALGORITHMS]
        self.assertEqual(len(set(values)), 4)
        self.assertEqual(values, [4211836502, 4089982009, 4190077063, 4235073208])

    def test_algorithm_id_exact_string_and_case(self):
        self.assertNotEqual(algorithm_seed(C1_REP0, "Greedy"), algorithm_seed(C1_REP0, "greedy"))
        self.assertNotEqual(algorithm_seed(C1_REP0, "GA"), algorithm_seed(C1_REP0, "GA "))
        self.assertEqual(algorithm_seed(C1_REP0, "SVCA"), derive_seed("algorithm|1593276916|SVCA"))

    def test_algorithm_seed_depends_on_scenario_seed(self):
        other = scenario_seed(42, "C1", 1)
        self.assertNotEqual(algorithm_seed(C1_REP0, "MILP"), algorithm_seed(other, "MILP"))

    def test_algorithm_seed_is_deterministic_and_in_range(self):
        for algorithm_id in ALGORITHMS:
            first = algorithm_seed(C1_REP0, algorithm_id)
            self.assertEqual(first, algorithm_seed(C1_REP0, algorithm_id))
            self.assertTrue(0 <= first <= 4294967295)

    def test_reward_vector(self):
        self.assertEqual(reward_seed(C1_REP0, REWARD_VECTOR[1]), REWARD_VECTOR[2])

    def test_reward_seed_is_shared_across_algorithms(self):
        # The reward seed has no algorithm input: the same scenario seed and label
        # give the same value no matter which algorithm ran before it.
        results = []
        for algorithm_id in ALGORITHMS:
            algorithm_seed(C1_REP0, algorithm_id)
            results.append(reward_seed(C1_REP0, "reward_stage"))
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(results[0], 581375934)

    def test_reward_seed_depends_on_scenario_seed_and_label(self):
        self.assertNotEqual(reward_seed(C1_REP0, "reward_stage"), reward_seed(C1_REP0, "Reward_Stage"))
        self.assertNotEqual(
            reward_seed(C1_REP0, "reward_stage"),
            reward_seed(scenario_seed(42, "C1", 1), "reward_stage"),
        )

    def test_reward_seed_differs_from_algorithm_seeds(self):
        self.assertNotIn(
            reward_seed(C1_REP0, "reward_stage"), [algorithm_seed(C1_REP0, a) for a in ALGORITHMS]
        )

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            algorithm_seed(-1, "MILP")
        with self.assertRaises(ValueError):
            algorithm_seed(C1_REP0, "")
        with self.assertRaises(TypeError):
            algorithm_seed("1593276916", "MILP")
        with self.assertRaises(ValueError):
            reward_seed(C1_REP0, "")
        with self.assertRaises(TypeError):
            reward_seed(C1_REP0, None)
        with self.assertRaises(ValueError):
            reward_seed(SEED_MAX + 1, "reward_stage")


class TestTemplatesComeFromTheYamlFile(ConfigFileTestCase):
    def test_default_path_is_the_repository_config(self):
        expected = Path(seeding.__file__).resolve().parents[2] / "configs" / "experiments_config.yaml"
        self.assertEqual(DEFAULT_CONFIG_PATH, expected)
        self.assertEqual(resolve_config_path(), expected)
        self.assertTrue(expected.is_file())

    def test_templates_match_the_real_configuration_file(self):
        real = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))["seeds"]["messages"]
        templates = get_seed_templates()
        self.assertEqual(templates.scenario, real["scenario"])
        self.assertEqual(templates.algorithm, real["algorithm"])
        self.assertEqual(templates.reward, real["reward"])

    def test_custom_templates_are_used(self):
        document = valid_document()
        document["seeds"]["messages"] = {
            "scenario": "S/{config_id}/{base_seed}/{repetition_index}",
            "algorithm": "A/{algorithm_id}/{scenario_seed}",
            "reward": "R/{reward_stage_label}/{scenario_seed}",
        }
        path = self.write_config(document)
        self.assertEqual(
            scenario_seed(7, "CX", 3, config_path=path), derive_seed("S/CX/7/3")
        )
        self.assertEqual(
            algorithm_seed(11, "MILP", config_path=path), derive_seed("A/MILP/11")
        )
        self.assertEqual(
            reward_seed(11, "lbl", config_path=path), derive_seed("R/lbl/11")
        )
        # and the values differ from the default-scheme values
        self.assertNotEqual(scenario_seed(42, "C1", 0, config_path=path), C1_REP0)

    def test_equivalent_file_reproduces_the_vectors(self):
        path = self.write_config(valid_document())
        self.assertEqual(scenario_seed(42, "C1", 0, config_path=path), C1_REP0)
        self.assertEqual(algorithm_seed(C1_REP0, "GA", config_path=path), 4089982009)
        self.assertEqual(reward_seed(C1_REP0, "reward_stage", config_path=path), 581375934)

    def test_templates_are_not_duplicated_in_the_source_code(self):
        source = Path(seeding.__file__).read_text(encoding="utf-8")
        for fragment in ("scenario|", "algorithm|", "reward|", "|{base_seed}", "|{config_id}",
                         "|{scenario_seed}", "|{algorithm_id}", "|{reward_stage_label}"):
            self.assertNotIn(fragment, source)

    def test_real_yaml_file_is_not_modified(self):
        before = hashlib.sha256(DEFAULT_CONFIG_PATH.read_bytes()).hexdigest()
        scenario_seed(42, "C1", 0)
        algorithm_seed(C1_REP0, "MILP")
        reward_seed(C1_REP0, "reward_stage")
        after = hashlib.sha256(DEFAULT_CONFIG_PATH.read_bytes()).hexdigest()
        self.assertEqual(before, after)

    def test_relative_path_is_resolved_against_the_repository_root(self):
        repo_root = DEFAULT_CONFIG_PATH.parents[1]
        self.assertEqual(
            resolve_config_path("configs/experiments_config.yaml"), repo_root / "configs" / "experiments_config.yaml"
        )
        self.assertEqual(resolve_config_path("some/where.yaml"), repo_root / "some" / "where.yaml")
        absolute = self.dir / "x.yaml"
        self.assertEqual(resolve_config_path(absolute), absolute)

    def test_default_config_does_not_depend_on_the_working_directory(self):
        previous = os.getcwd()
        os.chdir(str(self.dir))
        self.addCleanup(os.chdir, previous)
        clear_seed_template_cache()
        self.assertEqual(scenario_seed(42, "C1", 0), C1_REP0)
        self.assertEqual(
            resolve_config_path("configs/experiments_config.yaml"),
            DEFAULT_CONFIG_PATH,
        )


class TestConfigurationCaching(ConfigFileTestCase):
    def test_file_is_read_once_for_many_seed_calculations(self):
        path = self.write_config(valid_document())
        real_safe_load = yaml.safe_load
        with mock.patch("src.analytics.seeding.yaml.safe_load", side_effect=real_safe_load) as spy:
            for rep in range(10):
                scenario_seed(42, "C1", rep, config_path=path)
                algorithm_seed(C1_REP0, "MILP", config_path=path)
                reward_seed(C1_REP0, "reward_stage", config_path=path)
        self.assertEqual(spy.call_count, 1)

    def test_changed_file_is_reloaded(self):
        path = self.write_config(valid_document())
        first = scenario_seed(42, "C1", 0, config_path=path)
        document = valid_document()
        document["seeds"]["messages"]["scenario"] = "changed|{base_seed}|{config_id}|{repetition_index}"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        stat = path.stat()
        os.utime(str(path), ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
        second = scenario_seed(42, "C1", 0, config_path=path)
        self.assertEqual(first, C1_REP0)
        self.assertEqual(second, derive_seed("changed|42|C1|0"))

    def test_cache_can_be_cleared(self):
        path = self.write_config(valid_document())
        get_seed_templates(path)
        real_safe_load = yaml.safe_load
        clear_seed_template_cache()
        with mock.patch("src.analytics.seeding.yaml.safe_load", side_effect=real_safe_load) as spy:
            get_seed_templates(path)
        self.assertEqual(spy.call_count, 1)

    def test_failed_load_is_not_cached(self):
        path = self.write_config("seeds: [unclosed")
        with self.assertRaises(SeedConfigError):
            get_seed_templates(path)
        path.write_text(yaml.safe_dump(valid_document()), encoding="utf-8")
        self.assertEqual(scenario_seed(42, "C1", 0, config_path=path), C1_REP0)


class TestConfigurationErrors(ConfigFileTestCase):
    def assert_config_error(self, document, pattern=""):
        path = self.write_config(document)
        clear_seed_template_cache()
        with self.assertRaisesRegex(SeedConfigError, pattern):
            scenario_seed(42, "C1", 0, config_path=path)

    def test_missing_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            scenario_seed(42, "C1", 0, config_path=self.dir / "does_not_exist.yaml")

    def test_malformed_yaml(self):
        self.assert_config_error("seeds: [unclosed", "Malformed YAML")
        self.assert_config_error("seeds: {a: 1\n  b: : 2", "Malformed YAML")

    def test_top_level_must_be_a_mapping(self):
        self.assert_config_error("- a\n- b\n", "top level")
        self.assert_config_error("", "top level")

    def test_missing_or_invalid_seeds_section(self):
        self.assert_config_error({"experiment": {"id": "x"}}, "'seeds' section")
        self.assert_config_error({"seeds": "text"}, "'seeds' section")

    def test_missing_messages_section(self):
        document = valid_document()
        del document["seeds"]["messages"]
        self.assert_config_error(document, "seeds.messages")
        document["seeds"]["messages"] = ["a"]
        self.assert_config_error(document, "seeds.messages")

    def test_missing_template_keys(self):
        for kind in ("scenario", "algorithm", "reward"):
            with self.subTest(kind=kind):
                document = valid_document()
                del document["seeds"]["messages"][kind]
                self.assert_config_error(document, "missing seeds.messages.%s" % kind)

    def test_missing_parameter_keys(self):
        for key in ("derivation_method", "seed_bytes", "byte_order", "message_encoding", "seed_max"):
            with self.subTest(key=key):
                document = valid_document()
                del document["seeds"][key]
                self.assert_config_error(document, "missing seeds.%s" % key)

    def test_parameters_that_disagree_with_the_implementation(self):
        for key, bad in (("derivation_method", "md5"), ("hash_function", "sha1"), ("seed_bytes", 8),
                         ("byte_order", "little"), ("signed", True), ("seed_max", 255),
                         ("message_encoding", "latin-1")):
            with self.subTest(key=key):
                document = valid_document()
                document["seeds"][key] = bad
                self.assert_config_error(document, "seeds.%s" % key)

    def test_template_must_be_a_string(self):
        for bad in (5, None, ["a"], {"x": 1}):
            with self.subTest(bad=bad):
                document = valid_document()
                document["seeds"]["messages"]["scenario"] = bad
                self.assert_config_error(document, "must be a string")

    def test_invalid_placeholders(self):
        cases = {
            "missing placeholder": ("scenario", "scenario|{base_seed}|{config_id}", "missing=\\['repetition_index'\\]"),
            "unknown placeholder": ("scenario", "scenario|{base_seed}|{config_id}|{repetition_index}|{extra}", "unexpected=\\['extra'\\]"),
            "renamed placeholder": ("algorithm", "algorithm|{scenario_seed}|{algorithm}", "missing=\\['algorithm_id'\\]"),
            "wrong kind": ("reward", "reward|{scenario_seed}|{algorithm_id}", "unexpected=\\['algorithm_id'\\]"),
            "no placeholders": ("reward", "reward", "must use exactly"),
            "positional": ("algorithm", "algorithm|{}|{algorithm_id}", "unsupported placeholder"),
            "numbered": ("algorithm", "algorithm|{0}|{algorithm_id}", "unsupported placeholder"),
            "attribute access": ("scenario", "s|{base_seed}|{config_id.x}|{repetition_index}", "unsupported placeholder"),
            "index access": ("scenario", "s|{base_seed}|{config_id[0]}|{repetition_index}", "unsupported placeholder"),
            "format spec": ("scenario", "s|{base_seed}|{config_id}|{repetition_index:02d}", "format spec"),
            "conversion": ("scenario", "s|{base_seed}|{config_id!r}|{repetition_index}", "format spec or conversion"),
            "unbalanced brace": ("scenario", "s|{base_seed}|{config_id}|{repetition_index", "not a valid template"),
            "stray close brace": ("algorithm", "algorithm|{scenario_seed}|{algorithm_id}}", "not a valid template"),
        }
        for label, (kind, template, pattern) in cases.items():
            with self.subTest(label):
                document = valid_document()
                document["seeds"]["messages"][kind] = template
                path = self.write_config(document)
                clear_seed_template_cache()
                with self.assertRaisesRegex(SeedConfigError, pattern):
                    get_seed_templates(path)

    def test_no_silent_fallback_to_default_templates(self):
        # A broken file must raise; it must not fall back to the approved scheme.
        document = valid_document()
        document["seeds"]["messages"]["scenario"] = "scenario|{base_seed}"
        path = self.write_config(document)
        clear_seed_template_cache()
        with self.assertRaises(SeedConfigError):
            scenario_seed(42, "C1", 0, config_path=path)

    def test_errors_do_not_modify_the_file(self):
        path = self.write_config("seeds: [unclosed")
        before = path.read_bytes()
        with self.assertRaises(SeedConfigError):
            get_seed_templates(path)
        self.assertEqual(path.read_bytes(), before)

    def test_valid_document_is_not_mutated_by_loading(self):
        document = valid_document()
        snapshot = copy.deepcopy(document)
        self.write_config(document)
        self.assertEqual(document, snapshot)


if __name__ == "__main__":
    unittest.main()

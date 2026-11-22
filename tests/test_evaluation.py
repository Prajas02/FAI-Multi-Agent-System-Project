"""Unit tests for src/analytics/evaluation.py (design Sections 8.1-8.4).

Small hand-computed scenarios are built directly from the core types, so the
expected values do not depend on the scenario generator. The generator is used
only for properties that need a non-trivial, non-round scenario.
"""

import copy
import math
import types
import unittest
from collections.abc import Mapping
from unittest import mock

import numpy as np

from src.analytics.evaluation import (
    ENERGY_TOLERANCE_RELATIVE,
    PENALTY_STRATEGIES,
    SANITY_TOLERANCE,
    EnergyFeasibility,
    EnergyPenalty,
    EvaluationError,
    EvaluationResult,
    ValidationResult,
    compute_energy_feasibility,
    compute_energy_tolerances,
    compute_remaining_energy,
    energy_penalty_phi,
    evaluate_allocation,
    validate_assignments,
)
from src.core.agent import Agent
from src.core.task import Task
from src.core.types import Position
from src.simulation.environment import Environment
from src.simulation.generator import ScenarioData, ScenarioGenerator

NAN = float("nan")
INF = float("inf")
QUADRATIC = {"strategy": "quadratic", "lambda_E": 1.0}


def make_scenario(e1=1000.0, e2=500.0, capacity1=None, capacity2=None):
    """Two robots, three tasks; every number below is chosen for hand calculation.

    Robot 1 at (0, 0), v = 10, usage penalty 2.   Robot 2 at (100, 100), v = 5, penalty 1.
    Task 1 at (3, 4):    exec 10 s, priority 3.   (5 m from robot 1)
    Task 2 at (3, 16):   exec 20 s, priority 2.   (12 m from task 1)
    Task 3 at (100, 160): exec 30 s, priority 4.  (60 m from robot 2)
    """
    agents = [
        Agent(agent_id=1, position=Position(0.0, 0.0), velocity=10.0,
              energy_capacity=capacity1 or e1, current_energy=e1, usage_penalty=2.0),
        Agent(agent_id=2, position=Position(100.0, 100.0), velocity=5.0,
              energy_capacity=capacity2 or e2, current_energy=e2, usage_penalty=1.0),
    ]
    tasks = [
        Task(task_id=1, location=Position(3.0, 4.0), execution_time=10.0, priority=3.0),
        Task(task_id=2, location=Position(3.0, 16.0), execution_time=20.0, priority=2.0),
        Task(task_id=3, location=Position(100.0, 160.0), execution_time=30.0, priority=4.0),
    ]
    return ScenarioData(map_width=500.0, map_height=500.0, seed=0, agents=agents, tasks=tasks)


PLAN = {1: [1, 2], 2: [3]}

# Hand-computed (default BatteryDecayModel: 1 energy per metre, 0.1 per second of execution):
#   robot 1: eta1 = 5/10 + 10 = 10.5;  eta2 = 10.5 + 12/10 + 20 = 31.7;  d1 = 5 + 12 = 17
#            energy = (5 + 1) + (12 + 2) = 20
#   robot 2: eta3 = 60/5 + 30 = 42;  d2 = 60;  energy = 60 + 3 = 63
#   base = 3*10.5 + 2*31.7 + 4*42 + 2*17 + 1*60 = 262.9 + 94 = 356.9
BASE_COST = 356.9


class TestValidateAssignments(unittest.TestCase):
    def setUp(self):
        self.scenario = make_scenario()

    def assert_invalid(self, plan, pattern=None):
        result = validate_assignments(plan, self.scenario)
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.valid)
        self.assertIsNone(result.normalized)
        self.assertTrue(result.errors)
        if pattern is not None:
            self.assertTrue(any(pattern in message for message in result.errors), result.errors)
        return result

    def test_valid_allocation(self):
        result = validate_assignments(PLAN, self.scenario)
        self.assertTrue(result.valid)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.normalized, {1: [1, 2], 2: [3]})

    def test_result_structure(self):
        result = validate_assignments(PLAN, self.scenario)
        self.assertEqual(ValidationResult._fields, ("valid", "errors", "normalized"))
        valid, errors, normalized = result
        self.assertIs(valid, True)
        self.assertIsInstance(errors, list)
        self.assertIsInstance(normalized, dict)

    def test_order_of_tasks_is_preserved(self):
        result = validate_assignments({1: [2, 1], 2: [3]}, self.scenario)
        self.assertEqual(result.normalized[1], [2, 1])

    def test_tuples_are_accepted_and_converted_to_lists(self):
        result = validate_assignments({1: (1, 2), 2: (3,)}, self.scenario)
        self.assertTrue(result.valid)
        self.assertEqual(result.normalized, {1: [1, 2], 2: [3]})
        self.assertIsInstance(result.normalized[1], list)

    def test_omitted_robot_is_idle(self):
        result = validate_assignments({1: [1, 2, 3]}, self.scenario)
        self.assertTrue(result.valid)
        self.assertEqual(result.normalized, {1: [1, 2, 3], 2: []})

    def test_empty_list_robot_is_idle(self):
        result = validate_assignments({1: [1, 2, 3], 2: []}, self.scenario)
        self.assertTrue(result.valid)
        self.assertEqual(result.normalized, {1: [1, 2, 3], 2: []})

    def test_normalized_contains_every_robot_in_scenario_order(self):
        result = validate_assignments({2: [1, 2, 3]}, self.scenario)
        self.assertEqual(list(result.normalized), [1, 2])
        self.assertEqual(result.normalized, {1: [], 2: [1, 2, 3]})

    def test_scenario_without_tasks_and_empty_plan_is_valid(self):
        scenario = make_scenario()
        scenario.tasks = []
        result = validate_assignments({}, scenario)
        self.assertTrue(result.valid)
        self.assertEqual(result.normalized, {1: [], 2: []})

    def test_normalized_is_a_fresh_structure(self):
        plan = {1: [1, 2], 2: [3]}
        result = validate_assignments(plan, self.scenario)
        self.assertIsNot(result.normalized, plan)
        for robot in plan:
            self.assertIsNot(result.normalized[robot], plan[robot])
        result.normalized[1].append(99)
        self.assertEqual(plan, {1: [1, 2], 2: [3]})

    def test_input_is_not_modified(self):
        plan = {1: [1, 2], 2: [3]}
        snapshot = copy.deepcopy(plan)
        validate_assignments(plan, self.scenario)
        validate_assignments({1: [1, 1], 99: [4]}, self.scenario)
        self.assertEqual(plan, snapshot)

    def test_scenario_is_not_modified(self):
        before = self.scenario.to_dict()
        validate_assignments(PLAN, self.scenario)
        validate_assignments({1: [9]}, self.scenario)
        self.assertEqual(self.scenario.to_dict(), before)

    def test_missing_task(self):
        result = self.assert_invalid({1: [1, 2], 2: []}, "not assigned")
        self.assertTrue(any("[3]" in message for message in result.errors))

    def test_all_tasks_missing(self):
        result = self.assert_invalid({}, "not assigned")
        self.assertTrue(any("[1, 2, 3]" in message for message in result.errors))

    def test_duplicate_task_within_one_robot(self):
        self.assert_invalid({1: [1, 1, 2], 2: [3]}, "task 1 is assigned 2 times")

    def test_duplicate_task_across_robots(self):
        result = self.assert_invalid({1: [1, 2], 2: [2, 3]}, "task 2 is assigned 2 times")
        self.assertTrue(any("[1, 2]" in message for message in result.errors))

    def test_task_assigned_three_times(self):
        self.assert_invalid({1: [1, 1, 2, 3], 2: [1]}, "task 1 is assigned 3 times")

    def test_unknown_robot_id(self):
        self.assert_invalid({1: [1, 2], 2: [3], 99: []}, "unknown robot id 99")

    def test_unknown_task_id(self):
        self.assert_invalid({1: [1, 2, 99], 2: [3]}, "unknown task id 99")

    def test_non_mapping_input(self):
        for value in (None, [], [(1, [1])], "plan", 5, {1, 2}):
            with self.subTest(value=value):
                self.assert_invalid(value, "Mapping")

    def test_non_integer_robot_ids(self):
        for key in ("1", 1.0, None, 1.5, (1,)):
            with self.subTest(key=key):
                self.assert_invalid({key: [1, 2, 3]}, "not an integer")

    def test_boolean_robot_id_is_rejected(self):
        self.assert_invalid({True: [1, 2, 3]}, "not an integer")
        self.assert_invalid({False: [1, 2, 3]}, "not an integer")

    def test_non_list_values(self):
        for value in ("123", 5, None, {1, 2, 3}, {1: 1}, (x for x in [1, 2, 3]), 1.5):
            with self.subTest(value=value):
                self.assert_invalid({1: value, 2: []}, "list or tuple")

    def test_non_integer_task_ids(self):
        for task in ("1", 1.0, 2.5, None, [1], (1,)):
            with self.subTest(task=task):
                self.assert_invalid({1: [task, 2, 3], 2: []}, "not an integer")

    def test_boolean_task_ids_are_rejected(self):
        self.assert_invalid({1: [True, 2, 3], 2: []}, "not an integer")
        self.assert_invalid({1: [2, 3], 2: [False]}, "not an integer")

    def test_numpy_integers_are_accepted_and_normalized_to_int(self):
        result = validate_assignments(
            {np.int64(1): [np.int32(1), np.int64(2)], np.int64(2): [np.int64(3)]}, self.scenario
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.normalized, {1: [1, 2], 2: [3]})
        for robot, tasks in result.normalized.items():
            self.assertIs(type(robot), int)
            for task in tasks:
                self.assertIs(type(task), int)

    def test_all_violations_are_reported_together(self):
        result = self.assert_invalid({99: [1, 1], "x": 5, 1: [True]})
        text = " ".join(result.errors)
        for fragment in ("unknown robot id 99", "assigned 2 times", "not an integer", "list or tuple",
                         "not assigned"):
            self.assertIn(fragment, text)

    def test_mapping_subclasses_are_supported(self):
        result = validate_assignments(types.MappingProxyType({1: [1, 2], 2: [3]}), self.scenario)
        self.assertTrue(result.valid)

    def test_misbehaving_mapping_does_not_crash(self):
        class Broken(Mapping):
            def __getitem__(self, key):
                raise KeyError(key)

            def __iter__(self):
                raise RuntimeError("boom")

            def __len__(self):
                return 1

        result = validate_assignments(Broken(), self.scenario)
        self.assertFalse(result.valid)
        self.assertTrue(result.errors)

    def test_infeasible_energy_does_not_make_a_plan_invalid(self):
        scenario = make_scenario(e1=10.0)
        result = validate_assignments(PLAN, scenario)
        self.assertTrue(result.valid)


class TestComputeRemainingEnergy(unittest.TestCase):
    def test_initial_minus_used(self):
        scenario = make_scenario()
        remaining = compute_remaining_energy(scenario, {1: 20.0, 2: 63.0})
        self.assertEqual(remaining, {1: 980.0, 2: 437.0})

    def test_uses_current_energy_not_capacity(self):
        scenario = make_scenario(e1=400.0, capacity1=1000.0)
        remaining = compute_remaining_energy(scenario, {1: 100.0})
        self.assertEqual(remaining[1], 300.0)  # 400 - 100, not 1000 - 100

    def test_missing_robot_usage_is_zero(self):
        scenario = make_scenario()
        self.assertEqual(compute_remaining_energy(scenario, {1: 20.0}), {1: 980.0, 2: 500.0})
        self.assertEqual(compute_remaining_energy(scenario, {2: 63.0}), {1: 1000.0, 2: 437.0})

    def test_empty_usage_returns_initial_energy_for_every_robot(self):
        scenario = make_scenario()
        self.assertEqual(compute_remaining_energy(scenario, {}), {1: 1000.0, 2: 500.0})

    def test_overdrawn_energy_is_negative(self):
        scenario = make_scenario(e1=10.0)
        self.assertEqual(compute_remaining_energy(scenario, {1: 20.0})[1], -10.0)

    def test_values_are_floats_for_every_robot_ordered_by_id(self):
        scenario = make_scenario(e1=1000, e2=500)  # ints in, floats out
        remaining = compute_remaining_energy(scenario, {2: 1})
        self.assertEqual(list(remaining), [1, 2])
        for value in remaining.values():
            self.assertIs(type(value), float)

    def test_numpy_values_are_accepted(self):
        remaining = compute_remaining_energy(make_scenario(), {1: np.float64(20.0), 2: np.int64(3)})
        self.assertEqual(remaining, {1: 980.0, 2: 497.0})

    def test_unknown_robot_in_usage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown robot ids"):
            compute_remaining_energy(make_scenario(), {1: 1.0, 7: 2.0})

    def test_invalid_usage_values_are_rejected(self):
        for value in (NAN, INF, -INF, -1.0):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    compute_remaining_energy(make_scenario(), {1: value})
        for value in ("5", None, True, [1]):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    compute_remaining_energy(make_scenario(), {1: value})

    def test_non_mapping_usage_is_rejected(self):
        with self.assertRaises(TypeError):
            compute_remaining_energy(make_scenario(), [1.0, 2.0])

    def test_scenario_is_not_modified(self):
        scenario = make_scenario()
        before = scenario.to_dict()
        compute_remaining_energy(scenario, {1: 5.0})
        self.assertEqual(scenario.to_dict(), before)


class TestEnergyTolerancesAndFeasibility(unittest.TestCase):
    def test_tolerance_formula(self):
        scenario = make_scenario(e1=1000.0, e2=0.5)
        tolerances = compute_energy_tolerances(scenario)
        self.assertEqual(ENERGY_TOLERANCE_RELATIVE, 1e-9)
        self.assertAlmostEqual(tolerances[1], 1e-6, places=18)   # 1e-9 * max(1, 1000)
        self.assertEqual(tolerances[2], 1e-9)                    # 1e-9 * max(1, 0.5)

    def test_tolerance_uses_current_energy(self):
        scenario = make_scenario(e1=100.0, capacity1=5000.0)
        self.assertAlmostEqual(compute_energy_tolerances(scenario)[1], 1e-7, places=18)

    def test_feasible_when_no_robot_overdraws(self):
        result = compute_energy_feasibility({1: 980.0, 2: 437.0}, 1e-9)
        self.assertIsInstance(result, EnergyFeasibility)
        self.assertTrue(result.feasible)
        self.assertEqual(result.deficit, {1: 0.0, 2: 0.0})

    def test_exactly_zero_remaining_is_feasible(self):
        self.assertTrue(compute_energy_feasibility({1: 0.0}, 1e-9).feasible)

    def test_infeasible_reports_per_robot_deficits_for_every_robot(self):
        result = compute_energy_feasibility({1: -10.0, 2: 5.0, 3: -0.25}, 1e-9)
        self.assertFalse(result.feasible)
        self.assertEqual(result.deficit, {1: 10.0, 2: 0.0, 3: 0.25})

    def test_tuple_unpacking(self):
        feasible, deficit = compute_energy_feasibility({1: -2.0}, 0.0)
        self.assertFalse(feasible)
        self.assertEqual(deficit, {1: 2.0})

    def test_floating_point_noise_within_tolerance_counts_as_zero(self):
        scenario = make_scenario(e1=0.3, e2=500.0)
        remaining = compute_remaining_energy(scenario, {1: 0.1 + 0.2})   # 0.30000000000000004
        self.assertLess(remaining[1], 0.0)                               # genuine float noise
        result = compute_energy_feasibility(remaining, compute_energy_tolerances(scenario))
        self.assertTrue(result.feasible)
        self.assertEqual(result.deficit[1], 0.0)

    def test_tolerance_scales_with_initial_energy(self):
        scenario = make_scenario(e1=1000.0)
        tolerances = compute_energy_tolerances(scenario)           # ~1e-6 for robot 1
        within = compute_energy_feasibility({1: -5e-7, 2: 0.0}, tolerances)
        self.assertTrue(within.feasible)
        beyond = compute_energy_feasibility({1: -2e-6, 2: 0.0}, tolerances)
        self.assertFalse(beyond.feasible)
        self.assertEqual(beyond.deficit[1], 2e-6)                  # the full deficit, not reduced

    def test_deficit_just_beyond_a_small_tolerance_is_reported(self):
        scenario = make_scenario(e1=0.3, e2=500.0)
        result = compute_energy_feasibility({1: -1e-6, 2: 0.0}, compute_energy_tolerances(scenario))
        self.assertFalse(result.feasible)

    def test_uniform_and_per_robot_tolerances(self):
        remaining = {1: -0.5, 2: -0.5}
        self.assertTrue(compute_energy_feasibility(remaining, 1.0).feasible)
        mixed = compute_energy_feasibility(remaining, {1: 1.0, 2: 0.1})
        self.assertFalse(mixed.feasible)
        self.assertEqual(mixed.deficit, {1: 0.0, 2: 0.5})

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            compute_energy_feasibility({1: -1.0}, -1e-9)
        with self.assertRaises(ValueError):
            compute_energy_feasibility({1: -1.0}, NAN)
        with self.assertRaises(ValueError):
            compute_energy_feasibility({1: -1.0}, {2: 1e-9})          # tolerance missing for robot 1
        with self.assertRaises(ValueError):
            compute_energy_feasibility({1: -1.0}, {1: -1.0})
        for energy in (NAN, INF, -INF):
            with self.subTest(energy=energy):
                with self.assertRaises(ValueError):
                    compute_energy_feasibility({1: energy}, 1e-9)
        with self.assertRaises(TypeError):
            compute_energy_feasibility({1: "5"}, 1e-9)
        with self.assertRaises(TypeError):
            compute_energy_feasibility([1.0], 1e-9)
        with self.assertRaises(TypeError):
            compute_energy_feasibility({1: 1.0}, True)

    def test_empty_input_is_feasible(self):
        result = compute_energy_feasibility({}, 1e-9)
        self.assertTrue(result.feasible)
        self.assertEqual(result.deficit, {})


class TestEnergyPenaltyPhi(unittest.TestCase):
    DEFICIT = {1: 10.0, 2: 13.0, 3: 0.0}

    def test_quadratic(self):
        # 0.5 * (100 + 169 + 0) = 134.5
        self.assertEqual(energy_penalty_phi(self.DEFICIT, "quadratic", 0.5), 134.5)

    def test_linear(self):
        # 0.5 * (10 + 13 + 0) = 11.5
        self.assertEqual(energy_penalty_phi(self.DEFICIT, "linear", 0.5), 11.5)

    def test_zero(self):
        self.assertEqual(energy_penalty_phi(self.DEFICIT, "zero", 0.5), 0.0)
        self.assertEqual(energy_penalty_phi(self.DEFICIT, "zero", 0.0), 0.0)

    def test_lambda_scales_linearly(self):
        base = energy_penalty_phi(self.DEFICIT, "quadratic", 1.0)
        self.assertEqual(energy_penalty_phi(self.DEFICIT, "quadratic", 3.0), 3.0 * base)
        self.assertEqual(energy_penalty_phi(self.DEFICIT, "quadratic", 0.0), 0.0)

    def test_no_deficit_gives_zero_penalty_for_every_strategy(self):
        for strategy in PENALTY_STRATEGIES:
            self.assertEqual(energy_penalty_phi({1: 0.0, 2: 0.0}, strategy, 7.0), 0.0)
            self.assertEqual(energy_penalty_phi({}, strategy, 7.0), 0.0)

    def test_quadratic_penalises_large_deficits_more_than_linear(self):
        self.assertGreater(energy_penalty_phi({1: 10.0}, "quadratic", 1.0),
                           energy_penalty_phi({1: 10.0}, "linear", 1.0))
        self.assertLess(energy_penalty_phi({1: 0.5}, "quadratic", 1.0),
                        energy_penalty_phi({1: 0.5}, "linear", 1.0))

    def test_supports_exactly_three_strategies(self):
        self.assertEqual(PENALTY_STRATEGIES, ("quadratic", "linear", "zero"))

    def test_iterable_of_deficits_is_accepted(self):
        self.assertEqual(energy_penalty_phi([10.0, 13.0], "linear", 1.0), 23.0)
        self.assertEqual(energy_penalty_phi((x for x in [3.0, 4.0]), "quadratic", 1.0), 25.0)

    def test_numpy_values_are_accepted(self):
        self.assertEqual(energy_penalty_phi({1: np.float64(3.0)}, "quadratic", np.float64(2.0)), 18.0)

    def test_independent_of_feasibility_computation(self):
        # phi only looks at the deficits it is given.
        self.assertEqual(energy_penalty_phi({1: 2.0}, "quadratic", 1.0), 4.0)

    def test_invalid_strategies(self):
        for strategy in ("Quadratic", "QUADRATIC", "cubic", "", " linear", "none"):
            with self.subTest(strategy=strategy):
                with self.assertRaises(ValueError):
                    energy_penalty_phi({1: 1.0}, strategy, 1.0)
        for strategy in (None, 1, ["linear"], b"linear"):
            with self.subTest(strategy=strategy):
                with self.assertRaises(TypeError):
                    energy_penalty_phi({1: 1.0}, strategy, 1.0)

    def test_invalid_lambda(self):
        for value in (-1.0, NAN, INF, -INF):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    energy_penalty_phi({1: 1.0}, "quadratic", value)
        for value in (None, "1.0", True, [1.0]):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    energy_penalty_phi({1: 1.0}, "quadratic", value)

    def test_lambda_is_validated_even_for_the_zero_strategy(self):
        with self.assertRaises(ValueError):
            energy_penalty_phi({1: 1.0}, "zero", NAN)

    def test_invalid_deficits(self):
        for value in (-1.0, NAN, INF):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    energy_penalty_phi({1: value}, "quadratic", 1.0)
        for value in ("1", None, True):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    energy_penalty_phi({1: value}, "linear", 1.0)
        for bad in (5, 5.0, "12", None):
            with self.subTest(deficit=bad):
                with self.assertRaises(TypeError):
                    energy_penalty_phi(bad, "linear", 1.0)

    def test_non_finite_result_is_rejected(self):
        with self.assertRaises(ValueError):
            energy_penalty_phi({1: 1e200}, "quadratic", 1.0)  # squares to infinity


class TestEvaluateAllocation(unittest.TestCase):
    def setUp(self):
        self.scenario = make_scenario()

    def evaluate(self, scenario=None, plan=None, penalty=QUADRATIC):
        return evaluate_allocation(scenario or self.scenario, plan or copy.deepcopy(PLAN), penalty)

    def test_result_structure_has_no_runtime_or_solver_fields(self):
        self.assertEqual(
            EvaluationResult._fields,
            ("allocation_cost", "base_cost", "energy_penalty_value", "energy_feasible",
             "remaining_energy", "deficit", "task_completion_times", "agent_distances"),
        )
        for forbidden in ("runtime_seconds", "solver_name", "total_cost"):
            self.assertNotIn(forbidden, EvaluationResult._fields)

    def test_hand_computed_feasible_plan(self):
        result = self.evaluate()
        self.assertAlmostEqual(result.task_completion_times[1], 10.5, places=9)
        self.assertAlmostEqual(result.task_completion_times[2], 31.7, places=9)
        self.assertAlmostEqual(result.task_completion_times[3], 42.0, places=9)
        self.assertAlmostEqual(result.agent_distances[1], 17.0, places=9)
        self.assertAlmostEqual(result.agent_distances[2], 60.0, places=9)
        self.assertAlmostEqual(result.base_cost, BASE_COST, places=9)
        self.assertTrue(result.energy_feasible)
        self.assertEqual(result.deficit, {1: 0.0, 2: 0.0})
        self.assertAlmostEqual(result.remaining_energy[1], 980.0, places=9)   # 1000 - 20
        self.assertAlmostEqual(result.remaining_energy[2], 437.0, places=9)   # 500 - 63
        self.assertEqual(result.energy_penalty_value, 0.0)
        self.assertAlmostEqual(result.allocation_cost, BASE_COST, places=9)

    def test_allocation_cost_is_base_plus_penalty(self):
        result = self.evaluate(scenario=make_scenario(e1=10.0))
        self.assertAlmostEqual(result.allocation_cost, result.base_cost + result.energy_penalty_value, places=12)

    def test_infeasible_plan_quadratic_linear_zero(self):
        # Robot 1 has 10 energy but needs 20: deficit 10. Travel and times are unchanged, so the base cost is too.
        scenario = make_scenario(e1=10.0)
        expected = {
            "quadratic": 2.0 * 10.0 ** 2,   # 200
            "linear": 2.0 * 10.0,           # 20
            "zero": 0.0,
        }
        for strategy, penalty in expected.items():
            with self.subTest(strategy=strategy):
                result = self.evaluate(scenario, penalty={"strategy": strategy, "lambda_E": 2.0})
                self.assertAlmostEqual(result.base_cost, BASE_COST, places=9)
                self.assertAlmostEqual(result.deficit[1], 10.0, places=9)
                self.assertEqual(result.deficit[2], 0.0)
                self.assertAlmostEqual(result.energy_penalty_value, penalty, places=9)
                self.assertAlmostEqual(result.allocation_cost, BASE_COST + penalty, places=9)

    def test_two_infeasible_robots(self):
        scenario = make_scenario(e1=10.0, e2=50.0)      # deficits 10 and 63 - 50 = 13
        result = self.evaluate(scenario, penalty={"strategy": "quadratic", "lambda_E": 0.5})
        self.assertAlmostEqual(result.deficit[1], 10.0, places=9)
        self.assertAlmostEqual(result.deficit[2], 13.0, places=9)
        self.assertAlmostEqual(result.energy_penalty_value, 134.5, places=9)   # 0.5 * (100 + 169)
        self.assertAlmostEqual(result.allocation_cost, BASE_COST + 134.5, places=9)

    def test_energy_feasibility_is_independent_of_the_penalty_strategy(self):
        for scenario, expected in ((make_scenario(), True), (make_scenario(e1=10.0), False)):
            results = [
                self.evaluate(scenario, penalty={"strategy": s, "lambda_E": 1.0}) for s in PENALTY_STRATEGIES
            ]
            self.assertEqual({r.energy_feasible for r in results}, {expected})
            self.assertEqual(len({tuple(sorted(r.deficit.items())) for r in results}), 1)
            self.assertEqual(len({r.base_cost for r in results}), 1)
        zero = self.evaluate(make_scenario(e1=10.0), penalty={"strategy": "zero", "lambda_E": 1.0})
        self.assertFalse(zero.energy_feasible)      # infeasible even though phi is zero
        self.assertEqual(zero.energy_penalty_value, 0.0)

    def test_exactly_zero_remaining_energy_is_feasible(self):
        result = self.evaluate(make_scenario(e1=20.0))        # robot 1 uses exactly 20
        self.assertTrue(result.energy_feasible)
        self.assertEqual(result.deficit[1], 0.0)

    def test_infeasible_plans_remain_valid_allocations(self):
        scenario = make_scenario(e1=10.0)
        self.assertTrue(validate_assignments(PLAN, scenario).valid)
        result = self.evaluate(scenario)
        self.assertFalse(result.energy_feasible)
        self.assertGreater(result.allocation_cost, result.base_cost)

    def test_idle_robot(self):
        plan = {1: [1, 2, 3], 2: []}
        result = self.evaluate(plan=plan)
        d_13 = math.hypot(100.0 - 3.0, 160.0 - 16.0)
        eta3 = 31.7 + d_13 / 10.0 + 30.0
        expected = 3.0 * 10.5 + 2.0 * 31.7 + 4.0 * eta3 + 2.0 * (17.0 + d_13)
        self.assertAlmostEqual(result.base_cost, expected, places=9)
        self.assertEqual(result.agent_distances[2], 0.0)
        self.assertEqual(result.remaining_energy[2], 500.0)
        self.assertAlmostEqual(result.remaining_energy[1], 1000.0 - (20.0 + d_13 + 3.0), places=9)

    def test_robot_omitted_from_a_valid_plan_is_handled(self):
        validated = validate_assignments({1: [1, 2, 3]}, self.scenario)
        by_omission = evaluate_allocation(self.scenario, {1: [1, 2, 3]}, QUADRATIC)
        by_empty_list = evaluate_allocation(self.scenario, validated.normalized, QUADRATIC)
        self.assertEqual(by_omission, by_empty_list)

    def test_unrounded_base_cost_and_sanity_agreement_with_the_simulator(self):
        scenario = ScenarioGenerator(seed=2024).generate_scenario(4, 9)
        tasks = [t.task_id for t in scenario.tasks]
        robots = [a.agent_id for a in scenario.agents]
        plan = {r: [] for r in robots}
        for index, task_id in enumerate(tasks):
            plan[robots[index % len(robots)]].append(task_id)

        result = evaluate_allocation(scenario, plan, {"strategy": "zero", "lambda_E": 0.0})

        env = Environment()
        env.load_scenario(scenario)
        sim, _ = env.execute_allocation(copy.deepcopy(plan), energy_violation_penalty_factor=0.0)

        priority = {t.task_id: t.priority for t in scenario.tasks}
        penalty = {a.agent_id: a.usage_penalty for a in scenario.agents}
        manual = math.fsum(
            [priority[t] * sim.task_completion_times[t] for t in sim.task_completion_times]
            + [penalty[r] * sim.agent_distances[r] for r in sim.agent_distances]
        )
        self.assertAlmostEqual(result.base_cost, manual, places=9)         # unrounded reconstruction
        self.assertNotEqual(result.base_cost, round(result.base_cost, 2))   # not a 2-decimal value
        self.assertLessEqual(abs(round(result.base_cost, 2) - sim.total_cost), SANITY_TOLERANCE)
        self.assertLessEqual(abs(result.base_cost - sim.total_cost), 0.005 + 1e-9)
        self.assertEqual(sim.metadata["energy_penalty"], 0.0)

    def test_matches_simulator_rounded_cost_within_documented_tolerance(self):
        env = Environment()
        env.load_scenario(self.scenario)
        sim, _ = env.execute_allocation(copy.deepcopy(PLAN), energy_violation_penalty_factor=0.0)
        result = self.evaluate()
        self.assertEqual(SANITY_TOLERANCE, 0.011)
        self.assertLessEqual(abs(round(result.base_cost, 2) - sim.total_cost), SANITY_TOLERANCE)
        self.assertEqual(sim.total_cost, 356.9)

    def test_simulator_is_called_with_penalty_factor_zero(self):
        original = Environment.execute_allocation
        returned = []

        def spy(self_, assignments, **kwargs):
            outcome = original(self_, assignments, **kwargs)
            returned.append(outcome[0])
            return outcome

        with mock.patch.object(Environment, "execute_allocation", autospec=True, side_effect=spy) as patched:
            self.evaluate(make_scenario(e1=10.0))                      # an energy-infeasible plan
        self.assertEqual(patched.call_count, 1)
        self.assertEqual(patched.call_args.kwargs, {"energy_violation_penalty_factor": 0.0})
        self.assertEqual(returned[0].metadata["energy_penalty"], 0.0)  # simulator penalty really off
        self.assertEqual(returned[0].total_cost, round(BASE_COST, 2))

    def test_simulator_penalty_is_never_combined_with_ours(self):
        result = self.evaluate(make_scenario(e1=10.0), penalty={"strategy": "linear", "lambda_E": 3.0})
        # 30 from our linear penalty; the simulator's default factor of 1000 would add 10000.
        self.assertAlmostEqual(result.allocation_cost, BASE_COST + 30.0, places=9)

    def test_fresh_environment_per_evaluation(self):
        environments = []
        original_init = Environment.__init__

        def spy_init(self_, *args, **kwargs):
            environments.append(self_)
            original_init(self_, *args, **kwargs)

        with mock.patch.object(Environment, "__init__", spy_init):
            self.evaluate()
            self.evaluate()
        self.assertEqual(len(environments), 2)
        self.assertIsNot(environments[0], environments[1])

    def test_scenario_is_not_mutated(self):
        scenario = make_scenario(e1=10.0)
        before = scenario.to_dict()
        task_state = [(t.completed, t.completion_time, t.assigned_agent_id) for t in scenario.tasks]
        agent_state = [(a.current_energy, a.position) for a in scenario.agents]
        self.evaluate(scenario)
        self.evaluate(scenario)
        self.assertEqual(scenario.to_dict(), before)
        self.assertEqual([(t.completed, t.completion_time, t.assigned_agent_id) for t in scenario.tasks], task_state)
        self.assertEqual([(a.current_energy, a.position) for a in scenario.agents], agent_state)

    def test_assignment_dictionaries_are_not_aliased(self):
        caller_plan = {1: [1, 2], 2: [3]}
        snapshot = copy.deepcopy(caller_plan)
        seen = []
        original = Environment.execute_allocation

        def spy(self_, assignments, **kwargs):
            seen.append(assignments)
            return original(self_, assignments, **kwargs)

        with mock.patch.object(Environment, "execute_allocation", autospec=True, side_effect=spy):
            evaluate_allocation(self.scenario, caller_plan, QUADRATIC)
        self.assertEqual(len(seen), 1)
        self.assertIsNot(seen[0], caller_plan)
        for robot in caller_plan:
            self.assertIsNot(seen[0][robot], caller_plan[robot])
        self.assertEqual(caller_plan, snapshot)

    def test_same_normalized_plan_can_be_reused(self):
        normalized = validate_assignments(PLAN, self.scenario).normalized
        snapshot = copy.deepcopy(normalized)
        first = evaluate_allocation(self.scenario, normalized, QUADRATIC)
        second = evaluate_allocation(self.scenario, normalized, QUADRATIC)
        self.assertEqual(first, second)
        self.assertEqual(normalized, snapshot)

    def test_returned_dictionaries_are_independent_copies(self):
        first = self.evaluate()
        first.task_completion_times[1] = -1.0
        first.agent_distances[1] = -1.0
        first.remaining_energy[1] = -1.0
        second = self.evaluate()
        self.assertAlmostEqual(second.task_completion_times[1], 10.5, places=9)
        self.assertAlmostEqual(second.agent_distances[1], 17.0, places=9)
        self.assertAlmostEqual(second.remaining_energy[1], 980.0, places=9)

    def test_deterministic(self):
        self.assertEqual(self.evaluate(make_scenario(e1=10.0)), self.evaluate(make_scenario(e1=10.0)))

    def test_penalty_config_formats(self):
        as_tuple = self.evaluate(make_scenario(e1=10.0), penalty=EnergyPenalty("quadratic", 1.0))
        as_mapping = self.evaluate(make_scenario(e1=10.0), penalty={"strategy": "quadratic", "lambda_E": 1.0})
        as_config = self.evaluate(
            make_scenario(e1=10.0),
            penalty={"status": "provisional_pending_team_confirmation", "strategy": "quadratic", "lambda_E": 1.0},
        )
        self.assertEqual(as_tuple, as_mapping)
        self.assertEqual(as_tuple, as_config)
        self.assertAlmostEqual(as_tuple.energy_penalty_value, 100.0, places=9)

    def test_invalid_penalty_configuration_is_rejected_before_the_simulator_runs(self):
        bad = [
            {"strategy": "cubic", "lambda_E": 1.0},
            {"strategy": "quadratic", "lambda_E": NAN},
            {"strategy": "quadratic", "lambda_E": -1.0},
            {"strategy": "quadratic"},
            {"lambda_E": 1.0},
        ]
        with mock.patch.object(Environment, "execute_allocation", side_effect=AssertionError("simulator called")):
            for penalty in bad:
                with self.subTest(penalty=penalty):
                    with self.assertRaises(ValueError):
                        evaluate_allocation(self.scenario, copy.deepcopy(PLAN), penalty)
            for penalty in (None, "quadratic", 1.0, ["quadratic", 1.0]):
                with self.subTest(penalty=penalty):
                    with self.assertRaises(TypeError):
                        evaluate_allocation(self.scenario, copy.deepcopy(PLAN), penalty)

    def test_invalid_allocations_are_rejected_without_calling_the_simulator(self):
        invalid_plans = [
            {1: [1, 2], 2: []},          # missing task
            {1: [1, 2], 2: [2, 3]},      # duplicate across robots
            {1: [1, 1, 2, 3]},           # duplicate within a robot
            {1: [1, 2, 3], 99: []},      # unknown robot
            {1: [1, 2, 3, 99]},          # unknown task
            {1: [True, 2, 3]},           # boolean id
            [1, 2, 3],                   # not a mapping
        ]
        with mock.patch.object(Environment, "execute_allocation", side_effect=AssertionError("simulator called")):
            for plan in invalid_plans:
                with self.subTest(plan=plan):
                    with self.assertRaisesRegex(ValueError, "invalid allocation"):
                        evaluate_allocation(self.scenario, plan, QUADRATIC)

    def test_inconsistent_simulator_total_raises_evaluation_error(self):
        original = Environment.execute_allocation

        def wrong_total(self_, assignments, **kwargs):
            result, traces = original(self_, assignments, **kwargs)
            result.total_cost += 5.0
            return result, traces

        with mock.patch.object(Environment, "execute_allocation", wrong_total):
            with self.assertRaisesRegex(EvaluationError, "disagrees"):
                self.evaluate()

    def test_total_just_inside_the_sanity_tolerance_is_accepted(self):
        original = Environment.execute_allocation

        def slightly_off(self_, assignments, **kwargs):
            result, traces = original(self_, assignments, **kwargs)
            result.total_cost += 0.01
            return result, traces

        with mock.patch.object(Environment, "execute_allocation", slightly_off):
            self.assertAlmostEqual(self.evaluate().base_cost, BASE_COST, places=9)

    def test_missing_completion_time_raises_evaluation_error(self):
        original = Environment.execute_allocation

        def drop_task(self_, assignments, **kwargs):
            result, traces = original(self_, assignments, **kwargs)
            del result.task_completion_times[3]
            return result, traces

        with mock.patch.object(Environment, "execute_allocation", drop_task):
            with self.assertRaises(EvaluationError):
                self.evaluate()

    def test_missing_distance_raises_evaluation_error(self):
        original = Environment.execute_allocation

        def drop_robot(self_, assignments, **kwargs):
            result, traces = original(self_, assignments, **kwargs)
            del result.agent_distances[2]
            return result, traces

        with mock.patch.object(Environment, "execute_allocation", drop_robot):
            with self.assertRaises(EvaluationError):
                self.evaluate()

    def test_non_finite_simulator_values_raise_evaluation_error(self):
        original = Environment.execute_allocation

        def make_nan(self_, assignments, **kwargs):
            result, traces = original(self_, assignments, **kwargs)
            result.agent_distances[1] = NAN
            return result, traces

        with mock.patch.object(Environment, "execute_allocation", make_nan):
            with self.assertRaises(EvaluationError):
                self.evaluate()

    def test_simulator_runtime_and_solver_name_are_ignored(self):
        original = Environment.execute_allocation

        def garbage_metadata(self_, assignments, **kwargs):
            result, traces = original(self_, assignments, **kwargs)
            result.runtime_seconds = 123456.0
            result.solver_name = "SomethingElse"
            return result, traces

        baseline = self.evaluate()                       # computed without the patch
        with mock.patch.object(Environment, "execute_allocation", garbage_metadata):
            self.assertEqual(self.evaluate(), baseline)


class TestModuleBoundaries(unittest.TestCase):
    def test_no_reward_or_logging_functionality_is_defined_here(self):
        import src.analytics.evaluation as module
        for name in ("standardize_robot_rewards", "compute_robot_reward_gini", "append_benchmark_results"):
            self.assertFalse(hasattr(module, name), name)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for analytics metrics (currently: robot-reward Gini index)."""

import math
import unittest

from src.analytics.metrics import GiniResult, compute_robot_reward_gini


def _rows(*rewards):
    """Build (robot_id, reward) pairs with ids 0..n-1."""
    return list(enumerate(rewards))


class TestRobotRewardGini(unittest.TestCase):
    def test_equal_positive_rewards(self):
        result = compute_robot_reward_gini(_rows(5.0, 5.0, 5.0, 5.0))
        self.assertEqual(result.gini_index, 0.0)
        self.assertFalse(result.invalid_reward_flag)

    def test_unequal_positive_rewards_hand_calculated(self):
        # x = [1, 2, 3, 4], R = 4, sum = 10.
        # Ordered pair differences: pairs (1,2)=1,(1,3)=2,(1,4)=3,(2,3)=1,
        # (2,4)=2,(3,4)=1 -> unordered sum 10 -> double sum 20.
        # G = 20 / (2 * 4 * 10) = 0.25.
        result = compute_robot_reward_gini(_rows(1.0, 2.0, 3.0, 4.0))
        self.assertAlmostEqual(result.gini_index, 0.25, places=12)
        self.assertFalse(result.invalid_reward_flag)

    def test_two_robots_one_gets_everything(self):
        # x = [0, 10]: double sum = 20, G = 20 / (2*2*10) = 0.5 = 1 - 1/R.
        result = compute_robot_reward_gini(_rows(0.0, 10.0))
        self.assertAlmostEqual(result.gini_index, 0.5, places=12)
        self.assertFalse(result.invalid_reward_flag)

    def test_input_order_does_not_matter(self):
        a = compute_robot_reward_gini([(1, 4.0), (2, 1.0), (3, 3.0), (4, 2.0)])
        b = compute_robot_reward_gini(_rows(1.0, 2.0, 3.0, 4.0))
        self.assertAlmostEqual(a.gini_index, b.gini_index, places=12)

    def test_string_robot_ids_and_generator_input(self):
        pairs = (("r1", 1.0), ("r2", 2.0), ("r3", 3.0), ("r4", 4.0))
        result = compute_robot_reward_gini(p for p in pairs)
        self.assertAlmostEqual(result.gini_index, 0.25, places=12)

    def test_all_zero_rewards(self):
        result = compute_robot_reward_gini(_rows(0.0, 0.0, 0.0))
        self.assertEqual(result.gini_index, 0.0)
        self.assertFalse(result.invalid_reward_flag)

    def test_single_robot_positive_reward(self):
        result = compute_robot_reward_gini([("r1", 7.5)])
        self.assertEqual(result.gini_index, 0.0)
        self.assertFalse(result.invalid_reward_flag)

    def test_single_robot_zero_reward(self):
        result = compute_robot_reward_gini([("r1", 0.0)])
        self.assertEqual(result.gini_index, 0.0)
        self.assertFalse(result.invalid_reward_flag)

    def test_single_robot_negative_reward_is_invalid(self):
        # Negative rewards are invalid even for a single robot.
        result = compute_robot_reward_gini([("r1", -1.0)])
        self.assertTrue(math.isnan(result.gini_index))
        self.assertTrue(result.invalid_reward_flag)

    def test_negative_reward_is_invalid_and_not_shifted(self):
        result = compute_robot_reward_gini(_rows(-1.0, 2.0, 3.0))
        self.assertTrue(math.isnan(result.gini_index))
        self.assertTrue(result.invalid_reward_flag)

    def test_negative_reward_with_zero_total_is_invalid(self):
        result = compute_robot_reward_gini(_rows(-2.0, 2.0))
        self.assertTrue(math.isnan(result.gini_index))
        self.assertTrue(result.invalid_reward_flag)

    def test_input_is_not_modified(self):
        data = [(0, -1.0), (1, 2.0)]
        compute_robot_reward_gini(data)
        self.assertEqual(data, [(0, -1.0), (1, 2.0)])

    def test_none_rewards_are_missing(self):
        result = compute_robot_reward_gini(None)
        self.assertTrue(math.isnan(result.gini_index))
        self.assertFalse(result.invalid_reward_flag)

    def test_empty_rewards_are_missing(self):
        result = compute_robot_reward_gini([])
        self.assertTrue(math.isnan(result.gini_index))
        self.assertFalse(result.invalid_reward_flag)

    def test_nan_reward_is_unavailable(self):
        result = compute_robot_reward_gini(_rows(1.0, float("nan"), 3.0))
        self.assertTrue(math.isnan(result.gini_index))
        self.assertFalse(result.invalid_reward_flag)

    def test_negative_reward_takes_precedence_over_nan_reward(self):
        result = compute_robot_reward_gini(_rows(-1.0, float("nan")))
        self.assertTrue(math.isnan(result.gini_index))
        self.assertTrue(result.invalid_reward_flag)

    def test_duplicate_robot_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            compute_robot_reward_gini([(1, 1.0), (2, 2.0), (1, 3.0)])

    def test_duplicate_robot_ids_with_identical_rewards_are_rejected(self):
        # Must not be silently deduplicated.
        with self.assertRaises(ValueError):
            compute_robot_reward_gini([(1, 2.0), (1, 2.0)])

    def test_duplicate_robot_ids_rejected_even_with_negative_reward(self):
        with self.assertRaises(ValueError):
            compute_robot_reward_gini([(1, -1.0), (1, 2.0)])

    def test_malformed_entries_are_rejected(self):
        with self.assertRaises(ValueError):
            compute_robot_reward_gini([1.0, 2.0])
        with self.assertRaises(ValueError):
            compute_robot_reward_gini([(1, "a"), (2, 2.0)])
        with self.assertRaises(ValueError):
            compute_robot_reward_gini([(1, float("inf")), (2, 2.0)])

    def test_return_types(self):
        for rewards in (
            _rows(1.0, 2.0, 3.0),   # normal
            _rows(0.0, 0.0),        # all zero
            [("r1", 4.0)],          # single robot
            _rows(-1.0, 1.0),       # invalid
            [],                     # missing
        ):
            result = compute_robot_reward_gini(rewards)
            self.assertIsInstance(result, GiniResult)
            self.assertIs(type(result.gini_index), float)
            self.assertIs(type(result.invalid_reward_flag), bool)

    def test_result_unpacks_as_tuple(self):
        gini, invalid = compute_robot_reward_gini(_rows(1.0, 3.0))
        # x = [1, 3]: double sum = 4, G = 4 / (2 * 2 * 4) = 0.25.
        self.assertAlmostEqual(gini, 0.25, places=12)
        self.assertFalse(invalid)

    def test_gini_within_bounds_for_nonnegative_rewards(self):
        rewards = _rows(0.0, 0.0, 0.0, 12.0)
        result = compute_robot_reward_gini(rewards)
        # Upper bound for non-negative rewards is 1 - 1/R = 0.75.
        self.assertAlmostEqual(result.gini_index, 0.75, places=12)


if __name__ == "__main__":
    unittest.main()

"""Headless tests for exact saved-plan animation playback."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.core.agent import Agent
from src.core.task import Task
from src.core.types import Position
from src.simulation.generator import ScenarioData
from src.visualization.animation import (
    PlaybackBundleError,
    PlaybackEngine,
    interpolate,
    load_bundle,
    load_playback_bundles,
    run_pygame,
    world_to_screen,
    write_bundle,
)


def scenario() -> ScenarioData:
    return ScenarioData(
        100.0, 100.0, 7,
        [Agent(1, Position(0, 0), 10.0, 1000.0, 1000.0, 1.0),
         Agent(2, Position(0, 100), 10.0, 1000.0, 1000.0, 1.0)],
        [Task(1, Position(100, 0), 5.0, 1.0), Task(2, Position(100, 100), 5.0, 1.0),
         Task(3, Position(50, 0), 5.0, 1.0)],
    )


class TestAnimationBundle(unittest.TestCase):
    def setUp(self):
        self.scenario = scenario()
        self.assignments = {1: [3, 1], 2: [2]}

    def test_bundle_round_trip_preserves_assignment_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            write_bundle(path, self.scenario, {"Greedy": self.assignments}, {"Greedy": {"gini_index": 0.2}})
            loaded, allocations, metrics = load_bundle(path)
        self.assertEqual(loaded.to_dict(), self.scenario.to_dict())
        self.assertEqual(allocations["Greedy"], self.assignments)
        self.assertEqual(allocations["Greedy"][1], [3, 1])
        self.assertEqual(metrics["Greedy"]["gini_index"], 0.2)

    def test_invalid_or_missing_assignment_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid.json"
            for assignments in ({1: [1], 2: [2]}, {}, {1: [1, 1], 2: [2, 3]}):
                with self.subTest(assignments=assignments), self.assertRaises(PlaybackBundleError):
                    write_bundle(output, self.scenario, {"Greedy": assignments})

    def test_load_rejects_missing_empty_and_incomplete_allocations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            path.write_text('{"version": 1, "scenario": {}, "allocations": {}}', encoding="utf-8")
            with self.assertRaises(PlaybackBundleError):
                load_bundle(path)
            path.write_text(json.dumps({"version": 1, "scenario": self.scenario.to_dict(),
                                        "allocations": {"Greedy": {}}}), encoding="utf-8")
            with self.assertRaises(PlaybackBundleError):
                load_bundle(path)

    def test_world_mapping_and_interpolation(self):
        self.assertEqual(world_to_screen(Position(0, 0), 100, 100, 200, 200, 20), (20, 180))
        self.assertEqual(world_to_screen(Position(100, 100), 100, 100, 200, 200, 20), (180, 20))
        self.assertEqual(interpolate(Position(0, 0), Position(100, 50), 0.25), Position(25, 12.5))
        self.assertEqual(interpolate(Position(0, 0), Position(100, 50), 2.0), Position(100, 50))

    def test_robots_move_simultaneously_and_preserve_routes(self):
        engine = PlaybackEngine(self.scenario, self.assignments)
        engine.paused = False
        engine.speed = 1.0
        engine.update(2.0)
        self.assertEqual(engine.states[1].phase, "travelling")
        self.assertEqual(engine.states[2].phase, "travelling")
        self.assertEqual(engine.states[1].position, Position(20, 0))
        self.assertEqual(engine.states[2].position, Position(20, 100))
        self.assertEqual(engine.assignments[1], [3, 1])

    def test_multiple_saved_scenarios_load_in_cli_order(self):
        other = scenario()
        other.seed = 8
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "one.json", Path(directory) / "two.json"
            write_bundle(first, self.scenario, {"Greedy": self.assignments})
            write_bundle(second, other, {"Greedy": self.assignments, "GA": self.assignments})
            choices = load_playback_bundles([first, second])
        self.assertEqual([choice[0].name for choice in choices], ["one.json", "two.json"])
        self.assertEqual([choice[1].seed for choice in choices], [7, 8])
        self.assertIn("GA", choices[1][2])

    def test_requested_initial_algorithm_must_exist_in_first_bundle(self):
        # Validation is kept outside the Pygame event loop so a bad CLI selection
        # is rejected instead of silently showing a different saved allocation.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.json"
            write_bundle(path, self.scenario, {"Greedy": self.assignments})
            with self.assertRaisesRegex(PlaybackBundleError, "not in"):
                run_pygame(path, "GA")

    def test_playback_engine_does_not_call_any_allocator(self):
        engine = PlaybackEngine(self.scenario, self.assignments)
        engine.paused = False
        with mock.patch("src.analytics.allocation_adapters.greedy_adapter", side_effect=AssertionError("must not run")):
            engine.update(1.0)
        self.assertEqual(engine.assignments, self.assignments)

    def test_task_changes_state_only_after_execution_finishes(self):
        engine = PlaybackEngine(self.scenario, self.assignments)
        engine.paused = False
        engine.speed = 1.0
        engine.update(5.0)  # robot 1 reaches task 3 but has not completed its 5s execution
        self.assertEqual(engine.states[1].phase, "executing")
        self.assertNotIn(3, engine.completed_tasks)
        energy_before = engine.states[1].energy
        engine.update(5.0)
        self.assertIn(3, engine.completed_tasks)
        self.assertEqual(engine.states[1].route_index, 1)
        self.assertLess(engine.states[1].energy, energy_before)
        self.assertAlmostEqual(engine.states[1].energy, 949.5)


if __name__ == "__main__":
    unittest.main()

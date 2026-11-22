"""Saved-plan playback model for the 2D multi-robot allocation demonstration.

Playback consumes a JSON bundle containing a serialized ``ScenarioData`` and
the exact ordered assignment lists returned by an allocator.  It never invokes
an allocator or changes an assignment.  Pygame is deliberately imported only
by :func:`run_pygame`, leaving the model usable in tests and headless tools.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from src.analytics.evaluation import validate_assignments
from src.core.types import Position
from src.simulation.generator import ScenarioData
from src.simulation.kinematics import BatteryDecayModel, KinematicsModel

BUNDLE_VERSION = 1
PADDING = 32
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e87ba4")


class PlaybackBundleError(ValueError):
    """A saved playback bundle is missing or has an invalid allocation plan."""


def world_to_screen(position: Position, map_width: float, map_height: float,
                    screen_width: int, screen_height: int, padding: int = PADDING) -> Tuple[int, int]:
    """Map a world position to a padded top-down screen point (world y is inverted)."""
    usable_width = screen_width - 2 * padding
    usable_height = screen_height - 2 * padding
    if map_width <= 0 or map_height <= 0 or usable_width <= 0 or usable_height <= 0:
        raise ValueError("map and screen dimensions must leave a positive drawable area")
    return (
        round(padding + position.x / map_width * usable_width),
        round(screen_height - padding - position.y / map_height * usable_height),
    )


def interpolate(start: Position, end: Position, alpha: float) -> Position:
    """Straight-line visual interpolation clamped to the current travel leg."""
    alpha = min(1.0, max(0.0, float(alpha)))
    return Position(start.x + alpha * (end.x - start.x), start.y + alpha * (end.y - start.y))


def _json_assignments(assignments: Mapping[int, List[int]]) -> Dict[str, List[int]]:
    return {str(robot_id): list(tasks) for robot_id, tasks in assignments.items()}


def _native_assignments(raw: Any) -> Dict[int, List[int]]:
    if not isinstance(raw, Mapping):
        raise PlaybackBundleError("bundle assignments must be a mapping of robot id to ordered task ids")
    parsed: Dict[int, List[int]] = {}
    for robot_id, tasks in raw.items():
        try:
            numeric_id = int(robot_id)
        except (TypeError, ValueError):
            raise PlaybackBundleError(f"invalid robot id in bundle: {robot_id!r}") from None
        if not isinstance(tasks, list):
            raise PlaybackBundleError(f"tasks for robot {robot_id!r} must be a list")
        parsed[numeric_id] = list(tasks)
    return parsed


def write_bundle(path: Path, scenario: ScenarioData, allocations: Mapping[str, Mapping[int, List[int]]],
                 metrics: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Path:
    """Persist exact allocation outputs for later playback without rerunning solvers."""
    if not allocations:
        raise PlaybackBundleError("at least one algorithm allocation is required")
    encoded: Dict[str, Dict[str, List[int]]] = {}
    for algorithm, assignments in allocations.items():
        validated = validate_assignments(assignments, scenario)
        if not validated.valid:
            raise PlaybackBundleError(f"invalid assignments for {algorithm}: {'; '.join(validated.errors)}")
        encoded[str(algorithm)] = _json_assignments(validated.normalized)
    document = {
        "version": BUNDLE_VERSION,
        "scenario": scenario.to_dict(),
        "allocations": encoded,
        "metrics": {str(k): dict(v) for k, v in (metrics or {}).items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def load_bundle(path: Path) -> Tuple[ScenarioData, Dict[str, Dict[int, List[int]]], Dict[str, Dict[str, Any]]]:
    """Load a bundle and validate every saved allocation against its scenario."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlaybackBundleError(f"cannot read playback bundle {path}: {exc}") from None
    if not isinstance(document, Mapping) or document.get("version") != BUNDLE_VERSION:
        raise PlaybackBundleError(f"unsupported or missing bundle version in {path}")
    try:
        scenario = ScenarioData.from_dict(document["scenario"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PlaybackBundleError(f"invalid scenario in bundle: {exc}") from None
    raw_allocations = document.get("allocations")
    if not isinstance(raw_allocations, Mapping) or not raw_allocations:
        raise PlaybackBundleError("bundle needs at least one saved allocation")
    allocations: Dict[str, Dict[int, List[int]]] = {}
    for algorithm, raw in raw_allocations.items():
        assignments = _native_assignments(raw)
        validated = validate_assignments(assignments, scenario)
        if not validated.valid:
            raise PlaybackBundleError(f"invalid assignments for {algorithm}: {'; '.join(validated.errors)}")
        allocations[str(algorithm)] = validated.normalized
    raw_metrics = document.get("metrics", {})
    if not isinstance(raw_metrics, Mapping):
        raise PlaybackBundleError("bundle metrics must be a mapping when present")
    return scenario, allocations, {str(k): dict(v) for k, v in raw_metrics.items() if isinstance(v, Mapping)}


def load_playback_bundles(paths: List[Path]) -> List[Tuple[Path, ScenarioData, Dict[str, Dict[int, List[int]]], Dict[str, Dict[str, Any]]]]:
    """Load scenario choices in CLI order; every bundle remains independently validated."""
    if not paths:
        raise PlaybackBundleError("at least one saved bundle is required")
    return [(path, *load_bundle(path)) for path in paths]


@dataclass
class RobotState:
    robot_id: int
    position: Position
    energy: float
    route_index: int = 0
    phase: str = "idle"
    leg_start: Optional[Position] = None
    leg_end: Optional[Position] = None
    travel_total: float = 0.0
    travel_elapsed: float = 0.0
    travel_velocity: float = 0.0
    execution_remaining: float = 0.0


class PlaybackEngine:
    """Concurrent visual execution of one immutable ordered allocation plan."""

    def __init__(self, scenario: ScenarioData, assignments: Mapping[int, List[int]]):
        validated = validate_assignments(assignments, scenario)
        if not validated.valid:
            raise PlaybackBundleError("invalid playback assignments: " + "; ".join(validated.errors))
        self.scenario = copy.deepcopy(scenario)
        self.assignments = copy.deepcopy(validated.normalized)
        self.agents = {agent.agent_id: agent for agent in self.scenario.agents}
        self.tasks = {task.task_id: task for task in self.scenario.tasks}
        self.battery = BatteryDecayModel()
        self.states = {robot_id: RobotState(robot_id, agent.position, agent.current_energy)
                       for robot_id, agent in self.agents.items()}
        self.completed_tasks: set[int] = set()
        self.simulation_time = 0.0
        self.paused = True
        self.speed = 20.0

    @property
    def done(self) -> bool:
        return len(self.completed_tasks) == len(self.tasks)

    def restart(self) -> None:
        self.__init__(self.scenario, self.assignments)

    def _start_next(self, state: RobotState) -> None:
        route = self.assignments[state.robot_id]
        if state.route_index >= len(route):
            state.phase = "done"
            return
        task = self.tasks[route[state.route_index]]
        agent = self.agents[state.robot_id]
        state.leg_start, state.leg_end = state.position, task.location
        state.travel_velocity = KinematicsModel.compute_effective_velocity(
            agent, state.energy / agent.energy_capacity
        )
        state.travel_total = KinematicsModel.compute_travel_time(
            state.position, task.location, state.travel_velocity
        )
        state.travel_elapsed = 0.0
        state.phase = "travelling" if state.travel_total > 0 else "executing"
        if state.phase == "executing":
            state.execution_remaining = task.execution_time

    def _advance_robot(self, state: RobotState, elapsed: float) -> None:
        remaining = elapsed
        while remaining > 0 and state.phase != "done":
            if state.phase == "idle":
                self._start_next(state)
                continue
            if state.phase == "travelling":
                take = min(remaining, state.travel_total - state.travel_elapsed)
                state.travel_elapsed += take
                remaining -= take
                state.position = interpolate(state.leg_start, state.leg_end, state.travel_elapsed / state.travel_total)
                if math.isclose(state.travel_elapsed, state.travel_total) or state.travel_elapsed >= state.travel_total:
                    task = self.tasks[self.assignments[state.robot_id][state.route_index]]
                    agent = self.agents[state.robot_id]
                    state.position = task.location
                    state.energy = max(0.0, state.energy - self.battery.compute_motion_energy(
                        state.leg_start.distance_to(task.location), state.travel_velocity
                    ))
                    state.execution_remaining = task.execution_time
                    state.phase = "executing"
                continue
            if state.phase == "executing":
                take = min(remaining, state.execution_remaining)
                state.execution_remaining -= take
                state.energy = max(0.0, state.energy - self.battery.compute_execution_energy(take))
                remaining -= take
                if state.execution_remaining <= 1e-9:
                    task_id = self.assignments[state.robot_id][state.route_index]
                    self.completed_tasks.add(task_id)
                    state.route_index += 1
                    state.phase = "idle"

    def update(self, wall_seconds: float) -> None:
        """Advance all robots concurrently; ``wall_seconds`` is scaled for demonstration only."""
        if self.paused or self.done:
            return
        elapsed = max(0.0, float(wall_seconds)) * self.speed
        for state in self.states.values():
            self._advance_robot(state, elapsed)
        self.simulation_time += elapsed


def run_pygame(bundle_path: Path | List[Path], initial_algorithm: Optional[str] = None) -> None:
    """Open the minimal interactive player for one or more saved scenario bundles."""
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("Pygame is required for playback; install requirements.txt first.") from exc
    bundle_paths = [bundle_path] if isinstance(bundle_path, Path) else list(bundle_path)
    bundles = load_playback_bundles(bundle_paths)
    if initial_algorithm is not None and initial_algorithm not in bundles[0][2]:
        raise PlaybackBundleError(
            f"algorithm {initial_algorithm!r} is not in {bundles[0][0]}"
        )
    scenario_index = 0

    def select_bundle(index: int, preferred_algorithm: Optional[str] = None):
        path, selected_scenario, selected_allocations, selected_metrics = bundles[index]
        selected_algorithms = list(selected_allocations)
        if preferred_algorithm in selected_allocations:
            algorithm_index = selected_algorithms.index(preferred_algorithm)
        elif initial_algorithm in selected_allocations:
            algorithm_index = selected_algorithms.index(initial_algorithm)
        else:
            algorithm_index = 0
        return path, selected_scenario, selected_allocations, selected_metrics, selected_algorithms, algorithm_index

    current_path, scenario, allocations, metrics, algorithms, selected = select_bundle(scenario_index)
    pygame.init()
    screen = pygame.display.set_mode((1100, 740))
    pygame.display.set_caption("FAI Multi-Robot Task Allocation Playback")
    clock, font, small = pygame.time.Clock(), pygame.font.Font(None, 24), pygame.font.Font(None, 18)
    engine = PlaybackEngine(scenario, allocations[algorithms[selected]])
    running = True
    while running:
        elapsed = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    engine.paused = not engine.paused
                elif event.key == pygame.K_r:
                    engine = PlaybackEngine(scenario, allocations[algorithms[selected]])
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    selected = (selected + (1 if event.key == pygame.K_RIGHT else -1)) % len(algorithms)
                    engine = PlaybackEngine(scenario, allocations[algorithms[selected]])
                elif event.key in (pygame.K_UP, pygame.K_DOWN):
                    previous_algorithm = algorithms[selected]
                    scenario_index = (scenario_index + (1 if event.key == pygame.K_DOWN else -1)) % len(bundles)
                    current_path, scenario, allocations, metrics, algorithms, selected = select_bundle(
                        scenario_index, previous_algorithm
                    )
                    engine = PlaybackEngine(scenario, allocations[algorithms[selected]])
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    engine.speed = max(1.0, engine.speed / 2)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    engine.speed = min(320.0, engine.speed * 2)
        engine.update(elapsed)
        screen.fill((250, 251, 252))
        map_rect = (0, 0, 760, 740)
        pygame.draw.rect(screen, (235, 237, 240), map_rect)
        for robot_id, route in allocations[algorithms[selected]].items():
            color = pygame.Color(PALETTE[(robot_id - 1) % len(PALETTE)])
            points = [world_to_screen(engine.states[robot_id].position, scenario.map_width, scenario.map_height, 760, 740)]
            start = engine.states[robot_id].position
            for task_id in route[engine.states[robot_id].route_index:]:
                target = engine.tasks[task_id].location
                points.append(world_to_screen(target, scenario.map_width, scenario.map_height, 760, 740))
                start = target
            if len(points) > 1:
                pygame.draw.lines(screen, color, False, points, 2)
        for task_id, task in engine.tasks.items():
            point = world_to_screen(task.location, scenario.map_width, scenario.map_height, 760, 740)
            owner = next(robot_id for robot_id, route in allocations[algorithms[selected]].items() if task_id in route)
            color = pygame.Color(PALETTE[(owner - 1) % len(PALETTE)])
            if task_id in engine.completed_tasks:
                color = pygame.Color(100, 170, 110)
            pygame.draw.circle(screen, color, point, 7)
            screen.blit(small.render(f"T{task_id}", True, (25, 25, 28)), (point[0] + 8, point[1] - 9))
        for robot_id, state in engine.states.items():
            point = world_to_screen(state.position, scenario.map_width, scenario.map_height, 760, 740)
            color = pygame.Color(PALETTE[(robot_id - 1) % len(PALETTE)])
            pygame.draw.circle(screen, color, point, 10)
            screen.blit(small.render(f"R{robot_id}", True, (15, 15, 18)), (point[0] + 11, point[1] - 9))
        panel_x = 780
        lines = [
            f"Algorithm: {algorithms[selected]}", f"State: {'paused' if engine.paused else 'playing'}",
            f"Scenario: {current_path.stem} (seed {scenario.seed})",
            f"Simulation time: {engine.simulation_time:.1f} s", f"Speed: {engine.speed:g}x",
            f"Tasks: {len(engine.completed_tasks)}/{len(engine.tasks)} complete", "",
            "Space play/pause | R restart", "Left/Right algorithm | Up/Down scenario | +/- speed", "",
        ]
        for robot_id, state in engine.states.items():
            lines.append(f"R{robot_id}: {state.phase}, energy {state.energy:.1f}")
        lines.extend(["", "Metrics (saved):"])
        for key in ("allocation_cost", "allocation_runtime_seconds", "reward_runtime_seconds", "gini_index"):
            lines.append(f"{key}: {metrics.get(algorithms[selected], {}).get(key, 'unavailable')}")
        for index, line in enumerate(lines):
            screen.blit(font.render(line, True, (25, 28, 32)), (panel_x, 30 + index * 25))
        pygame.display.flip()
    pygame.quit()

"""Allocation validation, cost evaluation and energy analytics.

Implements Sections 8.1-8.4 of docs/BENCHMARK_HARNESS_DESIGN.md:

- ``validate_assignments``: decide whether a plan is a valid allocation before
  the simulator sees it (the simulator does not validate plans).
- ``compute_remaining_energy`` / ``compute_energy_tolerances`` /
  ``compute_energy_feasibility``: energy feasibility (Eq. 7), independent of
  any penalty strategy.
- ``energy_penalty_phi``: the replaceable penalty component phi(U)
  (``quadratic`` | ``linear`` | ``zero``). The quadratic form and the
  coefficient are provisional engineering assumptions, not proposal formulas.
- ``evaluate_allocation``: runs the existing simulator (with its own energy
  penalty switched off) and reconstructs the unrounded objective
  ``A(U) = sum_t delta_t * eta_t + sum_r lambda_r * d_r + phi(U)``.

The simulator, core types and the other analytics modules are only *used*,
never modified. Reward standardization, Gini and logging are not part of this
module.

Owner: Yash Yeole (Benchmarking & Analytics).
"""

import copy
import math
from collections.abc import Mapping as MappingABC
from numbers import Integral, Real
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple, Union

from src.simulation.environment import Environment
from src.simulation.generator import ScenarioData

# Relative factor of the energy-feasibility tolerance:
# tolerance_r = ENERGY_TOLERANCE_RELATIVE * max(1, initial_energy_r)   (design Section 8.4)
ENERGY_TOLERANCE_RELATIVE = 1e-9

# The simulator's total_cost is rounded to 2 decimals; the reconstructed base
# cost may differ from it by at most this much (design Section 8.2).
SANITY_TOLERANCE = 0.011

PENALTY_STRATEGIES = ("quadratic", "linear", "zero")


class EvaluationError(RuntimeError):
    """The simulator output is inconsistent with the validated plan.

    This indicates a harness or simulator problem, not an algorithm failure.
    """


class ValidationResult(NamedTuple):
    """Result of :func:`validate_assignments`.

    Attributes:
        valid: ``True`` if the plan is a valid allocation.
        errors: Human-readable rule violations (empty when valid).
        normalized: A fresh ``{robot_id: [task_id, ...]}`` dictionary that
            contains **every** scenario robot (empty list for idle robots),
            or ``None`` when the plan is invalid.
    """
    valid: bool
    errors: List[str]
    normalized: Optional[Dict[int, List[int]]]


class EnergyFeasibility(NamedTuple):
    """Result of :func:`compute_energy_feasibility`."""
    feasible: bool
    deficit: Dict[int, float]


class EnergyPenalty(NamedTuple):
    """Penalty configuration (the ``evaluation.energy_penalty`` config section).

    ``evaluate_allocation`` also accepts a plain mapping with the keys
    ``strategy`` and ``lambda_E`` (other keys, such as ``status``, are ignored).
    """
    strategy: str
    lambda_E: float


class EvaluationResult(NamedTuple):
    """Result of :func:`evaluate_allocation` (all values unrounded).

    Attributes:
        allocation_cost: ``A(U) = base_cost + energy_penalty_value``.
        base_cost: ``sum_t delta_t * eta_t + sum_r lambda_r * d_r``.
        energy_penalty_value: ``phi(U)`` for the configured strategy.
        energy_feasible: ``True`` if no robot has an energy deficit.
        remaining_energy: ``E_r(U)`` per robot (may be negative).
        deficit: Energy deficit per robot (0.0 within tolerance).
        task_completion_times: ``eta_t`` per task (copied from the simulator).
        agent_distances: ``d_r`` per robot (copied from the simulator).
    """
    allocation_cost: float
    base_cost: float
    energy_penalty_value: float
    energy_feasible: bool
    remaining_energy: Dict[int, float]
    deficit: Dict[int, float]
    task_completion_times: Dict[int, float]
    agent_distances: Dict[int, float]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _is_int(value: Any) -> bool:
    """Integer, excluding booleans (NumPy integers are accepted)."""
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    """Real number, excluding booleans (NumPy floats are accepted)."""
    return isinstance(value, Real) and not isinstance(value, bool)


def _finite_float(name: str, value: Any, nonnegative: bool = False) -> float:
    if not _is_real(value):
        raise TypeError(f"{name} must be a real number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}.")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be >= 0, got {result!r}.")
    return result


def _robot_ids(scenario: ScenarioData) -> List[int]:
    return [agent.agent_id for agent in scenario.agents]


def _task_ids(scenario: ScenarioData) -> List[int]:
    return [task.task_id for task in scenario.tasks]


# ---------------------------------------------------------------------------
# A. Assignment validation (design Section 8.1)
# ---------------------------------------------------------------------------

def validate_assignments(assignments: Any, scenario: ScenarioData) -> ValidationResult:
    """Validate an allocation plan before it is given to the simulator.

    Rules (proposal Eqs. 5-6 and the simulator's input type):

    * ``assignments`` is a ``Mapping``.
    * Every robot id is an integer (not a bool) and belongs to the scenario.
    * Every value is a ``list`` or ``tuple`` of integer task ids (not bools).
    * Every task id belongs to the scenario.
    * No task appears more than once, within one list or across robots.
    * Every scenario task appears exactly once.
    * Robots omitted from the mapping (or given an empty list) are idle.

    Energy (Eq. 7) is **not** a validity rule: an energy-infeasible plan is
    still a valid allocation.

    Invalid content never raises; it returns ``valid=False`` with all the
    violations found. The input is not modified.

    Args:
        assignments: The raw object returned by an allocation algorithm.
        scenario: The scenario whose robot and task ids define the id sets.

    Returns:
        ``ValidationResult(valid, errors, normalized)``.
    """
    robot_ids = _robot_ids(scenario)
    task_ids = _task_ids(scenario)
    robot_set = set(robot_ids)
    task_set = set(task_ids)

    if not isinstance(assignments, MappingABC):
        return ValidationResult(
            False,
            [
                "assignments must be a Mapping of robot id -> list of task ids, "
                f"got {type(assignments).__name__}."
            ],
            None,
        )
    try:
        items = list(assignments.items())
    except Exception as exc:  # a misbehaving Mapping must not crash validation
        return ValidationResult(False, [f"assignments could not be read: {exc!r}."], None)

    errors: List[str] = []
    plan: Dict[int, List[int]] = {}
    assigned_to: Dict[int, List[Any]] = {}

    for robot, tasks in items:
        robot_ok = True
        if not _is_int(robot):
            errors.append(f"robot id {robot!r} is not an integer.")
            robot_ok = False
        elif int(robot) not in robot_set:
            errors.append(f"unknown robot id {int(robot)}.")
            robot_ok = False
        label = int(robot) if _is_int(robot) else robot

        if not isinstance(tasks, (list, tuple)):
            errors.append(
                f"tasks of robot {label!r} must be a list or tuple of task ids, "
                f"got {type(tasks).__name__}."
            )
            continue

        sequence: List[int] = []
        for position, task in enumerate(tasks):
            if not _is_int(task):
                errors.append(
                    f"task id {task!r} (robot {label!r}, position {position}) is not an integer."
                )
                continue
            task_id = int(task)
            if task_id not in task_set:
                errors.append(f"unknown task id {task_id} (robot {label!r}, position {position}).")
                continue
            sequence.append(task_id)
            assigned_to.setdefault(task_id, []).append(label)
        if robot_ok:
            plan[int(robot)] = sequence

    for task_id in sorted(t for t, owners in assigned_to.items() if len(owners) > 1):
        owners = assigned_to[task_id]
        errors.append(
            f"task {task_id} is assigned {len(owners)} times (robots {owners}); "
            "each task must be assigned exactly once."
        )

    missing = sorted(task_set - set(assigned_to))
    if missing:
        errors.append(f"tasks not assigned to any robot: {missing}.")

    if errors:
        return ValidationResult(False, errors, None)

    normalized = {robot_id: list(plan.get(robot_id, [])) for robot_id in robot_ids}
    return ValidationResult(True, [], normalized)


# ---------------------------------------------------------------------------
# B-C. Energy (design Section 8.4)
# ---------------------------------------------------------------------------

def compute_remaining_energy(
    scenario: ScenarioData, agent_energy_used: Mapping[int, float]
) -> Dict[int, float]:
    """Remaining energy ``E_r(U) = E_r(initial) - E_r(used)`` for every robot.

    The initial energy is each robot's ``current_energy`` (the simulator's start
    state), not ``energy_capacity``. A robot missing from ``agent_energy_used``
    (for example an idle robot) is assumed to have used zero energy.

    Args:
        scenario: Provides the robots and their initial energy.
        agent_energy_used: ``{robot_id: energy used}`` (unclamped sums from the
            simulator).

    Returns:
        ``{robot_id: remaining energy}`` for every scenario robot, ordered by
        robot id. Values can be negative (energy overdrawn).

    Raises:
        TypeError / ValueError: ``agent_energy_used`` is not a mapping, has an
            id that is not a scenario robot, or has a value that is not a
            finite non-negative number.
    """
    if not isinstance(agent_energy_used, MappingABC):
        raise TypeError("agent_energy_used must be a Mapping of robot id -> energy used.")
    initial = {agent.agent_id: agent.current_energy for agent in scenario.agents}
    unknown = sorted(str(r) for r in agent_energy_used if r not in initial)
    if unknown:
        raise ValueError(f"agent_energy_used contains unknown robot ids: {unknown}.")

    remaining: Dict[int, float] = {}
    for robot_id in sorted(initial):
        used = _finite_float(
            f"energy used by robot {robot_id}", agent_energy_used.get(robot_id, 0.0), nonnegative=True
        )
        remaining[robot_id] = float(initial[robot_id]) - used
    return remaining


def compute_energy_tolerances(scenario: ScenarioData) -> Dict[int, float]:
    """Per-robot feasibility tolerance ``1e-9 * max(1, initial_energy_r)``.

    ``initial_energy_r`` is the robot's ``current_energy``.
    """
    return {
        agent.agent_id: ENERGY_TOLERANCE_RELATIVE * max(1.0, float(agent.current_energy))
        for agent in sorted(scenario.agents, key=lambda a: a.agent_id)
    }


def compute_energy_feasibility(
    remaining_energy: Mapping[int, float],
    tolerance: Union[float, Mapping[int, float]],
) -> EnergyFeasibility:
    """Energy feasibility (Eq. 7) and the per-robot deficits.

    ``deficit_r = max(0, -E_r)``; a deficit not larger than the robot's
    tolerance counts as zero (floating-point noise). The plan is feasible if
    every deficit is zero. This function knows nothing about penalty
    strategies: feasibility is independent of them.

    Args:
        remaining_energy: ``{robot_id: E_r(U)}``.
        tolerance: One non-negative tolerance for every robot, or a mapping
            ``{robot_id: tolerance_r}`` (see :func:`compute_energy_tolerances`
            for the documented per-robot values).

    Returns:
        ``EnergyFeasibility(feasible, deficit)`` with a deficit for every robot.

    Raises:
        TypeError / ValueError: invalid, missing or non-finite inputs.
    """
    if not isinstance(remaining_energy, MappingABC):
        raise TypeError("remaining_energy must be a Mapping of robot id -> energy.")
    per_robot: Optional[Mapping[int, float]] = tolerance if isinstance(tolerance, MappingABC) else None
    uniform = None if per_robot is not None else _finite_float("tolerance", tolerance, nonnegative=True)

    deficit: Dict[int, float] = {}
    for robot_id in sorted(remaining_energy):
        energy = _finite_float(f"remaining energy of robot {robot_id}", remaining_energy[robot_id])
        if per_robot is not None:
            if robot_id not in per_robot:
                raise ValueError(f"tolerance is missing for robot {robot_id}.")
            limit = _finite_float(f"tolerance of robot {robot_id}", per_robot[robot_id], nonnegative=True)
        else:
            limit = uniform
        raw = max(0.0, -energy)
        deficit[robot_id] = 0.0 if raw <= limit else raw
    return EnergyFeasibility(all(value == 0.0 for value in deficit.values()), deficit)


# ---------------------------------------------------------------------------
# D. Penalty (design Section 8.4) - independent and replaceable
# ---------------------------------------------------------------------------

def _check_strategy(strategy: Any) -> str:
    if not isinstance(strategy, str):
        raise TypeError(f"strategy must be a str, got {strategy!r}.")
    if strategy not in PENALTY_STRATEGIES:
        raise ValueError(f"strategy must be one of {list(PENALTY_STRATEGIES)}, got {strategy!r}.")
    return strategy


def energy_penalty_phi(
    deficit: Union[Mapping[int, float], Iterable[float]],
    strategy: str,
    lambda_E: float,
) -> float:
    """Energy-violation penalty ``phi(U)`` for the chosen strategy.

    * ``quadratic``: ``lambda_E * sum_r deficit_r ** 2`` (provisional default)
    * ``linear``: ``lambda_E * sum_r deficit_r``
    * ``zero``: ``0``

    The quadratic form and the coefficient are provisional engineering
    assumptions pending team confirmation (design Section 8.4).

    Args:
        deficit: Per-robot energy deficits (a mapping or an iterable of
            numbers), each finite and ``>= 0``.
        strategy: ``"quadratic"``, ``"linear"`` or ``"zero"`` (exact strings).
        lambda_E: Finite coefficient ``>= 0``.

    Raises:
        TypeError / ValueError: unknown strategy, non-finite or negative
            inputs, or a result that is not finite.
    """
    strategy = _check_strategy(strategy)
    coefficient = _finite_float("lambda_E", lambda_E, nonnegative=True)
    if isinstance(deficit, (str, bytes)) or not (
        isinstance(deficit, MappingABC) or hasattr(deficit, "__iter__")
    ):
        raise TypeError("deficit must be a Mapping or an iterable of numbers.")
    values = deficit.values() if isinstance(deficit, MappingABC) else deficit
    checked = [_finite_float("deficit", value, nonnegative=True) for value in values]

    if strategy == "zero":
        return 0.0
    if strategy == "linear":
        total = math.fsum(checked)
    else:
        total = math.fsum(value * value for value in checked)
    penalty = coefficient * total
    if not math.isfinite(penalty):
        raise ValueError("energy penalty is not finite.")
    return penalty


def _coerce_energy_penalty(energy_penalty: Any) -> Tuple[str, float]:
    if isinstance(energy_penalty, EnergyPenalty):
        strategy, lambda_e = energy_penalty.strategy, energy_penalty.lambda_E
    elif isinstance(energy_penalty, MappingABC):
        missing = [key for key in ("strategy", "lambda_E") if key not in energy_penalty]
        if missing:
            raise ValueError(f"energy_penalty is missing keys: {missing}.")
        strategy, lambda_e = energy_penalty["strategy"], energy_penalty["lambda_E"]
    else:
        raise TypeError("energy_penalty must be an EnergyPenalty or a mapping with 'strategy' and 'lambda_E'.")
    return _check_strategy(strategy), _finite_float("lambda_E", lambda_e, nonnegative=True)


# ---------------------------------------------------------------------------
# E. Evaluation through the existing simulator (design Sections 8.2-8.3)
# ---------------------------------------------------------------------------

def evaluate_allocation(
    scenario: ScenarioData,
    normalized_assignments: Mapping[int, List[int]],
    energy_penalty: Union[EnergyPenalty, Mapping[str, Any]],
) -> EvaluationResult:
    """Evaluate a valid plan: ``A(U)``, energy feasibility and diagnostics.

    Steps: a fresh ``Environment`` loads the scenario; a deep copy of the plan
    is executed with ``energy_violation_penalty_factor=0.0`` (the simulator's
    hard-coded linear penalty is switched off, so the two penalty models are
    never combined); the **unrounded** base cost is rebuilt from the
    simulator's unrounded outputs; our configured penalty is added.

    The simulator's rounded ``total_cost`` is used only as a sanity check.
    ``AllocationResult.runtime_seconds`` and ``solver_name`` are ignored.
    Neither the scenario nor ``normalized_assignments`` is modified or
    aliased.

    Args:
        scenario: The scenario to evaluate on (not modified).
        normalized_assignments: A valid plan (see :func:`validate_assignments`).
        energy_penalty: ``EnergyPenalty`` or a mapping with ``strategy`` and
            ``lambda_E`` (the ``evaluation.energy_penalty`` config section).

    Returns:
        ``EvaluationResult`` with unrounded values.

    Raises:
        TypeError / ValueError: invalid ``energy_penalty``, or a plan that is
            not a valid allocation (the simulator is not called in that case).
        EvaluationError: the simulator output is inconsistent (missing tasks or
            robots, non-finite values, or the rounded total differs from the
            reconstructed base cost by more than ``SANITY_TOLERANCE``).
    """
    strategy, lambda_e = _coerce_energy_penalty(energy_penalty)

    validation = validate_assignments(normalized_assignments, scenario)
    if not validation.valid:
        raise ValueError("cannot evaluate an invalid allocation: " + " ".join(validation.errors))

    environment = Environment(map_width=scenario.map_width, map_height=scenario.map_height)
    environment.load_scenario(scenario)
    assignments = copy.deepcopy(validation.normalized)
    result, _traces = environment.execute_allocation(
        assignments, energy_violation_penalty_factor=0.0
    )

    priority = {task.task_id: float(task.priority) for task in scenario.tasks}
    usage_penalty = {agent.agent_id: float(agent.usage_penalty) for agent in scenario.agents}

    if set(result.task_completion_times) != set(priority):
        raise EvaluationError(
            "simulator returned completion times for tasks "
            f"{sorted(result.task_completion_times)}, expected {sorted(priority)}."
        )
    if not set(usage_penalty) <= set(result.agent_distances):
        raise EvaluationError("simulator did not return a distance for every robot.")

    terms = [
        priority[task_id] * _finite_eval("completion time", result.task_completion_times[task_id])
        for task_id in sorted(priority)
    ] + [
        usage_penalty[robot_id] * _finite_eval("distance", result.agent_distances[robot_id])
        for robot_id in sorted(usage_penalty)
    ]
    base_cost = math.fsum(terms)

    if abs(round(base_cost, 2) - float(result.total_cost)) > SANITY_TOLERANCE:
        raise EvaluationError(
            f"reconstructed base cost {base_cost!r} disagrees with the simulator's "
            f"rounded total_cost {result.total_cost!r} (tolerance {SANITY_TOLERANCE})."
        )

    try:
        remaining = compute_remaining_energy(scenario, result.agent_energy_used)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"invalid energy usage from the simulator: {exc}") from None
    feasibility = compute_energy_feasibility(remaining, compute_energy_tolerances(scenario))
    penalty = energy_penalty_phi(feasibility.deficit, strategy, lambda_e)

    return EvaluationResult(
        allocation_cost=base_cost + penalty,
        base_cost=base_cost,
        energy_penalty_value=penalty,
        energy_feasible=feasibility.feasible,
        remaining_energy=remaining,
        deficit=dict(feasibility.deficit),
        task_completion_times={t: float(v) for t, v in sorted(result.task_completion_times.items())},
        agent_distances={r: float(v) for r, v in sorted(result.agent_distances.items())},
    )


def _finite_eval(name: str, value: Any) -> float:
    try:
        return _finite_float(name, value)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"simulator returned an invalid {name}: {exc}") from None

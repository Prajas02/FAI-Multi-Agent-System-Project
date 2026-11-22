"""Fast, Environment-consistent evaluation of allocation plans (A(U), Eq. 4-7).

Every allocator in this package (Greedy, SVCA, GA, ...) is scored by the SAME
function so that cost differences come from the algorithm, not from how the cost
is computed.

Why not just call ``Environment.execute_allocation``?
    It deep-copies every agent and task on each call (``reset()``), which is far
    too slow inside a GA fitness loop or a local-search inner loop. This module
    replays exactly the same step-by-step physics, but reuses Prajas's own
    ``KinematicsModel`` / ``BatteryDecayModel`` for the formulas so that any change
    he makes to velocity decay or energy use is picked up automatically.
    ``tests/test_allocation.py`` checks parity against ``Environment`` on random plans.

Key property used by the solvers:
    A(U) is SEPARABLE across agents. A task's completion time depends only on the
    route of the agent that serves it, so A(U) = sum_r route_cost(r, route_r).
    Insertion / relocation moves therefore only need to re-evaluate the routes they touch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.types import AllocationResult
from src.simulation.generator import ScenarioData
from src.simulation.kinematics import BatteryDecayModel, KinematicsModel

Routes = Dict[int, List[int]]  # agent_id -> ordered list of task_ids


@dataclass(frozen=True)
class RouteStats:
    """Outcome of one agent executing one ordered route."""
    cost: float                         # priority cost + travel cost + energy penalty
    priority_cost: float                # sum_t delta_t * eta_t
    travel_cost: float                  # lambda_r * d_r
    penalty: float                      # phi(U) contribution (energy deficit * M)
    distance: float                     # d_r
    energy_used: float
    completion_times: Tuple[float, ...]  # eta_t for each task in route order


_EMPTY_CACHE_LIMIT = 400_000


class AllocationEvaluator:
    """Scores allocation plans for one scenario.

    Create one evaluator per ``solve()`` call (not one shared across solvers):
    the internal route cache would otherwise make later solvers look faster.
    """

    def __init__(
        self,
        scenario: ScenarioData,
        battery_model: Optional[BatteryDecayModel] = None,
        energy_penalty_factor: float = 1000.0,
    ):
        self.scenario = scenario
        self.battery_model = battery_model or BatteryDecayModel()
        self.penalty_factor = energy_penalty_factor

        self.agents = {a.agent_id: a for a in scenario.agents}
        self.tasks = {t.task_id: t for t in scenario.tasks}
        self.agent_ids: List[int] = sorted(self.agents)
        self.task_ids: List[int] = sorted(self.tasks)

        # Precomputed Euclidean distances (same Position.distance_to as Environment).
        self.d_at: Dict[int, Dict[int, float]] = {
            a: {t: self.agents[a].position.distance_to(self.tasks[t].location) for t in self.task_ids}
            for a in self.agent_ids
        }
        self.d_tt: Dict[int, Dict[int, float]] = {
            t1: {t2: self.tasks[t1].location.distance_to(self.tasks[t2].location) for t2 in self.task_ids}
            for t1 in self.task_ids
        }
        self._cache: Dict[Tuple[int, Tuple[int, ...]], RouteStats] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    # ------------------------------------------------------------------ routes
    def route_stats(self, agent_id: int, route: Sequence[int]) -> RouteStats:
        """Replay one agent's route exactly as Environment.execute_allocation does."""
        key = (agent_id, tuple(route))
        hit = self._cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            return hit
        self.cache_misses += 1

        agent = self.agents[agent_id]
        bm = self.battery_model
        capacity = agent.energy_capacity
        energy = agent.current_energy

        prev: Optional[int] = None
        now = dist = spent = penalty = prio = 0.0
        completions: List[float] = []
        for tid in route:
            task = self.tasks[tid]
            d = self.d_at[agent_id][tid] if prev is None else self.d_tt[prev][tid]

            ratio = energy / capacity if capacity > 0 else 1.0
            v = KinematicsModel.compute_effective_velocity(agent, ratio)
            travel = d / v if v > 0 else 0.0
            step = bm.compute_motion_energy(d, v) + bm.compute_execution_energy(task.execution_time)

            if energy < step:
                penalty += (step - energy) * self.penalty_factor
                energy = 0.0
            else:
                energy -= step

            dist += d
            spent += step
            now += travel + task.execution_time
            prio += task.priority * now
            completions.append(now)
            prev = tid

        travel_cost = agent.usage_penalty * dist
        stats = RouteStats(
            cost=prio + travel_cost + penalty,
            priority_cost=prio,
            travel_cost=travel_cost,
            penalty=penalty,
            distance=dist,
            energy_used=spent,
            completion_times=tuple(completions),
        )
        if len(self._cache) >= _EMPTY_CACHE_LIMIT:
            self._cache.clear()
        self._cache[key] = stats
        return stats

    def route_cost(self, agent_id: int, route: Sequence[int]) -> float:
        return self.route_stats(agent_id, route).cost

    # -------------------------------------------------------------- full plans
    def total_cost(self, assignments: Routes) -> float:
        """Unrounded A(U). Use this inside search loops."""
        return sum(self.route_stats(a, r).cost for a, r in assignments.items())

    def validate(self, assignments: Routes) -> List[str]:
        """Return a list of constraint violations (empty list = valid plan).

        Checks Eq. 5-6: every task is assigned to exactly one existing agent.
        NOTE: Environment.execute_allocation does NOT check this. A plan that leaves
        tasks unassigned gets a LOWER A(U) there, because unassigned tasks add no cost.
        Always validate before comparing solvers.
        """
        problems: List[str] = []
        seen: Dict[int, int] = {}
        for a, route in assignments.items():
            if a not in self.agents:
                problems.append(f"unknown agent {a}")
                continue
            for t in route:
                if t not in self.tasks:
                    problems.append(f"agent {a} has unknown task {t}")
                elif t in seen:
                    problems.append(f"task {t} assigned twice (agents {seen[t]} and {a})")
                else:
                    seen[t] = a
        missing = sorted(set(self.task_ids) - set(seen))
        if missing:
            problems.append(f"unassigned tasks: {missing}")
        return problems

    def evaluate(
        self,
        assignments: Routes,
        solver_name: str = "Evaluator",
        runtime_seconds: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AllocationResult:
        """Build the shared AllocationResult (same fields/rounding as Environment)."""
        completion: Dict[int, float] = {}
        distances: Dict[int, float] = {}
        energy: Dict[int, float] = {}
        prio = travel = pen = 0.0
        for a, route in assignments.items():
            st = self.route_stats(a, route)
            for tid, c in zip(route, st.completion_times):
                completion[tid] = c
            distances[a] = st.distance
            energy[a] = st.energy_used
            prio += st.priority_cost
            travel += st.travel_cost
            pen += st.penalty

        meta: Dict[str, Any] = {
            "priority_completion_cost": round(prio, 2),
            "travel_cost": round(travel, 2),
            "energy_penalty": round(pen, 2),
            "valid": not self.validate(assignments),
        }
        if metadata:
            meta.update(metadata)
        return AllocationResult(
            assignments={a: list(r) for a, r in assignments.items()},
            task_completion_times=completion,
            agent_distances=distances,
            agent_energy_used=energy,
            total_cost=round(prio + travel + pen, 2),
            runtime_seconds=runtime_seconds,
            solver_name=solver_name,
            metadata=meta,
        )

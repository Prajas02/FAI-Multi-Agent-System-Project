"""Coalition model of Martin et al. (2023), "Multi-robot task allocation clustering based on
game theory", Robotics and Autonomous Systems 161:104314 (paper [1] of the proposal).

    J(S)  = psi(S) * ( sum_{i in S} J_i  -  alpha1 * D_RT(S)  -  alpha2 * |S|  +  rho )      (Eq. 5)
    J_i   = alpha3*E_i + alpha4*v_i - alpha5*lambda_i      if i is a robot                    (Eq. 6)
            -(alpha6*delta_i + alpha7*tau_i)               if i is a task
    psi(S)= 0 if S is empty, robots-only or tasks-only; 1 otherwise                           (Eq. 7)
    V(C)  = Omega(C) * sum_{C_h in C} J(C_h)                                                  (Eq. 8)
            Omega(C) = 0 if some cluster contains only tasks, else 1.

D_RT(S) is the AVERAGE robot-task distance inside S (paper Sec. 2.2).

Choices the paper leaves open (documented here, not hidden):
  * rho. The paper only says it is "an arbitrarily large number that guarantees J(S) is positive".
    Default ``rho=None`` uses the smallest bound that provably guarantees positivity for EVERY
    mixed coalition:  rho = sum_i max(0, -J_i) + alpha1 * max_distance + alpha2 * N + 1.
    Because V(C) adds rho once per mixed cluster, rho influences how many clusters V(C)
    prefers. ``rho`` / ``rho_multiplier`` let you study that (see the experiments).
  * Attribute scaling. The paper draws E in [0,100], v in [1,20], lambda in [1,5], tau in [0,10],
    delta in [1,5] and sets alpha3..alpha7 = 1. Our simulator uses different units (energy 500-2000,
    tau 10-60, ...), so with raw values energy would swamp everything. ``scale="paper"`` maps each
    attribute affinely from the simulator's range onto the paper's range, so alpha = 1 means the
    same thing as in the paper. ``scale="none"`` uses raw values.

Costs are incremental: ``PaperCoalitionState.add`` is O(|S|), ``cost`` is O(1), which keeps
Shapley permutation sampling cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.simulation.generator import ScenarioData

# Ranges used by the paper's case study (Sec. 5) and by our ScenarioGenerator defaults.
PAPER_RANGES: Dict[str, Tuple[float, float]] = {
    "E": (0.0, 100.0), "v": (1.0, 20.0), "lam": (1.0, 5.0), "tau": (0.0, 10.0), "delta": (1.0, 5.0),
}
SIM_RANGES: Dict[str, Tuple[float, float]] = {
    "E": (500.0, 2000.0), "v": (5.0, 15.0), "lam": (1.0, 3.0), "tau": (10.0, 60.0), "delta": (1.0, 5.0),
}


def _affine(x: float, src: Tuple[float, float], dst: Tuple[float, float]) -> float:
    lo, hi = src
    a, b = dst
    return a + (x - lo) * (b - a) / (hi - lo) if hi != lo else 0.5 * (a + b)


@dataclass
class PaperParams:
    """Weights of Eq. 5-6. Defaults are the paper's case study (alpha1=0.1, alpha2=10, rest 1)."""
    alpha1: float = 0.1
    alpha2: float = 10.0
    alpha3: float = 1.0   # robot energy
    alpha4: float = 1.0   # robot velocity
    alpha5: float = 1.0   # robot usage penalty
    alpha6: float = 1.0   # task priority
    alpha7: float = 1.0   # task operation time
    rho: Optional[float] = None
    rho_multiplier: float = 1.0
    scale: str = "paper"                       # "paper" | "none"
    sim_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(SIM_RANGES))


class PaperCoalitionModel:
    """Static per-scenario data for J(S) and V(C). Entities 0..R-1 are robots, R..R+T-1 tasks."""

    def __init__(self, scenario: ScenarioData, params: Optional[PaperParams] = None):
        if not scenario.agents:
            raise ValueError("SVCA needs at least one robot")
        self.params = p = params or PaperParams()
        self.agents = list(scenario.agents)
        self.tasks = list(scenario.tasks)
        self.R = len(self.agents)
        self.T = len(self.tasks)
        self.N = self.R + self.T

        # d[j][i]: distance task j <-> robot i
        self.d: List[List[float]] = [
            [a.position.distance_to(t.location) for a in self.agents] for t in self.tasks
        ]
        self.d_max = max((x for row in self.d for x in row), default=0.0)

        def sc(name: str, x: float) -> float:
            if p.scale == "none":
                return x
            return _affine(x, p.sim_ranges[name], PAPER_RANGES[name])

        self.J_i: List[float] = [
            p.alpha3 * sc("E", a.current_energy) + p.alpha4 * sc("v", a.velocity) - p.alpha5 * sc("lam", a.usage_penalty)
            for a in self.agents
        ]
        self.J_i += [
            -(p.alpha6 * sc("delta", t.priority) + p.alpha7 * sc("tau", t.execution_time))
            for t in self.tasks
        ]
        if p.rho is not None:
            self.rho = float(p.rho)
        else:
            neg = sum(max(0.0, -x) for x in self.J_i)
            self.rho = p.rho_multiplier * (neg + p.alpha1 * self.d_max + p.alpha2 * self.N + 1.0)

    # ------------------------------------------------------------------ helpers
    def entity_label(self, e: int) -> str:
        return f"R{self.agents[e].agent_id}" if e < self.R else f"T{self.tasks[e - self.R].task_id}"

    def new_state(self) -> "PaperCoalitionState":
        return PaperCoalitionState(self)

    def cost_of(self, members: Iterable[int]) -> float:
        """J(S) from scratch (Eq. 5)."""
        robots = [e for e in members if e < self.R]
        tasks = [e - self.R for e in members if e >= self.R]
        if not robots or not tasks:
            return 0.0                                   # psi(S) = 0
        p = self.params
        sum_d = 0.0
        for j in tasks:
            row = self.d[j]
            for i in robots:
                sum_d += row[i]
        size = len(robots) + len(tasks)
        sum_J = sum(self.J_i[e] for e in robots) + sum(self.J_i[self.R + j] for j in tasks)
        return sum_J - p.alpha1 * sum_d / (len(robots) * len(tasks)) - p.alpha2 * size + self.rho

    def has_robot(self, members: Iterable[int]) -> bool:
        return any(e < self.R for e in members)

    def value_of_clustering(self, clusters: Sequence[Sequence[int]]) -> float:
        """V(C) (Eq. 8). Omega(C)=0 if any non-empty cluster has no robot (task-only)."""
        total = 0.0
        for members in clusters:
            if not members:
                continue
            if not self.has_robot(members):
                return 0.0
            total += self.cost_of(members)
        return total


class PaperCoalitionState:
    """Incrementally maintained coalition (order-independent up to float rounding)."""

    __slots__ = ("m", "robots", "tasks", "sum_J", "sum_d")

    def __init__(self, model: PaperCoalitionModel):
        self.m = model
        self.robots: List[int] = []
        self.tasks: List[int] = []
        self.sum_J = 0.0
        self.sum_d = 0.0

    def add(self, e: int) -> None:
        m = self.m
        self.sum_J += m.J_i[e]
        if e < m.R:
            for j in self.tasks:
                self.sum_d += m.d[j][e]
            self.robots.append(e)
        else:
            j = e - m.R
            row = m.d[j]
            for i in self.robots:
                self.sum_d += row[i]
            self.tasks.append(j)

    def cost(self) -> float:
        nr, nt = len(self.robots), len(self.tasks)
        if nr == 0 or nt == 0:
            return 0.0
        p = self.m.params
        return self.sum_J - p.alpha1 * self.sum_d / (nr * nt) - p.alpha2 * (nr + nt) + self.m.rho

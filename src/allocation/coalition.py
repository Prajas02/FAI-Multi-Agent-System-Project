"""Coalition cost J(S) used by SVCA (proposal Eq. 11).

    J(S) = psi(S) * ( sum_{i in S} J_i  -  alpha1 * D_RT(S)  -  alpha2 * |S|  +  rho  [+ load term] )

The proposal does not define psi, J_i or D_RT, and the paper [1] has not been
checked yet, so the concrete choices below are ASSUMPTIONS, isolated in this file so they can be
swapped without touching the clustering code:

  Entities      N = R u T. Indices 0..R-1 are agents, R..R+T-1 are tasks (scenario order).
  J_i  (task)   delta_t * (tau_t + d_t^min / v_bar): stand-alone cost of serving task t
                with the nearest agent in the whole team (d_t^min) at mean speed v_bar.
  J_i  (agent)  lambda_r: fixed usage cost of deploying the agent.
  D_RT(S)       Proximity benefit: sum over tasks in S of max(0, D_ref - d(t, nearest agent in S)),
                where D_ref is the mean agent-task distance. Large when S's tasks sit near S's agents.
  psi(S)        1 for a feasible coalition; ``psi_penalty`` (>1) if S contains tasks but no agent
                (nobody can execute them). This is how infeasible coalitions get expensive.
  rho           Fixed cost of forming one coalition (makes merging attractive).
  load term     EXTENSION beyond Eq. 11 (weight ``load_weight``, 0 disables it). Each task in S is
                assigned to its NEAREST agent in S (ties: lower index) ONLY to estimate travel:
                  L_r  ~ d(r, centroid(O_r)) + c * sigma(O_r) * (n-1)/sqrt(n)    (tour-length estimate;
                          sigma = rms spread of the owned tasks, from running moment sums)
                Let m = number of agents that own at least one task, L = sum_r L_r, v = their mean speed:
                  load = (sum_t delta_t) * (sum_t tau_t + L / v) / (2 m)  +  sum_r lambda_r * L_r
                The first part is the queueing delay with the work shared evenly among the m agents that
                are actually in use (average completion ~ half the route time); the second is the usage
                cost lambda_r * d_r from Eq. 4. This makes agent count, agent positions and task
                compactness matter. Without it, adding a second agent never helps a cluster and SVCA
                cannot balance agents against tasks.
                (Two cruder versions were tried first and rejected because they ranked candidate
                 partitions backwards vs. the true cost: charging every task the full agent-task
                 distance, and charging each agent only the queue of ITS nearest-neighbour tasks.)

Costs are incremental: ``CoalitionState.add`` is O(T), ``cost`` is O(R). A full Shapley
permutation therefore costs O(N * (T + R)) instead of O(N^2 * T).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List

from src.simulation.generator import ScenarioData

_INF = math.inf


@dataclass
class CoalitionParams:
    alpha1: float = 0.05
    alpha2: float = 1.5
    rho: float = 10.0
    psi_penalty: float = 10.0
    load_weight: float = 1.0
    tour_const: float = 2.7     # c in the tour-length estimate


class CoalitionModel:
    """Static, per-scenario data for J(S)."""

    def __init__(self, scenario: ScenarioData, params: CoalitionParams):
        if not scenario.agents:
            raise ValueError("SVCA needs at least one agent")
        self.params = params
        self.agents = list(scenario.agents)
        self.tasks = list(scenario.tasks)
        self.R = len(self.agents)
        self.T = len(self.tasks)
        self.N = self.R + self.T

        # d[j][i]: distance from task j to agent i
        self.d: List[List[float]] = [
            [a.position.distance_to(t.location) for a in self.agents] for t in self.tasks
        ]
        self.v_bar = sum(a.velocity for a in self.agents) / self.R
        pairs = [x for row in self.d for x in row]
        self.D_ref = sum(pairs) / len(pairs) if pairs else 0.0
        self.dmin_global = [min(row) for row in self.d]

        self.J_i: List[float] = [a.usage_penalty for a in self.agents]
        for j, t in enumerate(self.tasks):
            self.J_i.append(t.priority * (t.execution_time + self.dmin_global[j] / self.v_bar))

    # entity helpers -----------------------------------------------------------
    def entity_label(self, e: int) -> str:
        return f"R{self.agents[e].agent_id}" if e < self.R else f"T{self.tasks[e - self.R].task_id}"

    def new_state(self) -> "CoalitionState":
        return CoalitionState(self)

    def cost_of(self, members: Iterable[int]) -> float:
        st = self.new_state()
        for e in members:
            st.add(e)
        return st.cost()


class CoalitionState:
    """Incrementally maintained coalition. Cost does not depend on insertion order
    (up to float rounding)."""

    def __init__(self, model: CoalitionModel):
        self.m = model
        self.agent_idx: List[int] = []
        self.tasks: List[int] = []              # local task indices present
        self.sum_J = 0.0
        self.sum_delta = 0.0                    # sum of priorities of all tasks (used if no agent)
        self.sum_tau = 0.0
        self.sum_dmin_global = 0.0
        self.sum_prox = 0.0                     # D_RT(S), valid when agents are present
        self.dmin: Dict[int, float] = {}        # task j -> distance to its owner (inf if none)
        self.owner: Dict[int, int] = {}         # task j -> owning agent index
        self.a_delta: Dict[int, float] = {}     # per-agent: sum of owned task priorities
        self.a_tau: Dict[int, float] = {}       # per-agent: sum of owned execution times
        self.a_n: Dict[int, int] = {}           # per-agent: number of owned tasks
        self.a_sx: Dict[int, float] = {}        # per-agent: sum of owned task x
        self.a_sy: Dict[int, float] = {}
        self.a_sq: Dict[int, float] = {}        # per-agent: sum of x^2 + y^2

    def clone(self) -> "CoalitionState":
        c = CoalitionState.__new__(CoalitionState)
        c.m = self.m
        c.agent_idx = list(self.agent_idx)
        c.tasks = list(self.tasks)
        c.sum_J = self.sum_J
        c.sum_delta = self.sum_delta
        c.sum_tau = self.sum_tau
        c.sum_dmin_global = self.sum_dmin_global
        c.sum_prox = self.sum_prox
        c.dmin = dict(self.dmin)
        c.owner = dict(self.owner)
        c.a_delta = dict(self.a_delta)
        c.a_tau = dict(self.a_tau)
        c.a_n = dict(self.a_n)
        c.a_sx = dict(self.a_sx)
        c.a_sy = dict(self.a_sy)
        c.a_sq = dict(self.a_sq)
        return c

    @property
    def n_agents(self) -> int:
        return len(self.agent_idx)

    @property
    def size(self) -> int:
        return len(self.agent_idx) + len(self.tasks)

    def _give(self, j: int, i: int, dj: float) -> None:
        t = self.m.tasks[j]
        x, y = t.location.x, t.location.y
        self.a_delta[i] += t.priority
        self.a_tau[i] += t.execution_time
        self.a_n[i] += 1
        self.a_sx[i] += x
        self.a_sy[i] += y
        self.a_sq[i] += x * x + y * y
        self.owner[j] = i
        self.dmin[j] = dj

    def _take_away(self, j: int) -> None:
        t = self.m.tasks[j]
        x, y = t.location.x, t.location.y
        i = self.owner[j]
        self.a_delta[i] -= t.priority
        self.a_tau[i] -= t.execution_time
        self.a_n[i] -= 1
        self.a_sx[i] -= x
        self.a_sy[i] -= y
        self.a_sq[i] -= x * x + y * y

    def _route_length(self, i: int) -> float:
        """Estimated tour length of agent i over the tasks it owns."""
        n = self.a_n[i]
        if n <= 0:
            return 0.0
        cx, cy = self.a_sx[i] / n, self.a_sy[i] / n
        pos = self.m.agents[i].position
        d0 = math.hypot(pos.x - cx, pos.y - cy)
        var = max(0.0, self.a_sq[i] / n - cx * cx - cy * cy)
        return d0 + self.m.params.tour_const * math.sqrt(var) * (n - 1) / math.sqrt(n)

    def add(self, e: int) -> None:
        m = self.m
        D_ref = m.D_ref
        self.sum_J += m.J_i[e]
        if e < m.R:
            self.agent_idx.append(e)
            self.a_delta[e] = self.a_tau[e] = 0.0
            self.a_n[e] = 0
            self.a_sx[e] = self.a_sy[e] = self.a_sq[e] = 0.0
            for j in self.tasks:
                dj = m.d[j][e]
                old = self.dmin[j]
                if dj < old or (dj == old and e < self.owner[j]):
                    if old == _INF:
                        self.sum_prox += max(0.0, D_ref - dj)
                    else:
                        self._take_away(j)
                        self.sum_prox += max(0.0, D_ref - dj) - max(0.0, D_ref - old)
                    self._give(j, e, dj)
        else:
            j = e - m.R
            t = m.tasks[j]
            self.tasks.append(j)
            self.sum_delta += t.priority
            self.sum_tau += t.execution_time
            self.sum_dmin_global += m.dmin_global[j]
            if self.agent_idx:
                dj, i = min((m.d[j][i], i) for i in self.agent_idx)
                self._give(j, i, dj)
                self.sum_prox += max(0.0, D_ref - dj)
            else:
                self.dmin[j] = _INF

    def cost(self) -> float:
        p = self.m.params
        size = self.size
        if size == 0:
            return 0.0
        psi = 1.0
        prox = 0.0
        load = 0.0
        if self.tasks:
            if not self.agent_idx:
                psi = p.psi_penalty
                workload = self.sum_tau + self.sum_dmin_global / self.m.v_bar
                load = self.sum_delta * workload / 2.0
            else:
                prox = self.sum_prox
                agents = self.m.agents
                m_use = 0
                L_tot = 0.0
                v_sum = 0.0
                for i in self.agent_idx:
                    if self.a_n[i] > 0:
                        L = self._route_length(i)
                        m_use += 1
                        L_tot += L
                        v_sum += agents[i].velocity
                        load += agents[i].usage_penalty * L
                load += self.sum_delta * (self.sum_tau + L_tot * m_use / v_sum) / (2.0 * m_use)
            load *= p.load_weight
        bracket = self.sum_J - p.alpha1 * prox - p.alpha2 * size + p.rho + load
        return psi * max(bracket, 0.0)

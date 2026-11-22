"""Characteristic functions V(S) for the reward-sharing game (Stage 2).

Players are the *agents* of the team. A coalition ``S`` is a subset of agents and
``V(S)`` is the net reward that coalition can generate from the task set, using
the same ingredients as the MRTA objective A(U) (Eq. 4) with the sign flipped:

    V(S) = sum_t  delta_t * (H - eta_t)  -  sum_r  lambda_r * d_r

i.e. "priority-weighted time saved, minus usage/travel penalty". ``H`` is a
common time horizon so that finishing a task at all is worth something and
finishing it sooner is worth more. V(empty) = 0 by construction.

Two ready-made games are provided:

* :class:`FacilityLocationValue` - every task is served by the *best* agent in
  the coalition (monotone, submodular, very cheap: one numpy ``max``). This is
  the default for large N (10..80) where MCMC needs many evaluations.
* :class:`RoutingValue` - a richer greedy-routing game where agents in the
  coalition serve tasks sequentially, with energy limits and queueing delays.

Both memoise values by coalition bitmask, so repeated MCMC visits are free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.core.agent import Agent
from src.core.task import Task


class CharacteristicFunction(ABC):
    """Base class: a cached TU-game value function over ``n_players`` players.

    Players are addressed by *index* ``0..n-1`` internally; ``player_ids`` maps an
    index back to the agent id used in the rest of the project.
    """

    def __init__(self, player_ids: Sequence[int], max_cache: int = 2_000_000):
        self.player_ids: List[int] = list(player_ids)
        self.n_players: int = len(self.player_ids)
        self._cache: Dict[int, float] = {0: 0.0}
        self._max_cache = max_cache
        self.num_evaluations = 0   # true (uncached) evaluations
        self.num_calls = 0         # all calls, cached or not

    # ------------------------------------------------------------------ API
    @abstractmethod
    def _evaluate(self, members: np.ndarray) -> float:
        """Compute V for a non-empty coalition given sorted player indices."""

    def value_of_mask(self, mask: int) -> float:
        """V(S) where S is encoded as an int bitmask over player indices."""
        self.num_calls += 1
        cached = self._cache.get(mask)
        if cached is not None:
            return cached
        members = np.fromiter(
            (i for i in range(self.n_players) if (mask >> i) & 1), dtype=np.int64
        )
        val = float(self._evaluate(members))
        self.num_evaluations += 1
        if len(self._cache) >= self._max_cache:
            self._cache = {0: 0.0}
        self._cache[mask] = val
        return val

    def __call__(self, coalition: Iterable[int]) -> float:
        """V(S) for an iterable of player *indices*."""
        mask = 0
        for i in coalition:
            mask |= 1 << int(i)
        return self.value_of_mask(mask)

    def prefix_values(self, perm: Sequence[int]) -> np.ndarray:
        """V of every prefix {perm[0..k]} for k = 0..n-1 (length-n array).

        Subclasses override this with a vectorised version when one exists; the
        default walks the ordering using the cached ``value_of_mask``.
        """
        out = np.empty(len(perm))
        mask = 0
        for k, p in enumerate(perm):
            mask |= 1 << int(p)
            out[k] = self.value_of_mask(mask)
        return out

    def grand_coalition_value(self) -> float:
        return self.value_of_mask((1 << self.n_players) - 1)

    @property
    def cache_hit_rate(self) -> float:
        return 0.0 if self.num_calls == 0 else 1.0 - self.num_evaluations / self.num_calls


# --------------------------------------------------------------------------- #
# Utility helpers shared by the concrete games
# --------------------------------------------------------------------------- #
def _agent_task_arrays(agents: Sequence[Agent], tasks: Sequence[Task]):
    ap = np.array([[a.position.x, a.position.y] for a in agents], dtype=float)
    tp = np.array([[t.location.x, t.location.y] for t in tasks], dtype=float)
    dist = np.linalg.norm(ap[:, None, :] - tp[None, :, :], axis=2)   # (R, T)
    return ap, tp, dist


def _default_horizon(agents, tasks, dist, factor: float) -> float:
    vel = np.array([max(a.velocity, 1e-9) for a in agents])[:, None]
    tau = np.array([t.execution_time for t in tasks])[None, :]
    return float(factor * (dist / vel + tau).max())


def natural_horizon(agents: Sequence[Agent], tasks: Sequence[Task], factor: float = 1.25) -> float:
    """The time horizon H a game over (agents, tasks) would pick by default.

    Compute it once on the *full* scenario and pass it as ``horizon=`` to every
    sub-game (e.g. per-cluster games) so payoffs stay comparable across games.
    """
    _, _, dist = _agent_task_arrays(agents, tasks)
    return _default_horizon(agents, tasks, dist, factor)


class FacilityLocationValue(CharacteristicFunction):
    """Each task is served by the single best agent in the coalition.

        u[r, t] = max(0, delta_t * (H - eta_rt) - lambda_r * d_rt)   if feasible else 0
        V(S)    = sum_t max_{r in S} u[r, t]

    ``eta_rt = d_rt / v_r + tau_t`` is the completion time, and feasibility is the
    agent's own energy check (``Agent.can_undertake``).
    """

    def __init__(
        self,
        agents: Sequence[Agent],
        tasks: Sequence[Task],
        horizon_factor: float = 1.25,
        reward_scale: float = 1.0,
        horizon: Optional[float] = None,
    ):
        super().__init__([a.agent_id for a in agents])
        self.agents = list(agents)
        self.tasks = list(tasks)
        _, _, dist = _agent_task_arrays(self.agents, self.tasks)
        vel = np.array([max(a.velocity, 1e-9) for a in self.agents])[:, None]
        tau = np.array([t.execution_time for t in self.tasks])[None, :]
        delta = np.array([t.priority for t in self.tasks])[None, :]
        lam = np.array([a.usage_penalty for a in self.agents])[:, None]
        eta = dist / vel + tau
        if horizon is None:
            horizon = _default_horizon(self.agents, self.tasks, dist, horizon_factor)
        self.horizon = horizon

        util = reward_scale * (delta * (horizon - eta) - lam * dist)
        feasible = np.array(
            [[a.can_undertake(dist[r, t], self.tasks[t].execution_time)
              for t in range(len(self.tasks))]
             for r, a in enumerate(self.agents)]
        )
        self.utility = np.where(feasible, np.maximum(util, 0.0), 0.0)   # (R, T)

    def _evaluate(self, members: np.ndarray) -> float:
        return float(self.utility[members].max(axis=0).sum())

    def prefix_values(self, perm: Sequence[int]) -> np.ndarray:
        """All prefix values at once via a running max: O(n * T), no cache needed."""
        perm = np.asarray(perm, dtype=np.int64)
        self.num_calls += len(perm)
        self.num_evaluations += len(perm)
        return np.maximum.accumulate(self.utility[perm], axis=0).sum(axis=1)

    @classmethod
    def from_scenario(cls, scenario, task_ids: Optional[Iterable[int]] = None, **kw):
        tasks = _filter_tasks(scenario.tasks, task_ids)
        return cls(scenario.agents, tasks, **kw)


class RoutingValue(CharacteristicFunction):
    """Greedy sequential-routing game with energy limits and queueing.

    Agents in the coalition repeatedly pick the (agent, task) pair with the best
    positive net gain

        gain = delta_t * (H - eta) - lambda_r * d

    where ``eta`` includes the time the agent has already spent (so busy agents
    become less attractive), skipping pairs the agent cannot afford energetically.
    Agent position, clock and remaining energy are updated after each pick.
    """

    def __init__(
        self,
        agents: Sequence[Agent],
        tasks: Sequence[Task],
        horizon_factor: float = 2.0,
        horizon: Optional[float] = None,
    ):
        super().__init__([a.agent_id for a in agents])
        self.agents = list(agents)
        self.tasks = list(tasks)
        _, _, dist = _agent_task_arrays(self.agents, self.tasks)
        self.horizon = (horizon if horizon is not None
                        else _default_horizon(self.agents, self.tasks, dist, horizon_factor))
        self._pos0 = np.array([[a.position.x, a.position.y] for a in self.agents])
        self._vel = np.array([max(a.velocity, 1e-9) for a in self.agents])
        self._energy0 = np.array([a.current_energy for a in self.agents])
        self._lam = np.array([a.usage_penalty for a in self.agents])
        self._cpm = np.array([a.consumption_per_meter for a in self.agents])
        self._cps = np.array([a.consumption_per_sec for a in self.agents])
        self._tpos = np.array([[t.location.x, t.location.y] for t in self.tasks])
        self._tau = np.array([t.execution_time for t in self.tasks])
        self._delta = np.array([t.priority for t in self.tasks])

    def _evaluate(self, members: np.ndarray) -> float:
        pos = self._pos0[members].copy()
        clock = np.zeros(len(members))
        energy = self._energy0[members].copy()
        vel, lam = self._vel[members], self._lam[members]
        cpm, cps = self._cpm[members], self._cps[members]
        open_task = np.ones(len(self._tau), dtype=bool)
        total = 0.0
        for _ in range(len(self._tau)):
            d = np.linalg.norm(pos[:, None, :] - self._tpos[None, :, :], axis=2)  # (m, T)
            eta = clock[:, None] + d / vel[:, None] + self._tau[None, :]
            need = d * cpm[:, None] + self._tau[None, :] * cps[:, None]
            gain = self._delta[None, :] * (self.horizon - eta) - lam[:, None] * d
            gain = np.where(open_task[None, :] & (need <= energy[:, None]), gain, -np.inf)
            flat = int(np.argmax(gain))
            r, t = divmod(flat, gain.shape[1])
            if not np.isfinite(gain[r, t]) or gain[r, t] <= 0.0:
                break
            total += gain[r, t]
            pos[r] = self._tpos[t]
            clock[r] = eta[r, t]
            energy[r] -= need[r, t]
            open_task[t] = False
        return float(total)

    @classmethod
    def from_scenario(cls, scenario, task_ids: Optional[Iterable[int]] = None, **kw):
        tasks = _filter_tasks(scenario.tasks, task_ids)
        return cls(scenario.agents, tasks, **kw)


class CallableValue(CharacteristicFunction):
    """Wrap any ``f(frozenset_of_player_indices) -> float`` as a cached game.

    Handy for unit tests, toy games, and for reusing the samplers on the SVCA
    coalition cost J(S) from Eq. (11).
    """

    def __init__(self, n_players: int, fn: Callable[[frozenset], float],
                 player_ids: Optional[Sequence[int]] = None):
        super().__init__(list(range(n_players)) if player_ids is None else player_ids)
        self._fn = fn

    def _evaluate(self, members: np.ndarray) -> float:
        return float(self._fn(frozenset(int(i) for i in members)))


def _filter_tasks(tasks: Sequence[Task], task_ids: Optional[Iterable[int]]) -> List[Task]:
    if task_ids is None:
        return list(tasks)
    keep = set(task_ids)
    return [t for t in tasks if t.task_id in keep]


def tasks_completed_in(allocation) -> List[int]:
    """Task ids that an :class:`AllocationResult` actually assigns to some agent."""
    return sorted({t for ts in allocation.assignments.values() for t in ts})

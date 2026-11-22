"""Robots choose their team: hedonic coalition formation under a reward-sharing rule.

Stage 1 (SVCA, GA, ...) proposes a partition of the robots and tasks into teams
(clusters). Here the robots react to it. Each team is a cooperative game: its value
V(C) is the FacilityLocation value of the team's robots on the team's tasks (one common
horizon H for every team), and that value is split among the team's robots with a
REWARD RULE:

    "equal"       V(C) / |C|                                (no notion of contribution)
    "shapley"     classical Shapley value of the team game  (alpha = 0)
    "preference"  preference-aware Shapley value, alpha > 0 (Eq. 9, preference graph W)

Robot dynamics (best-response / "switch" dynamics of a hedonic game):
    In every round, robots are visited in a random order. Robot i compares
      * its payoff in its current team,
      * the payoff it WOULD get after joining each other team (that team's game with i added),
      * staying idle (payoff 0, no tasks),
    and moves to the best option if it beats its current payoff by more than a small switching
    threshold (``min_gain_abs + min_gain_rel * |current payoff|``). The last robot of a team that
    still has tasks may not leave, so every task stays covered (same rule as SVCA's constraint 6).
    Rounds repeat until no robot moves (converged), the partition repeats (cycle) or
    ``max_rounds`` is reached.

Stability (the definition used in the report):
    A partition is NASH STABLE when no robot can raise its own payoff by moving to another
    team or going idle. ``initial_unstable_robots`` counts the robots that want to move away
    from the Stage-1 partition; ``converged`` says whether the dynamics reached a Nash-stable one.

Exact Shapley values are used for small teams (no Monte-Carlo noise that could trigger
spurious moves); larger teams fall back to sampling with a fixed seed per team.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.game_theory.characteristic import FacilityLocationValue, RoutingValue, natural_horizon
from src.game_theory.exact import exact_preference_shapley, exact_shapley
from src.game_theory.fairness import gini
from src.game_theory.mcmc import glauber_preference_shapley
from src.game_theory.preference import PreferenceGraph
from src.game_theory.sampling import permutation_shapley

RULES = ("equal", "shapley", "preference")
VALUE_TYPES = {"facility": (FacilityLocationValue, 1.25), "routing": (RoutingValue, 2.0)}
IDLE = -1


@dataclass
class TeamChoiceConfig:
    rule: str = "preference"
    alpha: float = 1.5                 # used by rule="preference"
    value_type: str = "routing"        # team value V(C): "routing" (route + battery aware) or "facility"
    max_rounds: int = 30
    min_gain_abs: float = 1e-6         # a move must beat the current payoff by this much ...
    min_gain_rel: float = 0.01         # ... plus this fraction of it (switching cost, absorbs MC noise)
    allow_idle: bool = True
    exact_preference_up_to: int = 8    # exact preference-aware Shapley for teams up to this size
    exact_shapley_up_to: int = 12
    num_samples: int = 400             # sampling budget for larger teams
    burn_in: int = 100
    max_log_gamma: Optional[float] = 3.0
    seed: Optional[int] = 42

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {self.rule!r}")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        if self.value_type not in VALUE_TYPES:
            raise ValueError(f"value_type must be one of {sorted(VALUE_TYPES)}")


@dataclass
class TeamChoiceResult:
    rule: str
    alpha: float
    initial_clusters: List[Dict[str, List[int]]]
    final_clusters: List[Dict[str, List[int]]]     # same order as initial; idle robots are listed separately
    idle_robots: List[int]
    moves: List[Dict[str, Any]]                     # round, robot, from, to (IDLE = -1), payoffs, gain
    rounds: int
    converged: bool
    cycled: bool
    initial_unstable_robots: List[int]
    initial_payoffs: Dict[int, float]
    final_payoffs: Dict[int, float]
    initial_value: float                            # sum of team values before / after
    final_value: float
    initial_gini: float
    final_gini: float
    games_solved: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_moves(self) -> int:
        return len(self.moves)

    @property
    def robots_moved(self) -> List[int]:
        return sorted({m["robot"] for m in self.moves})


class TeamGame:
    """Payoff oracle: shares of every robot in any (robots, tasks) team under one reward rule."""

    def __init__(self, scenario, config: TeamChoiceConfig, preference: Optional[PreferenceGraph] = None):
        self.cfg = config
        self.agents = {a.agent_id: a for a in scenario.agents}
        self.tasks = {t.task_id: t for t in scenario.tasks}
        self.value_cls, factor = VALUE_TYPES[config.value_type]
        self.horizon = natural_horizon(scenario.agents, scenario.tasks, factor)
        self.pref = preference or PreferenceGraph.from_agents(
            scenario.agents, seed=config.seed, max_log_gamma=config.max_log_gamma)
        self.index = {a.agent_id: i for i, a in enumerate(scenario.agents)}
        self._cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[float, Dict[int, float]]] = {}

    @property
    def games_solved(self) -> int:
        return len(self._cache)

    def solve(self, robots: Sequence[int], tasks: Sequence[int]) -> Tuple[float, Dict[int, float]]:
        """(team value V(C), {robot: share}) for the team, cached."""
        key = (tuple(sorted(robots)), tuple(sorted(tasks)))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        ids = list(key[0])
        if not ids or not key[1]:
            out = (0.0, {r: 0.0 for r in ids})
            self._cache[key] = out
            return out
        value = self.value_cls([self.agents[r] for r in ids], [self.tasks[t] for t in key[1]],
                                      horizon=self.horizon)
        v_total = float(value.grand_coalition_value())
        n, cfg = len(ids), self.cfg
        seed = None if cfg.seed is None else (cfg.seed * 1_000_003 + hash(key) % 1_000_003) % (2 ** 32)
        if n == 1 or cfg.rule == "equal":
            phi = np.full(n, v_total / n)
        elif cfg.rule == "shapley" or cfg.alpha == 0.0:
            phi = (exact_shapley(value) if n <= cfg.exact_shapley_up_to
                   else permutation_shapley(value, cfg.num_samples, seed=seed).phi)
        else:
            sel = [self.index[r] for r in ids]
            sub = PreferenceGraph(self.pref.W[np.ix_(sel, sel)], player_ids=ids, gamma_scale=self.pref.gamma_scale)
            if n <= cfg.exact_preference_up_to:
                phi = exact_preference_shapley(value, sub, cfg.alpha)
            else:
                phi = glauber_preference_shapley(value, sub, cfg.alpha, cfg.num_samples, cfg.burn_in,
                                                 None, seed).phi
        out = (v_total, {r: float(p) for r, p in zip(ids, phi)})
        self._cache[key] = out
        return out


def _payoffs(game: TeamGame, teams: List[Dict[str, Any]], idle: set, all_robots: Sequence[int]) -> Dict[int, float]:
    pay = {r: 0.0 for r in all_robots}
    for team in teams:
        if team["agents"]:
            pay.update(game.solve(sorted(team["agents"]), team["tasks"])[1])
    for r in idle:
        pay[r] = 0.0
    return pay


def _best_option(game: TeamGame, teams, idle, robot: int, where: int, current: float, cfg: TeamChoiceConfig):
    """Best alternative for ``robot`` (team index or IDLE) and its payoff; None if staying is best."""
    best_to, best_pay = None, current + cfg.min_gain_abs + cfg.min_gain_rel * abs(current)
    if where != IDLE:
        team = teams[where]
        if team["tasks"] and len(team["agents"]) == 1:      # last robot of a team with tasks must stay
            return None, current
    for k, team in enumerate(teams):
        if k == where:
            continue
        p = game.solve(sorted(team["agents"] | {robot}), team["tasks"])[1][robot]
        if p > best_pay or (best_to is not None and p == best_pay and k < best_to):
            best_to, best_pay = k, p
    if cfg.allow_idle and where != IDLE and 0.0 > best_pay:
        best_to, best_pay = IDLE, 0.0
    return best_to, best_pay


def _signature(teams, idle) -> Tuple:
    return tuple(tuple(sorted(t["agents"])) for t in teams) + (tuple(sorted(idle)),)


def choose_teams(scenario, clusters: Optional[Sequence[Mapping[str, Sequence[int]]]],
                 config: Optional[TeamChoiceConfig] = None,
                 preference: Optional[PreferenceGraph] = None) -> TeamChoiceResult:
    """Run the robots' team-choice dynamics starting from the Stage-1 ``clusters``.

    ``clusters`` is a list of ``{"agents": [...], "tasks": [...]}`` (e.g.
    ``AllocationResult.metadata["clusters"]``); None means one team with everything.
    Tasks never move: a team is defined by its tasks, robots choose which team to serve.
    Robots not in any cluster start idle.
    """
    cfg = config or TeamChoiceConfig()
    game = TeamGame(scenario, cfg, preference)
    robots = [a.agent_id for a in scenario.agents]
    if not clusters:
        clusters = [{"agents": robots, "tasks": [t.task_id for t in scenario.tasks]}]
    teams = [{"agents": set(int(a) for a in c["agents"]), "tasks": sorted(int(t) for t in c["tasks"])}
             for c in clusters]
    placed = set().union(*(t["agents"] for t in teams)) if teams else set()
    idle = set(robots) - placed
    where = {r: k for k, t in enumerate(teams) for r in t["agents"]}
    where.update({r: IDLE for r in idle})
    initial = [{"agents": sorted(t["agents"]), "tasks": list(t["tasks"])} for t in teams]

    def team_values():
        return sum(game.solve(sorted(t["agents"]), t["tasks"])[0] for t in teams if t["agents"])

    pay0 = _payoffs(game, teams, idle, robots)
    value0 = team_values()
    unstable = [r for r in robots if _best_option(game, teams, idle, r, where[r], pay0[r], cfg)[0] is not None]

    rng = np.random.default_rng(cfg.seed)
    moves: List[Dict[str, Any]] = []
    seen = {_signature(teams, idle)}
    converged = cycled = False
    rounds = 0
    for rounds in range(1, cfg.max_rounds + 1):
        moved = False
        for r in [robots[i] for i in rng.permutation(len(robots))]:
            src = where[r]
            current = (game.solve(sorted(teams[src]["agents"]), teams[src]["tasks"])[1][r]
                       if src != IDLE else 0.0)
            dst, new_pay = _best_option(game, teams, idle, r, src, current, cfg)
            if dst is None:
                continue
            if src == IDLE:
                idle.discard(r)
            else:
                teams[src]["agents"].discard(r)
            if dst == IDLE:
                idle.add(r)
            else:
                teams[dst]["agents"].add(r)
            where[r] = dst
            moves.append({"round": rounds, "robot": r, "from": src, "to": dst,
                          "payoff_before": round(current, 4), "payoff_after": round(new_pay, 4),
                          "gain": round(new_pay - current, 4)})
            moved = True
        if not moved:
            converged = True
            break
        sig = _signature(teams, idle)
        if sig in seen:
            cycled = True
            break
        seen.add(sig)

    pay1 = _payoffs(game, teams, idle, robots)
    return TeamChoiceResult(
        rule=cfg.rule, alpha=cfg.alpha if cfg.rule == "preference" else 0.0,
        initial_clusters=initial,
        final_clusters=[{"agents": sorted(t["agents"]), "tasks": list(t["tasks"])} for t in teams],
        idle_robots=sorted(idle), moves=moves, rounds=rounds, converged=converged, cycled=cycled,
        initial_unstable_robots=unstable,
        initial_payoffs=pay0, final_payoffs=pay1,
        initial_value=float(value0), final_value=float(team_values()),
        initial_gini=gini([pay0[r] for r in robots]), final_gini=gini([pay1[r] for r in robots]),
        games_solved=game.games_solved,
        metadata={"value_type": cfg.value_type, "horizon": game.horizon, "gamma_scale": game.pref.gamma_scale,
                  "min_gain_rel": cfg.min_gain_rel, "max_rounds": cfg.max_rounds},
    )

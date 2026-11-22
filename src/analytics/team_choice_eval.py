"""Stage 1 -> robots choose teams -> re-plan: measures how the reward rule changes the allocation.

``evaluate_team_choice`` takes a scenario and a Stage-1 partition, lets the robots react
under a reward rule (``src.game_theory.team_choice``), re-plans every team's routes with
the shared sub-solver, and returns one flat record with the numbers the report uses:

* stability: how many robots wanted to leave the Stage-1 teams, moves, rounds, convergence
* allocation: cost A(U) before / after (both re-planned with the SAME sub-solver, so the
  difference is due to the new teams only) and how many tasks changed robot
* fairness: Gini of the robots' payoffs before / after, total team value before / after
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.allocation.evaluator import AllocationEvaluator
from src.allocation.insertion import solve_subproblem
from src.game_theory.preference import PreferenceGraph
from src.game_theory.team_choice import TeamChoiceConfig, TeamChoiceResult, choose_teams

# (label, rule, alpha) used by the experiment script and the viewer.
DEFAULT_RULES = (("equal", "equal", 0.0), ("shapley", "shapley", 0.0),
                 ("preference_a1.5", "preference", 1.5), ("preference_a3", "preference", 3.0))


def replan(scenario, clusters: Sequence[Mapping[str, Sequence[int]]], idle: Sequence[int] = ()) -> Dict[int, List[int]]:
    """Routes for a fixed partition: every team solved with the shared greedy + local-search sub-solver."""
    ev = AllocationEvaluator(scenario)
    routes: Dict[int, List[int]] = {a: [] for a in ev.agent_ids}
    for c in clusters:
        if c["agents"] and c["tasks"]:
            routes.update(solve_subproblem(ev, list(c["agents"]), list(c["tasks"])))
    for r in idle:
        routes[r] = []
    return routes


def _owner(routes: Mapping[int, Sequence[int]]) -> Dict[int, int]:
    return {t: a for a, r in routes.items() for t in r}


def evaluate_team_choice(scenario, clusters, rule: str, alpha: float = 0.0, seed: Optional[int] = 42,
                         preference: Optional[PreferenceGraph] = None,
                         **config_overrides: Any) -> Dict[str, Any]:
    """Run team choice from ``clusters`` and return metrics plus the raw result under ``"result"``."""
    cfg = TeamChoiceConfig(rule=rule, alpha=alpha, seed=seed, **config_overrides)
    t0 = time.perf_counter()
    result: TeamChoiceResult = choose_teams(scenario, clusters, cfg, preference)
    t_choice = time.perf_counter() - t0
    ev = AllocationEvaluator(scenario)
    before = replan(scenario, result.initial_clusters)
    after = replan(scenario, result.final_clusters, result.idle_robots)
    cost_before, cost_after = ev.total_cost(before), ev.total_cost(after)
    own_b, own_a = _owner(before), _owner(after)
    n = len(scenario.agents)
    return {
        "rule": rule,
        "alpha": result.alpha,
        "num_robots": n,
        "num_tasks": len(scenario.tasks),
        "num_teams": len(result.initial_clusters),
        "unstable_robots": len(result.initial_unstable_robots),
        "unstable_fraction": len(result.initial_unstable_robots) / n if n else 0.0,
        "moves": result.num_moves,
        "robots_moved": len(result.robots_moved),
        "rounds": result.rounds,
        "converged": result.converged,
        "cycled": result.cycled,
        "idle_robots": len(result.idle_robots),
        "cost_before": round(cost_before, 2),
        "cost_after": round(cost_after, 2),
        "cost_change_pct": round(100.0 * (cost_after / cost_before - 1.0), 3) if cost_before > 0 else 0.0,
        "tasks_reassigned": sum(1 for t in own_b if own_a.get(t) != own_b[t]),
        "value_before": round(result.initial_value, 2),
        "value_after": round(result.final_value, 2),
        "gini_before": round(result.initial_gini, 4),
        "gini_after": round(result.final_gini, 4),
        "games_solved": result.games_solved,
        "runtime_seconds": round(t_choice, 4),
        "result": result,
        "routes_before": before,
        "routes_after": after,
    }

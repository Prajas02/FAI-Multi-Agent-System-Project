"""Interactive end-to-end pipeline view (scenario -> Stage 1 allocation -> Stage 2 rewards).

``build_pipeline_data`` runs the real solvers once and collects everything needed to *see*
what they did: SVCA Shapley values and clusters, every robot's route with an exact
per-leg timeline, the preference graph, the Glauber MCMC convergence and the reward
shares for a sweep of alpha. ``render_html`` embeds that data in a single self-contained
HTML page (no network access needed) that can be opened in any browser.

The per-leg timeline is computed with the same formulas as
``AllocationEvaluator.route_stats`` (velocity decays with the energy ratio, energy for
motion + execution is charged per task), so the animation matches the reported A(U).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.allocation.evaluator import AllocationEvaluator
from src.allocation.ga_clustering import GAClusteringAllocator, GAConfig
from src.allocation.ga_paper import PaperGAAllocator, PaperGAConfig
from src.allocation.greedy import GreedyAllocator
from src.allocation.milp import MILPAllocator, MILPConfig
from src.allocation.svca import SVCAAllocator, SVCAConfig
from src.allocation.svca_paper import PaperSVCAAllocator, PaperSVCAConfig
from src.analytics.team_choice_eval import DEFAULT_RULES, evaluate_team_choice
from src.core.types import AllocationResult
from src.game_theory.clustered import solve_by_clusters
from src.game_theory.mcmc import glauber_preference_shapley
from src.game_theory.solver import PreferenceAwareShapleySolver
from src.simulation.generator import ScenarioData
from src.simulation.kinematics import KinematicsModel

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("pipeline_template.html")

# Display name -> (allocator factory, one-line description shown in the page)
ALGORITHMS: Dict[str, Any] = {
    "SVCA (paper)": (
        lambda seed: PaperSVCAAllocator(PaperSVCAConfig.from_yaml(
            str(REPO_ROOT / "configs" / "svca_paper.yaml"), seed=seed)),
        "Shapley Value Clustering (Martin et al. 2023): rank robots and tasks by Shapley value, "
        "grow clusters round by round, keep the round with the best V(C), then solve each cluster.",
    ),
    "SVCA-ext": (
        lambda seed: SVCAAllocator(SVCAConfig(seed=seed)),
        "Extended SVCA: Shapley values of the true coalition cost J, cost-aware cluster building "
        "plus a refinement pass, then the same per-cluster sub-solver.",
    ),
    "GA (paper)": (
        lambda seed: PaperGAAllocator(PaperGAConfig(seed=seed)),
        "Genetic algorithm clustering with the paper's fitness V(C).",
    ),
    "GA-cost": (
        lambda seed: GAClusteringAllocator(GAConfig(seed=seed)),
        "Genetic algorithm clustering that directly minimises the true allocation cost A(U).",
    ),
    "Greedy": (
        lambda seed: GreedyAllocator(),
        "Baseline: repeatedly insert the cheapest (robot, task, position) with no clustering.",
    ),
}
MILP_NAME = "MILP"
MILP_DESCRIPTION = ("Exact mixed-integer program over all robots and tasks (HiGHS), warm-started "
                    "from Greedy + local search. Reference optimum when it proves optimality.")


def _r(x: float, nd: int = 3) -> float:
    return round(float(x), nd)


def build_timeline(scenario: ScenarioData, routes: Dict[int, List[int]],
                   evaluator: AllocationEvaluator) -> Dict[str, List[Dict[str, float]]]:
    """Exact per-leg schedule for every robot, mirroring ``AllocationEvaluator.route_stats``."""
    agents = {a.agent_id: a for a in scenario.agents}
    tasks = {t.task_id: t for t in scenario.tasks}
    bm = evaluator.battery_model
    out: Dict[str, List[Dict[str, float]]] = {}
    for aid, route in routes.items():
        agent = agents[aid]
        energy, now = agent.current_energy, 0.0
        x, y = agent.position.x, agent.position.y
        legs = []
        for tid in route:
            task = tasks[tid]
            d = math.hypot(task.location.x - x, task.location.y - y)
            ratio = energy / agent.energy_capacity if agent.energy_capacity > 0 else 1.0
            v = KinematicsModel.compute_effective_velocity(agent, ratio)
            travel = d / v if v > 0 else 0.0
            step = bm.compute_motion_energy(d, v) + bm.compute_execution_energy(task.execution_time)
            short = energy < step
            energy = max(0.0, energy - step)
            legs.append({
                "task": tid, "x0": _r(x, 2), "y0": _r(y, 2),
                "x1": _r(task.location.x, 2), "y1": _r(task.location.y, 2),
                "t0": _r(now), "ta": _r(now + travel), "tf": _r(now + travel + task.execution_time),
                "v": _r(v), "energy": _r(energy, 2), "short": short,
            })
            now += travel + task.execution_time
            x, y = task.location.x, task.location.y
        out[str(aid)] = legs
    return out


def _algorithm_record(name: str, description: str, result: AllocationResult, wall: float,
                      scenario: ScenarioData, utility: np.ndarray, agent_index: Dict[int, int],
                      task_index: Dict[int, int]) -> Dict[str, Any]:
    ev = AllocationEvaluator(scenario)
    routes = {a: list(r) for a, r in result.assignments.items()}
    meta = result.metadata
    timeline = build_timeline(scenario, routes, ev)
    # Work each robot actually does in THIS plan, measured with the Stage-2 task utility u[r, t].
    contribution = {
        str(a): _r(sum(utility[agent_index[a], task_index[t]] for t in r), 2) for a, r in routes.items()
    }
    rec: Dict[str, Any] = {
        "name": name,
        "solver_name": result.solver_name,
        "description": description,
        "cost": _r(result.total_cost, 2),
        "priority_cost": meta.get("priority_completion_cost"),
        "travel_cost": meta.get("travel_cost"),
        "energy_penalty": meta.get("energy_penalty"),
        "valid": bool(meta.get("valid", True)),
        "runtime": _r(wall, 4),
        "routes": {str(a): r for a, r in routes.items()},
        "timeline": timeline,
        "makespan": _r(max((legs[-1]["tf"] for legs in timeline.values() if legs), default=0.0), 2),
        "distance": {str(a): _r(d, 2) for a, d in result.agent_distances.items()},
        "energy_used": {str(a): _r(e, 2) for a, e in result.agent_energy_used.items()},
        "contribution": contribution,
        "clusters": meta.get("clusters"),
        "shapley": {k: _r(v, 3) for k, v in meta["shapley"].items()} if "shapley" in meta else None,
        "v_curve": meta.get("v_curve"),
        "extra": {k: meta[k] for k in ("q_used", "q_initial", "num_clusters", "proved_optimal",
                                       "milp_status", "mip_gap", "plan_source") if k in meta},
    }
    return rec


def _reward_section(scenario: ScenarioData, results: Dict[str, AllocationResult], cfg: Dict[str, Any],
                    alphas: Sequence[float], seed: int) -> Dict[str, Any]:
    """Stage-2 rewards exactly as the benchmark computes them (``reward.decomposition: clusters``):
    one preference-aware Shapley game per Stage-1 cluster, merged per robot."""
    num_samples = int(cfg.get("num_samples_Tw", 1000))
    burn_in = int(cfg.get("burn_in", 200))
    thin = cfg.get("thin")
    max_log_gamma = cfg.get("max_log_gamma", 3.0)
    alpha0 = float(cfg.get("preference_sensitivity_alpha", 1.5))
    alpha_grid = sorted(set([0.0, alpha0, *alphas]))

    # Global game: used for the preference graph and to show the MCMC sampler converging.
    base = PreferenceAwareShapleySolver.from_scenario(
        scenario, alpha=alpha0, seed=seed, max_log_gamma=max_log_gamma,
        num_samples=num_samples, burn_in=burn_in, thin=thin)
    value, pref = base.value, base.pref
    ids = [int(i) for i in value.player_ids]
    everyone = [{"agents": [a.agent_id for a in scenario.agents], "tasks": [t.task_id for t in scenario.tasks]}]

    by_algorithm: Dict[str, Any] = {}
    for name, result in results.items():
        clusters = result.metadata.get("clusters") or everyone
        sweep = []
        for alpha in alpha_grid:
            dist = solve_by_clusters(
                scenario, clusters=clusters, allocation=result, alpha=alpha, preference=pref,
                num_samples=num_samples, burn_in=burn_in, thin=thin, seed=seed)
            sweep.append({"alpha": alpha, "payoffs": [_r(dist.payoffs[i], 2) for i in ids],
                          "gini": _r(dist.gini_index, 4), "value": _r(dist.total_coalition_value, 2)})
        by_algorithm[name] = {"sweep": sweep, "num_games": len(clusters)}

    # Convergence of the Glauber estimate at the configured alpha (running mean of the draws).
    mc = glauber_preference_shapley(value, pref, alpha0, num_samples, burn_in, thin, seed)
    draws = mc.marginals
    running = np.cumsum(draws, axis=0) / np.arange(1, len(draws) + 1)[:, None]
    idx = np.unique(np.linspace(0, len(draws) - 1, min(120, len(draws))).astype(int))

    return {
        "player_ids": ids,
        "alpha_default": alpha0,
        "alphas": alpha_grid,
        "global_value": _r(value.grand_coalition_value(), 2),
        "by_algorithm": by_algorithm,
        "W": [[_r(w, 4) for w in row] for row in pref.W.tolist()],
        "gamma_scale": _r(pref.gamma_scale, 5),
        "trace": {"step": [int(i + 1) for i in idx],
                  "running": [[_r(v, 2) for v in running[i]] for i in idx],
                  "upsilon": [_r(mc.upsilon_trace[i], 4) for i in idx],
                  "acceptance": _r(mc.acceptance_rate, 3), "thin": mc.thin, "burn_in": burn_in},
        "utility_note": "u[r,t] = max(0, priority * (H - eta_rt) - lambda_r * d_rt), the per-task "
                        "value a robot creates in the FacilityLocation game.",
    }


def _team_choice_section(scenario: ScenarioData, results: Dict[str, AllocationResult],
                         seed: int) -> Dict[str, Any]:
    """Robots choose their team under each reward rule, starting from every allocator's clusters."""
    ev = AllocationEvaluator(scenario)
    out: Dict[str, Any] = {"rules": [{"label": l, "rule": r, "alpha": a} for l, r, a in DEFAULT_RULES], "by_algorithm": {}}
    for name, result in results.items():
        per = {}
        for label, rule, alpha in DEFAULT_RULES:
            rec = evaluate_team_choice(scenario, result.metadata.get("clusters"), rule, alpha, seed=seed)
            res = rec["result"]
            per[label] = {
                "initial": res.initial_clusters, "final": res.final_clusters, "idle": res.idle_robots,
                "moves": res.moves, "unstable": res.initial_unstable_robots,
                "rounds": res.rounds, "converged": res.converged, "cycled": res.cycled,
                "payoff_before": {str(k): _r(v, 2) for k, v in res.initial_payoffs.items()},
                "payoff_after": {str(k): _r(v, 2) for k, v in res.final_payoffs.items()},
                "value_before": rec["value_before"], "value_after": rec["value_after"],
                "gini_before": rec["gini_before"], "gini_after": rec["gini_after"],
                "cost_before": rec["cost_before"], "cost_after": rec["cost_after"],
                "cost_change_pct": rec["cost_change_pct"], "tasks_reassigned": rec["tasks_reassigned"],
                "timeline_after": build_timeline(scenario, rec["routes_after"], ev),
            }
        out["by_algorithm"][name] = per
    return out


def build_pipeline_data(scenario: ScenarioData, scenario_name: str, seed: int = 42,
                        include_milp: bool = False, milp_time_limit: float = 20.0,
                        alphas: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0),
                        config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run every allocator plus the reward stage on one scenario and collect the view data."""
    import time
    import yaml

    config_path = config_path or REPO_ROOT / "configs" / "default_config.yaml"
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    reward_cfg = config.get("mcmc_reward", {}) or {}

    # Stage-2 task utility, used to measure what each robot actually contributes in a plan.
    probe = PreferenceAwareShapleySolver.from_scenario(scenario, alpha=0.0, seed=seed)
    utility = np.asarray(probe.value.utility)
    agent_index = {a.agent_id: i for i, a in enumerate(scenario.agents)}
    task_index = {t.task_id: j for j, t in enumerate(scenario.tasks)}

    entries = [(n, f(seed), d) for n, (f, d) in ALGORITHMS.items()]
    if include_milp:
        entries.append((MILP_NAME, MILPAllocator(MILPConfig(time_limit=milp_time_limit)), MILP_DESCRIPTION))

    algorithms, results = [], {}
    for name, allocator, description in entries:
        t0 = time.perf_counter()
        result = allocator.solve(scenario)
        wall = time.perf_counter() - t0
        results[name] = result
        algorithms.append(_algorithm_record(name, description, result, wall, scenario,
                                            utility, agent_index, task_index))

    reward = _reward_section(scenario, results, reward_cfg, alphas, seed)
    team_choice = _team_choice_section(scenario, results, seed)

    return {
        "scenario": {
            "name": scenario_name, "seed": scenario.seed,
            "width": scenario.map_width, "height": scenario.map_height,
            "agents": [{"id": a.agent_id, "x": a.position.x, "y": a.position.y, "v": a.velocity,
                        "capacity": a.energy_capacity, "energy": a.current_energy,
                        "penalty": a.usage_penalty} for a in scenario.agents],
            "tasks": [{"id": t.task_id, "x": t.location.x, "y": t.location.y,
                       "exec": t.execution_time, "priority": t.priority} for t in scenario.tasks],
        },
        "algorithms": algorithms,
        "reward": reward,
        "team_choice": team_choice,
        "run": {"seed": seed, "milp": include_milp},
    }


def render_html(data: Dict[str, Any], output: Path) -> Path:
    """Write the self-contained interactive page."""
    template = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("/*__PIPELINE_DATA__*/null", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output

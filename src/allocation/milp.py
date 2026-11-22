"""Exact MILP for Optimization Problem 1 (proposal Eq. 4-7), solved with HiGHS via scipy.

Why scipy and not PuLP: PuLP >= 4 no longer bundles a solver (PULP_CBC_CMD was removed and CBC must be
installed separately), whereas ``scipy.optimize.milp`` ships HiGHS and is already a project dependency.

Model (routing with big-M time linking; no return trip, routes end at the last task)
    x[r,i,j] in {0,1}   robot r drives from i to j.   i = start position of r, or a task; j = a task.
    C[t]     >= 0       completion time of task t (execution included).

    min   sum_t delta_t * C[t]  +  sum_r lambda_r * sum_{i,j} d_ij * x[r,i,j]

    (a) every task visited exactly once:               sum_{r,i} x[r,i,t] = 1
    (b) a task has a successor only if it is visited:  sum_k x[r,t,k] <= sum_i x[r,i,t]
    (c) each robot leaves its start at most once:      sum_j x[r,start,j] <= 1
    (d) time linking:  C[j] >= C[i] + d_ij / v_r + tau_j   if x[r,i,j] = 1   (big-M)
                       C[j] >= d_start,j / v_r + tau_j     if x[r,start,j] = 1
        Execution times are positive, so a cycle would need C[i] > C[i]: sub-tours are impossible.
    (e) energy (Eq. 7 as a HARD constraint):  sum x[r,i,j] * (e_r d_ij + c_exec tau_j) <= E_r

Known differences from the simulator (``AllocationEvaluator``), reported in the result metadata:
  * The simulator slows a robot down once its battery falls below 15%. The MILP uses the nominal speed.
    Every MILP plan is therefore RE-SCORED with the true evaluator; ``metadata['model_gap']`` is
    (true cost - MILP objective). It is 0 unless a battery gets that low.
  * Energy is a hard constraint here, whereas the simulator adds a penalty. A scenario in which no
    assignment is energy-feasible is reported as infeasible (the allocator then falls back to the
    greedy plan and says so in the metadata).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.allocation.base import Allocator
from src.allocation.evaluator import AllocationEvaluator, Routes
from src.allocation.insertion import greedy_insertion, local_search
from src.core.types import AllocationResult
from src.simulation.generator import ScenarioData

_STATUS = {0: "optimal", 1: "time_limit", 2: "infeasible", 3: "unbounded", 4: "error"}


@dataclass
class MILPConfig:
    time_limit: float = 60.0        # seconds per MILP solve
    mip_rel_gap: float = 1e-4       # stop when (incumbent - bound) / incumbent <= this
    fallback_local_search: bool = True


@dataclass
class MILPOutcome:
    routes: Optional[Routes]        # None if no feasible solution was found
    status: str                     # optimal | time_limit | infeasible | unbounded | error
    objective: Optional[float]      # MILP objective of the returned routes (nominal-speed model)
    bound: Optional[float]          # best proven lower bound
    gap: Optional[float]            # relative gap reported by HiGHS
    runtime: float
    num_vars: int
    num_constraints: int


def solve_milp(
    ev: AllocationEvaluator,
    agent_ids: Sequence[int],
    task_ids: Sequence[int],
    time_limit: float = 60.0,
    mip_rel_gap: float = 1e-4,
    cutoff: Optional[float] = None,
) -> MILPOutcome:
    """Solve Optimization Problem 1 exactly for the given robots and tasks.

    ``cutoff``: an upper bound on the optimal objective, e.g. the cost of a known feasible plan. It is added
    as the valid cut  objective <= cutoff  (it can never remove the optimum) and lets HiGHS prune early.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    t0 = time.perf_counter()
    agents = sorted(agent_ids)
    tasks = sorted(task_ids)
    n, m = len(tasks), len(agents)
    if n == 0:
        return MILPOutcome({a: [] for a in agents}, "optimal", 0.0, 0.0, 0.0, 0.0, 0, 0)
    if m == 0:
        return MILPOutcome(None, "infeasible", None, None, None, 0.0, 0, 0)

    bm = ev.battery_model
    START = -1                                           # pseudo task index for "robot start position"

    def dist(a: int, i: int, j: int) -> float:           # a = agent id, i = local index or START, j = local index
        return ev.d_at[agents[a]][tasks[j]] if i == START else ev.d_tt[tasks[i]][tasks[j]]

    # ---- variables: x[a,i,j] first, then C[j] -------------------------------------------
    xcol: Dict[Tuple[int, int, int], int] = {}
    for a in range(m):
        for i in [START] + list(range(n)):
            for j in range(n):
                if i != j:
                    xcol[(a, i, j)] = len(xcol)
    n_x = len(xcol)
    ccol = {j: n_x + j for j in range(n)}
    n_var = n_x + n

    speed = [ev.agents[a].velocity for a in agents]
    lam = [ev.agents[a].usage_penalty for a in agents]
    delta = [ev.tasks[t].priority for t in tasks]
    tau = [ev.tasks[t].execution_time for t in tasks]
    cap = [ev.agents[a].current_energy for a in agents]
    e_move = [bm.compute_motion_energy(1.0, speed[a]) for a in range(m)]   # energy per metre (linear in d)
    e_exec = [bm.compute_execution_energy(tau[j]) for j in range(n)]

    # ---- bounds on completion times (for big-M) ----------------------------------------
    d_all = [dist(a, START, j) for a in range(m) for j in range(n)]
    d_all += [dist(0, i, j) for i in range(n) for j in range(n) if i != j]
    d_max = max(d_all) if d_all else 0.0
    crude = sum(tau) + n * d_max / min(speed)            # no route can finish later than this
    # Tighter, per robot: under the hard energy budget a robot cannot drive farther than E_r / e_r metres.
    h_robot = [
        min(crude, sum(tau) + cap[a] / (e_move[a] * speed[a])) if e_move[a] > 0 else crude
        for a in range(m)
    ]
    horizon = max(h_robot)
    w_start = [[dist(a, START, j) / speed[a] + tau[j] for j in range(n)] for a in range(m)]
    lb_c = [min(w_start[a][j] for a in range(m)) for j in range(n)]

    c_obj = np.zeros(n_var)
    for j in range(n):
        c_obj[ccol[j]] = delta[j]
    for (a, i, j), col in xcol.items():
        c_obj[col] = lam[a] * dist(a, i, j)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    lo: List[float] = []
    hi: List[float] = []
    inf = np.inf

    def add_row(entries: List[Tuple[int, float]], lower: float, upper: float) -> None:
        r = len(lo)
        for col, v in entries:
            rows.append(r)
            cols.append(col)
            vals.append(v)
        lo.append(lower)
        hi.append(upper)

    # (a) each task exactly once
    for j in range(n):
        add_row([(xcol[(a, i, j)], 1.0) for a in range(m) for i in [START] + list(range(n)) if i != j], 1.0, 1.0)
    # (b) successor only if visited, (c) at most one departure from the start
    for a in range(m):
        for j in range(n):
            ent = [(xcol[(a, j, k)], 1.0) for k in range(n) if k != j]
            ent += [(xcol[(a, i, j)], -1.0) for i in [START] + list(range(n)) if i != j]
            add_row(ent, -inf, 0.0)
        add_row([(xcol[(a, START, j)], 1.0) for j in range(n)], -inf, 1.0)
    # (d) time linking (big-M)
    for (a, i, j), col in xcol.items():
        if i == START:
            w = w_start[a][j]
            big = max(0.0, w - lb_c[j])
            add_row([(ccol[j], 1.0), (col, -big)], w - big, inf)
        else:
            w = dist(a, i, j) / speed[a] + tau[j]
            big = w + h_robot[a] - lb_c[j]                    # C_i <= h_robot[a] whenever robot a visits i
            add_row([(ccol[j], 1.0), (ccol[i], -1.0), (col, -big)], w - big, inf)
    # (e) energy budget
    for a in range(m):
        ent = [(col, e_move[a] * dist(a, i, j) + e_exec[j]) for (aa, i, j), col in xcol.items() if aa == a]
        add_row(ent, -inf, cap[a])

    if cutoff is not None and np.isfinite(cutoff):
        nz = np.nonzero(c_obj)[0]
        add_row([(int(k), float(c_obj[k])) for k in nz], -inf, float(cutoff) * (1.0 + 1e-7) + 1e-6)

    A = coo_matrix((vals, (rows, cols)), shape=(len(lo), n_var)).tocsr()
    integrality = np.zeros(n_var)
    integrality[:n_x] = 1
    lower = np.zeros(n_var)
    upper = np.ones(n_var)
    for j in range(n):
        lower[ccol[j]] = lb_c[j]
        upper[ccol[j]] = horizon

    res = milp(
        c_obj,
        constraints=LinearConstraint(A, np.array(lo), np.array(hi)),
        integrality=integrality,
        bounds=Bounds(lower, upper),
        options={"time_limit": float(time_limit), "mip_rel_gap": float(mip_rel_gap), "disp": False},
    )
    status = _STATUS.get(int(res.status), "error")
    bound = getattr(res, "mip_dual_bound", None)
    gap = getattr(res, "mip_gap", None)

    routes: Optional[Routes] = None
    objective: Optional[float] = None
    if res.x is not None:
        x = res.x
        succ: Dict[Tuple[int, int], int] = {}
        for (a, i, j), col in xcol.items():
            if x[col] > 0.5:
                succ[(a, i)] = j
        routes = {}
        for a in range(m):
            route: List[int] = []
            cur = START
            seen = set()
            while (a, cur) in succ and succ[(a, cur)] not in seen:
                cur = succ[(a, cur)]
                seen.add(cur)
                route.append(tasks[cur])
            routes[agents[a]] = route
        objective = float(res.fun)
    return MILPOutcome(
        routes=routes, status=status, objective=objective,
        bound=None if bound is None else float(bound), gap=None if gap is None else float(gap),
        runtime=time.perf_counter() - t0, num_vars=n_var, num_constraints=len(lo),
    )


def solve_cluster(
    ev: AllocationEvaluator,
    agent_ids: Sequence[int],
    task_ids: Sequence[int],
    subsolver: str = "heuristic",
    improve: bool = True,
    local_search_passes: int = 20,
    milp_time_limit: float = 30.0,
    milp_gap: float = 1e-4,
) -> Tuple[Routes, Dict[str, Any]]:
    """Solve one cluster with the chosen sub-solver. Returns (routes, info).

    subsolver = "heuristic": greedy insertion (+ local search if ``improve``)
    subsolver = "milp":      exact MILP; if it finds no solution, falls back to the heuristic and says so.
    """
    if subsolver == "milp":
        incumbent = greedy_insertion(ev, agent_ids, task_ids)
        if any(incumbent.values()):
            incumbent = local_search(ev, incumbent, max_passes=local_search_passes)
        out = solve_milp(ev, agent_ids, task_ids, milp_time_limit, milp_gap, cutoff=ev.total_cost(incumbent))
        info = {"status": out.status, "gap": out.gap, "milp_runtime": out.runtime, "fallback": False}
        if out.routes is not None:
            return out.routes, info
        info["fallback"] = True
        return incumbent, info
    elif subsolver != "heuristic":
        raise ValueError(f"unknown subsolver {subsolver!r}")
    else:
        info = {"status": "heuristic", "gap": None, "milp_runtime": 0.0, "fallback": False}
    routes = greedy_insertion(ev, agent_ids, task_ids)
    if improve and any(routes.values()):
        routes = local_search(ev, routes, max_passes=local_search_passes)
    return routes, info


class MILPAllocator(Allocator):
    """Centralised exact allocation over ALL robots and tasks (the optimal-assignment reference)."""

    name = "MILP"

    def __init__(self, config: Optional[MILPConfig] = None):
        self.config = config or MILPConfig()

    def solve(self, scenario: ScenarioData) -> AllocationResult:
        cfg = self.config
        t0 = time.perf_counter()
        ev = AllocationEvaluator(scenario)
        # Greedy+LS is the starting incumbent: its cost becomes a valid cut for the MILP, and its plan is
        # the answer if the MILP cannot improve on it within the time limit.
        incumbent = greedy_insertion(ev, ev.agent_ids, ev.task_ids)
        if cfg.fallback_local_search:
            incumbent = local_search(ev, incumbent)
        out = solve_milp(ev, ev.agent_ids, ev.task_ids, cfg.time_limit, cfg.mip_rel_gap,
                         cutoff=ev.total_cost(incumbent))

        from_milp = out.routes is not None
        routes = out.routes if from_milp else incumbent
        runtime = time.perf_counter() - t0

        result = ev.evaluate(
            routes,
            solver_name=self.name,
            runtime_seconds=runtime,
        )
        result.metadata.update({
            "plan_source": "milp" if from_milp else "greedy_incumbent",   # greedy_incumbent: MILP found nothing better
            "proved_optimal": out.status == "optimal",                      # True only if HiGHS proved optimality
            "milp_status": out.status,            # optimal | time_limit | infeasible | ...
            "milp_objective": out.objective,      # objective of the nominal-speed model
            "model_gap": None if out.objective is None else ev.total_cost(routes) - out.objective,
            "mip_gap": out.gap,
            "mip_bound": out.bound,
            "num_vars": out.num_vars,
            "num_constraints": out.num_constraints,
            "config": asdict(cfg),
        })
        return result

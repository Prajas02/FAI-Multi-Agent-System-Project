"""Route-building primitives shared by Greedy, SVCA and GA.

``solve_subproblem`` is the single sub-solver used inside every cluster, so any
difference between SVCA and GA clustering comes from the PARTITION, not from the solver.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from src.allocation.evaluator import AllocationEvaluator, Routes

_EPS = 1e-9


def greedy_insertion(
    ev: AllocationEvaluator,
    agent_ids: Sequence[int],
    task_ids: Iterable[int],
) -> Routes:
    """Priority-ordered cheapest insertion.

    Tasks are taken in descending priority delta_t (ties: lower task_id). Each task is
    inserted at the (agent, position) with the smallest increase in that agent's route
    cost. Because route cost includes the energy-deficit penalty, infeasible insertions
    are avoided whenever a feasible one exists.
    """
    agents = sorted(agent_ids)
    tasks = sorted(task_ids, key=lambda t: (-ev.tasks[t].priority, t))
    if tasks and not agents:
        raise ValueError("Cannot allocate tasks without at least one agent")

    routes: Routes = {a: [] for a in agents}
    cost: Dict[int, float] = {a: 0.0 for a in agents}

    for t in tasks:
        best = None  # (delta, agent, position, new_cost)
        for a in agents:
            seq = routes[a]
            for pos in range(len(seq) + 1):
                cand = seq[:pos] + [t] + seq[pos:]
                new_cost = ev.route_cost(a, cand)
                delta = new_cost - cost[a]
                if best is None or delta < best[0] - _EPS:
                    best = (delta, a, pos, new_cost)
        _, a, pos, new_cost = best
        routes[a].insert(pos, t)
        cost[a] = new_cost
    return routes


def local_search(
    ev: AllocationEvaluator,
    routes: Routes,
    max_passes: int = 20,
) -> Routes:
    """First-improvement local search: task relocation (intra/inter-route) + inter-route swap.

    Strictly decreases A(U) on every accepted move, so it terminates.
    """
    routes = {a: list(r) for a, r in routes.items()}
    agents = sorted(routes)
    cost = {a: ev.route_cost(a, routes[a]) for a in agents}

    for _ in range(max_passes):
        improved = False

        # --- relocate one task to its best position anywhere ---------------------
        for a in agents:
            i = 0
            while i < len(routes[a]):
                t = routes[a][i]
                src_wo = routes[a][:i] + routes[a][i + 1:]
                src_wo_cost = ev.route_cost(a, src_wo)
                best = None  # (delta, b, cand, cand_cost)
                for b in agents:
                    base = src_wo if b == a else routes[b]
                    for pos in range(len(base) + 1):
                        if b == a and pos == i:
                            continue
                        cand = base[:pos] + [t] + base[pos:]
                        c = ev.route_cost(b, cand)
                        if b == a:
                            delta = c - cost[a]
                        else:
                            delta = (c - cost[b]) + (src_wo_cost - cost[a])
                        if delta < -_EPS and (best is None or delta < best[0]):
                            best = (delta, b, cand, c)
                if best is not None:
                    _, b, cand, c = best
                    if b == a:
                        routes[a], cost[a] = cand, c
                    else:
                        routes[a], cost[a] = src_wo, src_wo_cost
                        routes[b], cost[b] = cand, c
                    improved = True
                i += 1

        # --- swap two tasks between different routes ------------------------------
        for ai in range(len(agents)):
            for bi in range(ai + 1, len(agents)):
                a, b = agents[ai], agents[bi]
                for i in range(len(routes[a])):
                    for j in range(len(routes[b])):
                        ra = list(routes[a])
                        rb = list(routes[b])
                        ra[i], rb[j] = rb[j], ra[i]
                        ca, cb = ev.route_cost(a, ra), ev.route_cost(b, rb)
                        if (ca + cb) - (cost[a] + cost[b]) < -_EPS:
                            routes[a], routes[b] = ra, rb
                            cost[a], cost[b] = ca, cb
                            improved = True

        if not improved:
            break
    return routes


def solve_subproblem(
    ev: AllocationEvaluator,
    agent_ids: Sequence[int],
    task_ids: Iterable[int],
    improve: bool = True,
    max_passes: int = 20,
) -> Routes:
    """Allocate ``task_ids`` among ``agent_ids``: greedy insertion, then optional local search."""
    routes = greedy_insertion(ev, agent_ids, task_ids)
    if improve and any(routes.values()):
        routes = local_search(ev, routes, max_passes=max_passes)
    return routes

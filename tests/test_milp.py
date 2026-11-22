"""Tests for the exact MILP (scipy/HiGHS) and its use as a per-cluster sub-solver."""

import itertools
import unittest

from src.allocation import (
    AllocationEvaluator,
    PaperGAAllocator,
    PaperGAConfig,
    PaperSVCAAllocator,
    PaperSVCAConfig,
)
from src.allocation.milp import MILPAllocator, MILPConfig, solve_cluster, solve_milp
from src.simulation.environment import Environment
from src.simulation.generator import ScenarioGenerator


def make_scenario(R, T, seed=0):
    return ScenarioGenerator(seed=seed).generate_scenario(R, T)


def brute_force(ev, agent_ids, task_ids):
    """Exhaustive optimum over all assignments and orders (tiny instances only)."""
    agents = sorted(agent_ids)
    best = float("inf")
    tasks = list(task_ids)
    for perm in itertools.permutations(tasks):
        for cuts in itertools.combinations_with_replacement(range(len(tasks) + 1), len(agents) - 1):
            bounds = (0,) + cuts + (len(tasks),)
            plan = {a: list(perm[bounds[k]:bounds[k + 1]]) for k, a in enumerate(agents)}
            best = min(best, ev.total_cost(plan))
    return best


class TestMILP(unittest.TestCase):
    def test_matches_brute_force_on_tiny_instances(self):
        for seed in range(4):
            sc = make_scenario(2, 4, seed)
            ev = AllocationEvaluator(sc)
            res = MILPAllocator().solve(sc)
            self.assertEqual(res.metadata["milp_status"], "optimal")
            self.assertTrue(res.metadata["proved_optimal"])
            self.assertAlmostEqual(res.total_cost, brute_force(ev, ev.agent_ids, ev.task_ids), places=1)

    def test_three_robots(self):
        sc = make_scenario(3, 4, seed=7)
        ev = AllocationEvaluator(sc)
        res = MILPAllocator().solve(sc)
        self.assertAlmostEqual(res.total_cost, brute_force(ev, ev.agent_ids, ev.task_ids), places=1)

    def test_valid_matches_environment_and_model_gap_is_zero(self):
        sc = make_scenario(3, 6, seed=2)
        res = MILPAllocator().solve(sc)
        self.assertEqual(AllocationEvaluator(sc).validate(res.assignments), [])
        env = Environment()
        env.load_scenario(sc)
        self.assertAlmostEqual(res.total_cost, env.execute_allocation(res.assignments)[0].total_cost, places=2)
        # batteries stay far above 15%, so the nominal-speed model equals the simulator
        self.assertAlmostEqual(res.metadata["model_gap"], 0.0, delta=1e-3)
        self.assertEqual(res.solver_name, "MILP")
        self.assertEqual(res.metadata["plan_source"], "milp")

    def test_never_worse_than_greedy_local_search(self):
        from src.allocation import GreedyAllocator
        for seed in range(3):
            sc = make_scenario(3, 7, seed)
            self.assertLessEqual(MILPAllocator(MILPConfig(time_limit=20)).solve(sc).total_cost,
                                 GreedyAllocator(use_local_search=True).solve(sc).total_cost + 0.01)

    def test_energy_is_a_hard_constraint(self):
        sc = make_scenario(2, 6, seed=3)
        ev0 = AllocationEvaluator(sc)
        free = MILPAllocator().solve(sc)
        # Shrink batteries so that one robot cannot serve everything; plan must stay penalty-free.
        used = max(free.agent_energy_used.values())
        for a in sc.agents:
            a.energy_capacity = a.current_energy = used * 0.9
        res = MILPAllocator().solve(sc)
        self.assertEqual(res.metadata["energy_penalty"], 0.0)
        self.assertEqual(AllocationEvaluator(sc).validate(res.assignments), [])
        for a, e in res.agent_energy_used.items():
            self.assertLessEqual(e, used * 0.9 + 1e-6)

    def test_energy_infeasible_instance_falls_back_to_greedy(self):
        sc = make_scenario(2, 4, seed=1)
        for a in sc.agents:
            a.energy_capacity = a.current_energy = 1.0     # nobody can do anything
        res = MILPAllocator().solve(sc)
        self.assertEqual(res.metadata["milp_status"], "infeasible")
        self.assertEqual(res.metadata["plan_source"], "greedy_incumbent")
        self.assertFalse(res.metadata["proved_optimal"])
        self.assertEqual(AllocationEvaluator(sc).validate(res.assignments), [])

    def test_edge_cases(self):
        res = MILPAllocator().solve(make_scenario(3, 0))
        self.assertEqual(res.total_cost, 0.0)
        sc = make_scenario(1, 3, seed=5)
        ev = AllocationEvaluator(sc)
        res = MILPAllocator().solve(sc)
        self.assertEqual(sorted(res.assignments[1]), [1, 2, 3])
        self.assertAlmostEqual(res.total_cost, brute_force(ev, [1], ev.task_ids), places=1)
        res = MILPAllocator().solve(make_scenario(2, 1))
        self.assertEqual(AllocationEvaluator(make_scenario(2, 1)).validate(res.assignments), [])

    def test_cutoff_is_a_valid_cut(self):
        sc = make_scenario(2, 4, seed=0)
        ev = AllocationEvaluator(sc)
        opt = brute_force(ev, ev.agent_ids, ev.task_ids)
        ok = solve_milp(ev, ev.agent_ids, ev.task_ids, cutoff=opt)         # cut at the optimum: still feasible
        self.assertIsNotNone(ok.routes)
        self.assertAlmostEqual(ok.objective, opt, places=4)
        bad = solve_milp(ev, ev.agent_ids, ev.task_ids, cutoff=opt * 0.9)  # below the optimum: infeasible
        self.assertIsNone(bad.routes)
        self.assertEqual(bad.status, "infeasible")


class TestClusterSolver(unittest.TestCase):
    def test_milp_cluster_optimal_and_heuristic_not_better(self):
        sc = make_scenario(2, 5, seed=4)
        ev = AllocationEvaluator(sc)
        exact, info = solve_cluster(ev, ev.agent_ids, ev.task_ids, subsolver="milp")
        heur, _ = solve_cluster(ev, ev.agent_ids, ev.task_ids, subsolver="heuristic")
        self.assertEqual(info["status"], "optimal")
        self.assertFalse(info["fallback"])
        self.assertAlmostEqual(ev.total_cost(exact), brute_force(ev, ev.agent_ids, ev.task_ids), places=1)
        self.assertLessEqual(ev.total_cost(exact), ev.total_cost(heur) + 1e-6)

    def test_unknown_subsolver_raises(self):
        sc = make_scenario(2, 3)
        ev = AllocationEvaluator(sc)
        with self.assertRaises(ValueError):
            solve_cluster(ev, ev.agent_ids, ev.task_ids, subsolver="magic")

    def test_empty_cluster(self):
        sc = make_scenario(2, 3)
        ev = AllocationEvaluator(sc)
        routes, info = solve_cluster(ev, ev.agent_ids, [], subsolver="milp")
        self.assertEqual(routes, {1: [], 2: []})


class TestExactSubsolverInClustering(unittest.TestCase):
    def test_svca_with_milp_subsolver(self):
        sc = make_scenario(4, 9, seed=6)
        heur = PaperSVCAAllocator().solve(sc)
        exact = PaperSVCAAllocator(PaperSVCAConfig(subsolver="milp", milp_time_limit=10)).solve(sc)
        self.assertEqual(AllocationEvaluator(sc).validate(exact.assignments), [])
        self.assertEqual(exact.metadata["subsolver"], "milp")
        self.assertEqual(sum(exact.metadata["subsolver_status"].values()), exact.metadata["num_clusters"]
                         - sum(1 for c in exact.metadata["clusters"] if not c["tasks"]))
        self.assertEqual(exact.metadata["clusters"], heur.metadata["clusters"])   # same partition
        self.assertLessEqual(exact.total_cost, heur.total_cost + 0.01)            # exact per cluster is never worse
        self.assertEqual(heur.metadata["subsolver"], "heuristic")

    def test_ga_with_milp_subsolver(self):
        sc = make_scenario(3, 6, seed=2)
        cfg = PaperGAConfig(population_size=10, max_generations=3, stall_generations=3,
                            subsolver="milp", milp_time_limit=10)
        res = PaperGAAllocator(cfg).solve(sc)
        self.assertEqual(AllocationEvaluator(sc).validate(res.assignments), [])
        self.assertEqual(res.metadata["subsolver"], "milp")


if __name__ == "__main__":
    unittest.main()

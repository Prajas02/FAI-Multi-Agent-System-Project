"""Tests for the paper-faithful SVCA and GA (Martin et al., 2023)."""

import os
import random
import unittest

from src.allocation import (
    AllocationEvaluator,
    PaperGAAllocator,
    PaperGAConfig,
    PaperParams,
    PaperSVCAAllocator,
    PaperSVCAConfig,
)
from src.allocation.paper_model import PaperCoalitionModel
from src.allocation.svca_paper import estimate_shapley, initial_q, svca_clusterings
from src.simulation.environment import Environment
from src.simulation.generator import ScenarioGenerator

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "svca_paper.yaml")


def make_scenario(R=6, T=14, seed=42, clustered=False):
    return ScenarioGenerator(seed=seed).generate_scenario(R, T, clustered_tasks=clustered)


def env_cost(scenario, plan):
    env = Environment()
    env.load_scenario(scenario)
    return env.execute_allocation(plan)[0].total_cost


class TestPaperModel(unittest.TestCase):
    def test_j_formula_by_hand(self):
        sc = make_scenario(1, 1, seed=3)
        p = PaperParams(rho=5.0)
        m = PaperCoalitionModel(sc, p)
        a, t = sc.agents[0], sc.tasks[0]
        # paper-scale attributes
        E = (a.current_energy - 500) / 1500 * 100
        v = 1 + (a.velocity - 5) / 10 * 19
        lam = 1 + (a.usage_penalty - 1) / 2 * 4
        tau = (t.execution_time - 10) / 50 * 10
        delta = t.priority
        d = a.position.distance_to(t.location)
        expected = (E + v - lam) - (delta + tau) - p.alpha1 * d - p.alpha2 * 2 + 5.0
        self.assertAlmostEqual(m.cost_of([0, 1]), expected, places=6)

    def test_psi_zero_for_empty_robots_only_tasks_only(self):
        sc = make_scenario(3, 5)
        m = PaperCoalitionModel(sc)
        self.assertEqual(m.cost_of([]), 0.0)
        self.assertEqual(m.cost_of([0, 1]), 0.0)                 # robots only
        self.assertEqual(m.cost_of([m.R, m.R + 1]), 0.0)         # tasks only
        self.assertNotEqual(m.cost_of([0, m.R]), 0.0)

    def test_automatic_rho_makes_every_mixed_coalition_positive(self):
        sc = make_scenario(5, 15, seed=2)
        m = PaperCoalitionModel(sc)
        rng = random.Random(0)
        for _ in range(300):
            members = rng.sample(range(m.N), rng.randint(2, m.N))
            if m.has_robot(members) and any(e >= m.R for e in members):
                self.assertGreater(m.cost_of(members), 0.0)

    def test_attribute_scaling_maps_simulator_ranges_to_paper_ranges(self):
        sc = make_scenario(2, 1)
        sc.agents[0].current_energy = 500.0     # -> E = 0
        sc.agents[1].current_energy = 2000.0    # -> E = 100
        only_energy = PaperParams(alpha1=0, alpha2=0, alpha4=0, alpha5=0, alpha6=0, alpha7=0, rho=0.0)
        m = PaperCoalitionModel(sc, only_energy)
        self.assertAlmostEqual(m.J_i[0], 0.0)
        self.assertAlmostEqual(m.J_i[1], 100.0)
        raw = PaperCoalitionModel(sc, PaperParams(scale="none", alpha1=0, alpha2=0, alpha4=0, alpha5=0,
                                                  alpha6=0, alpha7=0, rho=0.0))
        self.assertAlmostEqual(raw.J_i[1], 2000.0)

    def test_incremental_state_matches_scratch_in_any_order(self):
        m = PaperCoalitionModel(make_scenario(5, 12, seed=4))
        members = list(range(m.N))
        ref = m.cost_of(members)
        rng = random.Random(1)
        for _ in range(10):
            rng.shuffle(members)
            st = m.new_state()
            for e in members:
                st.add(e)
            self.assertAlmostEqual(st.cost(), ref, places=6)

    def test_clustering_value_omega(self):
        m = PaperCoalitionModel(make_scenario(3, 6))
        good = [[0, 3, 4], [1, 5, 6], [2, 7, 8]]
        self.assertGreater(m.value_of_clustering(good), 0.0)
        task_only = [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
        self.assertEqual(m.value_of_clustering(task_only), 0.0)              # Omega = 0
        robot_only_cluster = [[0, 3, 4, 5, 6, 7, 8], [1, 2]]
        self.assertAlmostEqual(m.value_of_clustering(robot_only_cluster), m.cost_of([0, 3, 4, 5, 6, 7, 8]), places=6)


class TestShapleyAlgorithm1(unittest.TestCase):
    def test_initial_q_matches_paper(self):
        self.assertEqual(initial_q(0.1, 0.08), 423)

    def test_efficiency_fixed_and_adaptive(self):
        m = PaperCoalitionModel(make_scenario(4, 9, seed=6))
        grand = m.cost_of(range(m.N))
        for cfg in (PaperSVCAConfig(adaptive=False, q_fixed=30), PaperSVCAConfig(l_max=3, q_max=500)):
            phi, q, _ = estimate_shapley(m, cfg, random.Random(0))
            self.assertAlmostEqual(sum(phi), grand, places=5)

    def test_adaptive_uses_at_least_q_ini_and_respects_cap(self):
        m = PaperCoalitionModel(make_scenario(4, 9, seed=6))
        _, q, q_ini = estimate_shapley(m, PaperSVCAConfig(), random.Random(0))
        self.assertEqual(q_ini, 423)
        self.assertGreaterEqual(q, q_ini)
        _, q_cap, _ = estimate_shapley(m, PaperSVCAConfig(l_max=10**6, q_max=q_ini + 5), random.Random(0))
        self.assertEqual(q_cap, q_ini + 5)

    def test_signs_match_paper_remark_6_at_default_weights(self):
        """Robots have positive, tasks negative Shapley values (paper Remark 6)."""
        sc = make_scenario(15, 45, seed=1)
        info = PaperSVCAAllocator().partition(sc)[1]
        phi = info["phi"]
        self.assertTrue(all(x > 0 for x in phi[:15]))
        self.assertTrue(all(x < 0 for x in phi[15:]))


class TestAlgorithm2(unittest.TestCase):
    # robots 0,1,2 with phi 10, 6, 2; tasks 3..6 with phi -1, -2, -3, -4
    PHI = [10.0, 6.0, 2.0, -1.0, -2.0, -3.0, -4.0]

    @staticmethod
    def as_sets(clustering):
        return sorted(sorted(c) for c in clustering)

    def test_hand_computed_example_lowest_abs(self):
        cl = svca_clusterings(3, 4, self.PHI, "lowest_abs")
        self.assertEqual(len(cl), 3)
        self.assertEqual(self.as_sets(cl[0]), [[0, 3, 4, 5], [1, 6], [2]])
        self.assertEqual(self.as_sets(cl[1]), [[0, 3, 4, 6], [1, 2, 5]])    # robots 1,2 merged (lowest |phi|)
        self.assertEqual(self.as_sets(cl[2]), [[0, 1, 2, 3, 4, 5, 6]])

    def test_hand_computed_example_most_demanding(self):
        cl = svca_clusterings(3, 4, self.PHI, "most_demanding")
        self.assertEqual(self.as_sets(cl[0]), [[0, 5, 6], [1, 3, 4], [2]])

    def test_unknown_order_raises(self):
        with self.assertRaises(ValueError):
            svca_clusterings(3, 4, self.PHI, "nonsense")

    def test_structure_on_random_scenario(self):
        sc = make_scenario(8, 20, seed=9)
        _, info, model = PaperSVCAAllocator().partition(sc)
        cl = svca_clusterings(model.R, model.T, info["phi"])
        self.assertEqual([len(c) for c in cl], list(range(model.R, 0, -1)))
        for clustering in cl:
            flat = sorted(e for c in clustering for e in c)
            self.assertEqual(flat, list(range(model.N)))              # a partition of all players
            for c in clustering:
                self.assertTrue(any(e < model.R for e in c))          # never a task-only cluster

    def test_no_tasks_and_single_robot(self):
        cl = svca_clusterings(3, 0, [3.0, 2.0, 1.0])
        self.assertEqual(len(cl), 3)
        cl = svca_clusterings(1, 4, [5.0, -1.0, -2.0, -3.0, -4.0])
        self.assertEqual(self.as_sets(cl[0]), [[0, 1, 2, 3, 4]])


class TestPaperSVCAAllocator(unittest.TestCase):
    def setUp(self):
        self.sc = make_scenario(8, 20, seed=5)

    def test_valid_matches_environment_and_metadata(self):
        res = PaperSVCAAllocator().solve(self.sc)
        self.assertEqual(AllocationEvaluator(self.sc).validate(res.assignments), [])
        self.assertAlmostEqual(res.total_cost, env_cost(self.sc, res.assignments), places=2)
        self.assertEqual(res.solver_name, "SVCA")
        m = res.metadata
        self.assertAlmostEqual(m["V"], max(c["V"] for c in m["v_curve"]), places=6)
        self.assertEqual(len(m["v_curve"]), 8)
        self.assertEqual(m["num_clusters"], len(m["clusters"]))
        self.assertEqual(m["q_initial"], 423)
        for key in ("t_clu", "t_all", "q_used", "rho", "shapley", "frac_tasks_positive_phi"):
            self.assertIn(key, m)

    def test_deterministic(self):
        a = PaperSVCAAllocator().solve(self.sc)
        b = PaperSVCAAllocator().solve(self.sc)
        self.assertEqual(a.assignments, b.assignments)
        self.assertEqual(a.metadata["q_used"], b.metadata["q_used"])

    def test_edge_cases(self):
        res = PaperSVCAAllocator().solve(make_scenario(3, 0))
        self.assertEqual(res.total_cost, 0.0)
        res = PaperSVCAAllocator().solve(make_scenario(1, 4))
        self.assertEqual(sorted(res.assignments[1]), [1, 2, 3, 4])

    def test_config_from_yaml(self):
        cfg = PaperSVCAConfig.from_yaml(CONFIG_PATH, l_max=3)
        self.assertEqual(cfg.params.alpha1, 0.1)
        self.assertEqual(cfg.params.alpha2, 10.0)
        self.assertIsNone(cfg.params.rho)
        self.assertEqual(cfg.theta, 0.1)
        self.assertEqual(cfg.xi, 0.08)
        self.assertEqual(cfg.l_max, 3)                  # override wins
        self.assertEqual(cfg.task_order, "lowest_abs")

    def test_rho_zero_prefers_grand_coalition(self):
        res = PaperSVCAAllocator(PaperSVCAConfig(params=PaperParams(rho_multiplier=0.0))).solve(self.sc)
        self.assertEqual(res.metadata["num_clusters"], 1)


class TestPaperGA(unittest.TestCase):
    def setUp(self):
        self.sc = make_scenario(6, 14, seed=8)
        self.cfg = PaperGAConfig(population_size=30, max_generations=15, stall_generations=10)

    def test_valid_and_matches_environment(self):
        res = PaperGAAllocator(self.cfg).solve(self.sc)
        self.assertEqual(AllocationEvaluator(self.sc).validate(res.assignments), [])
        self.assertAlmostEqual(res.total_cost, env_cost(self.sc, res.assignments), places=2)
        self.assertEqual(res.solver_name, "GA")

    def test_every_cluster_with_tasks_has_a_robot(self):
        res = PaperGAAllocator(self.cfg).solve(self.sc)
        for c in res.metadata["clusters"]:
            if c["tasks"]:
                self.assertGreaterEqual(len(c["agents"]), 1)
        self.assertGreater(res.metadata["V"], 0.0)       # Omega = 1

    def test_deterministic_and_v_never_decreases(self):
        a = PaperGAAllocator(self.cfg).solve(self.sc)
        b = PaperGAAllocator(self.cfg).solve(self.sc)
        self.assertEqual(a.assignments, b.assignments)
        h = a.metadata["v_history"]
        for x, y in zip(h, h[1:]):
            self.assertGreaterEqual(y, x - 1e-9)         # elitism


if __name__ == "__main__":
    unittest.main()

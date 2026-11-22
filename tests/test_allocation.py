"""Unit tests for the Allocation module (Dhairya Shah): evaluator, Greedy, SVCA, GA."""

import itertools
import os
import random
import unittest

from src.allocation import (
    AllocationEvaluator,
    GAClusteringAllocator,
    GAConfig,
    GreedyAllocator,
    SVCAAllocator,
    SVCAConfig,
)
from src.allocation.coalition import CoalitionModel, CoalitionParams
from src.allocation.insertion import greedy_insertion, local_search
from src.allocation.svca import estimate_shapley
from src.simulation.environment import Environment
from src.simulation.generator import ScenarioGenerator

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default_config.yaml")


def make_scenario(num_agents=6, num_tasks=14, seed=42, clustered=False):
    return ScenarioGenerator(seed=seed).generate_scenario(num_agents, num_tasks, clustered_tasks=clustered)


def random_plan(scenario, rng):
    """Every task assigned exactly once, random agent and random order."""
    plan = {a.agent_id: [] for a in scenario.agents}
    tasks = [t.task_id for t in scenario.tasks]
    rng.shuffle(tasks)
    for t in tasks:
        plan[rng.choice(list(plan))].append(t)
    return plan


def env_result(scenario, plan):
    env = Environment()
    env.load_scenario(scenario)
    result, _ = env.execute_allocation(plan)
    return result


class TestEvaluatorParity(unittest.TestCase):
    """The fast evaluator must reproduce Environment.execute_allocation."""

    def _check(self, scenario, seeds=10):
        ev = AllocationEvaluator(scenario)
        for s in range(seeds):
            plan = random_plan(scenario, random.Random(s))
            mine = ev.evaluate(plan)
            ref = env_result(scenario, plan)
            self.assertAlmostEqual(mine.total_cost, ref.total_cost, places=2)
            for t, c in ref.task_completion_times.items():
                self.assertAlmostEqual(mine.task_completion_times[t], c, places=6)
            for a, d in ref.agent_distances.items():
                self.assertAlmostEqual(mine.agent_distances[a], d, places=6)
            self.assertAlmostEqual(mine.metadata["energy_penalty"], ref.metadata["energy_penalty"], places=2)

    def test_parity_normal_energy(self):
        self._check(make_scenario(4, 12, seed=1))

    def test_parity_with_energy_violations_and_velocity_decay(self):
        sc = make_scenario(3, 12, seed=2)
        for a in sc.agents:          # tiny batteries: forces deficits and the <15% speed decay branch
            a.energy_capacity = a.current_energy = 120.0
        ev = AllocationEvaluator(sc)
        self.assertGreater(ev.evaluate(random_plan(sc, random.Random(0))).metadata["energy_penalty"], 0.0)
        self._check(sc)

    def test_cache_does_not_change_results(self):
        sc = make_scenario(4, 10, seed=3)
        ev = AllocationEvaluator(sc)
        plan = random_plan(sc, random.Random(5))
        first = ev.total_cost(plan)
        second = ev.total_cost(plan)
        self.assertEqual(first, second)
        self.assertGreater(ev.cache_hits, 0)

    def test_validate_catches_bad_plans(self):
        sc = make_scenario(2, 4, seed=4)
        ev = AllocationEvaluator(sc)
        self.assertEqual(ev.validate({1: [1, 2], 2: [3, 4]}), [])
        self.assertTrue(any("unassigned" in p for p in ev.validate({1: [1, 2], 2: [3]})))
        self.assertTrue(any("twice" in p for p in ev.validate({1: [1, 2], 2: [2, 3, 4]})))
        self.assertTrue(any("unknown agent" in p for p in ev.validate({9: [1, 2, 3, 4]})))
        # Documented pitfall: Environment happily scores an incomplete plan as CHEAPER.
        full = env_result(sc, {1: [1, 2], 2: [3, 4]}).total_cost
        partial = env_result(sc, {1: [1, 2], 2: [3]}).total_cost
        self.assertLess(partial, full)


class TestGreedy(unittest.TestCase):
    def test_valid_and_matches_environment(self):
        sc = make_scenario(6, 14)
        res = GreedyAllocator().solve(sc)
        self.assertEqual(AllocationEvaluator(sc).validate(res.assignments), [])
        self.assertTrue(res.metadata["valid"])
        self.assertAlmostEqual(res.total_cost, env_result(sc, res.assignments).total_cost, places=2)
        self.assertEqual(res.solver_name, "Greedy")

    def test_deterministic(self):
        sc = make_scenario(6, 14)
        a = GreedyAllocator().solve(sc)
        b = GreedyAllocator().solve(sc)
        self.assertEqual(a.assignments, b.assignments)
        self.assertEqual(a.total_cost, b.total_cost)

    def test_local_search_never_worse(self):
        for seed in range(5):
            sc = make_scenario(5, 15, seed=seed)
            g = GreedyAllocator().solve(sc).total_cost
            gl = GreedyAllocator(use_local_search=True).solve(sc).total_cost
            self.assertLessEqual(gl, g + 1e-6)

    def test_close_to_brute_force_optimum_on_tiny_instances(self):
        for seed in range(4):
            sc = make_scenario(2, 5, seed=seed)
            ev = AllocationEvaluator(sc)
            best = float("inf")
            tasks = ev.task_ids
            for perm in itertools.permutations(tasks):
                for cut in range(len(tasks) + 1):
                    plan = {ev.agent_ids[0]: list(perm[:cut]), ev.agent_ids[1]: list(perm[cut:])}
                    best = min(best, ev.total_cost(plan))
            greedy = GreedyAllocator().solve(sc).total_cost
            with_ls = GreedyAllocator(use_local_search=True).solve(sc).total_cost
            # results are rounded to 2 decimals (like Environment), hence the 0.01 tolerance
            self.assertGreaterEqual(greedy, best - 0.01)      # nothing beats the optimum
            self.assertLessEqual(with_ls, greedy + 0.01)
            self.assertLessEqual(with_ls, 1.15 * best)        # LS should be close to optimal here

    def test_no_agents_raises(self):
        sc = make_scenario(2, 3)
        ev = AllocationEvaluator(sc)
        with self.assertRaises(ValueError):
            greedy_insertion(ev, [], ev.task_ids)

    def test_no_tasks_and_single_agent(self):
        res = GreedyAllocator().solve(make_scenario(3, 0))
        self.assertEqual(res.total_cost, 0.0)
        res = GreedyAllocator().solve(make_scenario(1, 4))
        self.assertEqual(sorted(res.assignments[1]), [1, 2, 3, 4])


class TestCoalitionAndShapley(unittest.TestCase):
    def setUp(self):
        self.sc = make_scenario(5, 12, seed=11)
        self.model = CoalitionModel(self.sc, CoalitionParams())

    def test_cost_independent_of_insertion_order(self):
        members = list(range(self.model.N))
        ref = self.model.cost_of(members)
        rng = random.Random(0)
        for _ in range(10):
            rng.shuffle(members)
            self.assertAlmostEqual(self.model.cost_of(members), ref, places=6)

    def test_empty_coalition_costs_nothing(self):
        self.assertEqual(self.model.cost_of([]), 0.0)

    def test_tasks_without_an_agent_are_penalised(self):
        R = self.model.R
        with_agent = self.model.cost_of([0, R, R + 1])
        without = self.model.cost_of([R, R + 1])
        self.assertGreater(without, with_agent)

    def test_shapley_efficiency(self):
        """Marginals telescope per permutation, so sum_i phi_i must equal J(N)."""
        for antithetic in (True, False):
            phi = estimate_shapley(self.model, 20, random.Random(1), antithetic)
            self.assertAlmostEqual(sum(phi), self.model.cost_of(range(self.model.N)), places=6)

    def test_shapley_deterministic_given_seed(self):
        a = estimate_shapley(self.model, 10, random.Random(3))
        b = estimate_shapley(self.model, 10, random.Random(3))
        self.assertEqual(a, b)


class TestSVCA(unittest.TestCase):
    def setUp(self):
        self.sc = make_scenario(12, 30, seed=5, clustered=True)
        self.cfg = SVCAConfig(target_cluster_size=14)

    def test_partition_covers_every_entity_once_and_every_cluster_has_an_agent(self):
        clusters, shapley, model = SVCAAllocator(self.cfg).partition(self.sc)
        self.assertGreater(len(clusters), 1)
        agents = sorted(a for c in clusters for a in c.agent_ids)
        tasks = sorted(t for c in clusters for t in c.task_ids)
        self.assertEqual(agents, sorted(a.agent_id for a in self.sc.agents))
        self.assertEqual(tasks, sorted(t.task_id for t in self.sc.tasks))
        for c in clusters:
            if c.task_ids:
                self.assertGreaterEqual(len(c.agent_ids), 1)
        self.assertEqual(len(shapley), model.N)

    def test_cluster_sizes_respect_limit(self):
        import math
        cfg = SVCAConfig(target_cluster_size=14, size_slack=0.3)
        clusters, _, model = SVCAAllocator(cfg).partition(self.sc)
        limit = max(math.ceil(model.N / len(clusters)), math.ceil(14 * 1.3))
        for c in clusters:
            self.assertLessEqual(c.size, limit)

    def test_solution_valid_and_matches_environment(self):
        res = SVCAAllocator(self.cfg).solve(self.sc)
        self.assertEqual(AllocationEvaluator(self.sc).validate(res.assignments), [])
        self.assertAlmostEqual(res.total_cost, env_result(self.sc, res.assignments).total_cost, places=2)
        self.assertEqual(res.solver_name, "SVCA")
        self.assertEqual(res.metadata["num_clusters"], len(res.metadata["clusters"]))
        for key in ("runtime_shapley", "runtime_clustering", "runtime_subsolve", "shapley"):
            self.assertIn(key, res.metadata)

    def test_deterministic_for_fixed_seed(self):
        a = SVCAAllocator(self.cfg).solve(self.sc)
        b = SVCAAllocator(self.cfg).solve(self.sc)
        self.assertEqual(a.assignments, b.assignments)
        self.assertEqual(a.total_cost, b.total_cost)

    def test_single_cluster_equals_greedy_with_local_search(self):
        """K=1 means no partitioning, so SVCA must reduce to the unclustered solver."""
        sc = make_scenario(6, 14)
        svca = SVCAAllocator(SVCAConfig(target_cluster_size=1000)).solve(sc)
        self.assertEqual(svca.metadata["num_clusters"], 1)
        self.assertEqual(svca.total_cost, GreedyAllocator(use_local_search=True).solve(sc).total_cost)

    def test_refinement_never_increases_sum_of_cluster_costs(self):
        sc = make_scenario(20, 60, seed=7)
        def total_j(cfg):
            clusters, _, model = SVCAAllocator(cfg).partition(sc)
            aidx = {a.agent_id: i for i, a in enumerate(sc.agents)}
            tidx = {t.task_id: model.R + i for i, t in enumerate(sc.tasks)}
            return sum(model.cost_of([aidx[a] for a in c.agent_ids] + [tidx[t] for t in c.task_ids]) for c in clusters)
        self.assertLessEqual(total_j(SVCAConfig(refine_passes=10)), total_j(SVCAConfig(refine_passes=0)) + 1e-6)

    def test_edge_cases(self):
        res = SVCAAllocator().solve(make_scenario(3, 0))
        self.assertEqual(res.total_cost, 0.0)
        res = SVCAAllocator().solve(make_scenario(1, 4))
        self.assertEqual(sorted(res.assignments[1]), [1, 2, 3, 4])

    def test_config_from_yaml(self):
        cfg = SVCAConfig.from_yaml(CONFIG_PATH, num_permutations_q=8)
        self.assertEqual(cfg.alpha1, 0.05)
        self.assertEqual(cfg.alpha2, 1.5)
        self.assertEqual(cfg.rho, 10.0)
        self.assertEqual(cfg.num_permutations_q, 8)   # explicit override wins
        self.assertEqual(cfg.seed, 42)


class TestGAClustering(unittest.TestCase):
    def setUp(self):
        self.sc = make_scenario(10, 24, seed=8)
        self.cfg = GAConfig(target_cluster_size=12, population_size=12, generations=6)

    def test_valid_and_matches_environment(self):
        res = GAClusteringAllocator(self.cfg).solve(self.sc)
        self.assertEqual(AllocationEvaluator(self.sc).validate(res.assignments), [])
        self.assertAlmostEqual(res.total_cost, env_result(self.sc, res.assignments).total_cost, places=2)
        self.assertEqual(res.solver_name, "GA")

    def test_every_cluster_with_tasks_has_an_agent(self):
        res = GAClusteringAllocator(self.cfg).solve(self.sc)
        for c in res.metadata["clusters"]:
            if c["tasks"]:
                self.assertGreaterEqual(len(c["agents"]), 1)

    def test_repair_always_yields_feasible_chromosomes(self):
        alloc = GAClusteringAllocator(self.cfg)
        alloc._setup(AllocationEvaluator(self.sc))
        rng = random.Random(0)
        for _ in range(50):
            genes = alloc._random_individual(rng)
            for agents, tasks in alloc._clusters(genes):
                if tasks:
                    self.assertTrue(agents)

    def test_deterministic_and_history_non_increasing(self):
        a = GAClusteringAllocator(self.cfg).solve(self.sc)
        b = GAClusteringAllocator(self.cfg).solve(self.sc)
        self.assertEqual(a.assignments, b.assignments)
        h = a.metadata["fitness_history"]
        self.assertEqual(len(h), self.cfg.generations + 1)
        for x, y in zip(h, h[1:]):
            self.assertLessEqual(y, x + 1e-9)       # elitism

    def test_same_k_as_svca_by_default(self):
        svca = SVCAAllocator(SVCAConfig(target_cluster_size=12)).solve(self.sc)
        ga = GAClusteringAllocator(self.cfg).solve(self.sc)
        self.assertEqual(svca.metadata["num_clusters"], ga.metadata["num_clusters"])


class TestLocalSearch(unittest.TestCase):
    def test_never_worsens_and_keeps_all_tasks(self):
        for seed in range(5):
            sc = make_scenario(4, 14, seed=seed)
            ev = AllocationEvaluator(sc)
            plan = random_plan(sc, random.Random(seed))
            improved = local_search(ev, plan)
            self.assertLessEqual(ev.total_cost(improved), ev.total_cost(plan) + 1e-9)
            self.assertEqual(ev.validate(improved), [])


if __name__ == "__main__":
    unittest.main()

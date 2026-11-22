"""Tests for the game-theory / reward-solver module (Goral Mashru)."""

import unittest

import numpy as np

from src.game_theory import (
    compare_global_vs_clustered,
    solve_by_clusters,
    CallableValue,
    FacilityLocationValue,
    GlauberSampler,
    PreferenceAwareShapleySolver,
    PreferenceGraph,
    RoutingValue,
    core_stability,
    exact_preference_shapley,
    exact_shapley,
    gini,
    glauber_preference_shapley,
    permutation_shapley,
    stratified_shapley,
)
from src.simulation.generator import ScenarioGenerator


def glove_game(n_left=2, n_right=1):
    """Classic glove market: V(S) = min(#left, #right). Known Shapley for (2L,1R): [1/6,1/6,2/3]."""
    left = set(range(n_left))
    return CallableValue(
        n_left + n_right,
        lambda S: min(len(S & left), len(S - left)),
    )


def additive_game(weights):
    return CallableValue(len(weights), lambda S: sum(weights[i] for i in S))


class TestExactShapley(unittest.TestCase):
    def test_glove_game_known_values(self):
        phi = exact_shapley(glove_game())
        np.testing.assert_allclose(phi, [1 / 6, 1 / 6, 2 / 3], atol=1e-12)

    def test_additive_game_returns_weights(self):
        w = [1.0, 2.0, 3.5, 0.5]
        np.testing.assert_allclose(exact_shapley(additive_game(w)), w, atol=1e-12)

    def test_efficiency_and_symmetry(self):
        v = glove_game(2, 2)
        phi = exact_shapley(v)
        self.assertAlmostEqual(phi.sum(), v.grand_coalition_value())
        self.assertAlmostEqual(phi[0], phi[1])
        self.assertAlmostEqual(phi[2], phi[3])


class TestPreferenceGraph(unittest.TestCase):
    def test_distribution_is_valid(self):
        pref = PreferenceGraph.random(5, seed=1)
        perms, probs = pref.exact_distribution(alpha=1.5)
        self.assertEqual(perms.shape, (120, 5))
        self.assertAlmostEqual(probs.sum(), 1.0)
        self.assertTrue((probs > 0).all())

    def test_alpha_zero_is_uniform(self):
        pref = PreferenceGraph.random(4, seed=2)
        _, probs = pref.exact_distribution(alpha=0.0)
        np.testing.assert_allclose(probs, 1 / 24)

    def test_upsilon_matches_definition(self):
        pref = PreferenceGraph.random(5, mean_weight=0.2, seed=3)
        perm = np.array([3, 0, 4, 1, 2])
        direct = sum(np.exp(pref.log_gamma(perm[: k + 1])) for k in range(5))
        self.assertAlmostEqual(pref.upsilon(perm), direct, places=9)
        # vectorised batch version agrees
        self.assertAlmostEqual(pref._upsilon_batch(perm[None, :])[0], direct, places=9)

    def test_weights_validated(self):
        with self.assertRaises(ValueError):
            PreferenceGraph(np.full((3, 3), 1.5))

    def test_no_overflow_for_large_dense_graph(self):
        pref = PreferenceGraph(np.full((80, 80), 0.9))
        self.assertTrue(np.isfinite(pref.upsilon(np.arange(80))))


class TestExactPreferenceShapley(unittest.TestCase):
    def test_alpha_zero_matches_classical(self):
        v = glove_game(2, 2)
        pref = PreferenceGraph.random(4, seed=0)
        np.testing.assert_allclose(
            exact_preference_shapley(v, pref, 0.0), exact_shapley(v), atol=1e-12
        )

    def test_efficiency_holds_for_any_alpha(self):
        v = glove_game(3, 2)
        pref = PreferenceGraph.random(5, mean_weight=0.4, seed=5)
        for alpha in (0.0, 0.5, 3.0):
            phi = exact_preference_shapley(v, pref, alpha)
            self.assertAlmostEqual(phi.sum(), v.grand_coalition_value(), places=9)

    def test_preferences_change_payoffs(self):
        v = glove_game(2, 2)
        pref = PreferenceGraph.random(4, mean_weight=0.5, seed=6)
        a = exact_preference_shapley(v, pref, 0.0)
        b = exact_preference_shapley(v, pref, 5.0)
        self.assertGreater(np.abs(a - b).max(), 1e-6)


class TestGlauberMCMC(unittest.TestCase):
    def test_incremental_state_matches_recompute(self):
        """After many steps the cached prefix log-Gamma / Upsilon must equal a fresh computation."""
        v = CallableValue(7, lambda S: len(S) ** 1.5 + (2 if {0, 1} <= S else 0))
        pref = PreferenceGraph.random(7, mean_weight=0.3, seed=4)
        for gap in (None, 1, 3):
            sampler = GlauberSampler(v, pref, alpha=1.0, seed=7, max_gap=gap)
            for _ in range(2000):
                sampler.step()
            self.assertEqual(sorted(sampler.perm.tolist()), list(range(7)))
            np.testing.assert_allclose(sampler.s, pref.prefix_log_gamma(sampler.perm), atol=1e-8)
            self.assertAlmostEqual(sampler.ups, pref.upsilon(sampler.perm), places=5)
            self.assertGreater(sampler.accepted, 0)

    def test_marginals_telescope_to_grand_value(self):
        v = FacilityLocationValue.from_scenario(
            ScenarioGenerator(seed=2).generate_scenario(6, 10))
        sampler = GlauberSampler(v, PreferenceGraph.random(6, seed=1), alpha=1.0, seed=0)
        for _ in range(50):
            sampler.step()
        self.assertAlmostEqual(sampler.marginal_contributions().sum(),
                               v.grand_coalition_value(), places=9)

    def test_fast_prefix_values_match_generic(self):
        v = FacilityLocationValue.from_scenario(
            ScenarioGenerator(seed=4).generate_scenario(7, 12))
        perm = np.random.default_rng(0).permutation(7)
        fast = v.prefix_values(perm)
        slow = np.array([v(perm[: k + 1]) for k in range(7)])
        np.testing.assert_allclose(fast, slow, atol=1e-9)

    def test_efficiency_exact(self):
        v = glove_game(3, 2)
        pref = PreferenceGraph.random(5, seed=8)
        res = glauber_preference_shapley(v, pref, 1.5, num_samples=300, burn_in=50, seed=1)
        self.assertAlmostEqual(res.phi.sum(), v.grand_coalition_value(), places=9)

    def test_matches_exact_alpha_zero(self):
        v = glove_game(2, 2)
        pref = PreferenceGraph.random(4, seed=0)
        res = glauber_preference_shapley(v, pref, 0.0, num_samples=6000, burn_in=200, seed=3)
        np.testing.assert_allclose(res.phi, exact_shapley(v), atol=0.05)

    def test_matches_exact_preference_aware(self):
        """MCMC estimate converges to the brute-force preference-aware value."""
        rng_game = np.random.default_rng(11)
        base = rng_game.random(6) * 3
        v = CallableValue(6, lambda S: sum(base[i] for i in S) + 0.6 * len(S) ** 1.7)
        pref = PreferenceGraph.random(6, mean_weight=0.25, seed=12)
        alpha = 1.5
        truth = exact_preference_shapley(v, pref, alpha)
        res = glauber_preference_shapley(v, pref, alpha, num_samples=20000, burn_in=500, seed=5)
        np.testing.assert_allclose(res.phi, truth, atol=0.08)

    def test_empirical_distribution_matches_mu(self):
        """Visit frequencies of orderings approach the softmax mu (n=4), for both neighbourhoods."""
        for gap in (None, 1):
            self._check_distribution(gap)

    def _check_distribution(self, gap):
        pref = PreferenceGraph.random(4, mean_weight=0.5, seed=21)
        alpha = 0.7
        perms, probs = pref.exact_distribution(alpha)
        index = {tuple(p): i for i, p in enumerate(perms)}
        sampler = GlauberSampler(glove_game(2, 2), pref, alpha, seed=9, max_gap=gap)
        for _ in range(500):
            sampler.step()
        counts = np.zeros(len(perms))
        steps = 200000    # error shrinks like 1/sqrt(steps); mu has a 29% mode so mixing is slow
        for _ in range(steps):
            sampler.step()
            counts[index[tuple(sampler.perm)]] += 1
        self.assertLess(np.abs(counts / steps - probs).max(), 0.01)

    def test_tempering_caps_log_gamma(self):
        W = np.full((30, 30), 0.5)
        pref = PreferenceGraph(W, max_log_gamma=4.0)
        self.assertLessEqual(pref.log_gamma(range(30)), 4.0 + 1e-9)
        self.assertLess(pref.gamma_scale, 1.0)

    def test_two_seeds_agree_at_moderate_n(self):
        """Mixing check: independent chains give statistically consistent estimates."""
        scen = ScenarioGenerator(seed=5).generate_scenario(25, 25)
        v = FacilityLocationValue.from_scenario(scen)
        pref = PreferenceGraph.from_agents(scen.agents, seed=0, max_log_gamma=3.0)
        r1 = glauber_preference_shapley(v, pref, 1.5, 1500, 300, seed=1)
        r2 = glauber_preference_shapley(v, pref, 1.5, 1500, 300, seed=2)
        z = np.abs(r1.phi - r2.phi) / np.maximum(np.sqrt(r1.std_error ** 2 + r2.std_error ** 2), 1e-9)
        self.assertLess(np.median(z), 1.5)
        self.assertLess(z.max(), 5.0)

    def test_reproducible_with_seed(self):
        v, pref = glove_game(3, 2), PreferenceGraph.random(5, seed=1)
        a = glauber_preference_shapley(v, pref, 1.0, 200, 50, seed=4).phi
        b = glauber_preference_shapley(v, pref, 1.0, 200, 50, seed=4).phi
        np.testing.assert_array_equal(a, b)


class TestMonteCarloBaselines(unittest.TestCase):
    def test_stratified_close_to_exact(self):
        v = glove_game(3, 3)
        # C(5, s) <= 10 for every stratum, so every stratum is enumerated exactly
        res = stratified_shapley(v, samples_per_stratum=10, seed=0)
        np.testing.assert_allclose(res.phi, exact_shapley(v), atol=1e-9)

    def test_stratified_sampling_regime(self):
        base = np.linspace(1, 4, 10)
        v = CallableValue(10, lambda S: sum(base[i] for i in S) + 0.3 * len(S) ** 2)
        truth = exact_shapley(v)
        for alloc in ("uniform", "neyman"):
            res = stratified_shapley(v, samples_per_stratum=20, allocation=alloc, seed=2)
            np.testing.assert_allclose(res.phi, truth, atol=0.15)
            self.assertAlmostEqual(res.phi.sum(), v.grand_coalition_value(), places=9)

    def test_permutation_mc(self):
        v = glove_game(2, 2)
        res = permutation_shapley(v, num_permutations=3000, seed=1)
        np.testing.assert_allclose(res.phi, exact_shapley(v), atol=0.05)
        self.assertAlmostEqual(res.phi.sum(), v.grand_coalition_value(), places=9)


class TestFairness(unittest.TestCase):
    def test_gini_bounds(self):
        self.assertAlmostEqual(gini([1, 1, 1, 1]), 0.0)
        self.assertAlmostEqual(gini([0, 0, 0, 10]), 0.75)
        self.assertEqual(gini([]), 0.0)

    def test_shapley_of_symmetric_convex_game_is_in_core(self):
        v = CallableValue(5, lambda S: len(S) ** 2)            # convex -> Shapley in core
        stab = core_stability(v, exact_shapley(v))
        self.assertLessEqual(stab["max_excess"], 1e-9)
        self.assertEqual(stab["stability_score"], 1.0)

    def test_unfair_split_is_unstable(self):
        v = CallableValue(4, lambda S: float(len(S)))
        stab = core_stability(v, [4.0, 0.0, 0.0, 0.0])
        self.assertGreater(stab["max_excess"], 0.0)
        self.assertLess(stab["stability_score"], 1.0)

    def test_sampled_regime_runs(self):
        v = CallableValue(20, lambda S: float(len(S)) ** 1.2)
        stab = core_stability(v, np.full(20, v.grand_coalition_value() / 20), num_samples=300, seed=0)
        self.assertEqual(stab["exhaustive"], 0.0)


class TestCharacteristicFunctions(unittest.TestCase):
    def setUp(self):
        self.scenario = ScenarioGenerator(seed=7).generate_scenario(num_agents=6, num_tasks=12)

    def test_basic_properties(self):
        for cls in (FacilityLocationValue, RoutingValue):
            v = cls.from_scenario(self.scenario)
            self.assertEqual(v([]), 0.0)
            self.assertGreaterEqual(v.grand_coalition_value(), 0.0)
            # monotone in coalition growth for the facility-location game
            if cls is FacilityLocationValue:
                self.assertGreaterEqual(v([0, 1, 2]), v([0, 1]) - 1e-12)

    def test_caching(self):
        v = FacilityLocationValue.from_scenario(self.scenario)
        v([0, 1]); n1 = v.num_evaluations
        v([1, 0]); self.assertEqual(v.num_evaluations, n1)

    def test_task_subset_restricts_value(self):
        full = FacilityLocationValue.from_scenario(self.scenario)
        part = FacilityLocationValue.from_scenario(self.scenario, task_ids=[0, 1, 2])
        self.assertLessEqual(part.grand_coalition_value(), full.grand_coalition_value())


class TestSolver(unittest.TestCase):
    def setUp(self):
        self.scenario = ScenarioGenerator(seed=3).generate_scenario(num_agents=7, num_tasks=15)

    def test_returns_reward_distribution_contract(self):
        solver = PreferenceAwareShapleySolver.from_scenario(
            self.scenario, alpha=1.5, num_samples=300, burn_in=50, seed=0)
        dist = solver.solve("glauber", compute_stability=True)
        ids = [a.agent_id for a in self.scenario.agents]
        self.assertEqual(sorted(dist.payoffs), sorted(ids))
        self.assertAlmostEqual(sum(dist.payoffs.values()), dist.total_coalition_value, places=6)
        self.assertEqual(dist.solver_name, "GlauberMCMC")
        self.assertTrue(0.0 <= dist.gini_index <= 1.0)
        self.assertIn("stability", dist.metadata)

    def test_mcmc_vs_exact_on_generated_scenario(self):
        solver = PreferenceAwareShapleySolver.from_scenario(
            self.scenario, alpha=1.5, num_samples=6000, burn_in=300, seed=0)
        mcmc, exact = solver.solve("glauber"), solver.solve("exact")
        a = np.array([mcmc.payoffs[k] for k in exact.payoffs])
        b = np.array(list(exact.payoffs.values()))
        rel = np.abs(a - b).max() / max(b.max(), 1e-9)
        self.assertLess(rel, 0.08)

    def test_all_methods_run_and_are_efficient(self):
        solver = PreferenceAwareShapleySolver.from_scenario(
            self.scenario, alpha=1.0, num_samples=200, burn_in=50, seed=1)
        for name, dist in solver.compare().items():
            self.assertAlmostEqual(sum(dist.payoffs.values()), dist.total_coalition_value,
                                   places=5, msg=name)

    def test_scales_to_80_agents(self):
        scen = ScenarioGenerator(seed=5).generate_scenario(num_agents=80, num_tasks=80)
        solver = PreferenceAwareShapleySolver.from_scenario(
            scen, alpha=1.5, num_samples=50, burn_in=100, seed=0)
        dist = solver.solve("glauber")
        self.assertEqual(len(dist.payoffs), 80)
        self.assertAlmostEqual(sum(dist.payoffs.values()), dist.total_coalition_value, places=5)


class TestClusteredRewards(unittest.TestCase):
    def setUp(self):
        self.scen = ScenarioGenerator(seed=11).generate_scenario(8, 16)
        ids = [a.agent_id for a in self.scen.agents]
        tids = [t.task_id for t in self.scen.tasks]
        self.clusters = [{"agents": ids[:4], "tasks": tids[:9]},
                         {"agents": ids[4:], "tasks": tids[9:]}]
        self.fast = dict(num_samples=300, burn_in=50, seed=0)

    def test_single_cluster_equals_global_game(self):
        ids = [a.agent_id for a in self.scen.agents]
        tids = [t.task_id for t in self.scen.tasks]
        one = solve_by_clusters(self.scen, [{"agents": ids, "tasks": tids}], method="exact", alpha=1.0)
        direct = PreferenceAwareShapleySolver.from_scenario(self.scen, alpha=1.0, seed=42).solve("exact")
        for k in direct.payoffs:
            self.assertAlmostEqual(one.payoffs[k], direct.payoffs[k], places=8)

    def test_efficiency_per_cluster_and_overall(self):
        d = solve_by_clusters(self.scen, self.clusters, **self.fast)
        self.assertAlmostEqual(sum(d.payoffs.values()), d.total_coalition_value, places=6)
        for c in d.metadata["clusters"]:
            self.assertAlmostEqual(sum(d.payoffs[a] for a in c["agents"]), c["value"], places=6)
        self.assertEqual(d.metadata["num_clusters"], 2)
        self.assertEqual(d.metadata["uncovered_agents"], [])

    def test_clustering_cannot_create_value(self):
        res = compare_global_vs_clustered(self.scen, self.clusters, **self.fast)
        self.assertLessEqual(res["value_retained"], 1.0 + 1e-9)
        self.assertEqual(set(res), {"global", "clustered", "gini_delta", "value_retained",
                                    "runtime_ratio", "payoff_shift", "rank_correlation"})

    def test_edge_cases_no_tasks_and_singleton_cluster(self):
        ids = [a.agent_id for a in self.scen.agents]
        tids = [t.task_id for t in self.scen.tasks]
        cl = [{"agents": ids[:1], "tasks": tids},          # singleton agent
              {"agents": ids[1:3], "tasks": []},           # no tasks -> no reward
              ]                                            # remaining agents not covered
        d = solve_by_clusters(self.scen, cl, **self.fast)
        self.assertEqual(d.payoffs[ids[1]], 0.0)
        self.assertEqual(d.payoffs[ids[5]], 0.0)
        self.assertEqual(d.metadata["uncovered_agents"], ids[3:])
        solo = FacilityLocationValue([a for a in self.scen.agents if a.agent_id == ids[0]],
                                     self.scen.tasks,
                                     horizon=d.metadata["horizon"])
        self.assertAlmostEqual(d.payoffs[ids[0]], solo.grand_coalition_value(), places=6)

    def test_allocation_restricts_tasks_and_stability_reported(self):
        first4 = [t.task_id for t in self.scen.tasks[:4]]

        class Alloc:     # minimal stand-in for AllocationResult
            assignments = {self.scen.agents[0].agent_id: first4}
            metadata = {}
        d = solve_by_clusters(self.scen, None, allocation=Alloc(), compute_stability=True, **self.fast)
        self.assertEqual(sorted(d.metadata["clusters"][0]["tasks"]), sorted(first4))
        self.assertIn("global_stability_of_merged_payoffs", d.metadata)

    def test_reads_clusters_from_allocation_metadata(self):
        class Alloc:
            assignments = {a.agent_id: [] for a in self.scen.agents}
            assignments[self.scen.agents[0].agent_id] = [t.task_id for t in self.scen.tasks]
        Alloc.metadata = {"clusters": self.clusters}
        d = solve_by_clusters(self.scen, None, allocation=Alloc(), **self.fast)
        self.assertEqual(d.metadata["num_clusters"], 2)


if __name__ == "__main__":
    unittest.main()

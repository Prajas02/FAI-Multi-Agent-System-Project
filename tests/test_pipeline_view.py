"""Tests for the interactive pipeline viewer and the integration fixes that came with it."""

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from src.allocation.evaluator import AllocationEvaluator
from src.allocation.greedy import GreedyAllocator
from src.analytics import allocation_adapters as adapters
from src.analytics import seeding
from src.analytics.evaluation import validate_assignments
from src.simulation.generator import ScenarioGenerator
from src.visualization.pipeline_view import build_pipeline_data, build_timeline, render_html

ROOT = Path(__file__).resolve().parents[1]


class TestTimeline(unittest.TestCase):
    def test_timeline_matches_the_evaluator_completion_times_and_energy(self):
        scenario = ScenarioGenerator(seed=7).generate_scenario(4, 12)
        result = GreedyAllocator().solve(scenario)
        ev = AllocationEvaluator(scenario)
        timeline = build_timeline(scenario, result.assignments, ev)
        for aid, route in result.assignments.items():
            legs = timeline[str(aid)]
            self.assertEqual([leg["task"] for leg in legs], route)
            stats = ev.route_stats(aid, route)
            for leg, eta in zip(legs, stats.completion_times):
                self.assertAlmostEqual(leg["tf"], eta, places=2)
            if legs:
                start = next(a.current_energy for a in scenario.agents if a.agent_id == aid)
                self.assertAlmostEqual(legs[-1]["energy"], max(0.0, start - stats.energy_used), places=1)
            for prev, nxt in zip(legs, legs[1:]):          # legs are contiguous in space and time
                self.assertEqual((prev["x1"], prev["y1"]), (nxt["x0"], nxt["y0"]))
                self.assertAlmostEqual(prev["tf"], nxt["t0"], places=2)


class TestPipelineData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = ScenarioGenerator(seed=11).generate_scenario(4, 10)
        cls.data = build_pipeline_data(cls.scenario, "unit", seed=3, alphas=(1.0,))

    def test_every_algorithm_produces_a_valid_complete_plan(self):
        names = [a["name"] for a in self.data["algorithms"]]
        self.assertEqual(names, ["SVCA (paper)", "SVCA-ext", "GA (paper)", "GA-cost", "Greedy"])
        for algo in self.data["algorithms"]:
            routes = {int(a): r for a, r in algo["routes"].items()}
            self.assertTrue(validate_assignments(routes, self.scenario).valid, algo["name"])
            self.assertGreater(algo["makespan"], 0)

    def test_reward_section_is_consistent(self):
        reward = self.data["reward"]
        ids = reward["player_ids"]
        self.assertEqual(ids, [a.agent_id for a in self.scenario.agents])
        self.assertEqual(reward["alphas"], [0.0, 1.0, 1.5])
        self.assertEqual(set(reward["by_algorithm"]), {a["name"] for a in self.data["algorithms"]})
        for algo in self.data["algorithms"]:
            per = reward["by_algorithm"][algo["name"]]
            self.assertEqual(per["num_games"], len(algo["clusters"]) if algo["clusters"] else 1)
            for entry in per["sweep"]:
                self.assertEqual(len(entry["payoffs"]), len(ids))
                # efficiency: the merged Shapley shares add up to the value of the per-cluster games
                self.assertAlmostEqual(sum(entry["payoffs"]), entry["value"], delta=0.05 * len(ids))
        self.assertEqual(len(reward["W"]), len(ids))

    def test_render_html_embeds_the_data(self):
        tmp = Path(tempfile.mkdtemp(prefix="pipeline_view_"))
        self.addCleanup(shutil.rmtree, str(tmp), True)
        out = render_html(self.data, tmp / "page.html")
        html = out.read_text(encoding="utf-8")
        self.assertNotIn("/*__PIPELINE_DATA__*/", html)
        payload = re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1)
        self.assertEqual(json.loads(payload)["scenario"]["name"], "unit")


class TestIntegrationFixes(unittest.TestCase):
    def test_milp_adapter_returns_a_valid_plan(self):
        scenario = ScenarioGenerator(seed=5).generate_scenario(2, 4)
        payload = adapters.milp_adapter(scenario, 0, 10.0)
        self.assertTrue(validate_assignments(payload["assignments"], scenario).valid)

    def test_reward_adapter_uses_the_stage1_clusters(self):
        from src.analytics.reward_adapters import RewardStageInput, preference_aware_reward_adapter
        scenario = ScenarioGenerator(seed=21).generate_scenario(6, 14)
        assignments = GreedyAllocator().solve(scenario).assignments
        options = {"method": "glauber", "num_samples_Tw": 200, "burn_in": 50}
        ids = [a.agent_id for a in scenario.agents]
        tasks = [t.task_id for t in scenario.tasks]
        split = [{"agents": ids[:3], "tasks": tasks[:7]}, {"agents": ids[3:], "tasks": tasks[7:]}]
        whole = preference_aware_reward_adapter(RewardStageInput(scenario, assignments, options), 1, 60)
        clustered = preference_aware_reward_adapter(RewardStageInput(scenario, assignments, options, split), 1, 60)
        self.assertEqual(set(whole.payoffs), set(ids))
        self.assertEqual(set(clustered.payoffs), set(ids))
        self.assertEqual(clustered.metadata["num_clusters"], 2)
        self.assertNotEqual([round(whole.payoffs[i], 6) for i in ids], [round(clustered.payoffs[i], 6) for i in ids])
        glob = preference_aware_reward_adapter(
            RewardStageInput(scenario, assignments, {**options, "decomposition": "global"}, split), 1, 60)
        self.assertEqual(glob.solver_name, "GlauberMCMC")             # old single-game path still available

    def test_adapters_expose_clusters_only_when_the_solver_made_them(self):
        scenario = ScenarioGenerator(seed=4).generate_scenario(4, 8)
        self.assertNotIn("clusters", adapters.greedy_adapter(scenario, 1, 30.0))
        payload = adapters.svca_paper_adapter(scenario, 1, 30.0)
        covered = sorted(t for c in payload["clusters"] for t in c["tasks"])
        self.assertEqual(covered, sorted(t.task_id for t in scenario.tasks))

    def test_benchmark_penalty_equals_the_allocator_objective(self):
        import yaml
        from src.analytics.evaluation import evaluate_allocation
        penalty = yaml.safe_load((ROOT / "configs" / "experiments_config.yaml").read_text(encoding="utf-8"))[
            "evaluation"]["energy_penalty"]
        scenario = ScenarioGenerator(seed=9).generate_scenario(3, 25)
        starved = {1: [t.task_id for t in scenario.tasks], 2: [], 3: []}     # forces an energy shortfall
        ev = AllocationEvaluator(scenario).evaluate(starved)
        self.assertGreater(ev.metadata["energy_penalty"], 0)
        self.assertAlmostEqual(evaluate_allocation(scenario, starved, penalty).allocation_cost, ev.total_cost, places=1)

    def test_run_plan_reward_seed_uses_the_given_config(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_benchmarks_fix", ROOT / "scripts" / "run_benchmarks.py")
        rb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rb)
        config_path = ROOT / "configs" / "experiments_config.yaml"
        config = rb.load_config(config_path)
        plan = rb.build_run_plan(config, config_path, only_configs=["C1"], repetitions=1)
        self.assertTrue(plan)
        for run in plan:
            self.assertEqual(run.reward_seed, seeding.reward_seed(
                run.scenario_seed, config["seeds"]["reward_stage_label"], config_path=config_path))


if __name__ == "__main__":
    unittest.main()

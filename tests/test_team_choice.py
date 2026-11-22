"""Tests for the robots-choose-their-team step (src/game_theory/team_choice.py, src/analytics/team_choice_eval.py)."""

import unittest

from src.allocation.svca_paper import PaperSVCAAllocator
from src.analytics.evaluation import validate_assignments
from src.analytics.team_choice_eval import evaluate_team_choice
from src.game_theory.team_choice import IDLE, TeamChoiceConfig, TeamGame, _best_option, choose_teams
from src.simulation.generator import ScenarioGenerator


def _scenario(seed=5, robots=10, tasks=16):
    return ScenarioGenerator(seed=seed).generate_scenario(robots, tasks)


def _clusters(scenario):
    return PaperSVCAAllocator().solve(scenario).metadata["clusters"]


class TestTeamChoiceDynamics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = _scenario()
        cls.clusters = _clusters(cls.scenario)

    def run_rule(self, rule, alpha=0.0, value_type="facility", **kw):
        cfg = TeamChoiceConfig(rule=rule, alpha=alpha, value_type=value_type, seed=3, **kw)
        return choose_teams(self.scenario, self.clusters, cfg), cfg

    def test_converged_partition_is_nash_stable(self):
        for rule, alpha in (("equal", 0.0), ("shapley", 0.0), ("preference", 1.5)):
            with self.subTest(rule=rule):
                res, cfg = self.run_rule(rule, alpha)
                if not res.converged:
                    continue
                game = TeamGame(self.scenario, cfg)
                teams = [{"agents": set(c["agents"]), "tasks": c["tasks"]} for c in res.final_clusters]
                idle = set(res.idle_robots)
                where = {r: k for k, t in enumerate(teams) for r in t["agents"]}
                where.update({r: IDLE for r in idle})
                for r, k in where.items():
                    cur = game.solve(sorted(teams[k]["agents"]), teams[k]["tasks"])[1][r] if k != IDLE else 0.0
                    self.assertIsNone(_best_option(game, teams, idle, r, k, cur, cfg)[0], f"R{r} still wants to move")

    def test_every_task_stays_covered_and_robots_are_conserved(self):
        res, _ = self.run_rule("equal")
        robots = sorted(a.agent_id for a in self.scenario.agents)
        placed = sorted([a for c in res.final_clusters for a in c["agents"]] + res.idle_robots)
        self.assertEqual(placed, robots)
        for c in res.final_clusters:
            if c["tasks"]:
                self.assertTrue(c["agents"], "a team with tasks lost its last robot")
        self.assertEqual(sorted(t for c in res.final_clusters for t in c["tasks"]),
                         sorted(t.task_id for t in self.scenario.tasks))

    def test_every_move_strictly_improves_the_mover(self):
        res, cfg = self.run_rule("shapley")
        for m in res.moves:
            self.assertGreater(m["payoff_after"], m["payoff_before"])

    def test_shapley_shares_sum_to_team_value(self):
        cfg = TeamChoiceConfig(rule="preference", alpha=1.5, value_type="facility", seed=1)
        game = TeamGame(self.scenario, cfg)
        for c in self.clusters:
            if c["agents"] and c["tasks"]:
                value, shares = game.solve(c["agents"], c["tasks"])
                self.assertAlmostEqual(sum(shares.values()), value, places=6)

    def test_deterministic_for_a_seed(self):
        a, _ = self.run_rule("preference", 1.5)
        b, _ = self.run_rule("preference", 1.5)
        self.assertEqual(a.moves, b.moves)
        self.assertEqual(a.final_clusters, b.final_clusters)

    def test_routing_value_runs_and_no_clusters_means_one_team(self):
        res = choose_teams(self.scenario, None, TeamChoiceConfig(rule="shapley", value_type="routing", seed=2))
        self.assertEqual(len(res.initial_clusters), 1)
        self.assertTrue(res.converged)

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            TeamChoiceConfig(rule="majority")
        with self.assertRaises(ValueError):
            TeamChoiceConfig(value_type="nope")
        with self.assertRaises(ValueError):
            TeamChoiceConfig(alpha=-1.0)


class TestTeamChoiceEvaluation(unittest.TestCase):
    def test_replanned_plan_is_valid_and_metrics_are_consistent(self):
        scenario = _scenario(seed=8, robots=8, tasks=14)
        rec = evaluate_team_choice(scenario, _clusters(scenario), "equal", seed=4, value_type="facility")
        self.assertTrue(validate_assignments(rec["routes_after"], scenario).valid)
        self.assertTrue(validate_assignments(rec["routes_before"], scenario).valid)
        self.assertEqual(rec["unstable_fraction"], rec["unstable_robots"] / rec["num_robots"])
        if rec["moves"] == 0:
            self.assertEqual(rec["cost_before"], rec["cost_after"])
            self.assertEqual(rec["tasks_reassigned"], 0)


if __name__ == "__main__":
    unittest.main()

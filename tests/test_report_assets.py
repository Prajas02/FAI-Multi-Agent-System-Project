"""Tests for the visualization layer and scripts/generate_report_assets.py.

All data here is a TEMPORARY SYNTHETIC FIXTURE written into temporary
directories. It exercises the code paths only: it is not a benchmark result and
is never written into the project's real result, figure or table folders.
"""

import contextlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analytics.logger import (
    BENCHMARK_RESULT_COLUMNS,
    aggregate_benchmark_results,
    append_benchmark_results,
    append_robot_rewards,
    write_experiment_metadata,
)
from src.analytics.metrics import compute_robot_reward_gini
from src.visualization import plots, tables

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "generate_report_assets.py"
_spec = importlib.util.spec_from_file_location("generate_report_assets", str(SCRIPT))
gra = importlib.util.module_from_spec(_spec)
sys.modules["generate_report_assets"] = gra
_spec.loader.exec_module(gra)

SIZES = [(5, 5), (5, 15), (10, 20)]


def record(algorithm, robots, tasks, rep, cost=100.0, allocation_runtime=1.0, success=True, allocation_done=True,
           reward_done=False, reward_runtime=None, total_runtime=None, gini=None, invalid=False, energy=True,
           experiment="exp", error=""):
    return dict(
        experiment_id=experiment, scenario_id="%dx%d_run%02d" % (robots, tasks, rep), algorithm=algorithm,
        num_robots=robots, num_tasks=tasks, total_entities=robots + tasks, random_seed=1,
        allocation_cost=cost if success else None, allocation_runtime_seconds=allocation_runtime,
        reward_runtime_seconds=reward_runtime, total_runtime_seconds=total_runtime,
        allocation_stage_completed=allocation_done, reward_stage_completed=reward_done, gini_index=gini,
        invalid_reward_flag=invalid, energy_feasible=energy if success else None, coalition_stability=None,
        success=success, error_message=error,
    )


def basic_records(algorithms=("MILP", "GA"), sizes=SIZES, reps=3, experiment="exp"):
    """Allocation-only records (no reward stage), costs grow with size."""
    rows = []
    for a_index, algorithm in enumerate(algorithms):
        for robots, tasks in sizes:
            for rep in range(reps):
                rows.append(record(algorithm, robots, tasks, rep, cost=(robots + tasks) * (10 + a_index) + rep,
                                   allocation_runtime=0.1 * (a_index + 1) * (robots + tasks) + 0.01 * rep,
                                   energy=(rep != 0), experiment=experiment))
    return rows


def summary_of(records):
    frame = pd.DataFrame(records, columns=list(BENCHMARK_RESULT_COLUMNS))
    return aggregate_benchmark_results(frame)


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="report_test_"))
        self.addCleanup(shutil.rmtree, str(self.dir), True)
        self.results = self.dir / "raw"
        self.figures = self.dir / "figures"
        self.tables = self.dir / "tables"

    def write_results(self, records, rewards=None):
        append_benchmark_results(records, self.results)
        if rewards:
            append_robot_rewards(rewards, self.results)

    def generate(self, **kwargs):
        return gra.generate_assets(self.results, figures_dir=self.figures, tables_dir=self.tables, **kwargs)

    def run_cli(self, *extra):
        argv = ["--results-dir", str(self.results), "--figures-dir", str(self.figures),
                "--tables-dir", str(self.tables)] + list(extra)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = gra.main(argv)
        return code, out.getvalue(), err.getvalue()

    def names(self, directory):
        return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


# ---------------------------------------------------------------------------
# plots.py
# ---------------------------------------------------------------------------

class TestMetricAvailability(unittest.TestCase):
    def test_allocation_only_data(self):
        status = plots.available_metrics(summary_of(basic_records()))
        for key in ("allocation_cost", "allocation_runtime", "energy_feasibility"):
            self.assertTrue(status[key].available, key)
        for key in ("reward_runtime", "total_runtime", "gini"):
            self.assertFalse(status[key].available, key)

    def test_coalition_stability_is_never_a_plottable_metric(self):
        self.assertFalse(any("coalition" in key for key in plots.METRIC_SPECS))
        self.assertFalse(any("coalition" in key for key in tables.TABLE_KEYS))

    def test_empty_summary_has_no_available_metric(self):
        empty = summary_of([])
        self.assertFalse(any(s.available for s in plots.available_metrics(empty).values()))

    def test_metric_with_only_nan_means_is_unavailable(self):
        rows = [record("MILP", 5, 5, rep, success=False, error="timeout", allocation_done=False)
                for rep in range(3)]
        status = plots.available_metrics(summary_of(rows))
        self.assertFalse(status["allocation_cost"].available)         # no successful run: no cost mean
        self.assertTrue(status["allocation_runtime"].available is False)   # stage flag False: excluded


def series_lines(axes):
    """The plotted series (errorbar caps and bars are extra, unlabeled Line2D objects)."""
    return [line for line in axes.lines if not line.get_label().startswith("_")]


class TestPlotMetric(unittest.TestCase):
    def setUp(self):
        self.summary = summary_of(basic_records(("MILP", "GA", "Greedy", "SVCA")))

    def axes_of(self, figure):
        return figure.axes[0]

    def test_unavailable_metric_raises(self):
        with self.assertRaises(plots.MetricUnavailable):
            plots.plot_metric(self.summary, "gini")
        with self.assertRaises(plots.MetricUnavailable):
            plots.plot_metric(self.summary, "total_runtime")

    def test_labels_titles_legend_and_one_line_per_algorithm(self):
        figure = plots.plot_metric(self.summary, "allocation_cost")
        axes = self.axes_of(figure)
        self.assertEqual([t.get_text() for t in axes.get_legend().get_texts()], ["MILP", "GA", "Greedy", "SVCA"])
        self.assertIn("Allocation cost", figure._suptitle.get_text())
        self.assertIn("A(U)", axes.get_ylabel())
        self.assertIn("total entities N", axes.get_xlabel())
        self.assertEqual(len(series_lines(axes)), 4)

    def test_x_axis_shows_n_robots_and_tasks_for_every_configuration(self):
        axes = self.axes_of(plots.plot_metric(self.summary, "allocation_cost"))
        labels = [t.get_text() for t in axes.get_xticklabels()]
        self.assertEqual(labels, ["N=10\n5R / 5T", "N=20\n5R / 15T", "N=30\n10R / 20T"])

    def test_consistent_style_per_algorithm_even_when_others_are_absent(self):
        full = self.axes_of(plots.plot_metric(self.summary, "allocation_cost"))
        subset = summary_of(basic_records(("GA", "SVCA")))
        part = self.axes_of(plots.plot_metric(subset, "allocation_cost"))
        full_style = {l.get_label(): (l.get_color(), l.get_marker()) for l in series_lines(full)}
        part_style = {l.get_label(): (l.get_color(), l.get_marker()) for l in series_lines(part)}
        self.assertEqual(part_style["GA"], full_style["GA"])
        self.assertEqual(part_style["SVCA"], full_style["SVCA"])
        self.assertEqual(len({v[0] for v in full_style.values()}), 4)
        self.assertEqual(len({v[1] for v in full_style.values()}), 4)         # marker shape is a second encoding

    def test_missing_cells_are_not_drawn_as_zero(self):
        rows = basic_records(("MILP", "GA"))
        rows = [r for r in rows if not (r["algorithm"] == "MILP" and r["num_tasks"] == 15)]   # MILP: no middle size
        axes = self.axes_of(plots.plot_metric(summary_of(rows), "allocation_cost"))
        milp = [l for l in series_lines(axes) if l.get_label() == "MILP"][0]
        values = np.asarray(milp.get_ydata(), dtype=float)
        self.assertTrue(np.isnan(values[1]))
        self.assertTrue(np.isfinite(values[[0, 2]]).all())
        for line in series_lines(axes):
            self.assertNotIn(0.0, [v for v in np.asarray(line.get_ydata(), dtype=float) if np.isfinite(v)])

    def test_failed_runs_are_excluded_from_the_cost_means(self):
        rows = basic_records(("MILP",), sizes=[(5, 5)], reps=3)
        rows += [record("MILP", 5, 5, 3, success=False, error="timeout", allocation_done=False),
                 record("MILP", 5, 5, 4, success=False, error="invalid allocation: x")]
        summary = summary_of(rows)
        costs = [r["allocation_cost"] for r in rows if r["success"]]
        axes = self.axes_of(plots.plot_metric(summary, "allocation_cost"))
        self.assertAlmostEqual(float(series_lines(axes)[0].get_ydata()[0]), sum(costs) / len(costs), places=9)
        self.assertEqual(int(summary.loc[0, "n_allocation_cost"]), 3)
        self.assertEqual(int(summary.loc[0, "n_runs"]), 5)

    def test_runtimes_respect_stage_completion(self):
        rows = [record("GA", 5, 5, 0, allocation_runtime=2.0),
                record("GA", 5, 5, 1, allocation_runtime=4.0),
                record("GA", 5, 5, 2, success=False, allocation_done=False, allocation_runtime=300.0,
                       error="TimeoutError")]       # elapsed time of a stopped stage: must not enter the mean
        axes = self.axes_of(plots.plot_metric(summary_of(rows), "allocation_runtime"))
        self.assertAlmostEqual(float(series_lines(axes)[0].get_ydata()[0]), 3.0)

    def test_error_bars_only_where_a_std_exists(self):
        rows = [record("MILP", 5, 5, 0), record("MILP", 5, 5, 1, cost=120.0), record("MILP", 5, 15, 0)]  # n=2 and n=1
        axes = self.axes_of(plots.plot_metric(summary_of(rows), "allocation_cost"))
        self.assertEqual(len(axes.containers), 1)
        segments = axes.containers[0].lines[2][0].get_segments()        # the error bars
        self.assertEqual(len(segments), 1)                              # only the n=2 cell has a std
        self.assertAlmostEqual(float(segments[0][0][0]), 0.0)           # ... at the first configuration

    def test_single_configuration_is_plottable(self):
        figure = plots.plot_metric(summary_of(basic_records(sizes=[(5, 5)])), "allocation_cost")
        self.assertEqual(len(self.axes_of(figure).get_xticklabels()), 1)

    def test_eight_configurations_keep_all_labels(self):
        sizes = [(5, 5), (5, 15), (10, 20), (15, 25), (20, 30), (25, 35), (30, 40), (35, 45)]
        axes = self.axes_of(plots.plot_metric(summary_of(basic_records(sizes=sizes)), "allocation_cost"))
        self.assertEqual([t.get_text().split("\n")[0] for t in axes.get_xticklabels()],
                         ["N=10", "N=20", "N=30", "N=40", "N=50", "N=60", "N=70", "N=80"])

    def test_log_scale_only_for_runtimes_spanning_orders_of_magnitude(self):
        wide = [record("MILP", 5, 5, 0, allocation_runtime=0.01), record("MILP", 5, 15, 0, allocation_runtime=100.0)]
        self.assertEqual(self.axes_of(plots.plot_metric(summary_of(wide), "allocation_runtime")).get_yscale(), "log")
        narrow = [record("MILP", 5, 5, 0, allocation_runtime=1.0), record("MILP", 5, 15, 0, allocation_runtime=3.0)]
        self.assertEqual(self.axes_of(plots.plot_metric(summary_of(narrow), "allocation_runtime")).get_yscale(),
                         "linear")
        self.assertEqual(self.axes_of(plots.plot_metric(summary_of(wide), "allocation_cost")).get_yscale(), "linear")

    def test_allocation_cost_uses_readable_log_scale_for_wide_positive_range(self):
        rows = [record("MILP", 5, 5, 0, cost=10.0), record("MILP", 5, 15, 1, cost=1_000_000.0)]
        axes = self.axes_of(plots.plot_metric(summary_of(rows), "allocation_cost"))
        self.assertEqual(axes.get_yscale(), "log")
        self.assertIn("logarithmic", axes.get_ylabel())
        self.assertNotIn("1e", " ".join(t.get_text() for t in axes.get_yticklabels()))

    def test_energy_feasibility_is_a_share_axis(self):
        axes = self.axes_of(plots.plot_metric(self.summary, "energy_feasibility"))
        self.assertEqual([t.get_text() for t in axes.get_yticklabels()], ["0%", "25%", "50%", "75%", "100%"])
        self.assertAlmostEqual(float(series_lines(axes)[0].get_ydata()[0]), 2 / 3, places=9)   # reps 1, 2 feasible of 3

    def test_footnote_is_shown(self):
        figure = plots.plot_metric(self.summary, "allocation_cost", footnote="PROVISIONAL penalty note")
        self.assertTrue(any("PROVISIONAL penalty note" in t.get_text() for t in figure.texts))
        self.assertTrue(any("not zero" in t.get_text() for t in figure.texts))

    def test_reward_metrics_plot_when_data_exists(self):
        rows = [record("GA", 5, 5, rep, reward_done=True, reward_runtime=2.0 + rep, total_runtime=10.0 + rep,
                       gini=0.2 + 0.1 * rep) for rep in range(3)]
        summary = summary_of(rows)
        for key in ("reward_runtime", "total_runtime", "gini"):
            self.assertTrue(plots.metric_status(summary, key).available, key)
            figure = plots.plot_metric(summary, key)
            self.assertAlmostEqual(float(series_lines(figure.axes[0])[0].get_ydata()[0]),
                                   {"reward_runtime": 3.0, "total_runtime": 11.0, "gini": 0.3}[key], places=9)


class TestProblemSizesAndSaving(TempDirTestCase):
    def test_problem_size_figure_shows_robots_and_tasks(self):
        figure = plots.plot_problem_sizes(summary_of(basic_records()))
        axes = figure.axes[0]
        heights = [p.get_height() for p in axes.patches]
        self.assertEqual(sorted(heights), sorted([5, 5, 5, 15, 10, 20]))
        self.assertIn("Robots", [t.get_text() for t in axes.get_legend().get_texts()][0])

    def test_problem_sizes_of_an_empty_summary(self):
        with self.assertRaises(plots.MetricUnavailable):
            plots.plot_problem_sizes(summary_of([]))

    def test_save_figure_writes_png_and_pdf_and_reruns_cleanly(self):
        figure = plots.plot_metric(summary_of(basic_records()), "allocation_cost")
        first = plots.save_figure(figure, self.dir / "figs" / "fig")
        self.assertEqual(sorted(p.name for p in first), ["fig.pdf", "fig.png"])
        self.assertEqual((self.dir / "figs" / "fig.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual((self.dir / "figs" / "fig.pdf").read_bytes()[:5], b"%PDF-")
        plots.save_figure(figure, self.dir / "figs" / "fig")                 # overwrite
        self.assertEqual(self.names(self.dir / "figs"), ["fig.pdf", "fig.png"])   # no temp files left

    def test_unsupported_format_is_rejected(self):
        figure = plots.plot_problem_sizes(summary_of(basic_records()))
        with self.assertRaises(ValueError):
            plots.save_figure(figure, self.dir / "x", ("svg",))
        self.assertEqual(self.names(self.dir), [])


# ---------------------------------------------------------------------------
# tables.py
# ---------------------------------------------------------------------------

class TestTables(TempDirTestCase):
    def test_cells_show_mean_std_and_valid_count(self):
        table = tables.build_table(summary_of(basic_records()), "allocation_cost")
        self.assertEqual(list(table.columns), ["N", "Robots", "Tasks", "MILP", "GA"])
        self.assertEqual(table.loc[0, "MILP"], "101.0 ± 1.0 (n=3)")      # costs 100, 101, 102
        self.assertEqual(table.loc[0, "GA"], "111.0 ± 1.0 (n=3)")

    def test_missing_cells_read_not_available_never_zero(self):
        rows = [r for r in basic_records() if not (r["algorithm"] == "GA" and r["num_tasks"] == 15)]
        table = tables.build_table(summary_of(rows), "allocation_cost")
        self.assertEqual(table.loc[1, "GA"], tables.NOT_AVAILABLE)
        self.assertNotIn("0.0 (n", table.loc[1, "GA"])

    def test_single_valid_run_has_no_std(self):
        table = tables.build_table(summary_of([record("MILP", 5, 5, 0, cost=50.0)]), "allocation_cost")
        self.assertEqual(table.loc[0, "MILP"], "50.0 (n=1)")

    def test_run_outcomes_table_shows_failed_runs(self):
        rows = basic_records(("MILP",), sizes=[(5, 5)]) + [record("MILP", 5, 5, 9, success=False, error="x")]
        table = tables.build_table(summary_of(rows), tables.RUN_OUTCOMES_KEY)
        self.assertEqual(table.loc[0, "MILP"], "3/4")

    def test_energy_feasibility_table(self):
        table = tables.build_table(summary_of(basic_records()), "energy_feasibility")
        self.assertEqual(table.loc[0, "MILP"], "67% (eligible successful runs)")

    def test_energy_feasibility_label_does_not_claim_n_success_as_the_denominator(self):
        # 3 successful runs, but one has no recorded feasibility flag: the aggregation's denominator is 2, not 3
        rows = [record("MILP", 5, 5, 0, energy=True), record("MILP", 5, 5, 1, energy=False),
                record("MILP", 5, 5, 2, energy=None)]
        summary = summary_of(rows)
        self.assertEqual(int(summary.loc[0, "n_success"]), 3)
        self.assertAlmostEqual(float(summary.loc[0, "energy_feasible_rate"]), 0.5)
        cell = tables.build_table(summary, "energy_feasibility").loc[0, "MILP"]
        self.assertEqual(cell, "50% (eligible successful runs)")
        self.assertNotIn("of 3", cell)
        self.assertNotIn("(of", cell)
        self.assertIn("recorded feasibility flag", tables.TABLE_TITLES["energy_feasibility"])

    def test_energy_feasibility_without_any_flag_is_not_available(self):
        table = tables.build_table(summary_of([record("MILP", 5, 5, 0, energy=None)]), "energy_feasibility")
        self.assertEqual(table.loc[0, "MILP"], tables.NOT_AVAILABLE)

    def test_unknown_table_is_rejected(self):
        with self.assertRaises(ValueError):
            tables.build_table(summary_of(basic_records()), "coalition_stability")

    def test_files_and_latex_escaping(self):
        table = tables.build_table(summary_of(basic_records()), "energy_feasibility")
        files = tables.write_table(table, self.dir / "t", "Share of runs (100% feasible) & more", "tab:x")
        self.assertEqual(sorted(p.suffix for p in files), [".csv", ".md", ".tex"])
        latex = (self.dir / "t.tex").read_text(encoding="utf-8")
        self.assertIn(r"\caption{Share of runs (100\% feasible) \& more}", latex)
        self.assertIn(r"67\% (eligible successful runs)", latex)
        self.assertIn(r"\begin{tabular}{rrrll}", latex)
        markdown = (self.dir / "t.md").read_text(encoding="utf-8")
        self.assertIn("| N | Robots | Tasks | MILP | GA |", markdown)
        self.assertEqual(self.names(self.dir), ["t.csv", "t.md", "t.tex"])
        before = {p.name: p.read_bytes() for p in files}
        tables.write_table(table, self.dir / "t", "Share of runs (100% feasible) & more", "tab:x")
        self.assertEqual({p.name: p.read_bytes() for p in files}, before)         # rerun is identical

    def test_plus_minus_sign_in_latex(self):
        table = tables.build_table(summary_of(basic_records()), "allocation_cost")
        tables.write_table(table, self.dir / "c", "Cost", "tab:c")
        self.assertIn(r"$\pm$", (self.dir / "c.tex").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# scripts/generate_report_assets.py
# ---------------------------------------------------------------------------

class TestNoData(TempDirTestCase):
    def assert_nothing_written(self):
        self.assertFalse(self.figures.exists())
        self.assertFalse(self.tables.exists())

    def test_no_result_file(self):
        with self.assertRaisesRegex(gra.NothingToReport, "no benchmark result file"):
            self.generate()
        code, out, err = self.run_cli()
        self.assertEqual(code, 0, err)
        self.assertIn("Nothing to report", out)
        self.assertIn("Reward fairness analysis is pending reward-solver integration", out)
        self.assert_nothing_written()

    def test_empty_result_files(self):
        self.results.mkdir()
        (self.results / "benchmark_results.csv").write_bytes(b"")                      # zero bytes
        self.assertEqual(self.run_cli()[0], 0)
        (self.results / "benchmark_results.csv").write_text(",".join(BENCHMARK_RESULT_COLUMNS) + "\n")   # header only
        code, out, err = self.run_cli()
        self.assertEqual(code, 0, err)
        self.assertIn("contains no records", out)
        self.assert_nothing_written()

    def test_unknown_experiment_and_ambiguous_experiments(self):
        self.write_results(basic_records(experiment="a") + basic_records(experiment="b"))
        code, _, err = self.run_cli()
        self.assertEqual(code, 2)
        self.assertIn("several experiments", err)
        code, _, err = self.run_cli("--experiment-id", "nope")
        self.assertEqual(code, 2)
        self.assertIn("no records", err)
        self.assert_nothing_written()
        self.assertEqual(self.run_cli("--experiment-id", "a")[0], 0)

    def test_corrupt_result_file(self):
        self.write_results(basic_records())
        path = self.results / "benchmark_results.csv"
        path.write_text(path.read_text().replace("True", "maybe", 1))
        code, _, err = self.run_cli()
        self.assertEqual(code, 4)
        self.assertIn("not a valid result file", err)
        self.assert_nothing_written()

    def test_invalid_formats_option(self):
        self.write_results(basic_records())
        self.assertEqual(self.run_cli("--formats", "svg")[0], 2)
        self.assert_nothing_written()


class TestGeneratedAssets(TempDirTestCase):
    def setUp(self):
        super().setUp()
        rows = basic_records(("MILP", "GA"))
        rows.append(record("MILP", 5, 5, 7, success=False, allocation_done=False, allocation_runtime=300.0,
                           error="TimeoutError: limit"))
        rows.append(record("GA", 5, 15, 7, success=False, error="invalid allocation: task 1 is assigned 2 times"))
        self.write_results(rows)
        write_experiment_metadata("exp", {"evaluation": {"energy_penalty": {"strategy": "quadratic", "lambda_E": 1.0}}},
                                  {"base_seed": 42}, {}, extra={"harness": {"backend": "subprocess"}},
                                  results_dir=self.results)

    def test_allocation_only_data_generates_available_metrics_only(self):
        report = self.generate()
        figures, tables_dir = self.names(self.figures), self.names(self.tables)
        for stem in ("allocation_cost", "allocation_runtime", "energy_feasibility", "problem_sizes"):
            self.assertIn("exp_%s.png" % stem, figures)
            self.assertIn("exp_%s.pdf" % stem, figures)
        for stem in ("reward_runtime", "total_runtime", "reward_fairness_gini", "coalition_stability"):
            self.assertFalse([f for f in figures if stem in f], stem)
        for key in ("allocation_cost", "allocation_runtime", "energy_feasibility", "run_outcomes"):
            for suffix in (".csv", ".tex", ".md"):
                self.assertIn("exp_table_%s%s" % (key, suffix), tables_dir)
        self.assertIn("summary_exp.csv", tables_dir)
        self.assertIn("exp_report_summary.json", tables_dir)
        self.assertFalse([f for f in tables_dir if "reward" in f or "gini" in f or "total_runtime" in f
                          or "coalition" in f])
        self.assertEqual(report["unavailable_metrics"], ["gini", "reward_runtime", "total_runtime"])

    def test_machine_readable_summary(self):
        self.generate()
        report = json.loads((self.tables / "exp_report_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["experiment_id"], "exp")
        self.assertEqual(report["algorithms"], ["MILP", "GA"])
        self.assertEqual(report["runs_per_algorithm"]["MILP"], {"runs": 10, "successful_runs": 9, "failed_runs": 1})
        self.assertEqual(report["runs_per_algorithm"]["GA"]["failed_runs"], 1)
        self.assertEqual(len(report["configurations"]), 3)
        for key in ("reward_runtime", "total_runtime", "gini"):
            self.assertFalse(report["metrics"][key]["available"])
            self.assertIn("pending reward-solver integration", report["metrics"][key]["reason"])
        self.assertTrue(report["metrics"]["allocation_cost"]["available"])
        self.assertIn("pending a formal definition", report["coalition_stability"])
        self.assertIn("pending reward-solver integration", report["reward_fairness_analysis"])
        self.assertIn("provisional", report["energy_penalty"]["status"])
        self.assertEqual(report["energy_penalty"]["lambda_E"], 1.0)
        notes = " ".join(report["notes"])
        self.assertIn("make no claim about which algorithm is better", notes)
        self.assertIn("PROVISIONAL", notes)
        self.assertNotIn("outperform", json.dumps(report).lower())

    def test_failed_runs_do_not_enter_cost_means_and_are_counted(self):
        self.generate()
        summary = pd.read_csv(self.tables / "summary_exp.csv")
        milp = summary[(summary["algorithm"] == "MILP") & (summary["num_tasks"] == 5)].iloc[0]
        self.assertEqual(milp["n_runs"], 4)
        self.assertEqual(milp["n_success"], 3)
        self.assertEqual(milp["n_allocation_cost"], 3)
        self.assertAlmostEqual(milp["allocation_cost_mean"], 101.0)        # costs 100, 101, 102
        self.assertEqual(milp["n_allocation_runtime"], 3)                  # the timed-out stage is excluded
        table = pd.read_csv(self.tables / "exp_table_run_outcomes.csv")
        self.assertEqual(table.loc[0, "MILP"], "3/4")

    def test_provisional_penalty_label_on_the_cost_figure_and_table(self):
        self.generate()
        report = json.loads((self.tables / "exp_report_summary.json").read_text(encoding="utf-8"))
        self.assertIn("PROVISIONAL energy penalty (quadratic, lambda_E=1.0)", " ".join(report["notes"]))
        latex = (self.tables / "exp_table_allocation_cost.tex").read_text(encoding="utf-8")
        self.assertIn("PROVISIONAL", latex)

    def test_penalty_note_without_metadata(self):
        (self.results / "metadata_exp.yaml").unlink()
        report = self.generate()
        self.assertIn("unknown", " ".join(report["notes"]))
        self.assertIn("PROVISIONAL", " ".join(report["notes"]))
        self.assertIn("unknown", report["energy_penalty"]["status"])

    def test_rerun_is_safe_and_deterministic(self):
        self.generate()
        first_names = (self.names(self.figures), self.names(self.tables))
        first_text = {p.name: p.read_bytes() for p in self.tables.iterdir()}
        self.generate()
        self.assertEqual((self.names(self.figures), self.names(self.tables)), first_names)
        self.assertEqual({p.name: p.read_bytes() for p in self.tables.iterdir()}, first_text)
        self.assertFalse([n for n in first_names[0] + first_names[1] if n.endswith(".tmp")])

    def test_png_only(self):
        self.generate(formats=("png",))
        self.assertFalse([f for f in self.names(self.figures) if f.endswith(".pdf")])

    def test_cli_prints_generated_and_unavailable(self):
        code, out, err = self.run_cli()
        self.assertEqual(code, 0, err)
        self.assertIn("exp_allocation_cost.png", out)
        self.assertIn("unavailable metrics", out)
        self.assertIn("reward_runtime: pending reward-solver integration", out)
        self.assertIn("coalition_stability: pending a formal definition", out)
        self.assertIn("reward fairness analysis: pending reward-solver integration", out)
        self.assertIn("PROVISIONAL", out)

    def test_nothing_is_written_to_the_real_project_folders(self):
        real = [REPO_ROOT / "outputs" / "figures", REPO_ROOT / "outputs" / "tables", REPO_ROOT / "data" / "results"]
        before = [sorted(str(p) for p in d.rglob("*")) for d in real]
        self.generate()
        self.assertEqual([sorted(str(p) for p in d.rglob("*")) for d in real], before)

    def test_additional_visualizations_and_tables_are_reported(self):
        report = self.generate()
        for key in ("allocation_cost_distribution", "energy_feasibility_heatmap", "run_reliability",
                    "cost_runtime_tradeoff"):
            self.assertTrue(report["visualizations"][key]["available"], key)
            self.assertTrue(any(key in Path(p).name for p in report["figures"][key]))
        self.assertFalse(report["visualizations"]["paired_cost_comparison"]["available"])
        self.assertIn("route_allocation", report["unavailable_visualizations"])
        for stem in ("allocation_cost_distribution", "reliability"):
            self.assertTrue((self.tables / f"exp_table_{stem}.csv").exists())

    def test_cost_distributions_ignore_failed_and_missing_costs(self):
        rows = [record("MILP", 5, 5, 0, cost=10.0),
                record("MILP", 5, 5, 1, success=False, error="failure"),
                record("MILP", 5, 5, 2, cost=None)]
        frame = pd.DataFrame(rows)
        fig = plots.plot_cost_distributions(frame)
        axes = fig.axes
        self.assertEqual(len(axes), 1)
        self.assertEqual(axes[0].get_xticklabels()[0].get_text(), "MILP")

    def test_paired_costs_require_scenario_match_and_positive_baseline(self):
        rows = [record("Greedy", 5, 5, 0, cost=10), record("GA", 5, 5, 0, cost=5),
                record("Greedy", 5, 5, 1, cost=0), record("GA", 5, 5, 1, cost=1),
                record("Greedy", 5, 5, 2, cost=10)]
        fig = plots.plot_paired_costs(pd.DataFrame(rows))
        self.assertIn("Per-scenario cost relative", fig._suptitle.get_text())
        self.assertEqual(len(fig.axes[0].collections), 1)  # only the matched, positive-baseline scenario


class TestRewardData(TempDirTestCase):
    def reward_run(self, algorithm, rep, rewards, robots=3, stored_gini="recompute", invalid=False, experiment="exp"):
        gini = (compute_robot_reward_gini(list(enumerate(rewards))).gini_index
                if stored_gini == "recompute" else stored_gini)
        main = record(algorithm, robots, 7, rep, reward_done=True, reward_runtime=2.0, total_runtime=9.0,
                      gini=None if invalid or (isinstance(gini, float) and math.isnan(gini)) else gini,
                      invalid=invalid, experiment=experiment)
        rows = [dict(experiment_id=experiment, scenario_id=main["scenario_id"], algorithm=algorithm, robot_id=i,
                     reward=float(x)) for i, x in enumerate(rewards)]
        return main, rows

    def write(self, runs):
        append_robot_rewards([row for _, rows in runs for row in rows], self.results)
        append_benchmark_results([main for main, _ in runs], self.results)

    def test_valid_reward_records_enable_fairness_and_reward_figures(self):
        self.write([self.reward_run("GA", rep, [1.0, 2.0, 3.0 + rep]) for rep in range(3)])
        report = self.generate()
        for key in ("reward_runtime", "total_runtime", "gini"):
            self.assertTrue(report["metrics"][key]["available"], key)
        self.assertEqual(report["reward_fairness_analysis"], "available")
        self.assertIn("exp_reward_fairness_gini.png", self.names(self.figures))
        self.assertIn("exp_table_gini.csv", self.names(self.tables))
        self.assertEqual(report["source"]["runs_with_reward_records"], 3)
        self.assertEqual(report["warnings"], [])

    def test_gini_is_recomputed_from_reward_records(self):
        run = self.reward_run("GA", 0, [1.0, 2.0, 3.0], stored_gini=0.9)          # stored value disagrees
        self.write([run])
        report = self.generate()
        self.assertEqual(report["source"]["stored_gini_mismatching_records"], 1)
        self.assertTrue(any("recomputed" in w for w in report["warnings"]))
        table = pd.read_csv(self.tables / "exp_table_gini.csv")
        expected = compute_robot_reward_gini([(0, 1.0), (1, 2.0), (2, 3.0)]).gini_index
        self.assertTrue(str(table.loc[0, "GA"]).startswith("%.3f" % expected))

    def test_stored_gini_without_reward_records_is_excluded(self):
        main, _ = self.reward_run("GA", 0, [1.0, 2.0, 3.0])
        self.write_results([main])                                                  # no robot_rewards.csv at all
        report = self.generate()
        self.assertFalse(report["metrics"]["gini"]["available"])
        self.assertIn("no valid reward measurements", report["metrics"]["gini"]["reason"])
        self.assertEqual(report["source"]["stored_gini_without_reward_records"], 1)
        self.assertTrue(any("no robot reward records" in w for w in report["warnings"]))
        self.assertFalse([f for f in self.names(self.figures) if "gini" in f])
        self.assertEqual(report["reward_fairness_analysis"].startswith("pending"), True)
        self.assertTrue(report["metrics"]["reward_runtime"]["available"])           # runtimes are independent of Gini

    # --- completeness and eligibility of the reward records ------------------------------------

    def run_with_records(self, rewards, robots=3, reward_done=True, invalid=False):
        main, rows = self.reward_run("GA", 0, rewards, robots=robots, invalid=invalid)
        if not reward_done:
            main = dict(main, reward_stage_completed=False, reward_runtime_seconds=None, total_runtime_seconds=None,
                        gini_index=None)
        return main, rows

    def test_exactly_one_record_per_robot_is_used(self):
        self.write([self.run_with_records([1.0, 2.0, 3.0])])
        report = self.generate()
        self.assertTrue(report["metrics"]["gini"]["available"])
        self.assertEqual(report["source"]["runs_with_reward_records"], 1)
        for key in ("incomplete_reward_groups", "reward_groups_fewer_than_robots", "reward_groups_more_than_robots",
                    "reward_records_on_ineligible_runs"):
            self.assertEqual(report["source"][key], 0, key)
        self.assertEqual(report["warnings"], [])

    def test_fewer_reward_records_than_robots_give_no_gini(self):
        main, rows = self.run_with_records([1.0, 2.0, 3.0])
        self.write([(main, rows[:2])])                                    # 2 records for 3 robots
        report = self.generate()
        self.assertFalse(report["metrics"]["gini"]["available"])
        self.assertEqual(report["source"]["incomplete_reward_groups"], 1)
        self.assertEqual(report["source"]["reward_groups_fewer_than_robots"], 1)
        self.assertEqual(report["source"]["reward_groups_more_than_robots"], 0)
        self.assertEqual(report["source"]["runs_with_reward_records"], 0)
        self.assertTrue(any("exactly one record per robot" in w and "1 with fewer" in w
                            for w in report["warnings"]), report["warnings"])
        self.assertFalse([f for f in self.names(self.figures) if "gini" in f])

    def test_more_reward_records_than_robots_give_no_gini(self):
        main, rows = self.run_with_records([1.0, 2.0, 3.0])
        extra = dict(rows[0], robot_id=99, reward=5.0)
        self.write([(main, rows + [extra])])                              # 4 records for 3 robots
        report = self.generate()
        self.assertFalse(report["metrics"]["gini"]["available"])
        self.assertEqual(report["source"]["incomplete_reward_groups"], 1)
        self.assertEqual(report["source"]["reward_groups_more_than_robots"], 1)
        self.assertEqual(report["source"]["reward_groups_fewer_than_robots"], 0)
        self.assertTrue(any("1 with more" in w for w in report["warnings"]), report["warnings"])

    def test_incomplete_group_does_not_affect_other_valid_runs(self):
        good = self.reward_run("GA", 1, [1.0, 2.0, 4.0])
        bad_main, bad_rows = self.reward_run("GA", 2, [1.0, 2.0, 3.0])
        self.write([good, (bad_main, bad_rows[:1])])
        report = self.generate()
        self.assertTrue(report["metrics"]["gini"]["available"])
        summary = pd.read_csv(self.tables / "summary_exp.csv")
        self.assertEqual(summary.loc[0, "n_gini"], 1)                     # only the complete group counts
        self.assertEqual(report["source"]["runs_with_reward_records"], 1)

    def test_reward_records_on_a_run_whose_reward_stage_did_not_complete_are_not_used(self):
        main, rows = self.run_with_records([1.0, 2.0, 3.0], reward_done=False)
        self.write([(main, rows)])
        report = self.generate()
        self.assertFalse(report["metrics"]["gini"]["available"])
        self.assertEqual(report["source"]["reward_records_on_ineligible_runs"], 1)
        self.assertEqual(report["source"]["runs_with_reward_records"], 0)
        self.assertEqual(report["source"]["incomplete_reward_groups"], 0)
        self.assertTrue(any("did not complete or their reward flag is invalid" in w for w in report["warnings"]),
                        report["warnings"])

    def test_reward_records_on_a_run_with_an_invalid_reward_flag_are_not_used(self):
        main, rows = self.run_with_records([1.0, 2.0, 3.0], invalid=True)   # flag set although the values are valid
        self.write([(main, rows)])
        report = self.generate()
        self.assertFalse(report["metrics"]["gini"]["available"])
        self.assertEqual(report["source"]["reward_records_on_ineligible_runs"], 1)
        summary = pd.read_csv(self.tables / "summary_exp.csv")
        self.assertEqual(summary.loc[0, "n_invalid_reward"], 1)
        self.assertEqual(summary.loc[0, "n_gini"], 0)

    def test_stored_gini_on_an_ineligible_run_is_discarded(self):
        main, rows = self.run_with_records([1.0, 2.0, 3.0])
        frame = pd.DataFrame([dict(main, reward_stage_completed=False, gini_index=0.5)],
                             columns=list(BENCHMARK_RESULT_COLUMNS))
        checked, stats = gra.validate_gini_with_reward_records(frame, pd.DataFrame(rows))
        self.assertTrue(checked["gini_index"].isna().all())
        self.assertEqual(stats["reward_records_on_ineligible_runs"], 1)

    def test_eligible_set_uses_the_existing_gini_metric(self):
        main, rows = self.run_with_records([1.0, 2.0, 4.0])
        frame = pd.DataFrame([dict(main, gini_index=None)], columns=list(BENCHMARK_RESULT_COLUMNS))
        checked, stats = gra.validate_gini_with_reward_records(frame, pd.DataFrame(rows))
        expected = compute_robot_reward_gini([(0, 1.0), (1, 2.0), (2, 4.0)]).gini_index
        self.assertAlmostEqual(float(checked.loc[0, "gini_index"]), expected, places=12)
        self.assertEqual(stats["runs_with_reward_records"], 1)

    def test_orphan_groups_stay_excluded_alongside_the_new_checks(self):
        main, rows = self.run_with_records([1.0, 2.0, 3.0])
        orphan = [dict(r, scenario_id="never_recorded") for r in rows]
        append_robot_rewards(rows + orphan, self.results)
        append_benchmark_results([main], self.results)
        report = self.generate()
        self.assertEqual(report["source"]["orphan_reward_record_groups"], 1)
        self.assertEqual(report["source"]["runs_with_reward_records"], 1)
        self.assertEqual(report["source"]["incomplete_reward_groups"], 0)

    def test_invalid_negative_rewards_produce_no_gini(self):
        runs = [self.reward_run("GA", rep, [1.0, -2.0, 3.0], invalid=True) for rep in range(2)]
        self.write(runs)
        report = self.generate()
        self.assertFalse(report["metrics"]["gini"]["available"])
        summary = pd.read_csv(self.tables / "summary_exp.csv")
        self.assertEqual(summary.loc[0, "n_invalid_reward"], 2)
        self.assertEqual(summary.loc[0, "n_gini"], 0)

    def test_orphan_reward_records_are_ignored_with_a_warning(self):
        main, rows = self.reward_run("GA", 0, [1.0, 2.0, 3.0])
        orphan = dict(rows[0], scenario_id="never_recorded")
        append_robot_rewards(rows + [orphan], self.results)
        append_benchmark_results([main], self.results)
        report = self.generate()
        self.assertEqual(report["source"]["orphan_reward_record_groups"], 1)
        self.assertTrue(any("no matching main result record" in w for w in report["warnings"]))

    def test_rewards_of_another_experiment_are_not_used(self):
        main, rows = self.reward_run("GA", 0, [1.0, 2.0, 3.0], experiment="other")
        self.write_results(basic_records(experiment="exp"))
        append_robot_rewards(rows, self.results)
        report = self.generate(experiment_id="exp")
        self.assertEqual(report["source"]["runs_with_reward_records"], 0)

    def test_unavailable_reason_distinguishes_pending_from_invalid(self):
        self.write_results(basic_records())
        reason = self.generate()["metrics"]["gini"]["reason"]
        self.assertIn("pending reward-solver integration", reason)


if __name__ == "__main__":
    unittest.main()

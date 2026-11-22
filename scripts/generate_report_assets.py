"""Generate report figures and tables from the benchmark result files.

    python scripts/generate_report_assets.py                      # the single experiment in the result files
    python scripts/generate_report_assets.py --experiment-id ID   # a specific experiment

Reads ``data/results/raw/benchmark_results.csv`` (and ``robot_rewards.csv`` when
present), reuses the existing aggregation (``aggregate_benchmark_results``) and
writes, flat and named by experiment id:

    outputs/figures/<id>_<figure>.png / .pdf       one figure per AVAILABLE metric (+ problem sizes)
    outputs/tables/<id>_table_<table>.csv/.tex/.md report tables (cells: mean ± std (n=valid runs))
    outputs/tables/summary_<id>.csv                aggregated summary (existing logger writer)
    outputs/tables/<id>_report_summary.json        machine-readable summary (assets, availability, notes)

Rules:
- No result file, an empty one, or no valid measurement for a metric: nothing is
  drawn for it, a clear explanation is printed and the script exits cleanly.
- Missing values are never plotted or tabulated as zero.
- Failed runs, and stages that did not complete, are excluded by the aggregation's
  own eligibility rules; valid counts are shown in the tables.
- Gini is taken from valid robot reward records only (recomputed with the existing
  metric); with no valid reward measurement the fairness figure is not generated.
- Coalition stability is never plotted: it is pending a formal definition.
- The energy penalty is labelled PROVISIONAL unless the experiment config marks it confirmed, and is labelled
  as such wherever its effect (allocation cost) is shown.
- The assets describe measured values only. They make no ranking or "better algorithm" claim.

Rerunning overwrites the same files atomically. Exit codes: 0 done (or nothing to
do), 2 invalid arguments, 4 unreadable/invalid result data.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.analytics.logger import (  # noqa: E402
    BENCHMARK_RESULTS_FILENAME,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SUMMARY_DIR,
    ROBOT_REWARDS_FILENAME,
    DuplicateKeyError,
    SchemaValidationError,
    aggregate_benchmark_results,
    load_benchmark_results,
    load_robot_rewards,
    write_summary_table,
)
from src.analytics.metrics import compute_robot_reward_gini  # noqa: E402
from src.visualization import plots, tables  # noqa: E402

DEFAULT_FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
FIGURE_NAMES = {
    "allocation_cost": "allocation_cost",
    "allocation_runtime": "allocation_runtime",
    "reward_runtime": "reward_runtime",
    "total_runtime": "total_runtime",
    "energy_feasibility": "energy_feasibility",
    "gini": "reward_fairness_gini",
}
GINI_TOLERANCE = 1e-9
COALITION_NOTE = "pending a formal definition from the Game Theory owner; no values exist and none are plotted"
REWARD_PENDING = "pending reward-solver integration: no run has completed the reward stage"
GENERAL_NOTES = [
    "These assets describe measured values only; they make no claim about which algorithm is better.",
    "Means and standard deviations use only the runs eligible for each metric (valid counts are in the tables); "
    "failed runs and stages that did not complete are excluded; missing values are never treated as zero.",
]


class ReportError(RuntimeError):
    """The result data cannot be used (reported with exit code 4)."""


class NothingToReport(Exception):
    """No usable data: the message explains why (the script still exits cleanly)."""


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def validate_gini_with_reward_records(
    frame: pd.DataFrame, rewards: Optional[pd.DataFrame]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Return a copy of ``frame`` whose Gini comes from valid, complete robot reward records only.

    A run is eligible for Gini only if its main result has ``reward_stage_completed = True``
    and ``invalid_reward_flag = False`` AND it has exactly ``num_robots`` reward records.
    For an eligible run the Gini is recomputed with the existing ``compute_robot_reward_gini``.
    Every other run gets NaN, even if a Gini value is stored:

    - no reward records at all (``stored_gini_without_reward_records`` counts those with a stored value);
    - reward records on a run that is not eligible (stage not completed or invalid flag):
      ``reward_records_on_ineligible_runs``;
    - fewer or more reward records than robots: ``incomplete_reward_groups`` (split into
      ``reward_groups_fewer_than_robots`` and ``reward_groups_more_than_robots``).

    Orphan reward groups (no main record) are ignored and counted. ``runs_with_reward_records``
    counts the runs whose complete reward set was used.
    """
    frame = frame.copy()
    groups: Dict[Tuple[str, str, str], List[Tuple[Any, float]]] = {}
    if rewards is not None and not rewards.empty:
        for key, group in rewards.groupby(["experiment_id", "scenario_id", "algorithm"], sort=False):
            groups[key] = list(zip(group["robot_id"], group["reward"]))
    main_keys = set(zip(frame["experiment_id"], frame["scenario_id"], frame["algorithm"]))
    stats = {
        "runs_with_reward_records": 0,
        "stored_gini_without_reward_records": 0,
        "stored_gini_mismatching_records": 0,
        "orphan_reward_record_groups": len(set(groups) - main_keys),
        "reward_records_on_ineligible_runs": 0,
        "reward_stage_completed_without_reward_records": 0,
        "incomplete_reward_groups": 0,
        "reward_groups_fewer_than_robots": 0,
        "reward_groups_more_than_robots": 0,
    }
    nan = float("nan")
    recomputed = []
    for _, row in frame.iterrows():
        key = (row["experiment_id"], row["scenario_id"], row["algorithm"])
        stored = row["gini_index"]
        records = groups.get(key)
        if records is None:
            if bool(row["reward_stage_completed"]):
                stats["reward_stage_completed_without_reward_records"] += 1
            if pd.notna(stored):
                stats["stored_gini_without_reward_records"] += 1
            recomputed.append(nan)
            continue
        eligible = bool(row["reward_stage_completed"]) and not bool(row["invalid_reward_flag"])
        if not eligible:
            stats["reward_records_on_ineligible_runs"] += 1
            recomputed.append(nan)
            continue
        expected = int(row["num_robots"])
        if len(records) != expected:
            stats["incomplete_reward_groups"] += 1
            stats["reward_groups_fewer_than_robots" if len(records) < expected
                  else "reward_groups_more_than_robots"] += 1
            recomputed.append(nan)
            continue
        stats["runs_with_reward_records"] += 1
        value = compute_robot_reward_gini(records).gini_index
        if pd.notna(stored) and (pd.isna(value) or abs(stored - value) > GINI_TOLERANCE):
            stats["stored_gini_mismatching_records"] += 1
        recomputed.append(value)
    frame["gini_index"] = pd.Series(recomputed, index=frame.index, dtype="float64")
    return frame, stats


def _penalty_from_metadata(results_dir: Path, experiment_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(penalty settings, backend) from the experiment metadata, if it exists."""
    path = results_dir / f"metadata_{experiment_id}.yaml"
    if not path.exists():
        return None, None
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None, None
    penalty = ((document.get("config") or {}).get("evaluation") or {}).get("energy_penalty")
    backend = (((document.get("extra") or {}).get("harness")) or {}).get("backend")
    return (dict(penalty) if isinstance(penalty, dict) else None), backend


def _penalty_note(penalty: Optional[Dict[str, Any]]) -> str:
    if not penalty:
        return ("Allocation cost includes an energy penalty whose settings are unknown here (no metadata file); "
                "the penalty is a PROVISIONAL engineering assumption.")
    if _penalty_confirmed(penalty):
        return (f"Allocation cost includes the energy penalty ({penalty.get('strategy')}, "
                f"lambda_E={penalty.get('lambda_E')}), the same objective A(U) the allocators optimise.")
    return (f"Allocation cost includes the PROVISIONAL energy penalty ({penalty.get('strategy')}, "
            f"lambda_E={penalty.get('lambda_E')}): an unconfirmed engineering assumption, not a formally "
            "established part of the method.")


def _penalty_confirmed(penalty: Optional[Dict[str, Any]]) -> bool:
    """True when the experiment's config marks the penalty as agreed (status starts with 'confirmed')."""
    return bool(penalty) and str(penalty.get("status", "")).startswith("confirmed")


def _unavailable_reason(key: str, status: plots.MetricStatus, frame: pd.DataFrame) -> str:
    reward_stage_ran = bool(frame["reward_stage_completed"].any())
    if key in ("reward_runtime", "total_runtime") and not reward_stage_ran:
        return REWARD_PENDING + ("; total runtime needs both stages completed" if key == "total_runtime" else "")
    if key == "gini":
        return REWARD_PENDING if not reward_stage_ran else (
            "no valid reward measurements (rewards missing, invalid, or without robot reward records)")
    return status.reason


# ---------------------------------------------------------------------------
# Asset generation
# ---------------------------------------------------------------------------

def generate_assets(
    results_dir: Path,
    experiment_id: Optional[str] = None,
    figures_dir: Optional[Path] = None,
    tables_dir: Optional[Path] = None,
    formats: Sequence[str] = ("png", "pdf"),
    paired_baseline: str = "Greedy",
) -> Dict[str, Any]:
    """Generate the report assets; returns the machine-readable summary (also written as JSON).

    Raises:
        NothingToReport: no result file / empty file / unknown experiment (nothing is written).
        ReportError: the result files exist but are invalid.
        ValueError: ``experiment_id`` is needed but not given, or is unknown.
    """
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir) if figures_dir else DEFAULT_FIGURES_DIR
    tables_dir = Path(tables_dir) if tables_dir else DEFAULT_SUMMARY_DIR
    results_path = results_dir / BENCHMARK_RESULTS_FILENAME
    rewards_path = results_dir / ROBOT_REWARDS_FILENAME

    if not results_path.exists():
        raise NothingToReport(f"no benchmark result file found at {results_path}. Run the benchmark first "
                              "(python scripts/run_benchmarks.py).")
    try:
        frame = load_benchmark_results(results_path)
    except (SchemaValidationError, DuplicateKeyError) as exc:
        raise ReportError(f"{results_path} is not a valid result file: {exc}") from exc
    if frame.empty:
        raise NothingToReport(f"the result file {results_path} contains no records.")

    ids = sorted(frame["experiment_id"].unique())
    if experiment_id is None:
        if len(ids) != 1:
            raise ValueError(f"the result file contains several experiments {ids}; pass --experiment-id.")
        experiment_id = ids[0]
    elif experiment_id not in ids:
        raise ValueError(f"experiment {experiment_id!r} has no records; available experiments: {ids}.")
    frame = frame[frame["experiment_id"] == experiment_id].reset_index(drop=True)

    rewards = None
    if rewards_path.exists():
        try:
            rewards = load_robot_rewards(rewards_path)
        except (SchemaValidationError, DuplicateKeyError) as exc:
            raise ReportError(f"{rewards_path} is not a valid reward file: {exc}") from exc
        rewards = rewards[rewards["experiment_id"] == experiment_id]
    frame, reward_stats = validate_gini_with_reward_records(frame, rewards)

    summary = aggregate_benchmark_results(frame, experiment_id)     # existing aggregation, unchanged
    penalty, backend = _penalty_from_metadata(results_dir, experiment_id)
    penalty_note = _penalty_note(penalty)
    status = plots.available_metrics(summary)
    warnings: List[str] = []
    if reward_stats["stored_gini_without_reward_records"]:
        warnings.append(f"{reward_stats['stored_gini_without_reward_records']} run(s) have a stored Gini but no "
                        "robot reward records; those Gini values were excluded.")
    if reward_stats["stored_gini_mismatching_records"]:
        warnings.append(f"{reward_stats['stored_gini_mismatching_records']} stored Gini value(s) differ from the "
                        "Gini recomputed from the robot reward records; the recomputed values were used.")
    if reward_stats["incomplete_reward_groups"]:
        warnings.append(
            f"{reward_stats['incomplete_reward_groups']} reward record group(s) do not contain exactly one record "
            f"per robot ({reward_stats['reward_groups_fewer_than_robots']} with fewer, "
            f"{reward_stats['reward_groups_more_than_robots']} with more records than robots); "
            "their Gini was excluded.")
    if reward_stats["reward_records_on_ineligible_runs"]:
        warnings.append(
            f"{reward_stats['reward_records_on_ineligible_runs']} run(s) have reward records although their reward "
            "stage did not complete or their reward flag is invalid; those records were not used for Gini.")
    if reward_stats["reward_stage_completed_without_reward_records"]:
        warnings.append(f"{reward_stats['reward_stage_completed_without_reward_records']} completed reward stage(s) "
                        "have no robot reward records; their Gini values were unavailable.")
    if reward_stats["orphan_reward_record_groups"]:
        warnings.append(f"{reward_stats['orphan_reward_record_groups']} reward record group(s) have no matching "
                        "main result record and were ignored.")

    footnotes = {
        "allocation_cost": penalty_note,
        "allocation_runtime": "Only runs whose allocation stage completed are included (explicit stage flag).",
        "reward_runtime": "Only runs whose reward stage completed are included (explicit stage flag).",
        "total_runtime": ("Only runs with both stages completed are included."
                          + (" Includes worker start-up overhead of the subprocess backend." if backend == "subprocess"
                             else "")),
        "energy_feasibility": ("Feasible = no robot overdraws its energy (Eq. 7), computed over successful runs with a recorded feasibility flag; "
                               "reported separately from allocation success and independent of the penalty strategy."),
        "gini": "Gini of robot rewards, computed from valid robot reward records only.",
    }

    generated_figures: Dict[str, List[str]] = {}
    generated_tables: Dict[str, List[str]] = {}
    metrics_report: Dict[str, Any] = {}
    for key in plots.METRIC_SPECS:
        entry: Dict[str, Any] = {"available": bool(status[key].available)}
        if status[key].available:
            figure = plots.plot_metric(summary, key, footnote=footnotes[key])
            files = plots.save_figure(figure, figures_dir / f"{experiment_id}_{FIGURE_NAMES[key]}", formats)
            generated_figures[key] = [str(p) for p in files]
            table = tables.build_table(summary, key)
            table_files = tables.write_table(
                table, tables_dir / f"{experiment_id}_table_{key}",
                tables.TABLE_TITLES[key] + (f". {footnotes[key]}" if key in ("allocation_cost",) else ""),
                f"tab:{experiment_id}_{key}")
            generated_tables[key] = [str(p) for p in table_files]
            entry.update(detail=status[key].reason, figures=generated_figures[key], tables=generated_tables[key])
        else:
            entry["reason"] = _unavailable_reason(key, status[key], frame)
        metrics_report[key] = entry

    extra_status: Dict[str, Any] = {}
    extra_figures = {
        "allocation_cost_distribution": lambda: plots.plot_cost_distributions(frame, penalty_note),
        "energy_feasibility_heatmap": lambda: plots.plot_energy_heatmap(summary),
        "run_reliability": lambda: plots.plot_reliability(frame),
        "cost_runtime_tradeoff": lambda: plots.plot_cost_runtime_tradeoff(summary),
        "paired_cost_comparison": lambda: plots.plot_paired_costs(frame, paired_baseline),
    }
    for key, builder in extra_figures.items():
        try:
            fig = builder()
            paths = plots.save_figure(fig, figures_dir / f"{experiment_id}_{key}", formats)
            generated_figures[key] = [str(p) for p in paths]
            extra_status[key] = {"available": True, "figures": generated_figures[key]}
        except plots.MetricUnavailable as exc:
            extra_status[key] = {"available": False, "reason": str(exc)}

    if status["gini"].available:
        try:
            fig = plots.plot_gini_distributions(frame)
            paths = plots.save_figure(fig, figures_dir / f"{experiment_id}_gini_distributions", formats)
            generated_figures["gini_distributions"] = [str(p) for p in paths]
            extra_status["gini_distributions"] = {"available": True, "figures": generated_figures["gini_distributions"]}
            # One actual complete reward set, explicitly identified; never pool robots across runs.
            reward_groups = {}
            if rewards is not None:
                valid_keys = set(zip(frame.loc[frame.gini_index.notna(), "experiment_id"],
                                     frame.loc[frame.gini_index.notna(), "scenario_id"],
                                     frame.loc[frame.gini_index.notna(), "algorithm"]))
                for key, group in rewards.groupby(["experiment_id", "scenario_id", "algorithm"], sort=False):
                    if key in valid_keys:
                        row = frame[(frame.experiment_id == key[0]) & (frame.scenario_id == key[1]) & (frame.algorithm == key[2])].iloc[0]
                        if len(group) == int(row.num_robots):
                            reward_groups[key] = list(zip(group.robot_id, group.reward))
            if reward_groups:
                lorenz_key = sorted(reward_groups)[0]
                lorenz_label = " / ".join(map(str, lorenz_key[1:]))
                lorenz = plots.plot_lorenz_curve(reward_groups[lorenz_key], lorenz_label)
                paths = plots.save_figure(lorenz, figures_dir / f"{experiment_id}_lorenz_curve", formats)
                generated_figures["lorenz_curve"] = [str(p) for p in paths]
                extra_status["lorenz_curve"] = {"available": True, "figures": generated_figures["lorenz_curve"], "run": list(lorenz_key)}
            else:
                extra_status["lorenz_curve"] = {"available": False, "reason": "no valid complete robot reward set"}
        except plots.MetricUnavailable as exc:
            extra_status["gini_distributions"] = {"available": False, "reason": str(exc)}
    else:
        extra_status["gini_distributions"] = {"available": False, "reason": _unavailable_reason("gini", status["gini"], frame)}
        extra_status["lorenz_curve"] = {"available": False, "reason": _unavailable_reason("gini", status["gini"], frame)}
    extra_status["route_allocation"] = {"available": False, "reason": "benchmark schemas persist no robot start positions, task locations, assignments, or route sequences"}
    extra_status["reliability_categories"] = {"available": True, "categories": ["successful", "failed"], "reason": "failure category is not a structured schema field; error strings are not classified heuristically"}
    if extra_status.get("cost_runtime_tradeoff", {}).get("available"):
        eligibility = []
        for _, row in summary.iterrows():
            n_cost, n_runtime = int(row["n_allocation_cost"]), int(row["n_allocation_runtime"])
            eligibility.append({"algorithm": row["algorithm"], "robots": int(row["num_robots"]),
                                "tasks": int(row["num_tasks"]), "cost_runs": n_cost,
                                "runtime_runs": n_runtime, "same_count": n_cost == n_runtime})
        extra_status["cost_runtime_tradeoff"]["eligibility_counts"] = eligibility
        if any(not item["same_count"] for item in eligibility):
            warnings.append("Allocation cost and runtime trade-off values use different eligible run counts in some "
                            "algorithm/configuration groups; see visualizations.cost_runtime_tradeoff.eligibility_counts.")

    sizes_files = plots.save_figure(plots.plot_problem_sizes(summary), figures_dir / f"{experiment_id}_problem_sizes",
                                    formats)
    generated_figures["problem_sizes"] = [str(p) for p in sizes_files]
    outcome_files = tables.write_table(tables.build_table(summary, tables.RUN_OUTCOMES_KEY),
                                       tables_dir / f"{experiment_id}_table_{tables.RUN_OUTCOMES_KEY}",
                                       tables.TABLE_TITLES[tables.RUN_OUTCOMES_KEY], f"tab:{experiment_id}_outcomes")
    generated_tables[tables.RUN_OUTCOMES_KEY] = [str(p) for p in outcome_files]
    summary_csv = write_summary_table(summary, experiment_id, tables_dir)

    # Supporting run-level tables complement the existing metric tables.
    cost_rows = frame[frame["success"].astype(bool)].copy()
    cost_rows["allocation_cost"] = pd.to_numeric(cost_rows["allocation_cost"], errors="coerce")
    cost_rows = cost_rows.dropna(subset=["allocation_cost"])
    cost_stats = (cost_rows.groupby(["algorithm", "num_robots", "num_tasks", "total_entities"])["allocation_cost"]
                  .agg(n="count", median="median", q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75),
                       minimum="min", maximum="max").reset_index())
    cost_stats = cost_stats.rename(columns={"num_robots": "Robots", "num_tasks": "Tasks", "total_entities": "N",
                                            "algorithm": "Algorithm", "median": "Median cost", "q1": "Q1",
                                            "q3": "Q3", "minimum": "Minimum", "maximum": "Maximum", "n": "Count"})
    if not cost_stats.empty:
        generated_tables["allocation_cost_distribution"] = [str(p) for p in tables.write_table(
            cost_stats, tables_dir / f"{experiment_id}_table_allocation_cost_distribution",
            "Successful-run allocation-cost distribution statistics", f"tab:{experiment_id}_cost_distribution")]
    reliability = (frame.groupby(["algorithm", "num_robots", "num_tasks", "total_entities"], sort=False)
                   .agg(attempted=("success", "size"), successful=("success", "sum")).reset_index())
    reliability["failed"] = reliability.attempted - reliability.successful
    reliability["success_percent"] = reliability.successful / reliability.attempted * 100
    reliability = reliability.rename(columns={"num_robots": "Robots", "num_tasks": "Tasks", "total_entities": "N",
                                              "algorithm": "Algorithm", "attempted": "Attempted", "successful": "Successful",
                                              "failed": "Failed", "success_percent": "Success percent"})
    generated_tables["reliability"] = [str(p) for p in tables.write_table(
        reliability, tables_dir / f"{experiment_id}_table_reliability",
        "Run outcomes by algorithm and problem configuration", f"tab:{experiment_id}_reliability")]
    paired_table = pd.DataFrame(columns=["Algorithm", "N", "Matched scenarios", "Median relative cost", "Q1", "Q3"])
    if extra_status.get("paired_cost_comparison", {}).get("available"):
        try:
            paired_rows = []
            cost_pivot = cost_rows.pivot_table(index=["experiment_id", "scenario_id", "num_robots", "num_tasks", "total_entities"],
                                               columns="algorithm", values="allocation_cost", aggfunc="first")
            if paired_baseline in cost_pivot:
                for alg in plots.ordered_algorithms(frame):
                    if alg == paired_baseline or alg not in cost_pivot: continue
                    pair = cost_pivot[[paired_baseline, alg]].dropna(); pair = pair[pair[paired_baseline] > 0]
                    if not pair.empty:
                        ratios = pair[alg] / pair[paired_baseline]
                        for n, group in ratios.groupby(level="total_entities"):
                            paired_rows.append({"Algorithm": alg, "N": int(n), "Matched scenarios": len(group),
                                                "Median relative cost": group.median(), "Q1": group.quantile(.25), "Q3": group.quantile(.75)})
            paired_table = pd.DataFrame(paired_rows, columns=paired_table.columns)
        except (KeyError, ValueError):
            pass
    if not paired_table.empty:
        generated_tables["paired_cost_comparison"] = [str(p) for p in tables.write_table(
            paired_table, tables_dir / f"{experiment_id}_table_paired_cost_comparison",
            f"Matched scenario cost ratios relative to {paired_baseline}", f"tab:{experiment_id}_paired_cost")]
    if status["gini"].available:
        gini_stats = (frame.dropna(subset=["gini_index"]).groupby(["algorithm", "num_robots", "num_tasks", "total_entities"])["gini_index"]
                      .agg(n="count", median="median", q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75), minimum="min", maximum="max").reset_index())
        gini_stats = gini_stats.rename(columns={"num_robots": "Robots", "num_tasks": "Tasks", "total_entities": "N", "algorithm": "Algorithm", "n": "Count", "median": "Median Gini", "q1": "Q1", "q3": "Q3", "minimum": "Minimum", "maximum": "Maximum"})
        generated_tables["gini_distribution"] = [str(p) for p in tables.write_table(
            gini_stats, tables_dir / f"{experiment_id}_table_gini_distribution", "Valid per-run robot-reward Gini statistics", f"tab:{experiment_id}_gini_distribution")]

    per_algorithm = {
        algorithm: {"runs": int(group["n_runs"].sum()), "successful_runs": int(group["n_success"].sum()),
                    "failed_runs": int(group["n_runs"].sum() - group["n_success"].sum())}
        for algorithm, group in summary.groupby("algorithm", sort=False)
    }
    reward_available = metrics_report["gini"]["available"]
    report = {
        "experiment_id": experiment_id,
        "source": {"results_file": str(results_path), "records": int(len(frame)),
                   "reward_file_present": rewards_path.exists(), **reward_stats},
        "configurations": [{"total_entities": int(n), "robots": int(r), "tasks": int(t)} for n, r, t in
                           plots.problem_sizes(summary)[["total_entities", "num_robots", "num_tasks"]].values],
        "algorithms": plots.ordered_algorithms(summary),
        "runs_per_algorithm": per_algorithm,
        "energy_penalty": (dict(penalty) if _penalty_confirmed(penalty) else
                           {**penalty, "status": "provisional: engineering assumption pending team confirmation"}
                           if penalty else {"status": "unknown (no metadata file); provisional"}),
        "metrics": metrics_report,
        "visualizations": extra_status,
        "unavailable_metrics": sorted(k for k, v in metrics_report.items() if not v["available"]),
        "unavailable_visualizations": {k: v["reason"] for k, v in extra_status.items() if not v.get("available")},
        "coalition_stability": COALITION_NOTE,
        "reward_fairness_analysis": "available" if reward_available else REWARD_PENDING,
        "figures": generated_figures,
        "tables": {**generated_tables, "summary": [str(summary_csv)]},
        "warnings": warnings,
        "notes": GENERAL_NOTES + [penalty_note],
    }
    json_path = tables_dir / f"{experiment_id}_report_summary.json"
    tables.write_text_atomic(json_path, json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    report["tables"]["report_summary_json"] = [str(json_path)]
    return report


def print_report(report: Dict[str, Any]) -> None:
    print(f"experiment '{report['experiment_id']}': {report['source']['records']} result records, "
          f"{len(report['algorithms'])} algorithm(s), {len(report['configurations'])} configuration(s)")
    print("generated figures:")
    for key, files in report["figures"].items():
        for path in files:
            print(f"  [{key}] {path}")
    print("generated tables:")
    for key, files in report["tables"].items():
        for path in files:
            print(f"  [{key}] {path}")
    print("unavailable metrics (no figure or table was generated):")
    for key in report["unavailable_metrics"]:
        print(f"  - {key}: {report['metrics'][key]['reason']}")
    print(f"  - coalition_stability: {report['coalition_stability']}")
    if not report["metrics"]["gini"]["available"]:
        print("reward fairness analysis: " + report["reward_fairness_analysis"])
    for warning in report["warnings"]:
        print("warning: " + warning)
    print("note: " + report["notes"][-1])


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report figures and tables from the benchmark results.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                        help="folder with benchmark_results.csv (and robot_rewards.csv)")
    parser.add_argument("--experiment-id", default=None, help="experiment to report (needed if several exist)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--tables-dir", default=str(DEFAULT_SUMMARY_DIR))
    parser.add_argument("--formats", default="png,pdf", help="comma-separated figure formats (png, pdf)")
    parser.add_argument("--paired-baseline", default="Greedy", help="baseline algorithm for matched-scenario cost ratios")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    formats = [f for f in args.formats.split(",") if f]
    if not formats or not set(formats) <= {"png", "pdf"}:
        print("invalid --formats: use a comma-separated subset of png,pdf", file=sys.stderr)
        return 2
    try:
        report = generate_assets(Path(args.results_dir), args.experiment_id, Path(args.figures_dir),
                                 Path(args.tables_dir), formats, paired_baseline=args.paired_baseline)
    except NothingToReport as exc:
        print(f"Nothing to report: {exc}")
        print("No figures or tables were generated. Reward fairness analysis is pending reward-solver integration.")
        return 0
    except ValueError as exc:
        print(f"invalid request: {exc}", file=sys.stderr)
        return 2
    except ReportError as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return 4
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

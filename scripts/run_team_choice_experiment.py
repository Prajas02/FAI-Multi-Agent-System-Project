"""Experiment: do robots accept the Stage-1 teams, and how does the reward rule change that?

For every benchmark configuration (same scenarios and seeds as run_benchmarks.py) and every
clustering allocator (SVCA_Paper, SVCA, GA_Paper, GA), the robots choose their teams under
each reward rule (equal split, classical Shapley, preference-aware Shapley at alpha 1.5 and 3).
One CSV row per (scenario, allocator, rule); then summary tables and three figures:

  H1  fraction of robots that want to leave the Stage-1 teams, per rule      team_choice_instability
  H2  cost change after robots re-sort and teams re-plan, per rule          team_choice_cost_change
  H3  stability and Gini as alpha grows (0 = classical Shapley)             team_choice_alpha

Examples
    python scripts/run_team_choice_experiment.py --only-configs C1,C2,C3 --repetitions 3
    python scripts/run_team_choice_experiment.py                      # all C1..C8, 3 repetitions
    python scripts/run_team_choice_experiment.py --value-type facility  # ablation: battery-blind team value
"""
import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytics import allocation_adapters, seeding  # noqa: E402
from src.analytics.team_choice_eval import DEFAULT_RULES, evaluate_team_choice  # noqa: E402

ALLOCATORS = {
    "SVCA_Paper": allocation_adapters.svca_paper_adapter,
    "SVCA": allocation_adapters.svca_adapter,
    "GA_Paper": allocation_adapters.ga_paper_adapter,
    "GA": allocation_adapters.ga_adapter,
}
FIELDS = ["config_id", "scenario_id", "scenario_seed", "allocator", "rule_label", "rule", "alpha", "value_type",
          "num_robots", "num_tasks", "num_teams", "unstable_robots", "unstable_fraction", "moves",
          "robots_moved", "rounds", "converged", "cycled", "idle_robots", "cost_before", "cost_after",
          "cost_change_pct", "tasks_reassigned", "value_before", "value_after", "gini_before",
          "gini_after", "games_solved", "runtime_seconds"]

# Validated with the dataviz palette checker (light surface): rules = slots 1-4, allocators = slots 5-8.
RULE_COLORS = {"equal": "#2a78d6", "shapley": "#eb6834", "preference_a1.5": "#1baf7a", "preference_a3": "#eda100"}
RULE_NAMES = {"equal": "Equal split", "shapley": "Classical Shapley",
              "preference_a1.5": "Preference-aware, alpha 1.5", "preference_a3": "Preference-aware, alpha 3"}
ALLOC_COLORS = {"SVCA_Paper": "#e87ba4", "SVCA": "#008300", "GA_Paper": "#6250d6", "GA": "#e34948"}
ALLOC_MARKERS = {"SVCA_Paper": "o", "SVCA": "s", "GA_Paper": "^", "GA": "D"}


def _load_harness():
    spec = importlib.util.spec_from_file_location("run_benchmarks_tc", REPO_ROOT / "scripts" / "run_benchmarks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args):
    rb = _load_harness()
    config_path = REPO_ROOT / "configs" / "experiments_config.yaml"
    config = rb.load_config(config_path)
    only = set(args.only_configs.split(",")) if args.only_configs else None
    allocators = args.allocators.split(",") if args.allocators else list(ALLOCATORS)
    unknown = sorted(set(allocators) - set(ALLOCATORS))
    if unknown:
        raise SystemExit(f"unknown allocators: {', '.join(unknown)}")
    base_seed = config["experiment"]["base_seed"]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for entry in config["configurations"]:
            if only and entry["id"] not in only:
                continue
            for rep in range(args.repetitions):
                s_seed = seeding.scenario_seed(base_seed, entry["id"], rep, config_path=config_path)
                scenario_id = config["benchmark"]["scenario_id_format"].format(config_id=entry["id"], repetition_index=rep)
                spec = rb.RunSpec(entry["id"], entry["num_robots"], entry["num_tasks"], rep, scenario_id, s_seed,
                                  "team_choice", 0, 0.0)
                scenario = rb.generate_scenario(config, spec)
                for name in allocators:
                    payload = ALLOCATORS[name](scenario, seeding.algorithm_seed(s_seed, name, config_path=config_path), 300.0)
                    clusters = payload.get("clusters")
                    for label, rule, alpha in DEFAULT_RULES:
                        t0 = time.perf_counter()
                        rec = evaluate_team_choice(scenario, clusters, rule, alpha, seed=s_seed % (2 ** 31),
                                                   value_type=args.value_type)
                        row = {k: rec[k] for k in FIELDS if k in rec}
                        row.update(config_id=entry["id"], scenario_id=scenario_id, scenario_seed=s_seed,
                                   allocator=name, rule_label=label, value_type=args.value_type)
                        writer.writerow(row)
                        handle.flush()
                        rows.append(row)
                        print(f"{scenario_id} {name:<10} {label:<16} unstable {row['unstable_robots']:>2}/{row['num_robots']:<2} "
                              f"moves {row['moves']:>3} cost {row['cost_change_pct']:+7.2f}% "
                              f"gini {row['gini_before']:.3f}->{row['gini_after']:.3f} ({time.perf_counter() - t0:.1f}s)")
    print(f"saved {len(rows)} rows to {out}")
    return rows


def report(csv_path: Path, figures_dir: Path, tables_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        print("no rows to report")
        return
    vt = str(df["value_type"].iloc[0]) if "value_type" in df else "facility"
    suffix = f"_{vt}"
    vt_note = {"routing": "team value = routing game (battery aware)",
               "facility": "team value = facility game (ignores route battery limits)"}.get(vt, vt)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    rules = [r for r in RULE_COLORS if r in set(df["rule_label"])]
    allocs = [a for a in ALLOCATORS if a in set(df["allocator"])]

    summary = (df.groupby(["allocator", "rule_label"])
                 .agg(runs=("unstable_fraction", "size"),
                      unstable_fraction=("unstable_fraction", "mean"),
                      robots_moved=("robots_moved", "mean"),
                      rounds=("rounds", "mean"),
                      converged_rate=("converged", "mean"),
                      cost_change_pct=("cost_change_pct", "mean"),
                      cost_change_median=("cost_change_pct", "median"),
                      blowup_rate=("cost_change_pct", lambda s: float((s > 50).mean())),
                      tasks_reassigned=("tasks_reassigned", "mean"),
                      gini_before=("gini_before", "mean"),
                      gini_after=("gini_after", "mean"))
                 .reset_index())
    summary.to_csv(tables_dir / f"team_choice_summary{suffix}.csv", index=False)
    cols = [("allocator", "Allocator"), ("rule_label", "Reward rule"), ("runs", "Runs"),
            ("unstable_fraction", "Want to leave"), ("robots_moved", "Robots moved"), ("converged_rate", "Converged"),
            ("cost_change_median", "Median cost change \\%"), ("cost_change_pct", "Mean cost change \\%"),
            ("blowup_rate", "Runs $>$+50\\%"), ("gini_before", "Gini before"), ("gini_after", "Gini after")]
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\caption{Robots choosing their team under each reward rule; " + vt_note + " (means over scenarios).}",
             "\\label{tab:team_choice" + suffix.replace("_", "-") + "}", "\\begin{tabular}{ll" + "r" * (len(cols) - 2) + "}", "\\toprule",
             " & ".join(h for _, h in cols) + " \\\\", "\\midrule"]
    for _, rec in summary.iterrows():
        cells = []
        for key, _ in cols:
            v = rec[key]
            if key == "rule_label":
                v = RULE_NAMES.get(v, v)
            elif key in ("unstable_fraction", "converged_rate", "blowup_rate"):
                v = f"{100 * v:.0f}\\%"
            elif key == "runs":
                v = f"{int(v)}"
            elif isinstance(v, float):
                v = f"{v:.3f}" if key.startswith("gini") else f"{v:.2f}"
            cells.append(str(v).replace("_", "\\_"))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (tables_dir / f"team_choice_summary{suffix}.tex").write_text("\n".join(lines), encoding="utf-8")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e6e6", "grid.linewidth": 0.8})

    def grouped(metric, ylabel, title, fname, pct=False):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        width = 0.8 / len(rules)
        for j, rule in enumerate(rules):
            vals = [summary[(summary.allocator == a) & (summary.rule_label == rule)][metric].mean() for a in allocs]
            xs = [i + (j - (len(rules) - 1) / 2) * width for i in range(len(allocs))]
            bars = ax.bar(xs, vals, width * 0.92, color=RULE_COLORS[rule], label=RULE_NAMES[rule], zorder=3)
            ax.bar_label(bars, labels=[f"{v:.0%}" if pct else f"{v:+.1f}" for v in vals], fontsize=7, color="#444444", padding=2)
        ax.set_xticks(range(len(allocs)), allocs)
        ax.set_ylabel(ylabel)
        ax.set_title(title + "\n" + vt_note, loc="left", fontsize=10)
        if pct:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.12))
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figures_dir / f"{fname}{suffix}.{ext}", dpi=160)
        plt.close(fig)

    grouped("unstable_fraction", "robots that want to leave", "H1: robots wanting to leave the Stage-1 teams",
            "team_choice_instability", pct=True)
    grouped("cost_change_median", "median cost change after re-sorting (%)",
            "H2: allocation cost after robots choose teams (median over scenarios)", "team_choice_cost_change")

    alpha_of = {"shapley": 0.0, "preference_a1.5": 1.5, "preference_a3": 3.0}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, metric, ylabel in ((axes[0], "unstable_fraction", "robots that want to leave"),
                               (axes[1], "gini_after", "Gini of payoffs after choice")):
        for a in allocs:
            pts = sorted((alpha_of[r], summary[(summary.allocator == a) & (summary.rule_label == r)][metric].mean())
                         for r in alpha_of if r in rules)
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=ALLOC_COLORS[a], marker=ALLOC_MARKERS[a],
                    markersize=6, linewidth=2, label=a)
        ax.set_xlabel("alpha (0 = classical Shapley)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.2, 3.2)
        ax.set_xticks([0, 1.5, 3])
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_title("H3: stability vs alpha", loc="left", fontsize=11)
    axes[1].set_title("H3: fairness vs alpha", loc="left", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=len(labels), loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"team_choice_alpha{suffix}.{ext}", dpi=160)
    plt.close(fig)
    print(summary.round(3).to_string(index=False))
    print(f"figures -> {figures_dir}, tables -> {tables_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Robots choose their team: stability / cost / fairness experiment.")
    parser.add_argument("--only-configs", help="comma-separated config ids, e.g. C1,C2,C3 (default: all)")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--allocators", help=f"comma-separated subset of {', '.join(ALLOCATORS)}")
    parser.add_argument("--value-type", choices=("routing", "facility"), default="routing",
                        help="team value game: routing (battery aware, default) or facility (ablation)")
    parser.add_argument("--output", help="results CSV (default: data/results/team_choice/team_choice_results_<value-type>.csv)")
    parser.add_argument("--figures-dir", default=str(REPO_ROOT / "outputs" / "figures"))
    parser.add_argument("--tables-dir", default=str(REPO_ROOT / "outputs" / "tables"))
    parser.add_argument("--report-only", action="store_true", help="only rebuild figures/tables from the CSV")
    args = parser.parse_args(argv)
    if not args.output:
        args.output = str(REPO_ROOT / "data" / "results" / "team_choice" / f"team_choice_results_{args.value_type}.csv")
    if not args.report_only:
        run(args)
    report(Path(args.output), Path(args.figures_dir), Path(args.tables_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

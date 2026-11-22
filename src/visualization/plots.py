"""Benchmark figures built from the aggregated summary.

The input is the summary DataFrame produced by
``src.analytics.logger.aggregate_benchmark_results`` (columns ``SUMMARY_COLUMNS``);
nothing is re-aggregated here. Every metric is drawn from the mean/std columns
and metric-specific valid counts that the aggregation already computed, so
failed runs, incomplete stages and missing values are handled exactly as in the
approved analytics design.

Design rules followed (static report figures, light surface):

- One axis per chart (never a dual axis). The x axis is the *problem size*: one
  tick per experimental configuration, labelled with the total entity count
  ``N = robots + tasks`` and the robot/task counts, so the figure never hides
  that both counts change.
- Algorithm identity: fixed colour and marker per algorithm id (colours are the
  first four slots of the validated categorical palette, in a fixed order; a
  colour never depends on which series are present), a legend above the plot
  and a different marker shape per algorithm as a secondary encoding.
- Missing values are NaN, never zero: a missing point is simply not drawn and
  the line breaks there. Error bars (+/- 1 sample std) are drawn only where the
  std exists (at least two valid runs).
- Coalition stability is deliberately not available here: it has no formal
  definition yet, and no metric spec exists for it.

Figures are plain ``matplotlib.figure.Figure`` objects (no pyplot, no global
backend state), so they can be created and saved safely from scripts and tests.

Owner: Yash Yeole (Benchmarking & Analytics).
"""

import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, FuncFormatter

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#76756f"
GRID = "#e4e3df"

PREFERRED_ALGORITHM_ORDER = ("MILP", "GA", "Greedy", "SVCA")
# Categorical slots 1-4 (blue, orange, aqua, yellow); fixed per algorithm id.
ALGORITHM_COLORS = {"MILP": "#2a78d6", "GA": "#eb6834", "Greedy": "#1baf7a", "SVCA": "#eda100"}
ALGORITHM_MARKERS = {"MILP": "o", "GA": "s", "Greedy": "^", "SVCA": "D"}
# Slots 5-8 for any other algorithm id, assigned in sorted order (never cycled).
_FALLBACK_COLORS = ("#e87ba4", "#008300", "#6250d6", "#e34948")
_FALLBACK_MARKERS = ("v", "P", "X", "*")

ROBOT_COLOR, TASK_COLOR = "#2a78d6", "#eb6834"
FIGURE_SIZE = (8.4, 5.0)
SAVE_DPI = 200
LOG_SCALE_RATIO = 100.0     # use a log y axis when the largest/smallest positive mean differs this much


class MetricUnavailable(ValueError):
    """A metric has no valid measurement in the summary, so no figure is drawn."""


class MetricSpec(NamedTuple):
    """How one metric is read from the summary and labelled."""
    key: str
    title: str
    ylabel: str
    mean_column: str
    std_column: Optional[str]
    count_column: Optional[str]
    is_rate: bool = False
    allow_log: bool = False


METRIC_SPECS: Dict[str, MetricSpec] = {
    spec.key: spec
    for spec in (
        MetricSpec("allocation_cost", "Allocation cost by problem size", "Allocation cost A(U) (mean)",
                   "allocation_cost_mean", "allocation_cost_std", "n_allocation_cost", allow_log=True),
        MetricSpec("allocation_runtime", "Allocation runtime by problem size", "Allocation runtime, seconds (mean)",
                   "allocation_runtime_seconds_mean", "allocation_runtime_seconds_std", "n_allocation_runtime",
                   allow_log=True),
        MetricSpec("reward_runtime", "Reward-stage runtime by problem size", "Reward runtime, seconds (mean)",
                   "reward_runtime_seconds_mean", "reward_runtime_seconds_std", "n_reward_runtime",
                   allow_log=True),
        MetricSpec("total_runtime", "Total pipeline runtime by problem size", "Total runtime, seconds (mean)",
                   "total_runtime_seconds_mean", "total_runtime_seconds_std", "n_total_runtime", allow_log=True),
        MetricSpec("energy_feasibility", "Energy feasibility by problem size",
                   "Energy-feasible share of eligible successful runs", "energy_feasible_rate", None, "n_success",
                   is_rate=True),
        MetricSpec("gini", "Reward fairness (Gini index) by problem size", "Gini index of robot rewards (mean)",
                   "gini_index_mean", "gini_index_std", "n_gini"),
    )
}


class MetricStatus(NamedTuple):
    available: bool
    reason: str


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _require_columns(summary: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in summary.columns]
    if missing:
        raise ValueError(f"summary is missing columns: {missing}")


def ordered_algorithms(summary: pd.DataFrame) -> List[str]:
    """Known algorithms in the canonical order, then any others sorted."""
    present = list(dict.fromkeys(summary["algorithm"].tolist()))
    known = [a for a in PREFERRED_ALGORITHM_ORDER if a in present]
    return known + sorted(a for a in present if a not in PREFERRED_ALGORITHM_ORDER)


def algorithm_style(algorithm: str, algorithms: Sequence[str]) -> Tuple[str, str]:
    """(colour, marker) of an algorithm; fixed for the known ids, never cycled."""
    if algorithm in ALGORITHM_COLORS:
        return ALGORITHM_COLORS[algorithm], ALGORITHM_MARKERS[algorithm]
    others = sorted(a for a in algorithms if a not in ALGORITHM_COLORS)
    position = others.index(algorithm)
    if position >= len(_FALLBACK_COLORS):
        raise ValueError("too many distinct algorithm ids for the fixed categorical palette.")
    return _FALLBACK_COLORS[position], _FALLBACK_MARKERS[position]


def problem_sizes(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per experimental configuration, ordered by size, with a tick label."""
    _require_columns(summary, ["num_robots", "num_tasks", "total_entities"])
    sizes = (summary[["total_entities", "num_robots", "num_tasks"]]
             .drop_duplicates().sort_values(["total_entities", "num_robots", "num_tasks"]).reset_index(drop=True))
    sizes["label"] = [f"N={n}\n{r}R / {t}T" for n, r, t in
                      zip(sizes["total_entities"], sizes["num_robots"], sizes["num_tasks"])]
    return sizes


def metric_status(summary: pd.DataFrame, metric_key: str) -> MetricStatus:
    """Whether the summary holds at least one valid (non-NaN, counted) value for the metric."""
    spec = METRIC_SPECS[metric_key]
    _require_columns(summary, [spec.mean_column] + ([spec.count_column] if spec.count_column else []))
    if summary.empty:
        return MetricStatus(False, "the summary is empty")
    means = pd.to_numeric(summary[spec.mean_column], errors="coerce")
    if spec.is_rate:
        valid = means.notna()                      # a rate exists only if its denominator was > 0
    else:
        counts = pd.to_numeric(summary[spec.count_column], errors="coerce").fillna(0)
        valid = means.notna() & (counts > 0)
    if not valid.any():
        return MetricStatus(False, "no valid measurements in any configuration")
    return MetricStatus(True, f"{int(valid.sum())} of {len(summary)} algorithm/configuration cells have data")


def available_metrics(summary: pd.DataFrame) -> Dict[str, MetricStatus]:
    return {key: metric_status(summary, key) for key in METRIC_SPECS}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _new_figure() -> Tuple[Figure, "object"]:
    figure = Figure(figsize=FIGURE_SIZE, facecolor=SURFACE)
    figure.subplots_adjust(left=0.105, right=0.975, top=0.835, bottom=0.265)
    axes = figure.subplots()
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
        axes.spines[side].set_linewidth(1.0)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=8.5, length=3, width=0.8)
    axes.tick_params(which="minor", length=2, width=0.6, color=GRID)
    axes.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    axes.set_axisbelow(True)
    return figure, axes


def _frame_text(figure: Figure, title: str, footnote: Optional[str]) -> None:
    figure.suptitle(title, x=0.012, y=0.975, ha="left", va="top", fontsize=13, color=TEXT_PRIMARY)
    note = footnote or ""
    if note:
        figure.text(0.012, 0.012, note, ha="left", va="bottom", fontsize=7.2, color=TEXT_MUTED, wrap=True)


def _use_log_scale(means: np.ndarray) -> bool:
    positive = means[np.isfinite(means)]
    if positive.size == 0 or (positive <= 0).any():
        return False
    return positive.max() / positive.min() >= LOG_SCALE_RATIO


def plot_metric(summary: pd.DataFrame, metric_key: str, footnote: Optional[str] = None) -> Figure:
    """Line chart of one metric against problem size, one line per algorithm.

    Missing cells are not drawn (never plotted as zero); error bars (+/- 1
    sample std) appear only where the std exists.

    Raises:
        MetricUnavailable: the metric has no valid measurement.
    """
    spec = METRIC_SPECS[metric_key]
    status = metric_status(summary, metric_key)
    if not status.available:
        raise MetricUnavailable(f"{metric_key}: {status.reason}")

    sizes = problem_sizes(summary)
    algorithms = ordered_algorithms(summary)
    positions = {(n, r, t): i for i, (n, r, t) in
                 enumerate(zip(sizes["total_entities"], sizes["num_robots"], sizes["num_tasks"]))}
    x = np.arange(len(sizes), dtype=float)

    series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for algorithm in algorithms:
        mean = np.full(len(sizes), np.nan)
        std = np.full(len(sizes), np.nan)
        for _, row in summary[summary["algorithm"] == algorithm].iterrows():
            index = positions[(row["total_entities"], row["num_robots"], row["num_tasks"])]
            value = pd.to_numeric(row[spec.mean_column], errors="coerce")
            valid_count = spec.is_rate or pd.to_numeric(row[spec.count_column], errors="coerce") > 0
            if pd.notna(value) and valid_count:
                mean[index] = float(value)
                if spec.std_column is not None:
                    deviation = pd.to_numeric(row[spec.std_column], errors="coerce")
                    std[index] = float(deviation) if pd.notna(deviation) else np.nan
        series[algorithm] = (mean, std)

    figure, axes = _new_figure()
    all_means = np.concatenate([m for m, _ in series.values()])
    log_scale = spec.allow_log and _use_log_scale(all_means)
    for algorithm in algorithms:
        color, marker = algorithm_style(algorithm, algorithms)
        mean, std = series[algorithm]
        axes.plot(x, mean, color=color, marker=marker, linewidth=2.0, markersize=6.5, markeredgecolor=SURFACE,
                  markeredgewidth=1.2, label=algorithm, solid_capstyle="round", solid_joinstyle="round")
        bars = np.isfinite(mean) & np.isfinite(std)
        if bars.any():
            lower = mean[bars] - std[bars]
            if log_scale:
                lower = np.maximum(lower, mean[bars] * 1e-3)
            yerr = np.vstack([mean[bars] - lower, std[bars]])        # shape (2, n): lower and upper extents
            axes.errorbar(x[bars], mean[bars], yerr=yerr, fmt="none", ecolor=color,
                          elinewidth=1.1, capsize=3, alpha=0.85)

    axes.set_xticks(x)
    axes.set_xticklabels(sizes["label"], fontsize=8.5, color=TEXT_SECONDARY)
    axes.set_xlim(-0.4, len(sizes) - 0.6)
    axes.set_xlabel("Problem size per configuration: total entities N = robots (R) + tasks (T)",
                    fontsize=9, color=TEXT_SECONDARY, labelpad=6)
    axes.set_ylabel(spec.ylabel, fontsize=9, color=TEXT_SECONDARY)
    if log_scale:
        axes.set_yscale("log")
        if metric_key == "allocation_cost":
            axes.set_ylabel("Allocation cost A(U) (mean, logarithmic scale)", fontsize=9, color=TEXT_SECONDARY)
            axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    elif spec.is_rate:
        axes.set_ylim(-0.02, 1.05)
        axes.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axes.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    else:
        axes.set_ylim(bottom=0)
    handles, labels = axes.get_legend_handles_labels()
    legend = axes.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=len(labels),
                         frameon=False, fontsize=9, handlelength=2.2)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)

    notes = []
    if spec.std_column is not None:
        notes.append("Error bars: +/-1 sample std over valid runs (shown only when at least 2 runs are valid). "
                     "Missing cells are not drawn; they are not zero.")
    else:
        notes.append("Missing cells are not drawn; they are not zero.")
    if footnote:
        notes.append(footnote)
    _frame_text(figure, spec.title, "\n".join(notes))
    return figure


def plot_problem_sizes(summary: pd.DataFrame, footnote: Optional[str] = None) -> Figure:
    """Robots and tasks per configuration (shows how both counts change with N)."""
    sizes = problem_sizes(summary)
    if sizes.empty:
        raise MetricUnavailable("problem_sizes: the summary is empty")
    figure, axes = _new_figure()
    x = np.arange(len(sizes), dtype=float)
    width = 0.34
    axes.bar(x - width / 2 - 0.01, sizes["num_robots"], width=width, color=ROBOT_COLOR, label="Robots (R)")
    axes.bar(x + width / 2 + 0.01, sizes["num_tasks"], width=width, color=TASK_COLOR, label="Tasks (T)")
    for position, value in zip(x - width / 2 - 0.01, sizes["num_robots"]):
        axes.text(position, value, str(int(value)), ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY)
    for position, value in zip(x + width / 2 + 0.01, sizes["num_tasks"]):
        axes.text(position, value, str(int(value)), ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY)
    axes.set_xticks(x)
    axes.set_xticklabels(sizes["label"], fontsize=8.5, color=TEXT_SECONDARY)
    axes.set_xlabel("Experimental configuration (total entities N = R + T)", fontsize=9, color=TEXT_SECONDARY,
                    labelpad=6)
    axes.set_ylabel("Count", fontsize=9, color=TEXT_SECONDARY)
    axes.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes.set_ylim(bottom=0, top=float(max(sizes["num_robots"].max(), sizes["num_tasks"].max())) * 1.12)
    legend = axes.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)
    _frame_text(figure, "Problem size of each configuration", footnote)
    return figure


def _raw_configuration_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["algorithm", "num_robots", "num_tasks", "total_entities"]
    _require_columns(frame, required)
    return (frame[required].drop_duplicates().sort_values(
        ["total_entities", "num_robots", "num_tasks"]).reset_index(drop=True))


def _raw_chart(frame: pd.DataFrame, title: str, xlabel: str, ylabel: str,
               footnote: Optional[str] = None, figsize: Tuple[float, float] = (10.0, 5.6)):
    fig = Figure(figsize=figsize, facecolor=SURFACE)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.2)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    _frame_text(fig, title, footnote)
    return fig, ax


def _raw_log_if_wide(ax, values: Sequence[float], label: str) -> bool:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size and (vals <= 0).any():
        return False
    use_log = bool(vals.size and vals.max() / vals.min() >= LOG_SCALE_RATIO)
    if use_log:
        ax.set_yscale("log")
        ax.set_ylabel(label + " (log scale)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    return use_log


def plot_cost_distributions(frame: pd.DataFrame, footnote: Optional[str] = None) -> Figure:
    """Successful per-run costs by configuration, with observations overlaid."""
    _require_columns(frame, ["success", "allocation_cost", "algorithm", "num_robots", "num_tasks", "total_entities"])
    sizes = _raw_configuration_table(frame)
    if sizes.empty:
        raise MetricUnavailable("cost distributions: no configurations")
    algorithms = ordered_algorithms(frame)
    fig = Figure(figsize=(max(10, len(sizes) * 1.65), 5.8), facecolor=SURFACE)
    axes = fig.subplots(1, len(sizes), squeeze=False)
    fig.subplots_adjust(left=.08, right=.99, top=.82, bottom=.2, wspace=.25)
    allvalues=[]
    for pos, (_, size) in enumerate(sizes.iterrows()):
        ax=axes[0,pos]; ax.set_facecolor(SURFACE); ax.grid(axis="y", color=GRID, linewidth=.7); ax.set_axisbelow(True)
        data=[]
        for alg in algorithms:
            mask=(frame.algorithm.eq(alg)&frame.num_robots.eq(size.num_robots)&frame.num_tasks.eq(size.num_tasks)&frame.success.astype(bool))
            vals=pd.to_numeric(frame.loc[mask,"allocation_cost"],errors="coerce").dropna().to_numpy(float)
            data.append(vals); allvalues.extend(vals.tolist())
        nonempty=[(i+1,v) for i,v in enumerate(data) if len(v)]
        if nonempty:
            bp=ax.boxplot([v for _,v in nonempty], positions=[i for i,_ in nonempty], widths=.55,
                          patch_artist=True, showfliers=True, medianprops={"color":"#202020","linewidth":1.4})
            for patch,(i,_) in zip(bp["boxes"],nonempty): patch.set_facecolor(algorithm_style(algorithms[i-1],algorithms)[0]); patch.set_alpha(.55)
            for i,v in nonempty:
                color=algorithm_style(algorithms[i-1],algorithms)[0]
                jitter=np.linspace(-.09,.09,len(v)) if len(v)>1 else np.array([0.])
                ax.scatter(i+jitter,v,s=18,color=color,edgecolor="white",linewidth=.35,zorder=3)
        ax.set_xticks(range(1,len(algorithms)+1)); ax.set_xticklabels(algorithms,rotation=35,ha="right",fontsize=8)
        ax.set_title(f"N={int(size.total_entities)}\n{int(size.num_robots)}R / {int(size.num_tasks)}T",fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if not allvalues:
        raise MetricUnavailable("cost distributions: no successful runs with valid costs")
    _raw_log_if_wide(axes[0,0], allvalues, "Allocation cost")
    if axes[0,0].get_yscale()=="log":
        for ax in axes[0,1:]: ax.set_yscale("log"); ax.yaxis.set_major_formatter(FuncFormatter(lambda value,_:f"{value:,.0f}"))
    fig.suptitle("Per-run allocation-cost distributions",x=.012,y=.975,ha="left",fontsize=13,color=TEXT_PRIMARY)
    fig.text(.012,.012,"Boxes show median and interquartile range; whiskers and points show spread and outliers. Only successful runs with recorded costs are included."+("\n"+footnote if footnote else ""),fontsize=7.2,color=TEXT_MUTED,wrap=True)
    return fig


def plot_energy_heatmap(summary: pd.DataFrame) -> Figure:
    sizes=problem_sizes(summary); algorithms=ordered_algorithms(summary)
    matrix=np.full((len(algorithms),len(sizes)),np.nan)
    for i,a in enumerate(algorithms):
        for j,s in sizes.iterrows():
            m=summary[(summary.algorithm==a)&(summary.num_robots==s.num_robots)&(summary.num_tasks==s.num_tasks)]
            if len(m): matrix[i,j]=pd.to_numeric(m.iloc[0].energy_feasible_rate,errors="coerce")
    if not np.isfinite(matrix).any(): raise MetricUnavailable("energy heatmap: no eligible feasibility flags")
    fig=Figure(figsize=(max(8.5,len(sizes)*1.15),max(3.5,len(algorithms)*.65)),facecolor=SURFACE)
    ax=fig.subplots(); im=ax.imshow(np.ma.masked_invalid(matrix),vmin=0,vmax=1,cmap="YlGn",aspect="auto")
    ax.set_xticks(np.arange(len(sizes)),[f"C{i+1}\nN={int(s.total_entities)}" for i,(_,s) in enumerate(sizes.iterrows())])
    ax.set_yticks(np.arange(len(algorithms)),algorithms); ax.set_xlabel("Problem configuration"); ax.set_ylabel("Algorithm")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]): ax.text(j,i,"n/a" if np.isnan(matrix[i,j]) else f"{matrix[i,j]*100:.0f}%",ha="center",va="center",fontsize=9,color="#333")
    fig.colorbar(im,ax=ax,label="Energy-feasible rate"); _frame_text(fig,"Energy-feasibility rate by algorithm and configuration","Denominator: successful runs with a recorded feasibility flag.")
    return fig


def plot_reliability(frame: pd.DataFrame) -> Figure:
    _require_columns(frame,["algorithm","num_robots","num_tasks","total_entities","success"])
    sizes=_raw_configuration_table(frame); algorithms=ordered_algorithms(frame)
    configs=list(zip(sizes.num_robots,sizes.num_tasks)); x=np.arange(len(configs)); width=.8/max(1,len(algorithms))
    fig,ax=_raw_chart(frame,"Attempted-run outcomes","Problem configuration","Share of attempted runs","Failure categories are combined because the result schema does not store a structured failure type.",(max(9,len(configs)*1.05),5.4))
    for ai,alg in enumerate(algorithms):
        vals=[]; counts=[]
        for r,t in configs:
            g=frame[(frame.algorithm==alg)&(frame.num_robots==r)&(frame.num_tasks==t)]; counts.append(len(g))
            vals.append(float(g.success.astype(bool).mean()) if len(g) else np.nan)
        xpos=x+(ai-(len(algorithms)-1)/2)*width
        ax.bar(xpos,vals,width,color=algorithm_style(alg,algorithms)[0],label=alg)
        for xx,v,n in zip(xpos,vals,counts):
            if np.isfinite(v):
                ax.text(xx,v/2,f"{v*100:.0f}%\nn={n}",ha="center",va="center",fontsize=6.5)
                if v < .82: ax.text(xx,v+(1-v)/2,f"{(1-v)*100:.0f}%",ha="center",va="center",fontsize=6,color="#42413f")
    ax.set_ylim(0,1); ax.set_yticks([0,.25,.5,.75,1],["0%","25%","50%","75%","100%"])
    ax.set_xticks(x,[f"C{i+1} (N={int(s.total_entities)})" for i,(_,s) in enumerate(sizes.iterrows())]); ax.legend(frameon=False,ncol=min(4,len(algorithms)))
    return fig


def plot_cost_runtime_tradeoff(summary: pd.DataFrame) -> Figure:
    _require_columns(summary,["algorithm","num_robots","num_tasks","total_entities","allocation_cost_mean","allocation_runtime_seconds_mean","n_allocation_cost","n_allocation_runtime"])
    fig,ax=_raw_chart(summary,"Allocation cost and runtime trade-off","Allocation runtime (seconds)","Allocation cost A(U)","Cost uses successful valid measurements; runtime uses allocation-stage-completed runs. Their eligible run sets may differ.")
    sizes=problem_sizes(summary); algorithms=ordered_algorithms(summary); vals=[]
    markers=("o","s","^","D","v","P","X","*")
    for ai,a in enumerate(algorithms):
        g=summary[summary.algorithm==a]
        for _,row in g.iterrows():
            c=pd.to_numeric(row.allocation_cost_mean,errors="coerce"); r=pd.to_numeric(row.allocation_runtime_seconds_mean,errors="coerce")
            if pd.notna(c) and pd.notna(r):
                j=next((j for j,s in sizes.iterrows() if s.num_robots==row.num_robots and s.num_tasks==row.num_tasks),0)
                ax.scatter(r,c,color=algorithm_style(a,algorithms)[0],marker=markers[j%len(markers)],s=60,label=a if row.equals(g.iloc[0]) else "")
                vals.append(c)
                ax.annotate(f"C{j+1}",(r,c),xytext=(4,4),textcoords="offset points",fontsize=7)
    if not vals:
        raise MetricUnavailable("cost/runtime trade-off: no configuration has both eligible cost and runtime means")
    _raw_log_if_wide(ax,vals,"Allocation cost A(U)")
    runtimes=pd.to_numeric(summary.allocation_runtime_seconds_mean,errors="coerce").dropna().to_numpy(float)
    if len(runtimes) and (runtimes>0).all() and runtimes.max()/runtimes.min()>=LOG_SCALE_RATIO:
        ax.set_xscale("log"); ax.set_xlabel("Allocation runtime (seconds, log scale)")
    ax.legend(frameon=False,ncol=min(4,len(algorithms)))
    return fig


def plot_paired_costs(frame: pd.DataFrame, baseline: str = "Greedy") -> Figure:
    _require_columns(frame,["experiment_id","scenario_id","algorithm","num_robots","num_tasks","total_entities","success","allocation_cost"])
    algorithms=ordered_algorithms(frame)
    if baseline not in algorithms: raise MetricUnavailable(f"paired costs: baseline algorithm {baseline!r} is unavailable")
    data=frame[frame.success.astype(bool)].copy(); data["allocation_cost"]=pd.to_numeric(data.allocation_cost,errors="coerce")
    data=data.dropna(subset=["allocation_cost"])
    keys=["experiment_id","scenario_id","num_robots","num_tasks","total_entities"]
    pivot=data.pivot_table(index=keys,columns="algorithm",values="allocation_cost",aggfunc="first")
    if baseline not in pivot: raise MetricUnavailable(f"paired costs: no valid baseline {baseline!r}")
    rows=[]
    for a in algorithms:
        if a==baseline or a not in pivot: continue
        pair=pivot[[baseline,a]].dropna(); pair=pair[pair[baseline]>0]
        for key,row in pair.iterrows(): rows.append((a,key[2],key[3],key[4],row[a]/row[baseline]))
    if not rows: raise MetricUnavailable("paired costs: no matched scenarios with a positive baseline cost")
    paired=pd.DataFrame(rows,columns=["algorithm","robots","tasks","entities","relative"])
    sizes=paired[["robots","tasks","entities"]].drop_duplicates().sort_values("entities")
    fig,ax=_raw_chart(frame,f"Per-scenario cost relative to {baseline}","Problem configuration","Cost / baseline cost","Each point is matched by (experiment_id, scenario_id). Baseline is 1.0; lower relative cost is preferable for the allocation-cost objective.",(max(9,len(sizes)*1.2),5.4))
    for ai,a in enumerate([x for x in algorithms if x!=baseline]):
        for j,(_,s) in enumerate(sizes.iterrows()):
            vals=paired[(paired.algorithm==a)&(paired.robots==s.robots)&(paired.tasks==s.tasks)].relative.to_numpy()
            if len(vals):
                jitter=np.linspace(-.1,.1,len(vals)) if len(vals)>1 else np.array([0.])
                ax.scatter(j+ai*.12+jitter,vals,color=algorithm_style(a,algorithms)[0],s=22,alpha=.7,label=a if j==0 else "")
                ax.plot([j+ai*.12-.18,j+ai*.12+.18],[np.median(vals)]*2,color=algorithm_style(a,algorithms)[0],linewidth=2)
    ax.axhline(1,color="#333",linestyle="--",linewidth=1); ax.set_xticks(range(len(sizes)),[f"C{i+1}\nN={int(s.entities)}" for i,(_,s) in enumerate(sizes.iterrows())]); ax.legend(frameon=False)
    return fig


def plot_gini_distributions(frame: pd.DataFrame) -> Figure:
    _require_columns(frame,["algorithm","num_robots","num_tasks","total_entities","gini_index"])
    data=frame.copy(); data["gini_index"]=pd.to_numeric(data.gini_index,errors="coerce")
    data=data.dropna(subset=["gini_index"])
    if data.empty: raise MetricUnavailable("Gini distributions: no valid complete reward records")
    sizes=_raw_configuration_table(data); algorithms=ordered_algorithms(data)
    fig=Figure(figsize=(max(9,len(sizes)*1.6),5.2),facecolor=SURFACE); axes=fig.subplots(1,len(sizes),squeeze=False)
    fig.subplots_adjust(left=.08,right=.99,top=.82,bottom=.2,wspace=.25)
    for j,(_,s) in enumerate(sizes.iterrows()):
        ax=axes[0,j]; arr=[]; labels=[]; colors=[]
        for a in algorithms:
            vals=data[(data.algorithm==a)&(data.num_robots==s.num_robots)&(data.num_tasks==s.num_tasks)].gini_index.to_numpy()
            if len(vals): arr.append(vals); labels.append(a); colors.append(algorithm_style(a,algorithms)[0])
        if arr:
            bp=ax.boxplot(arr,patch_artist=True,showfliers=True); [p.set_facecolor(c) for p,c in zip(bp["boxes"],colors)]
            for i,(v,c) in enumerate(zip(arr,colors),1): ax.scatter(np.full(len(v),i),v,color=c,s=16,alpha=.7,zorder=3)
        ax.set_xticks(range(1,len(labels)+1),labels,rotation=35,ha="right",fontsize=8); ax.set_title(f"N={int(s.total_entities)}")
        ax.set_ylim(0,1); ax.grid(axis="y",color=GRID)
    fig.suptitle("Per-run robot-reward Gini distributions",x=.012,y=.975,ha="left",fontsize=13,color=TEXT_PRIMARY)
    fig.text(.012,.012,"Gini recomputed from complete, eligible robot reward records using the approved Gini function.",fontsize=7.2,color=TEXT_MUTED)
    return fig


def plot_lorenz_curve(reward_records: Sequence[Tuple[str, float]], label: str) -> Figure:
    values=np.asarray([float(v) for _,v in reward_records],dtype=float)
    if len(values)<1 or not np.isfinite(values).all() or (values<0).any(): raise MetricUnavailable("Lorenz curve: invalid or incomplete reward set")
    values=np.sort(values); cum=np.r_[0,np.cumsum(values)]
    y=cum/cum[-1] if cum[-1]>0 else np.linspace(0,1,len(values)+1)
    fig,ax=_raw_chart(None,"Robot-reward Lorenz curve", "Cumulative share of robots", "Cumulative share of reward", f"One eligible run: {label}. All-zero rewards follow the existing equality convention.")
    x=np.linspace(0,1,len(values)+1); ax.plot(x,y,color="#2a78d6",linewidth=2,label=label); ax.plot([0,1],[0,1],"--",color="#777",label="Equality"); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(frameon=False)
    return fig


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

_SAVE_METADATA = {"png": {"Software": None}, "pdf": {"CreationDate": None, "Creator": None, "Producer": None}}


def save_figure(figure: Figure, base_path: Path, formats: Sequence[str] = ("png", "pdf")) -> List[Path]:
    """Save ``figure`` as ``<base_path>.<format>`` for each format (overwrites; atomic per file)."""
    written: List[Path] = []
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        if fmt not in _SAVE_METADATA:
            raise ValueError(f"unsupported figure format {fmt!r}; use one of {sorted(_SAVE_METADATA)}.")
        target = base_path.parent / f"{base_path.name}.{fmt}"
        handle, tmp_name = tempfile.mkstemp(dir=str(base_path.parent), prefix="." + target.name + ".", suffix=".tmp")
        os.close(handle)
        try:
            figure.savefig(tmp_name, format=fmt, dpi=SAVE_DPI, facecolor=figure.get_facecolor(),
                           metadata=_SAVE_METADATA[fmt])
            os.replace(tmp_name, str(target))
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        written.append(target)
    return written

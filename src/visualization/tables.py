"""Report-ready tables built from the aggregated summary.

Each table has one row per experimental configuration (problem size N, robots,
tasks) and one column per algorithm. Cells read ``mean ± std (n=k)`` where ``k``
is the metric-specific valid-value count of the aggregation (the number of runs
that were actually eligible for that mean). A cell with no valid measurement
reads ``n/a``; a missing value is never shown as zero. The standard deviation is
the sample standard deviation and is omitted when fewer than two values exist.

Tables are written as CSV, LaTeX and Markdown. Coalition stability has no table:
it is pending a formal definition.

Owner: Yash Yeole (Benchmarking & Analytics).
"""

import csv
import io
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from src.visualization.plots import METRIC_SPECS, ordered_algorithms, problem_sizes

NOT_AVAILABLE = "n/a"
RUN_OUTCOMES_KEY = "run_outcomes"
TABLE_KEYS = tuple(METRIC_SPECS) + (RUN_OUTCOMES_KEY,)

TABLE_TITLES = {
    "allocation_cost": "Allocation cost A(U): mean ± sample std (n = valid runs)",
    "allocation_runtime": "Allocation runtime in seconds: mean ± sample std (n = runs whose allocation stage completed)",
    "reward_runtime": "Reward-stage runtime in seconds: mean ± sample std (n = runs whose reward stage completed)",
    "total_runtime": "Total pipeline runtime in seconds: mean ± sample std (n = runs with both stages completed)",
    "energy_feasibility": "Energy-feasible share of eligible successful runs (successful runs with a recorded "
                          "feasibility flag)",
    "gini": "Gini index of robot rewards: mean ± sample std (n = runs with valid reward records)",
    RUN_OUTCOMES_KEY: "Successful runs / attempted runs",
}


def _number(value: float, kind: str) -> str:
    if kind == "cost":
        return f"{value:,.1f}"
    if kind == "gini":
        return f"{value:.3f}"
    magnitude = abs(value)                      # runtimes
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    if magnitude >= 0.01:
        return f"{value:.3f}"
    return f"{value:.2e}"


def _cell(row: pd.Series, mean_column: str, std_column: Optional[str], count_column: str, kind: str) -> str:
    mean = pd.to_numeric(row[mean_column], errors="coerce")
    count = pd.to_numeric(row[count_column], errors="coerce")
    if pd.isna(mean) or pd.isna(count) or count <= 0:
        return NOT_AVAILABLE
    std = pd.to_numeric(row[std_column], errors="coerce") if std_column else float("nan")
    text = _number(float(mean), kind)
    if pd.notna(std):
        text += f" ± {_number(float(std), kind)}"
    return f"{text} (n={int(count)})"


def build_table(summary: pd.DataFrame, key: str) -> pd.DataFrame:
    """Formatted table for one metric (or ``run_outcomes``): sizes x algorithms of strings."""
    if key not in TABLE_KEYS:
        raise ValueError(f"unknown table {key!r}; choose from {list(TABLE_KEYS)}.")
    sizes = problem_sizes(summary)
    algorithms = ordered_algorithms(summary)
    rows = []
    for _, size in sizes.iterrows():
        row = {"N": int(size["total_entities"]), "Robots": int(size["num_robots"]), "Tasks": int(size["num_tasks"])}
        for algorithm in algorithms:
            match = summary[(summary["algorithm"] == algorithm)
                            & (summary["total_entities"] == size["total_entities"])
                            & (summary["num_robots"] == size["num_robots"])
                            & (summary["num_tasks"] == size["num_tasks"])]
            if match.empty:
                row[algorithm] = NOT_AVAILABLE
                continue
            data = match.iloc[0]
            if key == RUN_OUTCOMES_KEY:
                row[algorithm] = f"{int(data['n_success'])}/{int(data['n_runs'])}"
            elif key == "energy_feasibility":
                rate = pd.to_numeric(data["energy_feasible_rate"], errors="coerce")
                # The aggregation's denominator is the successful runs that have a non-missing
                # energy_feasible value; it is not n_success, so no count is claimed here.
                row[algorithm] = (NOT_AVAILABLE if pd.isna(rate)
                                  else f"{rate * 100:.0f}% (eligible successful runs)")
            else:
                spec = METRIC_SPECS[key]
                kind = {"allocation_cost": "cost", "gini": "gini"}.get(key, "runtime")
                row[algorithm] = _cell(data, spec.mean_column, spec.std_column, spec.count_column, kind)
        rows.append(row)
    return pd.DataFrame(rows, columns=["N", "Robots", "Tasks"] + algorithms)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text atomically (temp file in the same folder, then replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, str(path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _csv_text(table: pd.DataFrame) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(table.columns)
    writer.writerows(table.astype(str).values.tolist())
    return buffer.getvalue()


def _latex_escape(text: str) -> str:
    replacements = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                    ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("±", r"$\pm$")]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _latex_text(table: pd.DataFrame, caption: str, label: str) -> str:
    alignment = "rrr" + "l" * (len(table.columns) - 3)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{" + _latex_escape(caption) + "}", r"\label{" + label + "}",
             r"\begin{tabular}{" + alignment + "}", r"\hline",
             " & ".join(_latex_escape(str(c)) for c in table.columns) + r" \\", r"\hline"]
    for values in table.astype(str).values.tolist():
        lines.append(" & ".join(_latex_escape(v) for v in values) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _markdown_text(table: pd.DataFrame, caption: str) -> str:
    header = "| " + " | ".join(map(str, table.columns)) + " |"
    divider = "|" + "|".join(["---:"] * 3 + [":---"] * (len(table.columns) - 3)) + "|"
    body = ["| " + " | ".join(row) + " |" for row in table.astype(str).values.tolist()]
    return "\n".join([f"*{caption}*", "", header, divider] + body + [""])


def write_table(table: pd.DataFrame, base_path: Path, caption: str, label: str) -> List[Path]:
    """Write ``<base_path>.csv``, ``.tex`` and ``.md`` (each file atomically; reruns overwrite)."""
    base_path = Path(base_path)
    outputs = {
        ".csv": _csv_text(table),
        ".tex": _latex_text(table, caption, label),
        ".md": _markdown_text(table, caption),
    }
    written = []
    for suffix, text in outputs.items():
        target = base_path.parent / f"{base_path.name}{suffix}"
        write_text_atomic(target, text)
        written.append(target)
    return written

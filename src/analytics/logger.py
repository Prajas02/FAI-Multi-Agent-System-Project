"""Experiment logging, metadata and aggregation for the benchmarking harness.

Implements Sections 5 (result schemas), 5.4 (writing rules) and 6 (aggregation)
of docs/ANALYTICS_BENCHMARKING_PLAN.md.

- ``append_benchmark_results``: append-only writer for ``benchmark_results.csv``.
- ``append_robot_rewards``: append-only writer for ``robot_rewards.csv``.
  The two writers are independent: there is no cross-file transaction and no
  automatic orphan cleanup.
- ``write_experiment_metadata``: ``metadata_<experiment_id>.yaml``.
- ``aggregate_benchmark_results`` / ``write_summary_table`` /
  ``summarize_experiment``: per-experiment summary (Section 6).

The module works on standardized dictionaries and DataFrames only. It does not
compute Gini, costs or energy penalties and does not import solver classes.

Owner: Yash Yeole (Benchmarking & Analytics).
"""

import copy
import csv
import io
import math
import os
import platform
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

PathLike = Union[str, "os.PathLike[str]"]

# ---------------------------------------------------------------------------
# Schemas (Sections 5.1, 5.2 and 6)
# ---------------------------------------------------------------------------

BENCHMARK_RESULT_COLUMNS: Tuple[str, ...] = (
    "experiment_id",
    "scenario_id",
    "algorithm",
    "num_robots",
    "num_tasks",
    "total_entities",
    "random_seed",
    "allocation_cost",
    "allocation_runtime_seconds",
    "reward_runtime_seconds",
    "total_runtime_seconds",
    "allocation_stage_completed",
    "reward_stage_completed",
    "gini_index",
    "invalid_reward_flag",
    "energy_feasible",
    "coalition_stability",
    "success",
    "error_message",
)

ROBOT_REWARD_COLUMNS: Tuple[str, ...] = (
    "experiment_id",
    "scenario_id",
    "algorithm",
    "robot_id",
    "reward",
)

SUMMARY_COLUMNS: Tuple[str, ...] = (
    "algorithm",
    "num_robots",
    "num_tasks",
    "total_entities",
    "n_runs",
    "n_success",
    "n_allocation_cost",
    "n_allocation_runtime",
    "n_reward_runtime",
    "n_total_runtime",
    "n_gini",
    "allocation_cost_mean",
    "allocation_cost_std",
    "allocation_runtime_seconds_mean",
    "allocation_runtime_seconds_std",
    "reward_runtime_seconds_mean",
    "reward_runtime_seconds_std",
    "total_runtime_seconds_mean",
    "total_runtime_seconds_std",
    "energy_feasible_rate",
    "gini_index_mean",
    "gini_index_std",
    "n_invalid_reward",
    "coalition_stability_mean",
    "coalition_stability_std",
)

BENCHMARK_RESULTS_FILENAME = "benchmark_results.csv"
ROBOT_REWARDS_FILENAME = "robot_rewards.csv"

_BENCHMARK_KEY_COLUMNS = ("experiment_id", "scenario_id", "algorithm")
_ROBOT_REWARD_KEY_COLUMNS = ("experiment_id", "scenario_id", "algorithm", "robot_id")

_INT_COLUMNS = ("num_robots", "num_tasks", "total_entities", "random_seed")
_FLOAT_COLUMNS = (
    "allocation_cost",
    "allocation_runtime_seconds",
    "reward_runtime_seconds",
    "total_runtime_seconds",
    "gini_index",
    "coalition_stability",
)
_NONNEGATIVE_FLOAT_COLUMNS = (
    "allocation_runtime_seconds",
    "reward_runtime_seconds",
    "total_runtime_seconds",
)
_REQUIRED_BOOL_COLUMNS = (
    "allocation_stage_completed",
    "reward_stage_completed",
    "invalid_reward_flag",
    "success",
)
_STRING_COLUMNS = ("experiment_id", "scenario_id", "algorithm", "error_message")

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = _REPO_ROOT / "data" / "results" / "raw"
DEFAULT_SUMMARY_DIR = _REPO_ROOT / "outputs" / "tables"

_EXPERIMENT_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")


class SchemaValidationError(ValueError):
    """A record, file header or argument violates the approved schema."""


class DuplicateKeyError(ValueError):
    """A uniqueness key occurs more than once (in a batch or in the file)."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def validate_experiment_id(experiment_id: Any) -> str:
    """Return ``experiment_id`` if it is safe in file names, else raise.

    Allowed: letters, digits, ``.``, ``_`` and ``-`` (1-128 characters),
    and not consisting only of dots.
    """
    if (
        not isinstance(experiment_id, str)
        or _EXPERIMENT_ID_RE.fullmatch(experiment_id) is None
        or set(experiment_id) == {"."}
    ):
        raise SchemaValidationError(
            "experiment_id must be 1-128 characters from [A-Za-z0-9._-] and "
            f"not only dots, got {experiment_id!r}."
        )
    return experiment_id


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return math.isnan(float(value))
    return False


def _serialize_cell(value: Any) -> str:
    """Missing -> empty cell, bool -> True/False, float -> repr (exact)."""
    if _is_missing(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def _resolve_dir(directory: Optional[PathLike], default: Path) -> Path:
    return Path(directory) if directory is not None else default


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file + fsync + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix="." + path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, str(path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Field validators
# ---------------------------------------------------------------------------

def _check_str(name: str, value: Any, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SchemaValidationError(f"{name} must be a string, got {value!r}.")
    if not allow_empty and value == "":
        raise SchemaValidationError(f"{name} must not be empty.")
    return value


def _check_int(name: str, value: Any, minimum: Optional[int] = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise SchemaValidationError(f"{name} must be an integer, got {value!r}.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise SchemaValidationError(f"{name} must be >= {minimum}, got {result}.")
    return result


def _check_bool(name: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise SchemaValidationError(f"{name} must be a boolean, got {value!r}.")
    return bool(value)


def _check_optional_float(
    name: str, value: Any, nonnegative: bool = False
) -> Optional[float]:
    """Return a finite float, or None when the value is missing (None/NaN)."""
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise SchemaValidationError(f"{name} must be a number or missing, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaValidationError(f"{name} must be finite, got {result!r}.")
    if nonnegative and result < 0:
        raise SchemaValidationError(f"{name} must be >= 0, got {result!r}.")
    return result


def _check_exact_columns(
    record: Any, columns: Sequence[str], what: str
) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise SchemaValidationError(f"{what} must be a mapping, got {type(record).__name__}.")
    missing = [c for c in columns if c not in record]
    unknown = [str(k) for k in record if k not in columns]
    if missing or unknown:
        raise SchemaValidationError(
            f"{what} must have exactly the schema columns; "
            f"missing={missing}, unknown={unknown}."
        )
    return record


def _validate_benchmark_record(record: Any) -> Dict[str, Any]:
    """Validate one main-results record; return normalized values by column."""
    rec = _check_exact_columns(record, BENCHMARK_RESULT_COLUMNS, "benchmark record")
    out: Dict[str, Any] = {}

    out["experiment_id"] = validate_experiment_id(rec["experiment_id"])
    out["scenario_id"] = _check_str("scenario_id", rec["scenario_id"])
    out["algorithm"] = _check_str("algorithm", rec["algorithm"])
    for name in _INT_COLUMNS:
        out[name] = _check_int(name, rec[name], minimum=0 if name != "random_seed" else None)
    if out["total_entities"] != out["num_robots"] + out["num_tasks"]:
        raise SchemaValidationError(
            "total_entities must equal num_robots + num_tasks "
            f"({out['num_robots']} + {out['num_tasks']} != {out['total_entities']})."
        )
    for name in _FLOAT_COLUMNS:
        if name == "coalition_stability":
            continue
        out[name] = _check_optional_float(
            name, rec[name], nonnegative=name in _NONNEGATIVE_FLOAT_COLUMNS
        )
    for name in _REQUIRED_BOOL_COLUMNS:
        out[name] = _check_bool(name, rec[name])
    out["energy_feasible"] = (
        None if _is_missing(rec["energy_feasible"])
        else _check_bool("energy_feasible", rec["energy_feasible"])
    )
    if not _is_missing(rec["coalition_stability"]):
        raise SchemaValidationError(
            "coalition_stability must be missing/NaN until a definition is formally "
            f"agreed (Section 2.5), got {rec['coalition_stability']!r}."
        )
    out["coalition_stability"] = None
    out["error_message"] = (
        "" if rec["error_message"] is None
        else _check_str("error_message", rec["error_message"], allow_empty=True)
    )

    _check_cross_field_rules(out)
    return out


def _check_cross_field_rules(rec: Mapping[str, Any]) -> None:
    """The five cross-field rules of Section 5.4."""
    if rec["success"] and not rec["allocation_stage_completed"]:
        raise SchemaValidationError(
            "Rule 1: success=True requires allocation_stage_completed=True."
        )
    if rec["reward_stage_completed"] and not (
        rec["allocation_stage_completed"] and rec["success"]
    ):
        raise SchemaValidationError(
            "Rule 2: reward_stage_completed=True requires "
            "allocation_stage_completed=True and success=True."
        )
    if rec["invalid_reward_flag"] and not rec["reward_stage_completed"]:
        raise SchemaValidationError(
            "Rule 3: invalid_reward_flag=True requires reward_stage_completed=True."
        )
    if rec["gini_index"] is not None and not (
        rec["reward_stage_completed"] and not rec["invalid_reward_flag"]
    ):
        raise SchemaValidationError(
            "Rule 4: a non-NaN gini_index requires reward_stage_completed=True "
            "and invalid_reward_flag=False."
        )
    if not rec["success"] and rec["allocation_cost"] is not None:
        raise SchemaValidationError(
            "Rule 5: success=False requires allocation_cost to be missing/NaN."
        )


def _validate_robot_reward_record(record: Any) -> Dict[str, Any]:
    rec = _check_exact_columns(record, ROBOT_REWARD_COLUMNS, "robot reward record")
    out: Dict[str, Any] = {}
    out["experiment_id"] = validate_experiment_id(rec["experiment_id"])
    out["scenario_id"] = _check_str("scenario_id", rec["scenario_id"])
    out["algorithm"] = _check_str("algorithm", rec["algorithm"])

    robot_id = rec["robot_id"]
    if isinstance(robot_id, (bool, np.bool_)):
        raise SchemaValidationError(f"robot_id must be an int or string, got {robot_id!r}.")
    if isinstance(robot_id, (int, np.integer)):
        out["robot_id"] = int(robot_id)
    elif isinstance(robot_id, str) and robot_id != "":
        out["robot_id"] = robot_id
    else:
        raise SchemaValidationError(f"robot_id must be an int or non-empty string, got {robot_id!r}.")

    reward = rec["reward"]
    if (
        isinstance(reward, (bool, np.bool_))
        or not isinstance(reward, (int, float, np.integer, np.floating))
        or not math.isfinite(float(reward))
    ):
        raise SchemaValidationError(
            f"reward must be a finite number (negative values are kept as-is), got {reward!r}."
        )
    out["reward"] = float(reward)
    return out


# ---------------------------------------------------------------------------
# Strict parsing of CSV cells (used for existing files and for loading)
# ---------------------------------------------------------------------------

_INT_CELL_RE = re.compile(r"[+-]?[0-9]+(\.0+)?")
_FLOAT_CELL_RE = re.compile(r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?")
_INT64_MIN = -(2 ** 63)
_INT64_MAX = 2 ** 63 - 1


def _parse_int_cell(name: str, text: str) -> int:
    """Parse an integral cell. Plain integers (and integral ``5.0``) only."""
    if _INT_CELL_RE.fullmatch(text) is None:
        raise SchemaValidationError(f"{name}: invalid integer {text!r}.")
    value = int(text.split(".")[0])
    if not _INT64_MIN <= value <= _INT64_MAX:
        raise SchemaValidationError(f"{name}: integer {text!r} is out of range.")
    return value


def _parse_float_cell(name: str, text: str) -> float:
    """Parse a numeric cell (decimal notation). ``nan``/``inf`` text is rejected."""
    if _FLOAT_CELL_RE.fullmatch(text) is None:
        raise SchemaValidationError(f"{name}: invalid number {text!r}.")
    return float(text)


def _parse_bool_cell(name: str, text: str, allow_missing: bool = False) -> Optional[bool]:
    if text == "True":
        return True
    if text == "False":
        return False
    if text == "" and allow_missing:
        return None
    allowed = "True, False or empty" if allow_missing else "True or False"
    raise SchemaValidationError(f"{name}: expected {allowed}, got {text!r}.")


def _validate_benchmark_cells(cells: Mapping[str, str]) -> Dict[str, Any]:
    """Parse and validate one main-results CSV row (all cells are strings).

    Applies the same schema, type and cross-field validation as the writer.
    """
    parsed: Dict[str, Any] = {}
    for name in ("experiment_id", "scenario_id", "algorithm", "error_message"):
        parsed[name] = cells[name]
    for name in _INT_COLUMNS:
        parsed[name] = _parse_int_cell(name, cells[name])
    for name in _FLOAT_COLUMNS:
        if name == "coalition_stability":
            continue
        parsed[name] = None if cells[name] == "" else _parse_float_cell(name, cells[name])
    for name in _REQUIRED_BOOL_COLUMNS:
        parsed[name] = _parse_bool_cell(name, cells[name])
    parsed["energy_feasible"] = _parse_bool_cell(
        "energy_feasible", cells["energy_feasible"], allow_missing=True
    )
    # Any non-empty cell is passed on as a string and rejected by the validator.
    parsed["coalition_stability"] = None if cells["coalition_stability"] == "" else cells["coalition_stability"]
    return _validate_benchmark_record(parsed)


def _validate_robot_reward_cells(cells: Mapping[str, str]) -> Dict[str, Any]:
    """Parse and validate one robot-reward CSV row (all cells are strings)."""
    parsed = {
        "experiment_id": cells["experiment_id"],
        "scenario_id": cells["scenario_id"],
        "algorithm": cells["algorithm"],
        "robot_id": cells["robot_id"],
        "reward": _parse_float_cell("reward", cells["reward"]),
    }
    return _validate_robot_reward_record(parsed)


# ---------------------------------------------------------------------------
# CSV append machinery
# ---------------------------------------------------------------------------

def _read_existing_csv(
    path: Path,
    columns: Sequence[str],
    key_columns: Sequence[str],
    validate_row: Any = None,
) -> Tuple[bytes, List[Tuple[str, ...]], List[Dict[str, Any]]]:
    """Return (raw bytes, keys of existing rows, validated rows).

    Validates the header, the cell count of every row and, when ``validate_row``
    is given, every row's content (``validate_row`` maps a dict of cell strings
    to a normalized record and raises ``SchemaValidationError``).
    """
    if not path.exists():
        return b"", [], []
    raw = path.read_bytes()
    if len(raw) == 0:
        return b"", [], []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SchemaValidationError(f"{path} is not valid UTF-8: {exc}") from None
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader, None)
    if header is None or list(header) != list(columns):
        raise SchemaValidationError(
            f"{path} has an unexpected header. Expected {list(columns)}, got {header}."
        )
    key_idx = [columns.index(c) for c in key_columns]
    keys: List[Tuple[str, ...]] = []
    records: List[Dict[str, Any]] = []
    for line_no, row in enumerate(reader, start=2):
        if not row:
            continue
        if len(row) != len(columns):
            raise SchemaValidationError(
                f"{path} line {line_no} has {len(row)} cells, expected {len(columns)}."
            )
        if validate_row is not None:
            try:
                records.append(validate_row(dict(zip(columns, row))))
            except SchemaValidationError as exc:
                raise SchemaValidationError(f"{path} line {line_no}: {exc}") from None
        keys.append(tuple(row[i] for i in key_idx))
    return raw, keys, records


def _find_duplicates(keys: Sequence[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
    seen = set()
    dups: List[Tuple[str, ...]] = []
    for key in keys:
        if key in seen and key not in dups:
            dups.append(key)
        seen.add(key)
    return dups


def _append_rows(
    path: Path,
    columns: Sequence[str],
    key_columns: Sequence[str],
    rows: List[List[str]],
    validate_existing_row: Any = None,
) -> int:
    """Validate keys against the batch and the file, then atomically append.

    Existing rows are fully validated (schema, types, cross-field rules) when
    ``validate_existing_row`` is given; a malformed existing record rejects the
    append.
    """
    if not rows:
        return 0
    key_idx = [columns.index(c) for c in key_columns]
    new_keys = [tuple(row[i] for i in key_idx) for row in rows]

    batch_dups = _find_duplicates(new_keys)
    if batch_dups:
        raise DuplicateKeyError(f"Duplicate keys within the batch: {batch_dups}.")

    raw, existing_keys, _ = _read_existing_csv(
        path, columns, key_columns, validate_existing_row
    )
    existing_dups = _find_duplicates(existing_keys)
    if existing_dups:
        raise DuplicateKeyError(
            f"{path} already contains duplicate keys {existing_dups}; refusing to append."
        )
    existing_set = set(existing_keys)
    clash = [key for key in new_keys if key in existing_set]
    if clash:
        raise DuplicateKeyError(f"Keys already present in {path}: {clash}.")

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    if not raw:
        writer.writerow(columns)
    writer.writerows(rows)
    prefix = raw
    if prefix and not prefix.endswith(b"\n"):
        prefix += b"\n"
    _atomic_write_bytes(path, prefix + buffer.getvalue().encode("utf-8"))
    return len(rows)


def append_benchmark_results(
    records: Sequence[Mapping[str, Any]],
    results_dir: Optional[PathLike] = None,
) -> int:
    """Append main benchmark records (Section 5.1) to ``benchmark_results.csv``.

    The whole batch is validated first; any invalid record or duplicate key
    (within the batch or against the file) rejects the entire batch and
    nothing is written. Returns the number of rows appended (0 for an empty
    batch, in which case no file is created).

    Raises:
        SchemaValidationError: invalid record or unexpected existing header.
        DuplicateKeyError: duplicate (experiment_id, scenario_id, algorithm).
    """
    batch = list(records)
    rows: List[List[str]] = []
    for index, record in enumerate(batch):
        try:
            normalized = _validate_benchmark_record(record)
        except SchemaValidationError as exc:
            raise SchemaValidationError(f"record {index}: {exc}") from None
        rows.append([_serialize_cell(normalized[c]) for c in BENCHMARK_RESULT_COLUMNS])
    path = _resolve_dir(results_dir, DEFAULT_RESULTS_DIR) / BENCHMARK_RESULTS_FILENAME
    return _append_rows(
        path, BENCHMARK_RESULT_COLUMNS, _BENCHMARK_KEY_COLUMNS, rows, _validate_benchmark_cells
    )


def append_robot_rewards(
    records: Sequence[Mapping[str, Any]],
    results_dir: Optional[PathLike] = None,
) -> int:
    """Append standardized robot-level rewards (Section 5.2) to ``robot_rewards.csv``.

    Same all-or-nothing and atomic behaviour as :func:`append_benchmark_results`.
    The uniqueness key is (experiment_id, scenario_id, algorithm, robot_id).
    Rewards must be finite numbers; negative values are stored unchanged.
    This writer is independent of the main results writer.
    """
    batch = list(records)
    rows: List[List[str]] = []
    for index, record in enumerate(batch):
        try:
            normalized = _validate_robot_reward_record(record)
        except SchemaValidationError as exc:
            raise SchemaValidationError(f"record {index}: {exc}") from None
        rows.append([_serialize_cell(normalized[c]) for c in ROBOT_REWARD_COLUMNS])
    path = _resolve_dir(results_dir, DEFAULT_RESULTS_DIR) / ROBOT_REWARDS_FILENAME
    return _append_rows(
        path, ROBOT_REWARD_COLUMNS, _ROBOT_REWARD_KEY_COLUMNS, rows, _validate_robot_reward_cells
    )


# ---------------------------------------------------------------------------
# Metadata (Section 5.3)
# ---------------------------------------------------------------------------

def _git_output(args: List[str], cwd: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout


def collect_environment_info(repo_root: Optional[PathLike] = None) -> Dict[str, Any]:
    """Collect machine, library and git information.

    Hostname and user name are deliberately not recorded. Git values are
    ``None`` when git or the repository is unavailable.
    """
    from importlib import metadata as importlib_metadata

    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    libraries: Dict[str, Optional[str]] = {}
    for dist in ("numpy", "pandas", "PyYAML", "scipy", "PuLP"):
        try:
            libraries[dist] = importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            libraries[dist] = None

    commit_out = _git_output(["rev-parse", "HEAD"], root)
    commit = commit_out.strip() if commit_out and commit_out.strip() else None
    status_out = _git_output(["status", "--porcelain"], root)
    dirty = None if status_out is None else bool(status_out.strip())

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "libraries": libraries,
        "git_commit": commit,
        "git_dirty": dirty,
    }


def _to_plain(value: Any, where: str) -> Any:
    """Convert to YAML-safe plain Python data or raise SchemaValidationError."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        plain: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaValidationError(f"{where}: mapping keys must be strings, got {key!r}.")
            plain[key] = _to_plain(item, f"{where}.{key}")
        return plain
    if isinstance(value, (list, tuple)):
        return [_to_plain(item, f"{where}[{i}]") for i, item in enumerate(value)]
    raise SchemaValidationError(
        f"{where}: unsupported value type {type(value).__name__} for metadata."
    )


def write_experiment_metadata(
    experiment_id: str,
    config: Mapping[str, Any],
    seed_info: Mapping[str, Any],
    time_limits: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
    results_dir: Optional[PathLike] = None,
    overwrite: bool = False,
) -> Path:
    """Write ``metadata_<experiment_id>.yaml`` (Section 5.3) atomically.

    Args:
        experiment_id: Safe experiment id (see :func:`validate_experiment_id`).
        config: Copy of the experiment configuration (including the
            energy-penalty strategy and coefficient, supplied by the caller).
        seed_info: Base seed and seed-derivation rule, supplied by the caller.
        time_limits: Per-algorithm time limits, supplied by the caller.
        extra: Optional additional caller-supplied entries (for example
            solver versions beyond those collected automatically).
        results_dir: Output folder (default ``data/results/raw``).
        overwrite: Replace an existing metadata file only if True.

    Raises:
        FileExistsError: the file exists and ``overwrite`` is False.
        SchemaValidationError: invalid id or non-YAML-safe values.
    """
    validate_experiment_id(experiment_id)
    path = _resolve_dir(results_dir, DEFAULT_RESULTS_DIR) / f"metadata_{experiment_id}.yaml"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it.")

    document: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": _to_plain(config, "config"),
        "seed_info": _to_plain(seed_info, "seed_info"),
        "time_limits": _to_plain(time_limits, "time_limits"),
        "environment": collect_environment_info(),
    }
    if extra is not None:
        document["extra"] = _to_plain(extra, "extra")

    text = yaml.safe_dump(document, sort_keys=False, default_flow_style=False, allow_unicode=True)
    _atomic_write_bytes(path, text.encode("utf-8"))
    return path


_PROTECTED_METADATA_KEYS = ("experiment_id", "created_utc", "config", "seed_info", "time_limits", "environment")
_DOTTED_KEY_RE = re.compile(r"[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)*")


def _walk_extra(extra: Dict[str, Any], dotted: str, create_leaf_list: bool) -> Tuple[Dict[str, Any], str]:
    """Return (parent mapping, last key) for a dotted path inside ``extra``."""
    if _DOTTED_KEY_RE.fullmatch(dotted) is None:
        raise SchemaValidationError(f"invalid metadata path {dotted!r}.")
    *parents, leaf = dotted.split(".")
    node = extra
    for part in parents:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise SchemaValidationError(f"metadata path {dotted!r} crosses a non-mapping entry {part!r}.")
        node = child
    return node, leaf


def update_experiment_metadata(
    experiment_id: str,
    set_values: Optional[Mapping[str, Any]] = None,
    append_values: Optional[Mapping[str, Any]] = None,
    results_dir: Optional[PathLike] = None,
) -> Path:
    """Safely update an existing ``metadata_<experiment_id>.yaml`` (for example after ``--resume``).

    Only the ``extra`` section (and ``updated_utc``) can change. The experiment id,
    creation time, config copy, seed info, time limits and environment details of the
    original run are protected: they are never modified and the file is rejected if
    they would change. ``write_experiment_metadata`` keeps its overwrite protection;
    this function never replaces the document, it edits it in place and writes it
    atomically.

    Args:
        experiment_id: Safe experiment id of an existing metadata file.
        set_values: ``{"dotted.path": value}`` entries inside ``extra`` that are replaced
            (intermediate mappings are created).
        append_values: ``{"dotted.path": value}`` entries appended to a list inside
            ``extra`` (the list is created if absent).
        results_dir: Folder of the metadata file (default ``data/results/raw``).

    Raises:
        FileNotFoundError: the metadata file does not exist.
        SchemaValidationError: invalid id/path/value, a malformed file, a path that
            crosses a non-mapping, or an append target that is not a list.
    """
    validate_experiment_id(experiment_id)
    path = _resolve_dir(results_dir, DEFAULT_RESULTS_DIR) / f"metadata_{experiment_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; nothing to update.")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SchemaValidationError(f"cannot read {path}: {exc}") from None
    if not isinstance(document, dict) or document.get("experiment_id") != experiment_id:
        raise SchemaValidationError(f"{path} is not the metadata file of experiment {experiment_id!r}.")

    protected_before = {key: copy.deepcopy(document.get(key)) for key in _PROTECTED_METADATA_KEYS}
    extra = document.setdefault("extra", {})
    if not isinstance(extra, dict):
        raise SchemaValidationError(f"{path}: 'extra' must be a mapping.")

    for dotted, value in (set_values or {}).items():
        parent, leaf = _walk_extra(extra, dotted, create_leaf_list=False)
        parent[leaf] = _to_plain(value, dotted)
    for dotted, value in (append_values or {}).items():
        parent, leaf = _walk_extra(extra, dotted, create_leaf_list=True)
        target = parent.setdefault(leaf, [])
        if not isinstance(target, list):
            raise SchemaValidationError(f"metadata path {dotted!r} is not a list; cannot append.")
        target.append(_to_plain(value, dotted))

    if any(document.get(key) != protected_before[key] for key in _PROTECTED_METADATA_KEYS):
        raise SchemaValidationError("protected metadata entries must not be modified.")  # defensive
    document["updated_utc"] = datetime.now(timezone.utc).isoformat()
    text = yaml.safe_dump(document, sort_keys=False, default_flow_style=False, allow_unicode=True)
    _atomic_write_bytes(path, text.encode("utf-8"))
    return path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_validated_records(
    path: PathLike,
    columns: Sequence[str],
    key_columns: Sequence[str],
    validate_row: Any,
) -> List[Dict[str, Any]]:
    """Read a CSV, validating header, cell counts, every row's content and keys.

    Raises:
        DuplicateKeyError: the uniqueness key occurs more than once.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    _, keys, records = _read_existing_csv(path, columns, key_columns, validate_row)
    duplicates = _find_duplicates(keys)
    if duplicates:
        raise DuplicateKeyError(
            f"{path} contains duplicate keys {duplicates} "
            f"(key columns: {list(key_columns)})."
        )
    return records


def load_benchmark_results(path: Optional[PathLike] = None) -> pd.DataFrame:
    """Load ``benchmark_results.csv`` with typed columns.

    Every row is validated like a written record: exact header, strict integer,
    number and boolean cells, ``total_entities``, the five cross-field rules and
    the missing-only ``coalition_stability``. Malformed content raises
    ``SchemaValidationError`` (with the line number) and nothing is coerced.
    Duplicate (experiment_id, scenario_id, algorithm) keys raise
    ``DuplicateKeyError``.

    Integers are ``int64``, float columns are ``float64`` (empty -> NaN),
    required booleans are ``bool`` and ``energy_feasible`` is the nullable
    ``boolean`` dtype (empty -> ``<NA>``).
    """
    path = Path(path) if path is not None else DEFAULT_RESULTS_DIR / BENCHMARK_RESULTS_FILENAME
    records = _load_validated_records(
        path, BENCHMARK_RESULT_COLUMNS, _BENCHMARK_KEY_COLUMNS, _validate_benchmark_cells
    )
    frame = pd.DataFrame(records, columns=list(BENCHMARK_RESULT_COLUMNS))
    for name in _INT_COLUMNS:
        frame[name] = frame[name].astype("int64")
    for name in _FLOAT_COLUMNS:
        frame[name] = frame[name].astype("float64")
    for name in _REQUIRED_BOOL_COLUMNS:
        frame[name] = frame[name].astype(bool)
    frame["energy_feasible"] = frame["energy_feasible"].astype("boolean")
    return frame


def load_robot_rewards(path: Optional[PathLike] = None) -> pd.DataFrame:
    """Load ``robot_rewards.csv`` (``reward`` as float, ids as strings).

    Every row is validated like a written record; rewards must be finite
    numbers (``nan``/``inf`` text is rejected). Duplicate (experiment_id,
    scenario_id, algorithm, robot_id) keys raise ``DuplicateKeyError``. Note: this loads the raw file,
    including any orphan rows (Section 5.4). Consumers must restrict it to keys
    present in the main results.
    """
    path = Path(path) if path is not None else DEFAULT_RESULTS_DIR / ROBOT_REWARDS_FILENAME
    records = _load_validated_records(
        path, ROBOT_REWARD_COLUMNS, _ROBOT_REWARD_KEY_COLUMNS, _validate_robot_reward_cells
    )
    frame = pd.DataFrame(records, columns=list(ROBOT_REWARD_COLUMNS))
    frame["reward"] = frame["reward"].astype("float64")
    return frame


# ---------------------------------------------------------------------------
# Aggregation (Section 6)
# ---------------------------------------------------------------------------

def _is_na_scalar(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, (float, np.floating)) and math.isnan(float(value))


def _strict_bool_values(
    series: pd.Series, name: str, allow_missing: bool
) -> List[Optional[bool]]:
    """Return the values of ``series`` as bools; reject everything else.

    Accepts Python/NumPy booleans (so numpy ``bool`` and pandas nullable
    ``boolean`` columns work). Strings, integers and any other value are
    rejected, so ``"False"`` can never become ``True``. Missing values
    (None/NaN/``pd.NA``) are accepted only when ``allow_missing`` is True.
    """
    values: List[Optional[bool]] = []
    for value in series:
        if isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
        elif allow_missing and _is_na_scalar(value):
            values.append(None)
        else:
            expected = "booleans or missing values" if allow_missing else "booleans"
            raise SchemaValidationError(
                f"Column {name} must contain only {expected}, got {value!r}."
            )
    return values


def _bool_series(frame: pd.DataFrame, name: str) -> pd.Series:
    """Required boolean column: only real booleans, no missing values."""
    values = _strict_bool_values(frame[name], name, allow_missing=False)
    return pd.Series(values, index=frame.index, dtype=bool)


def _optional_bool_series(frame: pd.DataFrame, name: str) -> Tuple[pd.Series, pd.Series]:
    """Optional boolean column: returns (value present, value is True)."""
    values = _strict_bool_values(frame[name], name, allow_missing=True)
    present = pd.Series([v is not None for v in values], index=frame.index, dtype=bool)
    true = pd.Series([v is True for v in values], index=frame.index, dtype=bool)
    return present, true


def _mean_std(values: pd.Series) -> Tuple[int, float, float]:
    """(count, mean, sample std) of the non-NaN values."""
    valid = values.dropna().astype("float64")
    count = int(valid.shape[0])
    mean = float(valid.mean()) if count >= 1 else float("nan")
    std = float(valid.std(ddof=1)) if count >= 2 else float("nan")
    return count, mean, std


def aggregate_benchmark_results(
    results: pd.DataFrame, experiment_id: Optional[str] = None
) -> pd.DataFrame:
    """Aggregate main results per (algorithm, num_robots, num_tasks, total_entities).

    Implements Section 6: each metric excludes NaN values independently, has its
    own valid-value count, uses the sample standard deviation (``ddof=1``) and
    gives NaN for the std with fewer than two values (and for the mean with
    none). Runtime eligibility uses only the explicit stage-completion fields.
    Only the main results are read; ``robot_rewards.csv`` plays no part.

    Args:
        results: Main results (for example from :func:`load_benchmark_results`).
        experiment_id: Experiment to aggregate. May be omitted only if the data
            contains a single experiment.

    Returns:
        DataFrame with ``SUMMARY_COLUMNS``; empty (with those columns) if no
        records remain after filtering.

    Raises:
        SchemaValidationError: required columns are missing.
        ValueError: several experiments present and no ``experiment_id`` given.
    """
    missing = [c for c in BENCHMARK_RESULT_COLUMNS if c not in results.columns]
    if missing:
        raise SchemaValidationError(f"results is missing columns: {missing}.")

    frame = results
    if experiment_id is not None:
        validate_experiment_id(experiment_id)
        frame = frame[frame["experiment_id"] == experiment_id]
    else:
        ids = sorted(frame["experiment_id"].dropna().unique().tolist())
        if len(ids) > 1:
            raise ValueError(
                f"results contain several experiments {ids}; pass experiment_id."
            )
    frame = frame.reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))

    success = _bool_series(frame, "success")
    alloc_done = _bool_series(frame, "allocation_stage_completed")
    reward_done = _bool_series(frame, "reward_stage_completed")
    invalid = _bool_series(frame, "invalid_reward_flag")
    both_done = alloc_done & reward_done
    numeric = {
        name: pd.to_numeric(frame[name], errors="raise") for name in _FLOAT_COLUMNS
    }
    energy_present, energy_true = _optional_bool_series(frame, "energy_feasible")

    eligible = {
        "allocation_cost": success,
        "allocation_runtime_seconds": alloc_done,
        "reward_runtime_seconds": reward_done,
        "total_runtime_seconds": both_done,
        "gini_index": pd.Series(True, index=frame.index),
        "coalition_stability": pd.Series(True, index=frame.index),
    }

    group_cols = ["algorithm", "num_robots", "num_tasks", "total_entities"]
    rows = []
    for key, index in frame.groupby(group_cols, sort=False).groups.items():
        idx = list(index)
        row: Dict[str, Any] = dict(zip(group_cols, key))
        row["n_runs"] = len(idx)
        row["n_success"] = int(success.loc[idx].sum())
        stats = {}
        for name, mask in eligible.items():
            stats[name] = _mean_std(numeric[name].loc[idx].where(mask.loc[idx]))
        row["n_allocation_cost"] = stats["allocation_cost"][0]
        row["n_allocation_runtime"] = stats["allocation_runtime_seconds"][0]
        row["n_reward_runtime"] = stats["reward_runtime_seconds"][0]
        row["n_total_runtime"] = stats["total_runtime_seconds"][0]
        row["n_gini"] = stats["gini_index"][0]
        for name in (
            "allocation_cost",
            "allocation_runtime_seconds",
            "reward_runtime_seconds",
            "total_runtime_seconds",
            "gini_index",
            "coalition_stability",
        ):
            row[f"{name}_mean"] = stats[name][1]
            row[f"{name}_std"] = stats[name][2]
        denominator_mask = success.loc[idx] & energy_present.loc[idx]
        denominator = int(denominator_mask.sum())
        numerator = int((denominator_mask & energy_true.loc[idx]).sum())
        row["energy_feasible_rate"] = (
            numerator / denominator if denominator > 0 else float("nan")
        )
        row["n_invalid_reward"] = int(invalid.loc[idx].sum())
        rows.append(row)

    summary = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    summary = summary.sort_values(
        ["total_entities", "num_robots", "num_tasks", "algorithm"]
    ).reset_index(drop=True)
    return summary


def write_summary_table(
    summary: pd.DataFrame,
    experiment_id: str,
    output_dir: Optional[PathLike] = None,
) -> Path:
    """Write ``summary_<experiment_id>.csv`` atomically (CSV only, no LaTeX).

    The summary is derived data, so an existing file is replaced. Missing
    values (NaN) are written as empty cells.
    """
    validate_experiment_id(experiment_id)
    if list(summary.columns) != list(SUMMARY_COLUMNS):
        raise SchemaValidationError(
            f"summary must have exactly the columns {list(SUMMARY_COLUMNS)}."
        )
    path = _resolve_dir(output_dir, DEFAULT_SUMMARY_DIR) / f"summary_{experiment_id}.csv"
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(SUMMARY_COLUMNS)
    for values in summary.itertuples(index=False, name=None):
        writer.writerow([_serialize_cell(v) for v in values])
    _atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))
    return path


def summarize_experiment(
    experiment_id: str,
    results_dir: Optional[PathLike] = None,
    output_dir: Optional[PathLike] = None,
) -> Path:
    """Load the raw results, aggregate one experiment and write its summary CSV.

    Raises:
        ValueError: no records exist for ``experiment_id`` (no empty summary
            is written silently).
    """
    validate_experiment_id(experiment_id)
    source = _resolve_dir(results_dir, DEFAULT_RESULTS_DIR) / BENCHMARK_RESULTS_FILENAME
    summary = aggregate_benchmark_results(load_benchmark_results(source), experiment_id)
    if summary.empty:
        raise ValueError(f"No benchmark records found for experiment {experiment_id!r}.")
    return write_summary_table(summary, experiment_id, output_dir)

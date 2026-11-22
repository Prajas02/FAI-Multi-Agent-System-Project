"""Deterministic seed derivation for the benchmarking harness.

Implements Section 6.1 of docs/BENCHMARK_HARNESS_DESIGN.md and the ``seeds``
section of configs/experiments_config.yaml.

A seed is the first 4 bytes of ``SHA-256(message)`` read as an unsigned
big-endian integer (range 0 .. 4294967295). The message templates
(``seeds.messages.scenario``, ``.algorithm`` and ``.reward``) are read from the
YAML configuration and are never hard-coded here. Their placeholders are
``base_seed``, ``config_id``, ``repetition_index``, ``scenario_seed``,
``algorithm_id`` and ``reward_stage_label``. Repetition indices are zero-based.
Python's built-in ``hash()`` is never used because it is randomized per
process.

Owner: Yash Yeole (Benchmarking & Analytics).
"""

import hashlib
import os
import string
from pathlib import Path
from typing import Any, Dict, FrozenSet, NamedTuple, Optional, Tuple, Union

import yaml

PathLike = Union[str, "os.PathLike[str]"]

SEED_MIN = 0
SEED_MAX = 4294967295  # 2**32 - 1

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "experiments_config.yaml"

# Placeholder names each template must use (the template text itself is read
# from the configuration).
_EXPECTED_PLACEHOLDERS: Dict[str, FrozenSet[str]] = {
    "scenario": frozenset({"base_seed", "config_id", "repetition_index"}),
    "algorithm": frozenset({"scenario_seed", "algorithm_id"}),
    "reward": frozenset({"scenario_seed", "reward_stage_label"}),
}

# Derivation parameters of the ``seeds`` section that this module implements.
# The configuration must agree with them, otherwise the code and the documented
# scheme would silently diverge.
_EXPECTED_PARAMETERS: Dict[str, Any] = {
    "derivation_method": "sha256",
    "hash_function": "sha256",
    "seed_bytes": 4,
    "byte_order": "big",
    "signed": False,
    "seed_min": SEED_MIN,
    "seed_max": SEED_MAX,
    "message_encoding": "utf-8",
}

_FORMATTER = string.Formatter()


class SeedConfigError(ValueError):
    """The seed configuration is malformed, incomplete or inconsistent."""


class SeedTemplates(NamedTuple):
    """The three message templates read from the configuration."""
    scenario: str
    algorithm: str
    reward: str


# ---------------------------------------------------------------------------
# Core derivation
# ---------------------------------------------------------------------------

def derive_seed(message: str) -> int:
    """Derive a seed from ``message``.

    ``int.from_bytes(sha256(message.encode("utf-8")).digest()[:4], "big")``.

    Args:
        message: The message string.

    Returns:
        An integer in ``0 .. 4294967295``.

    Raises:
        TypeError: if ``message`` is not a ``str``.
    """
    if not isinstance(message, str):
        raise TypeError(f"message must be a str, got {type(message).__name__}.")
    digest = hashlib.sha256(message.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


# ---------------------------------------------------------------------------
# Configuration loading (cached, validated)
# ---------------------------------------------------------------------------

def resolve_config_path(config_path: Optional[PathLike] = None) -> Path:
    """Return the configuration file path.

    ``None`` gives ``<repository root>/configs/experiments_config.yaml``. A
    relative path is resolved against the **repository root**, not the current
    working directory. Absolute paths are used as given.
    """
    if config_path is None:
        return DEFAULT_CONFIG_PATH
    path = Path(config_path)
    return path if path.is_absolute() else _REPO_ROOT / path


def _validate_template(kind: str, template: Any, source: Path) -> str:
    if not isinstance(template, str):
        raise SeedConfigError(
            f"{source}: seeds.messages.{kind} must be a string, got {template!r}."
        )
    try:
        parsed = list(_FORMATTER.parse(template))
    except ValueError as exc:
        raise SeedConfigError(
            f"{source}: seeds.messages.{kind} is not a valid template ({exc}): {template!r}."
        ) from None

    found = set()
    for _literal, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if not field_name.isidentifier():
            raise SeedConfigError(
                f"{source}: seeds.messages.{kind} has an unsupported placeholder "
                f"{{{field_name}}} (only plain names are allowed): {template!r}."
            )
        if format_spec or conversion:
            raise SeedConfigError(
                f"{source}: seeds.messages.{kind} placeholder {{{field_name}}} must not use a "
                f"format spec or conversion (integers are written in plain decimal): {template!r}."
            )
        found.add(field_name)

    expected = _EXPECTED_PLACEHOLDERS[kind]
    if found != expected:
        raise SeedConfigError(
            f"{source}: seeds.messages.{kind} must use exactly the placeholders "
            f"{sorted(expected)}; missing={sorted(expected - found)}, "
            f"unexpected={sorted(found - expected)}: {template!r}."
        )
    return template


def _load_templates(path: Path) -> SeedTemplates:
    """Read and validate the seed templates from ``path`` (no caching)."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Seed configuration file not found: {path}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise SeedConfigError(f"Cannot read seed configuration {path}: {exc}") from None

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SeedConfigError(f"Malformed YAML in {path}: {exc}") from None

    if not isinstance(document, dict):
        raise SeedConfigError(f"{path}: the top level must be a mapping.")
    seeds = document.get("seeds")
    if not isinstance(seeds, dict):
        raise SeedConfigError(f"{path}: missing or invalid 'seeds' section.")

    for key, expected in _EXPECTED_PARAMETERS.items():
        if key not in seeds:
            raise SeedConfigError(f"{path}: missing seeds.{key}.")
        value = seeds[key]
        if value != expected or type(value) is not type(expected):
            raise SeedConfigError(
                f"{path}: seeds.{key} is {value!r}, but this implementation requires {expected!r}."
            )

    messages = seeds.get("messages")
    if not isinstance(messages, dict):
        raise SeedConfigError(f"{path}: missing or invalid 'seeds.messages' section.")
    values = {}
    for kind in _EXPECTED_PLACEHOLDERS:
        if kind not in messages:
            raise SeedConfigError(f"{path}: missing seeds.messages.{kind}.")
        values[kind] = _validate_template(kind, messages[kind], path)
    return SeedTemplates(**values)


_CACHE: Dict[str, Tuple[Tuple[int, int], SeedTemplates]] = {}


def get_seed_templates(config_path: Optional[PathLike] = None) -> SeedTemplates:
    """Return the validated templates, reading the file only when needed.

    The result is cached per file and reused until the file's modification time
    or size changes, so seed calculations do not re-read the YAML each time.

    Raises:
        FileNotFoundError: the file does not exist.
        SeedConfigError: unreadable or malformed YAML, missing keys, parameters
            that disagree with the implementation, or invalid templates.
    """
    path = resolve_config_path(config_path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        raise FileNotFoundError(f"Seed configuration file not found: {path}") from None
    except OSError as exc:
        raise SeedConfigError(f"Cannot access seed configuration {path}: {exc}") from None

    signature = (stat.st_mtime_ns, stat.st_size)
    key = str(path)
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    templates = _load_templates(path)
    _CACHE[key] = (signature, templates)
    return templates


def clear_seed_template_cache() -> None:
    """Forget all cached templates (the next call re-reads the file)."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Argument checks
# ---------------------------------------------------------------------------

def _check_seed_value(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {value!r}.")
    if not SEED_MIN <= value <= SEED_MAX:
        raise ValueError(f"{name} must be in {SEED_MIN}..{SEED_MAX}, got {value}.")
    return value


def _check_identifier(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {value!r}.")
    if value == "":
        raise ValueError(f"{name} must not be empty.")
    return value


def _check_repetition_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"repetition_index must be an int, got {value!r}.")
    if value < 0:
        raise ValueError(f"repetition_index is zero-based and must be >= 0, got {value}.")
    return value


# ---------------------------------------------------------------------------
# Public seed functions
# ---------------------------------------------------------------------------

def scenario_seed(
    base_seed: int,
    config_id: str,
    repetition_index: int,
    config_path: Optional[PathLike] = None,
) -> int:
    """Seed of one scenario: depends on base seed, configuration id and repetition.

    Shared by all algorithms evaluated on that scenario. ``repetition_index``
    is zero-based. ``config_path`` (optional) selects another YAML file, for
    example in tests; the default is the repository's experiments config.
    """
    message = get_seed_templates(config_path).scenario.format(
        base_seed=_check_seed_value("base_seed", base_seed),
        config_id=_check_identifier("config_id", config_id),
        repetition_index=_check_repetition_index(repetition_index),
    )
    return derive_seed(message)


def algorithm_seed(
    scenario_seed: int,
    algorithm_id: str,
    config_path: Optional[PathLike] = None,
) -> int:
    """Seed of one algorithm on one scenario: scenario seed plus algorithm id.

    The algorithm id is used exactly as given (case-sensitive, no
    normalization).
    """
    message = get_seed_templates(config_path).algorithm.format(
        scenario_seed=_check_seed_value("scenario_seed", scenario_seed),
        algorithm_id=_check_identifier("algorithm_id", algorithm_id),
    )
    return derive_seed(message)


def reward_seed(
    scenario_seed: int,
    reward_stage_label: str,
    config_path: Optional[PathLike] = None,
) -> int:
    """Seed of the reward stage: scenario seed plus a fixed reward-stage label.

    It does not depend on the algorithm, so it is the same for every algorithm
    evaluated on the same scenario.
    """
    message = get_seed_templates(config_path).reward.format(
        scenario_seed=_check_seed_value("scenario_seed", scenario_seed),
        reward_stage_label=_check_identifier("reward_stage_label", reward_stage_label),
    )
    return derive_seed(message)

"""Analytics-owned adapters from real allocation solvers to the benchmark contract.

The runner calls adapters as ``adapter(scenario, seed, time_limit_seconds)`` and
expects ``{"assignments": {robot_id: [task_id, ...]}}``, plus ``"clusters"`` (a list of
``{"agents": [...], "tasks": [...]}``) for allocators that partition the problem.  The allocators share
``solve(scenario) -> AllocationResult``; only the stochastic solvers consume the
algorithm seed.  The harness backend measures wall time and enforces the timeout
in subprocess mode, so solver-reported runtime is intentionally ignored here.
"""

from pathlib import Path
from typing import Any, Dict

from src.allocation.ga_clustering import GAClusteringAllocator, GAConfig
from src.allocation.ga_paper import PaperGAAllocator, PaperGAConfig
from src.allocation.greedy import GreedyAllocator
from src.allocation.milp import MILPAllocator, MILPConfig
from src.allocation.svca import SVCAAllocator, SVCAConfig
from src.allocation.svca_paper import PaperSVCAAllocator, PaperSVCAConfig
from src.core.types import AllocationResult
from src.simulation.generator import ScenarioData


def _assignments(result: AllocationResult) -> Dict[str, Any]:
    """Expose the result's plan in the harness's stable adapter payload shape."""
    if not isinstance(result, AllocationResult):
        raise TypeError(f"allocator returned {type(result).__name__}, expected AllocationResult")
    payload: Dict[str, Any] = {"assignments": result.assignments}
    clusters = result.metadata.get("clusters")
    if clusters:   # SVCA / GA coalitions: lets the reward stage solve one game per cluster
        payload["clusters"] = [{"agents": [int(a) for a in c["agents"]], "tasks": [int(t) for t in c["tasks"]]}
                               for c in clusters]
    return payload


def greedy_adapter(scenario: ScenarioData, seed: int, time_limit_seconds: float) -> Dict[str, Any]:
    """Run the deterministic Greedy baseline; seed is unused by this solver."""
    del seed, time_limit_seconds  # harness owns seeding policy and time measurement
    return _assignments(GreedyAllocator().solve(scenario))


def ga_adapter(scenario: ScenarioData, seed: int, time_limit_seconds: float) -> Dict[str, Any]:
    """Run GA with its config seed derived for this algorithm/scenario pair."""
    del time_limit_seconds
    return _assignments(GAClusteringAllocator(GAConfig(seed=seed)).solve(scenario))


def svca_adapter(scenario: ScenarioData, seed: int, time_limit_seconds: float) -> Dict[str, Any]:
    """Run SVCA with its permutation-sampling seed derived by the harness."""
    del time_limit_seconds
    return _assignments(SVCAAllocator(SVCAConfig(seed=seed)).solve(scenario))


def ga_paper_adapter(scenario: ScenarioData, seed: int, time_limit_seconds: float) -> Dict[str, Any]:
    """Run paper GA with its existing defaults and the harness-derived seed."""
    del time_limit_seconds
    return _assignments(PaperGAAllocator(PaperGAConfig(seed=seed)).solve(scenario))


def svca_paper_adapter(scenario: ScenarioData, seed: int, time_limit_seconds: float) -> Dict[str, Any]:
    """Run paper SVCA using the supplied YAML settings, overriding only its seed."""
    del time_limit_seconds
    config_path = Path(__file__).resolve().parents[2] / "configs" / "svca_paper.yaml"
    config = PaperSVCAConfig.from_yaml(str(config_path), seed=seed)
    return _assignments(PaperSVCAAllocator(config).solve(scenario))


def milp_adapter(scenario: ScenarioData, seed: int, time_limit_seconds: float) -> Dict[str, Any]:
    """Run the exact MILP baseline (deterministic; seed unused).

    The solver's own limit is set to 80% of the harness limit so HiGHS can return its
    incumbent before the backend's hard timeout kills the worker.
    """
    del seed
    limit = max(1.0, 0.8 * float(time_limit_seconds))
    return _assignments(MILPAllocator(MILPConfig(time_limit=limit)).solve(scenario))

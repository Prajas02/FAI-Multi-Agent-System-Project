"""Create a saved visualization bundle from one existing scenario and real allocator outputs.

This script runs an allocator only while exporting the bundle. Playback via
``play_simulation.py`` reads the saved bundle and never reruns allocation.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytics import allocation_adapters
from src.analytics.evaluation import evaluate_allocation
from src.analytics import seeding
from src.simulation.generator import ScenarioData
from src.visualization.animation import write_bundle

import yaml

# Score bundles with the benchmark's configured energy penalty so saved metrics match the CSVs.
_PENALTY = yaml.safe_load((REPO_ROOT / "configs" / "experiments_config.yaml").read_text(encoding="utf-8"))[
    "evaluation"]["energy_penalty"]

ADAPTERS = {
    "Greedy": allocation_adapters.greedy_adapter,
    "GA": allocation_adapters.ga_adapter,
    "SVCA": allocation_adapters.svca_adapter,
    "GA_Paper": allocation_adapters.ga_paper_adapter,
    "SVCA_Paper": allocation_adapters.svca_paper_adapter,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Save real allocation results for playback.")
    parser.add_argument("--scenario", required=True, help="saved ScenarioData JSON")
    parser.add_argument("--output", required=True, help="output playback JSON bundle")
    parser.add_argument("--algorithms", default="Greedy", help="comma-separated names, or all")
    parser.add_argument("--seed", type=int, default=42, help="base seed for reproducible per-algorithm exports")
    args = parser.parse_args(argv)
    scenario = ScenarioData.load_from_file(args.scenario)
    names = list(ADAPTERS) if args.algorithms == "all" else args.algorithms.split(",")
    unknown = sorted(set(names) - set(ADAPTERS))
    if unknown:
        parser.error(f"unknown algorithms: {', '.join(unknown)}")
    allocations, metrics = {}, {}
    for name in names:
        payload = ADAPTERS[name](scenario, seeding.algorithm_seed(args.seed, name), 300.0)
        assignments = payload["assignments"]
        allocations[name] = assignments
        evaluation = evaluate_allocation(scenario, assignments, _PENALTY)
        metrics[name] = {"allocation_cost": evaluation.allocation_cost, "energy_feasible": evaluation.energy_feasible}
    write_bundle(Path(args.output), scenario, allocations, metrics)
    print(f"saved {len(allocations)} exact allocation(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

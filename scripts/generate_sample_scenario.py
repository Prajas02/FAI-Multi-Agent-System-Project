"""Generate and save reproducible sample scenarios for the team.

Writes data/scenarios/toy_scenario.json (4 robots, 10 tasks) and
data/scenarios/clustered_scenario.json (6 robots, 14 clustered tasks).
Both come from ONE generator seeded with --seed, toy first, so the clustered
scenario is reproduced only by running this script (not from its stored seed alone).
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.simulation.generator import ScenarioGenerator


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regenerate the two sample scenarios in data/scenarios/.")
    parser.add_argument("--seed", type=int, default=42, help="generator seed (default 42)")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data" / "scenarios"),
                        help="folder for the JSON files (default data/scenarios)")
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generator = ScenarioGenerator(map_width=500.0, map_height=500.0, seed=args.seed)

    # 1. Default Toy Scenario (4 Agents, 10 Tasks)
    toy_scenario = generator.generate_scenario(num_agents=4, num_tasks=10)
    toy_path = out / "toy_scenario.json"
    toy_scenario.save_to_file(str(toy_path))
    print(f"[+] Saved toy scenario: {toy_path} (R=4, T=10)")

    # 2. Clustered Benchmark Scenario (6 Agents, 14 Tasks)
    clustered_scenario = generator.generate_scenario(
        num_agents=6,
        num_tasks=14,
        clustered_tasks=True,
        num_clusters=3,
    )
    clustered_path = out / "clustered_scenario.json"
    clustered_scenario.save_to_file(str(clustered_path))
    print(f"[+] Saved clustered scenario: {clustered_path} (R=6, T=14)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

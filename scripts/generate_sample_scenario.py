"""Generate and save reproducible sample scenarios for the team."""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.generator import ScenarioGenerator


def main():
    os.makedirs(os.path.join("data", "scenarios"), exist_ok=True)
    generator = ScenarioGenerator(map_width=500.0, map_height=500.0, seed=42)

    # 1. Default Toy Scenario (4 Agents, 10 Tasks)
    toy_scenario = generator.generate_scenario(num_agents=4, num_tasks=10)
    toy_path = os.path.join("data", "scenarios", "toy_scenario.json")
    toy_scenario.save_to_file(toy_path)
    print(f"[+] Saved toy scenario: {toy_path} (R=4, T=10)")

    # 2. Clustered Benchmark Scenario (6 Agents, 14 Tasks)
    clustered_scenario = generator.generate_scenario(
        num_agents=6,
        num_tasks=14,
        clustered_tasks=True,
        num_clusters=3,
    )
    clustered_path = os.path.join("data", "scenarios", "clustered_scenario.json")
    clustered_scenario.save_to_file(clustered_path)
    print(f"[+] Saved clustered scenario: {clustered_path} (R=6, T=14)")


if __name__ == "__main__":
    main()

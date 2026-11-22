"""Run Greedy, Greedy+LS, SVCA and GA on one scenario and cross-check against Environment.

Examples
    python scripts/run_allocation_demo.py                           # R=6, T=14 default config
    python scripts/run_allocation_demo.py --agents 20 --tasks 60 --seed 3
    python scripts/run_allocation_demo.py --agents 10 --tasks 30 --clustered
    python scripts/run_allocation_demo.py --scenario data/scenarios/toy_scenario.json
    python scripts/run_allocation_demo.py --agents 20 --tasks 60 --save outputs/allocation_demo.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.allocation import (  # noqa: E402
    AllocationEvaluator,
    GAClusteringAllocator,
    GAConfig,
    GreedyAllocator,
    SVCAAllocator,
    SVCAConfig,
)
from src.simulation.environment import Environment  # noqa: E402
from src.simulation.generator import ScenarioData, ScenarioGenerator  # noqa: E402

CONFIG_PATH = os.path.join("configs", "default_config.yaml")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", help="load a saved scenario JSON instead of generating one")
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--tasks", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clustered", action="store_true", help="hotspot-clustered task layout")
    ap.add_argument("--target-cluster-size", type=int, default=20, help="entities per sub-problem (sets K)")
    ap.add_argument("--q", type=int, default=None, help="Shapley permutations (default: from config)")
    ap.add_argument("--skip-ga", action="store_true")
    ap.add_argument("--save", help="write results (incl. clusters and Shapley values) to this JSON file")
    args = ap.parse_args()

    if args.scenario:
        scenario = ScenarioData.load_from_file(args.scenario)
    else:
        scenario = ScenarioGenerator(seed=args.seed).generate_scenario(
            args.agents, args.tasks, clustered_tasks=args.clustered
        )
    R, T = len(scenario.agents), len(scenario.tasks)

    overrides = {"target_cluster_size": args.target_cluster_size}
    if args.q is not None:
        overrides["num_permutations_q"] = args.q
    svca_cfg = (
        SVCAConfig.from_yaml(CONFIG_PATH, **overrides) if os.path.exists(CONFIG_PATH) else SVCAConfig(**overrides)
    )

    solvers = [
        GreedyAllocator(),
        GreedyAllocator(use_local_search=True),
        SVCAAllocator(svca_cfg),
    ]
    if not args.skip_ga:
        solvers.append(GAClusteringAllocator(GAConfig(target_cluster_size=args.target_cluster_size, seed=svca_cfg.seed)))

    ev = AllocationEvaluator(scenario)
    env = Environment()
    env.load_scenario(scenario)

    print(f"Scenario: R={R} agents, T={T} tasks (N={R + T})\n")
    print(f"{'solver':<11}{'cost A(U)':>12}{'runtime[s]':>12}{'K':>4}{'valid':>7}{'env check':>11}")
    results = []
    for solver in solvers:
        res = solver.solve(scenario)
        env_cost = env.execute_allocation(res.assignments)[0].total_cost
        ok = "OK" if abs(env_cost - res.total_cost) < 0.011 else f"MISMATCH ({env_cost})"
        k = res.metadata.get("num_clusters", "-")
        print(f"{res.solver_name:<11}{res.total_cost:>12.2f}{res.runtime_seconds:>12.3f}{k!s:>4}"
              f"{str(res.metadata['valid']):>7}{ok:>11}")
        results.append(res)

    svca = next(r for r in results if r.solver_name == "SVCA")
    print("\nSVCA clusters:")
    for i, c in enumerate(svca.metadata["clusters"]):
        print(f"  cluster {i}: agents={c['agents']}  tasks={c['tasks']}")
    print(f"  time split: shapley={svca.metadata['runtime_shapley']:.3f}s  "
          f"clustering={svca.metadata['runtime_clustering']:.3f}s  subsolve={svca.metadata['runtime_subsolve']:.3f}s")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        payload = [
            {
                "solver": r.solver_name,
                "total_cost": r.total_cost,
                "runtime_seconds": r.runtime_seconds,
                "assignments": {str(a): ts for a, ts in r.assignments.items()},
                "metadata": r.metadata,
            }
            for r in results
        ]
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved {args.save}")


if __name__ == "__main__":
    main()

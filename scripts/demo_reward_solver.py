"""Demo / smoke benchmark for the preference-aware reward solver (Goral Mashru).

Usage:
    python scripts/demo_reward_solver.py                 # small scenario, validates vs exact
    python scripts/demo_reward_solver.py --agents 80 --tasks 80 --samples 1000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import yaml

from src.game_theory import PreferenceAwareShapleySolver, equal_split
from src.simulation.generator import ScenarioGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=7)
    ap.add_argument("--tasks", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--value", choices=["facility", "routing"], default="facility")
    ap.add_argument("--config", default=os.path.join("configs", "default_config.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.alpha is not None:
        cfg["mcmc_reward"]["preference_sensitivity_alpha"] = args.alpha
    if args.samples is not None:
        cfg["mcmc_reward"]["num_samples_Tw"] = args.samples

    scen = ScenarioGenerator(seed=cfg["seed"]).generate_scenario(args.agents, args.tasks)
    solver = PreferenceAwareShapleySolver.from_config(scen, cfg, value_type=args.value)
    print(f"N={args.agents} agents, T={args.tasks} tasks, alpha={solver.alpha}, game={args.value}")

    results = solver.compare(compute_stability=True)
    exact = results.get("exact")
    print(f"\n{'method':<12}{'runtime(s)':>11}{'Gini':>8}{'sum phi':>11}{'V(N)':>11}"
          f"{'max|err| vs exact':>20}")
    for key, d in results.items():
        err = ""
        if exact is not None and key != "exact":
            err = f"{max(abs(d.payoffs[k] - exact.payoffs[k]) for k in exact.payoffs):.4f}"
        print(f"{d.solver_name:<12}{d.runtime_seconds:>11.3f}{d.gini_index:>8.3f}"
              f"{sum(d.payoffs.values()):>11.2f}{d.total_coalition_value:>11.2f}{err:>20}")
    eq = equal_split(solver.value)
    print(f"{eq.solver_name:<12}{'-':>11}{eq.gini_index:>8.3f}")

    g = results["glauber"]
    print(f"\nGlauber: acceptance={g.metadata['acceptance_rate']:.2f}, "
          f"cache hit rate={g.metadata['cache_hit_rate']:.2f}, "
          f"value evals={g.metadata['value_evaluations']}")
    if "stability" in g.metadata:
        s = g.metadata["stability"]
        print(f"Stability: max excess={s['max_excess']:.3f}, "
              f"score={s['stability_score']:.3f} over {int(s['num_checked'])} coalitions")


if __name__ == "__main__":
    main()

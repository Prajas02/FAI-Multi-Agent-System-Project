"""Build the interactive end-to-end pipeline viewer (one self-contained HTML file).

Examples
    python scripts/visualize_pipeline.py --scenario data/scenarios/clustered_scenario.json --open
    python scripts/visualize_pipeline.py --agents 12 --tasks 40 --clustered --open
    python scripts/visualize_pipeline.py --scenario data/scenarios/toy_scenario.json --milp

The page shows, for the chosen scenario: the robots and tasks, how SVCA (and the other
clustering methods) partition them, an animated playback of every robot's route with its
battery draining, the preference graph and the Glauber MCMC reward split for a sweep of
alpha, and a cost comparison of every allocation algorithm.
"""
import argparse
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.simulation.generator import ScenarioData, ScenarioGenerator
from src.visualization.pipeline_view import build_pipeline_data, render_html


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create the interactive pipeline viewer HTML page.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--scenario", help="saved ScenarioData JSON (default: data/scenarios/clustered_scenario.json)")
    src.add_argument("--agents", type=int, help="generate a new scenario with this many robots")
    parser.add_argument("--tasks", type=int, default=30, help="tasks for a generated scenario (default 30)")
    parser.add_argument("--clustered", action="store_true", help="generate spatially clustered tasks")
    parser.add_argument("--seed", type=int, default=42, help="seed for generation and for the solvers")
    parser.add_argument("--milp", action="store_true", help="also run the exact MILP (slow beyond ~10 tasks)")
    parser.add_argument("--milp-time-limit", type=float, default=20.0, help="MILP time limit in seconds")
    parser.add_argument("--output", help="output HTML path (default: outputs/visualization/pipeline_<name>.html)")
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = parser.parse_args(argv)

    if args.agents:
        scenario = ScenarioGenerator(seed=args.seed).generate_scenario(
            args.agents, args.tasks, clustered_tasks=args.clustered)
        name = f"generated_{args.agents}r_{args.tasks}t{'_clustered' if args.clustered else ''}_seed{args.seed}"
    else:
        path = Path(args.scenario or REPO_ROOT / "data" / "scenarios" / "clustered_scenario.json")
        scenario = ScenarioData.load_from_file(str(path))
        name = path.stem

    print(f"Running all allocators and the reward stage on '{name}' "
          f"({len(scenario.agents)} robots, {len(scenario.tasks)} tasks)...")
    data = build_pipeline_data(scenario, name, seed=args.seed, include_milp=args.milp,
                               milp_time_limit=args.milp_time_limit)
    for algo in data["algorithms"]:
        print(f"  {algo['name']:<13} cost {algo['cost']:>11,.1f}   runtime {algo['runtime']:.3f}s")

    output = Path(args.output) if args.output else REPO_ROOT / "outputs" / "visualization" / f"pipeline_{name}.html"
    render_html(data, output)
    print(f"Saved {output}")
    if args.open:
        webbrowser.open(output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

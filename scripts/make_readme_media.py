"""Generate the figures and animated GIFs used in README.md (written to docs/media/).

    python scripts/make_readme_media.py            # all media
    python scripts/make_readme_media.py --fast     # fewer animation frames

Every frame is drawn from real solver output: the scenarios are generated with the
project's ScenarioGenerator, plans come from the allocators, motion follows the exact
per-leg timeline of AllocationEvaluator (battery-dependent speed), rewards come from the
preference-aware Shapley solver and team changes from the team-choice dynamics.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from src.allocation.evaluator import AllocationEvaluator  # noqa: E402
from src.allocation.greedy import GreedyAllocator  # noqa: E402
from src.allocation.svca import SVCAAllocator  # noqa: E402
from src.allocation.svca_paper import PaperSVCAAllocator, PaperSVCAConfig  # noqa: E402
from src.analytics.team_choice_eval import evaluate_team_choice  # noqa: E402
from src.game_theory.clustered import solve_by_clusters  # noqa: E402
from src.game_theory.mcmc import glauber_preference_shapley  # noqa: E402
from src.game_theory.solver import PreferenceAwareShapleySolver  # noqa: E402
from src.simulation.generator import ScenarioGenerator  # noqa: E402
from src.visualization.pipeline_view import build_timeline  # noqa: E402

OUT = REPO_ROOT / "docs" / "media"
# Categorical slots of the validated dataviz reference palette (fixed order, never cycled past 8).
TEAM = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID, TASK = "#1f2328", "#6b7280", "#e8eaed", "#9aa1ab"


def style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#9aa1ab", "axes.titlelocation": "left", "axes.titlesize": 11,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def team_color(i):
    """Fixed categorical order; a 9th+ team folds into one neutral 'other' colour (never recycles a hue)."""
    if i is None or i < 0:
        return TASK
    return TEAM[i] if i < len(TEAM) else "#4b5563"


def hull(points):
    pts = sorted(set(map(tuple, points)))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def setup_map(ax, scenario, title):
    ax.set_xlim(-10, scenario.map_width + 10)
    ax.set_ylim(-10, scenario.map_height + 10)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#c9ced6")
    ax.set_title(title)


def draw_teams(ax, scenario, clusters, alpha=0.13):
    agents = {a.agent_id: a for a in scenario.agents}
    tasks = {t.task_id: t for t in scenario.tasks}
    for i, c in enumerate(clusters):
        pts = [(agents[a].position.x, agents[a].position.y) for a in c["agents"]] + \
              [(tasks[t].location.x, tasks[t].location.y) for t in c["tasks"]]
        if not pts:
            continue
        h = hull(pts)
        col = team_color(i)
        if len(h) >= 3:
            ax.add_patch(Polygon(h, closed=True, facecolor=col, alpha=alpha, edgecolor=col, linewidth=10,
                                 joinstyle="round"))
        else:
            xs, ys = zip(*h)
            ax.plot(xs, ys, color=col, alpha=alpha * 2, linewidth=14, solid_capstyle="round")


def draw_entities(ax, scenario, agent_team=None, task_team=None, positions=None, done=None, labels=True):
    max_p = max(t.priority for t in scenario.tasks)
    max_v = max(a.velocity for a in scenario.agents)
    for t in scenario.tasks:
        col = team_color((task_team or {}).get(t.task_id))
        filled = done is None or t.task_id in done
        ax.scatter(t.location.x, t.location.y, marker="D", s=25 + 45 * t.priority / max_p,
                   facecolor=col if filled else "white", edgecolor=col, linewidth=1.5, zorder=3)
    for a in scenario.agents:
        x, y = (positions or {}).get(a.agent_id, (a.position.x, a.position.y))
        col = team_color((agent_team or {}).get(a.agent_id, 0))
        ax.scatter(x, y, s=110 + 120 * a.velocity / max_v, color=col, edgecolor="white", linewidth=1.5, zorder=5)
        if labels:
            ax.text(x, y, str(a.agent_id), color="white", fontsize=7, fontweight="bold", ha="center", va="center", zorder=6)


def team_maps(clusters):
    at = {a: i for i, c in enumerate(clusters) for a in c["agents"]}
    tt = {t: i for i, c in enumerate(clusters) for t in c["tasks"]}
    return at, tt


# ----------------------------------------------------------------------------- figures
def fig_environment(scenario, paper, ext):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    setup_map(axes[0], scenario, "Scenario: 10 robots, 40 clustered tasks")
    draw_entities(axes[0], scenario, agent_team={a.agent_id: 0 for a in scenario.agents},
                  task_team={t.task_id: -1 for t in scenario.tasks})
    axes[0].text(0.01, -0.06, "circle size = robot speed, diamond size = task priority", transform=axes[0].transAxes,
                 fontsize=8, color=MUTED)
    for ax, res, name in ((axes[1], paper, "SVCA (paper)"), (axes[2], ext, "SVCA-ext")):
        cl = res.metadata["clusters"]
        setup_map(ax, scenario, f"{name}: {len(cl)} coalitions")
        draw_teams(ax, scenario, cl)
        at, tt = team_maps(cl)
        draw_entities(ax, scenario, at, tt)
    fig.tight_layout()
    fig.savefig(OUT / "fig_environment_clusters.png", dpi=150)
    plt.close(fig)


def fig_shapley_ranking(paper):
    phi = paper.metadata["shapley"]
    items = sorted(phi.items(), key=lambda kv: kv[1], reverse=True)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    xs = np.arange(len(items))
    cols = ["#2a78d6" if k.startswith("R") else "#9aa1ab" for k, _ in items]
    ax.bar(xs, [v for _, v in items], color=cols, width=0.8)
    ax.set_xticks(xs, [k for k, _ in items], rotation=90, fontsize=7)
    ax.axhline(0, color="#9aa1ab", linewidth=0.8)
    ax.set_ylabel("Shapley value of J")
    ax.set_title("SVCA ranks every robot (blue) and task (gray) by its Shapley value before growing clusters")
    ax.grid(axis="y", color=GRID)
    fig.tight_layout()
    fig.savefig(OUT / "fig_svca_shapley_ranking.png", dpi=150)
    plt.close(fig)


def fig_rewards(scenario, result):
    base = PreferenceAwareShapleySolver.from_scenario(scenario, alpha=1.5, seed=42)
    ids = [a.agent_id for a in scenario.agents]
    rows = {}
    for label, alpha in (("Classical Shapley (alpha 0)", 0.0), ("Preference-aware (alpha 1.5)", 1.5),
                         ("Preference-aware (alpha 3)", 3.0)):
        d = solve_by_clusters(scenario, allocation=result, alpha=alpha, preference=base.pref, seed=42)
        tot = sum(d.payoffs.values()) or 1.0
        rows[label] = [100 * d.payoffs[i] / tot for i in ids]
    rows["Equal split"] = [100 / len(ids)] * len(ids)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    w = 0.8 / len(rows)
    for j, (label, vals) in enumerate(rows.items()):
        ax.bar(np.arange(len(ids)) + (j - (len(rows) - 1) / 2) * w, vals, w * 0.92, label=label,
               color=["#9aa1ab", "#2a78d6", "#1baf7a", "#eda100"][j])
    ax.set_xticks(range(len(ids)), [f"R{i}" for i in ids])
    ax.set_ylabel("share of the reward (%)")
    ax.set_title("Reward split per robot (SVCA-ext coalitions)")
    ax.grid(axis="y", color=GRID)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    mc = glauber_preference_shapley(base.value, base.pref, 1.5, 1000, 200, None, 42)
    running = np.cumsum(mc.marginals, axis=0) / np.arange(1, len(mc.marginals) + 1)[:, None]
    for j in range(running.shape[1]):
        ax2.plot(running[:, j], color=TEAM[j % len(TEAM)] if j < len(TEAM) else "#9aa1ab", linewidth=1.5)
    ax2.set_xlabel("Glauber MCMC sample")
    ax2.set_ylabel("running estimate of phi_i")
    ax2.set_title(f"Glauber MCMC converges (acceptance {mc.acceptance_rate:.0%})")
    ax2.grid(axis="y", color=GRID)
    fig.tight_layout()
    fig.savefig(OUT / "fig_reward_sharing.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- animations
def robot_position(legs, start, t):
    for L in legs:
        if t < L["ta"]:
            f = (t - L["t0"]) / max(L["ta"] - L["t0"], 1e-9)
            return L["x0"] + f * (L["x1"] - L["x0"]), L["y0"] + f * (L["y1"] - L["y0"])
        if t < L["tf"]:
            return L["x1"], L["y1"]
    return (legs[-1]["x1"], legs[-1]["y1"]) if legs else start


def gif_execution(scenario, plans, fname, frames):
    """Side-by-side execution of several plans on the same scenario, on one shared clock."""
    ev = AllocationEvaluator(scenario)
    data = []
    for name, res in plans:
        tl = build_timeline(scenario, res.assignments, ev)
        cl = res.metadata.get("clusters") or [{"agents": [a.agent_id for a in scenario.agents],
                                               "tasks": [t.task_id for t in scenario.tasks]}]
        data.append((name, res, tl, *team_maps(cl)))
    horizon = max(L["tf"] for _, _, tl, _, _ in data for legs in tl.values() for L in legs)
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 5.4))
    axes = np.atleast_1d(axes)
    starts = {a.agent_id: (a.position.x, a.position.y) for a in scenario.agents}

    def draw(k):
        t = horizon * k / (frames - 1)
        for ax, (name, res, tl, at, tt) in zip(axes, data):
            ax.clear()
            setup_map(ax, scenario, f"{name}   cost A(U) = {res.total_cost:,.0f}")
            for aid, legs in tl.items():
                if legs:
                    xs = [legs[0]["x0"]] + [L["x1"] for L in legs]
                    ys = [legs[0]["y0"]] + [L["y1"] for L in legs]
                    ax.plot(xs, ys, color=team_color(at.get(int(aid), 0)), alpha=0.25, linewidth=1.2, linestyle="--")
            done = {L["task"] for legs in tl.values() for L in legs if L["tf"] <= t}
            pos = {int(a): robot_position(legs, starts[int(a)], t) for a, legs in tl.items()}
            draw_entities(ax, scenario, at, tt, positions=pos, done=done, labels=False)
            ax.set_xlabel(f"t = {t:4.0f} s    tasks done {len(done)}/{len(scenario.tasks)}", fontsize=9, color=INK)
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=frames)
    anim.save(OUT / fname, writer=PillowWriter(fps=12), dpi=80)
    plt.close(fig)


def gif_team_choice(scenario, start_clusters, record, fname, hold=6):
    res = record["result"]
    teams = [{"agents": list(c["agents"]), "tasks": list(c["tasks"])} for c in res.initial_clusters]
    states = [[{"agents": list(t["agents"]), "tasks": t["tasks"]} for t in teams]]
    for m in res.moves:
        if m["from"] >= 0:
            teams[m["from"]]["agents"].remove(m["robot"])
        if m["to"] >= 0:
            teams[m["to"]]["agents"].append(m["robot"])
        states.append([{"agents": list(t["agents"]), "tasks": t["tasks"]} for t in teams])
    seq = [0] * hold + [i for i in range(1, len(states)) for _ in range(3)] + [len(states) - 1] * (hold * 2)
    agents = {a.agent_id: a for a in scenario.agents}
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    unstable = set(res.initial_unstable_robots)

    def draw(frame):
        k = seq[frame]
        ax.clear()
        cl = states[k]
        head = "Stage-1 teams (SVCA paper)" if k == 0 else f"move {k}/{len(res.moves)}"
        setup_map(ax, scenario, f"Robots choose their team ({head})")
        draw_teams(ax, scenario, [c for c in cl])
        at, tt = team_maps(cl)
        draw_entities(ax, scenario, at, tt)
        if k == 0:
            for r in unstable:
                a = agents[r]
                ax.scatter(a.position.x, a.position.y, s=420, facecolor="none", edgecolor="#e34948",
                           linewidth=1.6, linestyle="--", zorder=4)
        else:
            m = res.moves[k - 1]
            a = agents[m["robot"]]
            ax.scatter(a.position.x, a.position.y, s=520, facecolor="none", edgecolor=INK, linewidth=2, zorder=4)
        last = k == len(states) - 1
        msg = (f"{len(unstable)}/{len(scenario.agents)} robots want to leave (red rings)" if k == 0 else
               f"R{res.moves[k - 1]['robot']}: payoff {res.moves[k - 1]['payoff_before']:.0f} -> "
               f"{res.moves[k - 1]['payoff_after']:.0f}")
        if last and k > 0:
            msg = (f"Nash stable after {res.rounds} rounds. Cost {record['cost_change_pct']:+.0f}%, "
                   f"Gini {record['gini_before']:.2f} -> {record['gini_after']:.2f}")
        ax.set_xlabel(msg, fontsize=9, color=INK)
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=len(seq))
    anim.save(OUT / fname, writer=PillowWriter(fps=4), dpi=80)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate README figures and GIFs into docs/media/.")
    parser.add_argument("--fast", action="store_true", help="fewer animation frames")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    style()

    s10 = ScenarioGenerator(seed=42).generate_scenario(10, 40, clustered_tasks=True)
    paper = PaperSVCAAllocator(PaperSVCAConfig.from_yaml(str(REPO_ROOT / "configs" / "svca_paper.yaml"), seed=42)).solve(s10)
    ext = SVCAAllocator().solve(s10)
    greedy = GreedyAllocator().solve(s10)
    fig_environment(s10, paper, ext)
    fig_shapley_ranking(paper)
    fig_rewards(s10, ext)
    print("figures done")
    frames = 60 if args.fast else 110
    gif_execution(s10, [("Greedy (no clustering)", greedy), ("SVCA-ext (3 coalitions)", ext)],
                  "anim_execution_greedy_vs_svca.gif", frames)
    gif_execution(s10, [("SVCA (paper)", paper)], "anim_execution_svca_paper.gif", frames)
    print("execution gifs done")

    s20 = ScenarioGenerator(seed=5).generate_scenario(20, 30)
    p20 = PaperSVCAAllocator(PaperSVCAConfig.from_yaml(str(REPO_ROOT / "configs" / "svca_paper.yaml"), seed=42)).solve(s20)
    rec = evaluate_team_choice(s20, p20.metadata["clusters"], "preference", 1.5, seed=42, value_type="routing")
    gif_team_choice(s20, p20.metadata["clusters"], rec, "anim_team_choice.gif")
    print(f"team choice gif done: {rec['moves']} moves, cost {rec['cost_change_pct']:+.1f}%")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}  {p.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

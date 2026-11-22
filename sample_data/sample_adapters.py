"""SYNTHETIC stand-ins for the missing teammate algorithms, for trying out the runner/report pipeline only.
They are simple heuristics, NOT real MILP/GA/Greedy/SVCA, and their numbers are NOT research results."""
import random

def _empty(scenario):
    return {a.agent_id: [] for a in scenario.agents}

def milp_stand_in(scenario, seed, time_limit_seconds):       # nearest-robot-from-its-last-position
    pos = {a.agent_id: a.position for a in scenario.agents}; plan = _empty(scenario)
    for t in scenario.tasks:
        r = min(pos, key=lambda k: (pos[k].distance_to(t.location), k)); plan[r].append(t.task_id); pos[r] = t.location
    return {"assignments": plan}

def ga_stand_in(scenario, seed, time_limit_seconds):         # seeded random assignment
    rng = random.Random(seed); robots = [a.agent_id for a in scenario.agents]; plan = _empty(scenario)
    for t in scenario.tasks: plan[rng.choice(robots)].append(t.task_id)
    return {"assignments": plan}

def greedy_stand_in(scenario, seed, time_limit_seconds):     # round robin
    robots = [a.agent_id for a in scenario.agents]; plan = _empty(scenario)
    for i, t in enumerate(scenario.tasks): plan[robots[i % len(robots)]].append(t.task_id)
    return {"assignments": plan}

def svca_stand_in(scenario, seed, time_limit_seconds):       # closest robot to each task's location (static start positions)
    plan = _empty(scenario)
    for t in sorted(scenario.tasks, key=lambda t: -t.priority):
        r = min(scenario.agents, key=lambda a: (a.position.distance_to(t.location), a.agent_id)).agent_id
        plan[r].append(t.task_id)
    return {"assignments": plan}

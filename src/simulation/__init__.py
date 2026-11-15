"""Simulation and Environment Engine Module.

Owner: Prajas Kulkarni (prajas.kulkarni@iitgn.ac.in)
Responsibilities:
- 2D continuous space environment
- Scenario generation (heterogeneous agents and distributed tasks)
- Kinematics and battery/velocity decay physics
"""

from src.simulation.environment import Environment, SimulationTrace
from src.simulation.generator import ScenarioGenerator, ScenarioData
from src.simulation.kinematics import BatteryDecayModel, KinematicsModel

__all__ = [
    "Environment",
    "SimulationTrace",
    "ScenarioGenerator",
    "ScenarioData",
    "BatteryDecayModel",
    "KinematicsModel",
]

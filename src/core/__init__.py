"""Core package exporting fundamental classes and contracts."""

from src.core.types import Position, AllocationResult, RewardDistribution
from src.core.agent import Agent
from src.core.task import Task

__all__ = ["Position", "AllocationResult", "RewardDistribution", "Agent", "Task"]

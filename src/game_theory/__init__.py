"""Game Theory and Preference-Aware Reward Solver Module.

Owner: Goral Mashru (goral.mashru@iitgn.ac.in)
Responsibilities:
- Characteristic function formulation V(S)
- Pairwise preference graph G = (N, W) and coalition harmony
- Glauber Dynamics MCMC sampling and preference-aware Shapley values
- Exact Shapley value validation
"""

from src.game_theory.characteristic import (
    CallableValue,
    CharacteristicFunction,
    FacilityLocationValue,
    RoutingValue,
)
from src.game_theory.clustered import compare_global_vs_clustered, solve_by_clusters
from src.game_theory.exact import exact_preference_shapley, exact_shapley
from src.game_theory.fairness import core_stability, gini
from src.game_theory.mcmc import GlauberSampler, MCMCResult, glauber_preference_shapley
from src.game_theory.preference import PreferenceGraph
from src.game_theory.sampling import permutation_shapley, stratified_shapley
from src.game_theory.solver import PreferenceAwareShapleySolver, equal_split

__all__ = [
    "CallableValue", "CharacteristicFunction", "FacilityLocationValue", "RoutingValue",
    "PreferenceGraph", "GlauberSampler", "MCMCResult", "glauber_preference_shapley",
    "exact_shapley", "exact_preference_shapley", "stratified_shapley", "permutation_shapley",
    "gini", "core_stability", "PreferenceAwareShapleySolver", "equal_split",
    "solve_by_clusters", "compare_global_vs_clustered",
]

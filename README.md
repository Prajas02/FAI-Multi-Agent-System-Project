# Preference-Aware Cooperative Task Allocation and Fair Reward Sharing in Heterogeneous Multi-Agent Teams

**Course Project:** Multi-Agent Systems (Foundations of AI) — Semester 7, IIT Gandhinagar  
**Proposal Date:** September 2026

---

## 1. Project Overview

Real-world multi-agent coordination requires two closely linked capabilities:
1. **Operational Task Allocation (MRTA):** Efficiently assigning distributed tasks to heterogeneous agents with varying speeds, payloads, battery reserves, and travel costs.
2. **Fair Reward Sharing:** Distributing collective rewards in proportion to individual contributions and inter-agent cooperation preferences.

This repository implements a two-stage game-theoretic framework:
- **Stage 1 (Allocation):** Uses the **Shapley Value Clustering Algorithm (SVCA)** to partition large-scale task allocation problems into tractable sub-problems, compared against Greedy, Genetic Algorithm (GA), and MILP baselines.
- **Stage 2 (Reward Sharing):** Uses **Cooperation Preference-Aware Shapley Values** with a pairwise preference graph $G=(N, W)$, approximated via **Glauber Dynamics MCMC** over the permutation space $\Omega$.

---

## 2. Team Member Ownership & Deliverables

| Member | Email | Role | Module | Primary Deliverables |
| :--- | :--- | :--- | :--- | :--- |
| **Prajas Kulkarni** | `prajas.kulkarni@iitgn.ac.in` | Simulation & Environment Engine | `src/simulation/`, `src/core/` | 2D continuous simulation, scenario generator, kinematics, battery/velocity decay models. |
| **Dhairya Shah** | `dhairya.shah@iitgn.ac.in` | Allocation & SVCA Clustering | `src/allocation/` | SVCA problem partitioning, Greedy baseline, GA clustering, and MILP solver. |
| **Goral Mashru** | `goral.mashru@iitgn.ac.in` | Game Theory & Reward Solver | `src/game_theory/` | Characteristic functions, preference graph, Glauber Dynamics MCMC, and Shapley reward solver. |
| **Yash Yeole** | `yash.yeole@iitgn.ac.in` | Benchmarking & Analytics | `src/analytics/`, `src/visualization/` | Experiment test harness, metrics evaluation (Gini index, cost, runtime), and plotting. |

---

## 3. Project Directory Structure

```text
FAI-Multi-Agent-System-Project/
├── .gitignore                          # Clean repository ignores
├── README.md                           # Project guide & documentation
├── requirements.txt                    # Project dependencies
├── pyproject.toml                      # Package installation config
│
├── configs/                            # Parameter configurations
│   └── default_config.yaml             # Simulation, SVCA, and MCMC defaults
│
├── src/                                # Source package
│   ├── core/                           # Shared entity primitives & contracts
│   │   ├── types.py                    # Position, AllocationResult, RewardDistribution
│   │   ├── agent.py                    # Agent state: position, velocity, energy, penalty
│   │   └── task.py                     # Task state: location, duration, priority
│   │
│   ├── simulation/                     # [Prajas Kulkarni] 2D space, kinematics & generator
│   ├── allocation/                     # [Dhairya Shah] SVCA, Greedy, GA & MILP solvers
│   ├── game_theory/                    # [Goral Mashru] Preference graph, MCMC & Shapley
│   ├── analytics/                      # [Yash Yeole] Metrics (Cost, Gini, Runtime) & logger
│   └── visualization/                  # [Yash Yeole] Trajectory plots & scaling charts
│
├── tests/                              # Unit testing suite
│   └── test_core.py                    # Core data contracts validation
│
├── scripts/                            # Demo and benchmarking execution scripts
├── data/                               # Saved benchmark scenarios & result dumps
└── outputs/                            # Exported figures and latex comparison tables
```

---

## 4. Setup & Quick Start

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/prajas-kulkarni/FAI-Multi-Agent-System-Project.git
cd FAI-Multi-Agent-System-Project

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2. Run Tests
```bash
python -m unittest discover tests
```

---

## 4b. Reward Solver (`src/game_theory/`, Goral Mashru)

```python
from src.simulation.generator import ScenarioGenerator
from src.game_theory import PreferenceAwareShapleySolver

scen = ScenarioGenerator(seed=42).generate_scenario(num_agents=20, num_tasks=30)
solver = PreferenceAwareShapleySolver.from_scenario(scen, allocation=None, alpha=1.5)
dist = solver.solve("glauber", compute_stability=True)   # -> core.types.RewardDistribution
# methods: "glauber" | "exact" (n<=9) | "stratified" | "permutation"
```

| File | Contents |
| :--- | :--- |
| `characteristic.py` | `V(S)` games: `FacilityLocationValue` (fast), `RoutingValue` (greedy routing + energy), `CallableValue` (wrap any `J(S)`, e.g. SVCA cost) |
| `preference.py` | `PreferenceGraph`: `W`, `Gamma`, `Upsilon`, exact `mu`, generators from agents |
| `mcmc.py` | `GlauberSampler`: heat-bath chain over orderings (Eq. 13), estimator Eq. 14, batch-means SE |
| `sampling.py` | `stratified_shapley` (uniform/Neyman), `permutation_shapley` — alpha=0 baselines and reusable for SVCA's Eq. 12 |
| `exact.py` | brute-force classical (n<=20) and preference-aware (n<=9) Shapley for validation |
| `fairness.py` | `gini`, `core_stability` (max excess / violation rate) |
| `solver.py` | `PreferenceAwareShapleySolver`, `equal_split` baseline |

Demo / smoke benchmark: `python scripts/demo_reward_solver.py --agents 80 --tasks 80`.

**Modelling note:** the literal `Gamma(C) = exp(sum w)` makes `mu` collapse onto one ordering for
larger teams, so the default graph is tempered via `max_log_gamma` (set `null` for the literal form).

## 5. Branching & Git Workflow for Team Members

To prevent merge conflicts, each member works on their respective feature branch:
- `Prajas`: `feature/simulation-engine`
- `Dhairya`: `feature/allocation-svca`
- `Goral`: `feature/game-theory-mcmc`
- `Yash`: `feature/benchmarking-viz`

Always pull from `main` before starting new work and ensure `python -m unittest discover tests` passes before creating a Pull Request.

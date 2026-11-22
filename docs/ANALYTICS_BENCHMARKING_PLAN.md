# Analytics & Benchmarking Plan

**Project:** Preference-Aware Cooperative Task Allocation and Fair Reward Sharing in Heterogeneous Multi-Agent Teams
**Workstream:** Benchmarking & Analytics (Yash Yeole)
**Status:** Design document. No code is implemented yet and no results exist yet. Every number in this document is either taken from the project proposal or is a configuration value. None of them are experimental results.

---

## 1. Project Overview and Responsibilities

### 1.1 Project summary

The project builds a two-stage pipeline for heterogeneous multi-robot teams:

1. **Allocation (Stage 1):** assign $T$ spatially distributed tasks to $R$ heterogeneous robots by minimising the allocation cost $A(U)$. The proposed method is the Shapley Value Clustering Algorithm (SVCA). The baselines are a centralised MILP solver, GA-based clustering and Greedy Allocation.
2. **Reward sharing (Stage 2):** split the team reward using preference-aware Shapley values, estimated with Monte Carlo sampling and Glauber Dynamics MCMC.

The proposal (Section 4.3) evaluates the framework in a Python 2D simulator on a 500 m × 500 m map for $N \in [10, 80]$. It uses four metrics: total allocation cost $A(U)$, computational runtime, reward fairness (Gini index $G$) and coalition stability.

### 1.2 My responsibilities (Benchmarking & Analytics)

| Area | What I own |
| :--- | :--- |
| Experimental harness | Driving every algorithm over the same generated scenarios, timing each stage, catching failures and recording seeds. |
| Evaluation metrics | Computing $A(U)$ the same way for every algorithm, tracking energy feasibility, measuring runtimes, computing the Gini index of robot rewards, and recording coalition stability once it is defined. |
| Data logging | Run-level and robot-level CSV records in fixed schemas (Section 5). |
| Adapters | Converting teammates' outputs into the standardized analytics schemas (Section 9). |
| Data aggregation | Mean and standard deviation per (algorithm, configuration) (Section 6). |
| Plotting | Plotting scripts and the figures/tables for the final report (Section 7). |
| Quantitative suite | The experiment configuration file and the scripts that reproduce every reported number. |

### 1.3 Boundaries (what I do **not** own)

- **No allocation algorithms.** Greedy, GA, MILP and SVCA belong to the Allocation & SVCA owner in `src/allocation/`. The harness only *calls* them through an agreed interface.
- **No reward-sharing or game-theory algorithms.** Characteristic functions, Monte Carlo/MCMC sampling and the preference-aware Shapley solver belong to the Game Theory owner in `src/game_theory/`.
- **No simulation logic.** Map generation, robot kinematics and battery/velocity models belong to the Simulation owner in `src/simulation/` and `src/core/`.
- **No changes to teammates' modules.** If the harness needs something from another module (a field, a hook, a seed argument), I raise it as a dependency (Section 10). Differences in output formats are absorbed by analytics-side adapters (Section 9), not by editing teammates' code.
- **No invented definitions.** Coalition stability has no formula in the proposal. Until the team formally agrees one with the Game Theory owner, the harness keeps a placeholder column for it (Section 2.5). The only non-proposal formula in this document is the provisional energy penalty (Section 2.2), and it is labelled as an engineering assumption.

---

## 2. Metrics

All metrics are computed **per run**: one algorithm on one scenario instance. They are aggregated later (Section 6).

### 2.1 Allocation cost $A(U)$

The formula is preserved exactly from proposal Eq. (4):

$$
A(U) = \sum_{t=1}^{T} \delta_t \, \eta_t(U) \;+\; \sum_{r=1}^{R} \lambda_r \, d_r(U) \;+\; \varphi(U)
$$

subject to (Eqs. 5–7):

$$
u_{r,t} \in \mathbb{Z}_+ \;\; \forall r, t, \qquad
\sum_{r=1}^{R} \mathbb{I}(u_{r,t} > 0) = 1 \;\; \forall t \in \mathcal{T}, \qquad
E_r(U) \ge 0 \;\; \forall r \in \mathcal{R}
$$

| Symbol | Meaning (from proposal) | Expected source |
| :--- | :--- | :--- |
| $U = [u_{r,t}]$ | Task execution sequences | Allocation output (`AllocationResult.assignments`) |
| $\delta_t$ | Priority of task $t$ | Task state $y_t = (q_t, \tau_t, \delta_t)$ |
| $\eta_t(U)$ | Completion time of task $t$ | Allocation output / simulator (`task_completion_times`) |
| $\lambda_r$ | Usage penalty of robot $r$ | Robot state $x_r = (p_r, v_r, E_r, \lambda_r)$ |
| $d_r(U)$ | Distance travelled by robot $r$ | Allocation output / simulator (`agent_distances`) |
| $E_r(U)$ | Remaining energy of robot $r$ after executing $U$ | Simulator / allocation output (initial $E_r$ minus `agent_energy_used`, to be confirmed) |
| $\varphi(U)$ | Penalty for energy violations | **Not defined in the proposal.** Provisional default in Section 2.2 |

**Benchmarking rule:** the harness computes $A(U)$ itself, with **one shared cost evaluator** applied to every algorithm's output. It does not report each solver's self-reported `total_cost`. This rules out differences caused by algorithms evaluating cost in slightly different ways. Where a solver also reports `total_cost`, the harness can compare it to the shared value as a consistency check and log any mismatch.

**Validity:** if an allocation breaks the single-assignment constraint (Eq. 6), it is not a valid allocation. The run is logged with `success = False` and an explanatory `error_message`, and its cost is not used in aggregates. Energy violations are handled differently: see Section 2.2.

### 2.2 Energy penalty $\varphi(U)$ and energy feasibility

The proposal says only that $\varphi(U)$ "accounts for energy violations". It gives no formula. To make $A(U)$ computable, this plan adopts a **provisional default**.

> **Engineering assumption, not an original proposal formula.** The default below was chosen by the analytics workstream so that the harness can run. It is configurable and replaceable, and it must be confirmed or replaced by the team before final results are reported.

**Provisional default: quadratic penalty**

$$
\varphi(U) = \lambda_E \sum_{r=1}^{R} \max\big(0,\; -E_r(U)\big)^2
$$

Here $\lambda_E \ge 0$ is an energy-penalty coefficient. It is a new parameter, separate from the per-robot usage penalty $\lambda_r$. A robot with $E_r(U) \ge 0$ contributes nothing. A robot that overdraws its energy contributes a cost that grows with the square of the deficit, so large violations are penalised much more than small ones.

**Documented alternatives**

| Strategy | Formula | Notes |
| :--- | :--- | :--- |
| `quadratic` (provisional default) | $\lambda_E \sum_r \max(0, -E_r(U))^2$ | Engineering assumption |
| `linear` | $\lambda_E \sum_r \max(0, -E_r(U))$ | Penalty proportional to the total energy deficit |
| `zero` | $\varphi(U) = 0$ | Cost ignores energy; feasibility is still reported separately |

**Configuration:** the strategy (`quadratic` / `linear` / `zero`) and the coefficient $\lambda_E$ are set in `configs/experiments_config.yaml`. No value for $\lambda_E$ has been chosen yet: it is a placeholder to be agreed with the team (Section 10, D4). Both values are copied into each experiment's metadata file, so every reported cost can be traced back to the penalty that produced it.

**Isolation:** the energy penalty is a separate, replaceable component of the cost evaluator (Section 9). Changing the strategy, or swapping in a team-agreed definition later, does not touch the rest of the cost evaluator, the harness or the logging.

**Feasibility is tracked separately from cost.** Whatever penalty strategy is chosen, every run also records whether the allocation satisfies Eq. (7):

- `energy_feasible = True` if $E_r(U) \ge 0$ for every robot $r$, otherwise `False`.
- An energy-infeasible allocation is still a *successful* run (`success = True`), and its cost (including $\varphi(U)$) is still recorded. Infeasibility is reported, not hidden inside the cost.
- This way the report can show cost and feasibility rate side by side. It also means results stay interpretable under the `zero` strategy, or if the penalty definition changes.

### 2.3 Runtime

Three runtimes are recorded per run, all in **seconds**, using a monotonic high-resolution timer (`time.perf_counter`):

| Field | What is timed |
| :--- | :--- |
| `allocation_runtime_seconds` | The allocation solver call only: from handing over the ready-made scenario until $U$ is returned. |
| `reward_runtime_seconds` | The reward-sharing solver call only: from handing over the reward-stage input until the reward output is returned. |
| `total_runtime_seconds` | The **full pipeline**, measured with **its own independent timer** that starts before the allocation stage and stops after the reward stage. It includes the hand-off between stages (for example preparing reward-stage input from the allocation output). |

**Stage completion** is recorded explicitly, as two boolean fields in the main results (Section 5.1). They describe only whether the solver call itself returned. They are **never inferred** from `success`, from `error_message`, or from the presence of rows in `robot_rewards.csv`.

| Field | `True` when | `False` when |
| :--- | :--- | :--- |
| `allocation_stage_completed` | The allocation solver call returned normally, regardless of whether the harness's later validation of the returned allocation succeeds. | The call never started, raised an exception, or timed out. |
| `reward_stage_completed` | The reward solver call returned normally, regardless of whether later reward adaptation, Gini computation or CSV writing succeeds. | The call never started, raised an exception, or timed out. |

**Pipeline completed** means both fields are `True`. `success` keeps its original meaning (the algorithm returned a valid allocation that the harness could evaluate), so `allocation_stage_completed = True` with `success = False` is possible, for example when the solver returned an allocation that violates Eq. (6).

Rules:

- `total_runtime_seconds` is **measured, never computed** as the sum of the other two. The difference between the total and the sum of the stages shows the pipeline overhead.
- Excluded from all three timers: scenario generation, the harness's own metric computation (cost, Gini, feasibility), conversion of results into log rows (for example the reward adapter producing `robot_rewards.csv` rows), and file I/O. These all happen after the pipeline timers stop or outside them.
- **Failure handling (applies to each stage runtime separately):**
  - If a stage **never starts** (for example the reward stage after the allocation stage failed or timed out), its runtime is `NaN` and its completion field is `False`.
  - If a stage **starts and then fails or times out**, its measured elapsed time is recorded (for a timeout, this is approximately the time limit) and its completion field is `False`. These elapsed times stay in the raw CSV but are excluded from the completed-stage runtime means (Section 6).
  - `total_runtime_seconds` is always the independently measured elapsed time of the full pipeline, **including when the pipeline fails or times out part-way**. It is `NaN` only if the pipeline itself never started.
- Each runtime is aggregated as mean and standard deviation over the repetitions of each (algorithm, configuration), using the eligibility rules and valid-value counts in Section 6.
- **Fairness of timing:** every algorithm runs on the same machine, in the same environment and under the same process conditions (one run at a time, no parallel benchmark jobs competing for CPU). The machine, CPU, Python version and solver/library versions are written to the experiment metadata.

### 2.4 Gini index of robot rewards

The Gini index is computed **only from standardized robot-level reward data**: the `reward` values for one (`experiment_id`, `scenario_id`, `algorithm`) in `robot_rewards.csv` (Section 5.2). It never reads the reward solver's internal objects directly. Robot rewards are $x_1, \dots, x_R$.

Standard formula, for non-negative rewards:

$$
G = \frac{\sum_{i=1}^{R} \sum_{j=1}^{R} |x_i - x_j|}{2 R \sum_{i=1}^{R} x_i}
$$

$G = 0$ means perfect equality. For non-negative rewards, $G$ lies in $[0, 1 - 1/R]$.

**Edge-case rules (to be implemented and unit-tested):**

| Case | `gini_index` | `invalid_reward_flag` |
| :--- | :--- | :--- |
| All rewards ≥ 0, total > 0, $R \ge 2$ | Standard formula | `False` |
| All rewards are zero | `0` (everyone receives the same amount, so the distribution is perfectly equal; this also avoids division by zero) | `False` |
| Exactly one robot ($R = 1$) | `0`, **by reporting convention** (inequality is undefined for a single robot) | `False` |
| **Any** reward is negative | `NaN` | `True`, with the reason noted in `error_message` |
| Rewards missing or unavailable (reward stage did not complete, or no usable robot-level rewards were produced) | `NaN` | `False` (the cause is recorded through the stage-completion fields and `error_message`) |

**Negative rewards are never shifted, clipped or rescaled** to force a Gini value. A run with any negative robot reward produces no Gini value. It is counted and reported through `invalid_reward_flag`, so the report can say how often this happened per algorithm.

**Completeness is not the Gini metric's job.** Before the Gini metric is called, the reward adapter / harness must ensure that a completed reward output contains **exactly one reward for every expected robot** of the scenario: no missing robots, no extra entities and no duplicates (Section 5.2). The Gini formula does not check this and does not know which robots were expected. If the check fails, the harness treats the run's rewards as unavailable (`gini_index = NaN`, `invalid_reward_flag = False`, with a reason in `error_message`) and does not pass incomplete data to the Gini metric.

Robot-level input with duplicate `robot_id` values for one run is rejected as invalid input (see the uniqueness key in Section 5.2); it never produces a Gini value.

Aggregation takes the **mean of per-run Gini values** over runs with a non-NaN value (counted by `n_gini`, Section 6). It does not compute one Gini over rewards pooled across runs.

Note: `RewardDistribution` in `src/core/types.py` has a `gini_index` field. The harness does not use it. To keep one source of truth, $G$ is always recomputed by the analytics Gini metric from the standardized robot-level data.

### 2.5 Coalition stability

**Status: pending formal agreement with the Game Theory & Reward Solver owner.**

The proposal lists coalition stability as an evaluation criterion but gives no formula. This plan deliberately does **not** define one. Until a definition is formally agreed:

- The main results CSV keeps a `coalition_stability` column, left empty (`NaN`) for every run. The logger **accepts only missing/NaN values** for this column and rejects any non-missing value, so no unagreed number can enter the raw data. This validation is relaxed only when the definition is formally agreed.
- No stability plot is produced.
- The coalition-stability metric exists only as a clearly marked placeholder component (Section 9). It returns `NaN` or reports "not implemented" and contains no improvised formula.

Before implementation, the agreed definition has to specify: what is evaluated (a coalition/cluster partition, a payoff vector, or both); what inputs are needed (for example characteristic-function values $J(S)$ or $V(S)$, cluster membership, payoffs); whether the result is a boolean, a score or a count; and how it is computed for large $N$.

---

## 3. Experimental Configuration

### 3.1 Fixed settings (from proposal)

| Setting | Value | Source |
| :--- | :--- | :--- |
| Environment | 2D, 500 m × 500 m | Proposal §4.3 |
| Problem size | $N = R + T$, with $N \in [10, 80]$ | Proposal §3, §4.3 |
| Repetitions per configuration | 10 independent runs | Project plan |
| Algorithms compared | Centralized MILP, GA-based clustering, Greedy Allocation, SVCA | Proposal §4.3 |

Both the number of robots $R$ and the number of tasks $T$ change across configurations.

### 3.2 Illustrative configurations

> **Initial proposals, adjustable.** The table below is a starting point. It can change once the simulation and allocation modules exist (for example, if MILP cannot finish at the larger sizes, or if the team prefers a different $R{:}T$ ratio). Every change goes through `configs/experiments_config.yaml`, not through code. Any replacement set must still keep $N = R + T$ within $[10, 80]$, vary both $R$ and $T$, and use 10 repetitions per configuration.

| Config | $R$ (robots) | $T$ (tasks) | $N = R + T$ |
| :---: | :---: | :---: | :---: |
| C1 | 5 | 5 | 10 |
| C2 | 5 | 15 | 20 |
| C3 | 10 | 20 | 30 |
| C4 | 15 | 25 | 40 |
| C5 | 20 | 30 | 50 |
| C6 | 25 | 35 | 60 |
| C7 | 30 | 40 | 70 |
| C8 | 35 | 45 | 80 |

Planned run count with these values: 8 configurations × 10 repetitions × 4 algorithms = **320 algorithm runs**, on 80 distinct scenario instances.

Possible later additions (only if time allows; also adjustable): one-factor sweeps that hold $R$ fixed while varying $T$, or the other way round, to separate the two effects.

### 3.3 Run-time limits

Exact MILP solving may become intractable at the larger sizes. The harness needs a **per-run time limit** for every algorithm. If a run hits it, it is recorded with `success = False` and `error_message = "timeout"` (or, if the solver returns an incumbent solution, with that flagged in the error message, as agreed with the Allocation owner). The elapsed time of a timed-out stage is recorded as described in Section 2.3. The limit value is **not fixed yet**. It will be a parameter in the experiment config (Section 10, D1 covers timeout behaviour).

---

## 4. Fair Benchmarking Protocol

Every algorithm is compared on **identical inputs**. For each configuration and repetition, the harness:

1. Generates **one** scenario instance with the Simulation owner's scenario generator, using a recorded seed.
2. Hands the **same** instance to all four algorithms (an identical object or a deep copy, so that no algorithm can change another algorithm's input).
3. Evaluates every output with the **same** cost evaluator, energy-penalty strategy, feasibility check and metric code.

Things that must be identical across algorithms for a given scenario:

| Category | Items |
| :--- | :--- |
| Robots | Count $R$, initial positions $p_r$, velocities $v_r$, energy $E_r$, usage penalties $\lambda_r$, capability/skill attributes |
| Tasks | Count $T$, locations $q_t$, execution times $\tau_t$, priorities $\delta_t$ |
| Environment | Map bounds (500 m × 500 m), distance metric |
| Constraints | Single assignment per task (Eq. 6), energy feasibility (Eq. 7), any capability-compatibility rules |
| Cost definition | Exactly $A(U)$ as in Section 2.1, with the same $\varphi(U)$ strategy and $\lambda_E$ for every algorithm within an experiment |
| Reward stage | The same reward solver and the same parameters (for example $\alpha$, sample counts) applied after each allocation, if rewards are compared across allocation algorithms |
| Compute | Same machine, same environment, runs executed one at a time |

**Seed management:**

- One base seed lives in `configs/experiments_config.yaml`.
- Each scenario gets its own seed, derived deterministically from the base seed, the configuration and the repetition index. It is stored in the `random_seed` column.
- Stochastic algorithms (GA, SVCA's permutation sampling) and the stochastic reward solver (MCMC) must accept a seed argument. Their seed is derived deterministically from the scenario seed, so any single run can be reproduced exactly. The exact derivation rule will be written down in the config/metadata.
- Library versions and the git commit hash of the code under test are recorded once per experiment in the metadata.

---

## 5. Result Schemas

Raw records are **always kept**. Aggregated tables are computed from them and never replace them. Both raw files are append-only across experiments and are told apart by `experiment_id`.

### 5.1 Main benchmark results (one row per run)

One row = one algorithm executed on one scenario instance (allocation + reward stage). Planned location: `data/results/raw/benchmark_results.csv`.

| Column | Type | Description |
| :--- | :--- | :--- |
| `experiment_id` | string | Identifies one benchmark campaign (one invocation of the harness with a given config file). |
| `scenario_id` | string | Identifies the generated scenario instance (configuration + repetition index). Shared by all four algorithms run on that instance, which allows paired comparisons. |
| `algorithm` | string | One of `MILP`, `GA`, `Greedy`, `SVCA` (exact labels to match the solvers' `solver_name`). |
| `num_robots` | int | $R$ |
| `num_tasks` | int | $T$ |
| `total_entities` | int | $N = R + T$ (stored explicitly and validated as `num_robots + num_tasks`). |
| `random_seed` | int | Seed used to generate this scenario (and from which algorithm seeds are derived). |
| `allocation_cost` | float | $A(U)$ from the shared cost evaluator, including $\varphi(U)$ under the configured strategy; `NaN` if the run failed. |
| `allocation_runtime_seconds` | float | Allocation solver runtime (Section 2.3). |
| `reward_runtime_seconds` | float | Reward solver runtime (Section 2.3); `NaN` if the reward stage never started; elapsed time if it started and then failed. |
| `total_runtime_seconds` | float | Independently measured full-pipeline runtime (Section 2.3). |
| `allocation_stage_completed` | bool | `True` if the allocation solver call returned normally, regardless of later allocation validation (Section 2.3). Always `True` or `False`, never empty. |
| `reward_stage_completed` | bool | `True` if the reward solver call returned normally, regardless of later reward adaptation or CSV writing (Section 2.3). Always `True` or `False`, never empty. |
| `gini_index` | float | Gini index of robot rewards (Section 2.4); `NaN` if unavailable or invalid. |
| `invalid_reward_flag` | bool | `True` if any robot reward was negative, so the Gini index is `NaN` by rule (Section 2.4). |
| `energy_feasible` | bool | `True` if $E_r(U) \ge 0$ for every robot (Eq. 7); tracked separately from cost (Section 2.2). Empty if the run failed. |
| `coalition_stability` | float / empty | **Empty (`NaN`) until a definition is formally agreed** (Section 2.5). The logger accepts only missing/NaN values and rejects any non-missing value. |
| `success` | bool | `True` if the algorithm returned a valid allocation that the harness could evaluate. |
| `error_message` | string | Empty on success; otherwise the exception, timeout or validation reason. Also used for non-fatal notes (for example the reason for `invalid_reward_flag`). |

Key: (`experiment_id`, `scenario_id`, `algorithm`) is unique.

The schema has **19 columns**, in the order shown. Changes from the earlier versions of this schema: `runtime_seconds` was replaced by the three runtime fields; `invalid_reward_flag`, `energy_feasible`, `allocation_stage_completed` and `reward_stage_completed` were added. All other fields are unchanged, and `success` keeps its original meaning.

### 5.2 Robot-level rewards (one row per robot per run)

Planned location: `data/results/raw/robot_rewards.csv`.

| Column | Type | Description |
| :--- | :--- | :--- |
| `experiment_id` | string | Same meaning as in 5.1 |
| `scenario_id` | string | Same meaning as in 5.1 |
| `algorithm` | string | Allocation algorithm whose output fed the reward stage |
| `robot_id` | int / string | Robot identifier, consistent with the scenario's robot ids |
| `reward` | float | Final reward of this robot, as produced by the reward solver (never shifted or rescaled) |

Each run whose reward output was successfully adapted contributes exactly $R$ rows. The rows join to the main results on (`experiment_id`, `scenario_id`, `algorithm`). A run whose reward solver returned (`reward_stage_completed = True`) but whose reward output could not be adapted into a complete robot-level set contributes **no** rows; the reason is recorded in `error_message`.

**Completeness responsibility:** the reward adapter / harness must ensure that a completed reward output contains **exactly one reward for every expected robot** (no missing robots, no extra entities, no duplicate ids) before the Gini metric is computed and before rows are written. This check is not part of the Gini formula (Section 2.4).

**Uniqueness key:** (`experiment_id`, `scenario_id`, `algorithm`, `robot_id`) is unique. **Duplicate rows for this key must be prevented.** The result writer (Section 9) checks the key before writing, and the Gini metric's input check rejects a run whose robot-level data contains duplicate `robot_id` values, so a duplicate can never silently change a Gini value. A violation, whether inside the batch or against rows already in the file, raises an error and rejects the whole batch (Section 5.4).

**How rows are produced:** the reward solver's output (currently expected to be `RewardDistribution` from `src/core/types.py`, whose `payoffs` may cover all entities in $\mathcal{N} = \mathcal{R} \cup \mathcal{T}$) is converted by a **replaceable reward adapter** on the analytics side (Section 9). The adapter selects robot entries, maps ids to `robot_id`, and emits rows in this schema. If the game-theory output format changes, only the adapter changes. The Gini metric and the reward-distribution plots consume this standardized robot-level data, never the solver's internal objects.

### 5.3 Experiment metadata

One metadata file per experiment, planned as `data/results/raw/metadata_<experiment_id>.yaml`. It records a copy of the experiment config (including the energy-penalty strategy and $\lambda_E$), the base seed and seed-derivation rule, time limits, library and solver versions, the git commit hash, and machine information.

These are planned output paths only. Creating them is part of implementation, and they sit inside the existing `data/results/` folder.

### 5.4 Writing rules and recovery

- **Two independent writers.** `benchmark_results.csv` and `robot_rewards.csv` are written by independent writers. There is **no cross-file transaction mechanism**.
- **All-or-nothing batches.** Each writer validates the whole batch (schema, types, duplicate keys within the batch, duplicate keys already in the file) before writing anything. If any record fails, the entire batch is rejected and nothing is written. Files are replaced atomically (temp file in the same folder, then replace), so a failure never leaves a partial file.
- **Cross-field record validation.** In addition to schema and type checks, the main-results writer rejects a record (and therefore the whole batch) that violates any of these rules:
  1. `success = True` requires `allocation_stage_completed = True`.
  2. `reward_stage_completed = True` requires both `allocation_stage_completed = True` and `success = True`.
  3. `invalid_reward_flag = True` requires `reward_stage_completed = True`.
  4. A non-NaN `gini_index` requires `reward_stage_completed = True` and `invalid_reward_flag = False`.
  5. `success = False` requires `allocation_cost` to be missing/NaN.

  The valid case `allocation_stage_completed = True` with `success = False` (the solver returned, but the allocation failed validation) is preserved: no rule forbids it.
- **Serialization.** Missing values are empty cells. Booleans are `True` / `False`. Floats round-trip exactly.
- **Write order.** The harness writes the run's robot rewards **first** and the main benchmark result **second**, so a main result row never exists without its reward rows (when rewards exist).
- **Orphan reward rows.** If the reward rows are written but the main result write then fails, **orphan reward rows** (rows with no matching main result record) may remain. The logger does not delete or overwrite them.
- **Retry behaviour.** Retrying the same (`experiment_id`, `scenario_id`, `algorithm`) will fail with a duplicate-key error when the reward rows are written again. This is intended: duplicate reward keys always raise an error, and rows are never silently overwritten or removed. Resolving an orphan (for example by re-running under a new `experiment_id`, or by a manual repair) is a deliberate human decision outside the logger.
- **Orphans never reach results.** Aggregation reads only the main results file (Section 6). Any consumer of `robot_rewards.csv` (for example plot P4) must use only reward rows whose (`experiment_id`, `scenario_id`, `algorithm`) matches a record in the main results.
- **Experiment ids and metadata.** `experiment_id` is restricted to letters, digits, `.`, `_` and `-`, so it is safe in file names. Metadata files are not overwritten by default. Metadata also records whether the git working tree had uncommitted changes (`git_dirty`) next to the commit hash.
- **Version control.** Raw result and metadata paths under `data/results/raw/` are added to `.gitignore` when this is implemented.

---

## 6. Data Aggregation

Aggregation groups main-result records by (`algorithm`, `num_robots`, `num_tasks`, `total_entities`). It produces:

| Output column | Definition |
| :--- | :--- |
| `n_runs` | Number of runs attempted (expected: 10) |
| `n_success` | Number of runs with `success = True` |
| `n_allocation_cost` | Number of values actually eligible and included in `allocation_cost_mean` and `allocation_cost_std`: runs with `success = True` and a non-NaN `allocation_cost` |
| `n_allocation_runtime` | Number of values actually eligible and included in the allocation runtime mean and std: runs with `allocation_stage_completed = True` and a non-NaN `allocation_runtime_seconds` |
| `n_reward_runtime` | Number of values actually eligible and included in the reward runtime mean and std: runs with `reward_stage_completed = True` and a non-NaN `reward_runtime_seconds` |
| `n_total_runtime` | Number of values actually eligible and included in the total runtime mean and std: runs with **both** stage-completion flags `True` and a non-NaN `total_runtime_seconds` |
| `n_gini` | Number of values actually eligible and included in `gini_index_mean` and `gini_index_std`: runs with a non-NaN `gini_index` |
| `allocation_cost_mean`, `allocation_cost_std` | Over runs with `success = True` and a non-NaN `allocation_cost` |
| `allocation_runtime_seconds_mean`, `allocation_runtime_seconds_std` | Over runs with `allocation_stage_completed = True` and a non-NaN `allocation_runtime_seconds` |
| `reward_runtime_seconds_mean`, `reward_runtime_seconds_std` | Over runs with `reward_stage_completed = True` and a non-NaN `reward_runtime_seconds` |
| `total_runtime_seconds_mean`, `total_runtime_seconds_std` | Over runs with **both** `allocation_stage_completed = True` and `reward_stage_completed = True` and a non-NaN `total_runtime_seconds` |
| `energy_feasible_rate` | (number of successful runs with `energy_feasible = True`) / (number of successful runs with a non-missing `energy_feasible` value). If the denominator is zero, the result is `NaN`. |
| `gini_index_mean`, `gini_index_std` | Over runs with a non-NaN Gini value (Section 2.4) |
| `n_invalid_reward` | Number of runs with `invalid_reward_flag = True` |
| `coalition_stability_mean`, `coalition_stability_std` | Only after a definition exists; otherwise left empty |

Rules:

- Standard deviation is the **sample** standard deviation ($\text{ddof} = 1$). The choice is stated in the report.
- `NaN` values are excluded from each metric's mean and std. The number of values actually eligible and included is always visible in that metric's own count (`n_allocation_cost`, `n_allocation_runtime`, `n_reward_runtime`, `n_total_runtime`, `n_gini`), so a mean over 3 values never looks like a mean over 10. `n_runs`, `n_success` and `n_invalid_reward` give the wider context.
- **Stage completion is never derived.** Aggregation uses only the explicit `allocation_stage_completed` and `reward_stage_completed` fields. It does not infer completion from `success`, from `error_message`, or from the existence of rows in `robot_rewards.csv`.
- Raw elapsed runtimes of failed or timed-out stages stay in the raw CSV (Section 2.3). They are excluded from the completed-stage runtime means above, and the report states how many runs were excluded through `n_runs` versus the metric counts. A completed stage with a `NaN` runtime is also excluded.
- `allocation_cost` and `n_success` still follow `success`, which keeps its original meaning. A run can therefore contribute to the allocation runtime mean (`allocation_stage_completed = True`) while being excluded from the cost mean (`success = False`).
- Aggregation reads **only** the main results file. It does not read `robot_rewards.csv`, so orphan reward rows (Section 5.4) cannot affect any aggregate.
- **One experiment per aggregation call:** `experiment_id` is used as a filter, not as a grouping column, so results of different campaigns are never pooled.
- If fewer than 2 valid values exist, std is `NaN`.
- Output: `outputs/tables/summary_<experiment_id>.csv`. A LaTeX version for the report is planned for the report-asset step (`scripts/generate_report_assets.py`) and is not part of the logging/aggregation implementation.

---

## 7. Planned Plots

All figures are generated by script from the CSV files, never by hand, and saved to `outputs/figures/`. Each algorithm keeps the same colour and marker in every figure. Error bars show ±1 standard deviation across the 10 repetitions.

| # | Figure | Axes / form | Status |
| :--- | :--- | :--- | :--- |
| P1 | Allocation cost vs problem size | x: $N$ (configs C1–C8); y: mean $A(U)$; one line per algorithm, error bars; energy-feasible rate annotated or shown alongside | Planned |
| P2 | Runtime vs problem size | x: $N$; y: mean runtime (s), likely log scale; one line per algorithm for allocation runtime and for total runtime (separate panels), error bars; timeouts marked | Planned |
| P3 | Gini index comparison | Grouped bars (or lines) of mean $G$ per algorithm across configurations, error bars; count of invalid-reward runs shown | Planned |
| P4 | Reward distributions | Box/violin plots of per-robot rewards per algorithm at selected configurations, from `robot_rewards.csv` (only rows associated with a main result record; orphans excluded, Section 5.4) | Planned |
| P5 | Coalition stability | Form depends on the agreed definition | **Blocked** until stability is formally defined |

Optional, if useful for the report: the success/timeout rate per algorithm vs $N$, a stacked breakdown of allocation vs reward runtime, and paired cost differences relative to MILP (on scenarios where MILP succeeded).

---

## 8. Future Files

None of these exist yet. They are planned for the implementation phase and are all owned by the Benchmarking & Analytics workstream.

| File | Purpose |
| :--- | :--- |
| `src/analytics/metrics.py` | Shared cost evaluator with the replaceable energy-penalty component, energy-feasibility check, Gini index with edge-case rules, coalition-stability placeholder, runtime helpers |
| `src/analytics/logger.py` | Reward and allocation adapters, writing main and robot-level CSV rows, metadata capture, aggregation helpers |
| `src/visualization/plots.py` | Plotting functions for P1–P5 |
| `scripts/run_benchmarks.py` | Harness entry point: load config → generate scenarios → run the pipeline with stage and total timers → log records |
| `scripts/generate_report_assets.py` | Aggregation → summary tables (CSV/LaTeX) → figures |
| `configs/experiments_config.yaml` | Configurations, repetitions, base seed, algorithm list, time limits, energy-penalty strategy and $\lambda_E$, output paths |
| `tests/test_metrics.py` | Unit tests for metrics (Gini edge cases including negative rewards, each energy-penalty strategy, feasibility, cost formula on hand-built cases, schema validation) |

> **Path note:** all planned source files use the existing repository layout: `src/analytics/`, `src/visualization/`, `configs/` and `tests/` already exist. `scripts/` is part of the layout described in the repository README but has not been created yet; it is only a location for new files. Likewise `data/results/raw/` (Section 5) is a new subfolder inside the existing `data/results/`. No existing directory is restructured, renamed or moved (Section 10, D8).

---

## 9. Replaceable Components and Interfaces

The harness is built from small components with fixed inputs and outputs, so any one of them can be replaced without changing the others. These are interface descriptions, not code.

| Component | Input | Output | Why replaceable |
| :--- | :--- | :--- | :--- |
| **Scenario provider** | ($R$, $T$, seed, map size) | One scenario: robots and tasks with all attributes from Section 4 | Wraps the Simulation owner's generator; only this wrapper changes if their API changes. |
| **Allocation solver adapter** | Scenario, algorithm name, seed, time limit | Standardized allocation: assignments, task completion times, robot distances, robot energy used, solver's own cost (optional) | Hides differences between the four solvers' call signatures and result objects (expected: `AllocationResult`). |
| **Cost evaluator** | Scenario, standardized allocation, energy-penalty component | $A(U)$ | One implementation of proposal Eq. (4) for every algorithm. |
| **Energy-penalty component** | Per-robot remaining energy $E_r(U)$, strategy, $\lambda_E$ | $\varphi(U)$ | Provisional (Section 2.2). Strategy is chosen by config; a team-agreed definition can replace it without touching the cost evaluator. |
| **Feasibility checker** | Per-robot remaining energy $E_r(U)$ | `energy_feasible` | Kept separate from the cost so feasibility is reported regardless of the penalty strategy. |
| **Reward adapter** | Game-theory reward output (expected: `RewardDistribution`), scenario robot ids | Robot-level rows: (`robot_id`, `reward`) for the `robot_rewards.csv` schema | The only place that knows the reward solver's format. If the format changes, only this adapter changes. |
| **Gini metric** | Standardized robot rewards for one run | (`gini_index`, `invalid_reward_flag`) | Pure function of robot-level data; never reads solver internals. |
| **Coalition-stability metric** | To be defined | To be defined | **Placeholder only, pending formal agreement with the Game Theory owner** (Section 2.5). Returns `NaN` until then. |
| **Pipeline timer** | The allocation call, the reward call, and the full pipeline | The three runtime fields | Stage timers and the independent total timer are kept separate from the solvers themselves. |
| **Result writers** | Run records, robot reward rows, metadata | Rows in `benchmark_results.csv` / `robot_rewards.csv`, metadata file | Schema and cross-field validation happen here, in one place, with all-or-nothing batches (Section 5.4). Rejects any non-missing `coalition_stability`. The two CSV writers are independent. |
| **Aggregator / plotter** | Raw CSV files only | Summary tables and figures | Work only from saved CSVs, so figures can be regenerated without rerunning experiments. Aggregation reads only the main results; plots that use robot rewards must exclude orphan rows. |

Interface rules:

- Analytics components depend only on the standardized formats above, never on teammates' internal classes. Teammates' outputs are converted once, in the adapters.
- Every replaceable choice (energy-penalty strategy, $\lambda_E$, time limits, configurations, algorithm list) is set in `configs/experiments_config.yaml` and copied into the experiment metadata.
- Adapters are analytics-side code. They do not require any change to teammates' modules.

---

## 10. Open Teammate Dependencies

| # | Dependency | Owner | Needed for | What I need |
| :--- | :--- | :--- | :--- | :--- |
| D1 | **Allocation result format** | Allocation & SVCA | Cost, runtime, harness | Confirmation that all four solvers return `AllocationResult` (from `src/core/types.py`) with `assignments`, `task_completion_times`, `agent_distances`, `agent_energy_used` filled in consistently. A shared solver call signature. Seed and time-limit arguments. Behaviour on timeout/infeasibility (exception vs flagged result), so the harness can tell "returned normally" from "raised or timed out" when setting `allocation_stage_completed`. Exact `solver_name` labels. |
| D2 | **Robot reward format** | Game Theory | Reward adapter, Gini, reward plots | Whether `RewardDistribution.payoffs` is keyed by robot id only or by all entities in $\mathcal{N} = \mathcal{R} \cup \mathcal{T}$, and how robot ids are told apart from task ids. Whether rewards can be negative (negative rewards make the Gini `NaN` by rule). A seed argument for MCMC. |
| D3 | **Coalition stability definition and data** | Game Theory (and Allocation for SVCA clusters) | Coalition stability | A formally agreed stability definition, plus the data needed to compute it: cluster/coalition membership, characteristic-function values, or a callable to evaluate them. Until then the logger accepts only missing/NaN values for `coalition_stability`. |
| D4 | **Cost-function inputs and the energy penalty** | Simulation + Allocation (+ whole team for $\varphi$) | $A(U)$, feasibility | How $\eta_t(U)$ is computed (does it include travel and execution time $\tau_t$?). How $E_r(U)$ is obtained (initial $E_r$ minus energy used?). Confirmation or replacement of the provisional quadratic $\varphi(U)$, and a value for $\lambda_E$. Access to $\delta_t$ and $\lambda_r$ from the scenario objects. Whether "capabilities" create compatibility constraints. |
| D5 | **Scenario generator interface** | Simulation | Fair benchmarking | A generator that takes ($R$, $T$, seed) and returns a reproducible scenario, so the same scenario can be given to all algorithms. |
| D6 | **Pipeline order for rewards** | Game Theory + Allocation | Reward runtime, Gini comparison | Whether the reward solver runs once per allocation algorithm's output (so the Gini differs per algorithm), and how the allocation output becomes reward-stage input. The cross-field rules in Section 5.4 assume that the reward stage runs only on a valid allocation (`reward_stage_completed = True` requires `success = True`); this needs confirmation. |
| D7 | **SVCA sub-problem solver** | Allocation | Interpreting results | Which allocator SVCA uses inside each cluster, so results are attributed correctly. |
| D8 | **Package layout** | Whole team | File locations | The existing repository layout must be preserved, with no restructuring or file movement. Analytics code lives in `src/analytics/` and `src/visualization/`. This is a constraint rather than an open question; it is listed so that teammates do not move these directories. |

Repository observation for the team (not changed by me): `src/core/types.py` currently seems to contain two concatenated versions of the module. The first defines a `BenchmarkRecord` dataclass whose field names (`run_id`, `problem_size_n`, `allocation_runtime_sec`, `is_feasible`, …) differ from the Section 5.1 schema. The owner of `src/core/` should decide whether `BenchmarkRecord` is updated to match this schema or removed in favour of the analytics result writers.

---

## 11. Ordered Implementation Checklist

1. [ ] Confirm that the existing repository layout is preserved (D8: no restructuring or file movement) and agree on both result schemas (Sections 5.1 and 5.2) with the team.
2. [ ] Agree on the interfaces: solver call signature and `AllocationResult` usage (D1), reward output format (D2), scenario generator API (D5), pipeline order (D6).
3. [ ] Get team confirmation (or a replacement) for the provisional energy penalty and a value for $\lambda_E$ (D4).
4. [ ] Write `configs/experiments_config.yaml`: configurations C1–C8, 10 repetitions, base seed, algorithm list, placeholder time limits, energy-penalty strategy and $\lambda_E$.
5. [ ] Implement the Gini metric with all edge-case rules from Section 2.4 (including the invalid-reward flag). Write its tests in `tests/test_metrics.py` first.
6. [ ] Implement the energy-penalty component (`quadratic`, `linear`, `zero`) and the feasibility checker, with tests for each strategy.
7. [ ] Implement the shared cost evaluator once the D4 inputs are confirmed. Unit-test it on small hand-computed scenarios.
8. [ ] Add the coalition-stability placeholder (`NaN` / not implemented) with a reference to this document.
9. [ ] Implement the allocation and reward adapters, and the result writers for `benchmark_results.csv`, `robot_rewards.csv` and the metadata file, with schema and cross-field validation (Section 5.4).
10. [ ] Implement `scripts/run_benchmarks.py`: seed derivation, one scenario per (config, repetition), same input to every algorithm, stage timers plus the independent total timer, explicit stage-completion flags, a completeness check of robot rewards before Gini (Section 5.2), robot rewards written before the main result (Section 5.4), time limits, error capture.
11. [ ] Do a dry run of the harness on the smallest configuration (C1) with whatever solvers are available. Check reproducibility by re-running with the same seeds.
12. [ ] Implement aggregation (Section 6) and the summary CSV export. LaTeX export belongs to the report-asset step (item 13).
13. [ ] Implement `plots.py` (P1–P4) and `scripts/generate_report_assets.py`.
14. [ ] Run the full suite once all four algorithms and the reward solver are merged. Review timeouts, feasibility rates and invalid-reward counts, and adjust the configurations through the YAML if needed.
15. [ ] Implement coalition stability and plot P5 once the definition is formally agreed (D3).
16. [ ] Produce the final report figures and tables from the recorded CSVs, and archive the raw data and metadata for reproducibility.

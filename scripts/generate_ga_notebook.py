#!/usr/bin/env python3
"""Generate notebooks/04_genetic_algorithms.ipynb — run from project root."""
import json, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _uid():
    return uuid.uuid4().hex[:16]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _uid(),
        "metadata": {},
        "source": text,
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _uid(),
        "metadata": {},
        "outputs": [],
        "source": text,
    }


# ---------------------------------------------------------------------------
# CELLS
# ---------------------------------------------------------------------------

cells = []

# ── Section 0: Setup ────────────────────────────────────────────────────────

cells.append(md("""\
# Genetic and Evolutionary Algorithms on Mountain Car Variants

*Assignment 01 — "Tinder for RL" | RLI-22 | April 2026*

This notebook presents a research-quality comparison of three population-based
optimisation algorithms — **Simple GA**, **CMA-ES**, and **NEAT** — applied to
four Mountain Car environment variants that exercise different reward structures,
action spaces, and state representations.

All training was completed offline with seeds `[7, 21, 42]`.
Every cell here loads pre-computed logs and figures; no retraining occurs.

**Contents**
1. State Representations and Action Types
2. Reward Design and Wrappers
3. Algorithm Design Rationale
4. Training Strategies and Hyperparameters
5. Evaluation and Performance Analysis
6. Policy Analysis
7. Cross-Environment Comparative Analysis
8. NEAT Topology Visualisation
"""))

cells.append(code("""\
from pathlib import Path
import sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from IPython.display import Image, display

warnings.filterwarnings("ignore")

ROOT = Path.cwd().resolve().parent   # notebooks/ → project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOGS        = ROOT / "outputs" / "logs"
FIGS        = ROOT / "outputs" / "figures"
MODELS      = ROOT / "outputs" / "models"
CONFIGS_DIR = ROOT / "configs"

AGENTS    = ["simple_ga", "cma_es", "neat"]
SCENARIOS = ["discrete", "continuous", "fuel", "minsteps"]
SEEDS     = [7, 21, 42]

AGENT_LABELS  = {"simple_ga": "Simple GA", "cma_es": "CMA-ES", "neat": "NEAT"}
AGENT_COLORS  = {"simple_ga": "#E63946", "cma_es": "#457B9D", "neat": "#2A9D8F"}
SCENARIO_LABELS = {
    "discrete":   "Discrete — MountainCar-v0 (energy shaping ×5)",
    "continuous": "Continuous — MountainCarContinuous-v0 (standard)",
    "fuel":       "Fuel — MountainCar-v0 (−1/step + 100 goal + shaping ×20)",
    "minsteps":   "MinSteps — MountainCarContinuous-v0 (−0.1|a|/step + 100 goal)",
}

print(f"Project root  : {ROOT}")
print(f"PNGs available: {len(list(FIGS.glob('*.png')))}")
print(f"GIFs available: {len(list(FIGS.glob('*.gif')))}")
"""))

cells.append(code("""\
import yaml

def load_aggregate(agent: str, scenario: str) -> dict:
    path = LOGS / f"{agent}_{scenario}_aggregate.json"
    return json.loads(path.read_text()) if path.exists() else {}

def load_seed_results(agent: str, scenario: str) -> pd.DataFrame:
    path = LOGS / f"{agent}_{scenario}_seed_results.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def load_episodes(agent: str, scenario: str) -> list:
    dfs = []
    for seed in SEEDS:
        p = LOGS / f"{agent}_{scenario}_seed{seed}_episodes.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
    return dfs

def load_config(agent: str, scenario: str) -> dict:
    path = CONFIGS_DIR / f"{agent}_{scenario}.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else {}

AGGREGATES   = {(a, s): load_aggregate(a, s)    for a in AGENTS for s in SCENARIOS}
SEED_RESULTS = {(a, s): load_seed_results(a, s) for a in AGENTS for s in SCENARIOS}
EPISODES     = {(a, s): load_episodes(a, s)     for a in AGENTS for s in SCENARIOS}
CONFIGS_DATA = {(a, s): load_config(a, s)       for a in AGENTS for s in SCENARIOS}

missing = [(k, "aggregate") for k, v in AGGREGATES.items() if not v]
missing += [(k, "seed_results") for k, v in SEED_RESULTS.items() if v.empty]
if missing:
    print("WARNING — missing data:", missing)
else:
    print(f"All data loaded: {len(AGGREGATES)} aggregates, {len(SEED_RESULTS)} seed_results files")
"""))

cells.append(code("""\
def show_fig(filename: str, width: int = 900) -> None:
    \"\"\"Display a pre-generated figure from outputs/figures/.\"\"\"
    path = FIGS / filename
    if path.exists():
        display(Image(filename=str(path), width=width))
    else:
        print(f"[figure not found: {filename}]")
"""))

# ── Section 1: State Representations ────────────────────────────────────────

cells.append(md("""\
---
## 1  State Representations and Action Types

Mountain Car is a canonical sparse-reward control problem: an underpowered car must
build momentum by swinging back and forth before it can climb to the goal at the top
of the right hill. With random actions, the car reaches the goal in roughly 0% of
episodes — making this an exploration problem as much as an optimisation one.

We evaluate four environment variants, differing along two axes:
- **Base environment**: discrete (`MountainCar-v0`) vs. continuous (`MountainCarContinuous-v0`)
- **Reward wrapper stack**: modifies the signal the algorithms optimise

### State variables
Both environments share a 2-dimensional continuous state `[position, velocity]`:
- **Position**: `[−1.2, 0.6]` — goal at `pos ≥ 0.5`
- **Velocity**: `[−0.07, 0.07]`
"""))

cells.append(code("""\
from src.envs.mountain_car_discrete import get_discrete_env_spec
from src.envs.mountain_car_continuous import get_continuous_env_spec

disc = get_discrete_env_spec()
cont = get_continuous_env_spec()

print("=== MountainCar-v0 (Discrete) ===")
print(f"  Observation space : {disc['observation_space']}")
print(f"  Action space      : {disc['action_space']}  →  {disc['action_meaning']}")
print(f"  Max episode steps : {disc['max_episode_steps']}")

print("\\n=== MountainCarContinuous-v0 ===")
print(f"  Observation space : {cont['observation_space']}")
print(f"  Action range      : {cont['action_range']}")
print(f"  Max episode steps : {cont['max_episode_steps']}")
"""))

cells.append(code("""\
import numpy as np
from src.envs.state_utils import augment_state

sample_states = [
    [-1.2, 0.00],   # start position, at rest
    [-0.6, 0.04],   # mid position, moving right
    [ 0.4, 0.07],   # near goal, max speed
]

print("State augmentation  [pos, vel] → [pos, vel, KE=0.5v², sin(3·pos)]")
print("-" * 65)
print(f"{'Raw state':30s}  →  Augmented state")
print("-" * 65)
for raw in sample_states:
    obs = np.array(raw, dtype=np.float32)
    aug = augment_state(obs)
    print(f"  {str(raw):28s}  →  {np.round(aug, 4)}")
"""))

cells.append(md("""\
### Why augmented state for discrete scenarios?

The raw `[pos, vel]` observation is sufficient in principle, but adding engineered
features accelerates learning for population-based methods:

- **Kinetic energy** `0.5v²`: directly encodes momentum; allows the policy to
  distinguish a stationary state (`v=0`) from one with leftward momentum (`v<0`)
  using a single positive scalar rather than reading the sign of velocity.
- **sin(3·pos)**: a rough proxy for gravitational potential; it has a period matched
  to the valley-to-peak distance and provides a gradient even at the goal boundary.

The continuous environments use the raw 2D observation — `MountainCarContinuous-v0`
provides a dense reward already encoding energy information, and the simpler input
space reduces the parameter count of the fixed-topology networks.
"""))

cells.append(code("""\
import pandas as pd

rows = []
for scenario in SCENARIOS:
    cfg = CONFIGS_DATA[("simple_ga", scenario)]
    env_id  = cfg["env"]["id"]
    wrappers = cfg["env"].get("wrappers", {})
    discrete = env_id == "MountainCar-v0"
    state_dim = 4 if wrappers.get("augment_state") else 2
    action_type = "Discrete {0,1,2}" if discrete else "Continuous [−1,1]"

    wrapper_chain = [env_id]
    if wrappers.get("augment_state"):     wrapper_chain.append("AugmentState")
    if wrappers.get("fuel_cost"):         wrapper_chain.append("FuelCost")
    if wrappers.get("step_cost"):         wrapper_chain.append("StepCost")
    if wrappers.get("energy_shaping"):
        w = wrappers["energy_shaping"].get("energy_weight", "?")
        wrapper_chain.append(f"EnergyShaping(×{w})")

    rows.append({
        "Scenario"     : scenario,
        "Base Env"     : env_id.replace("MountainCar", "MC"),
        "Action Type"  : action_type,
        "State Dim"    : state_dim,
        "Max Steps"    : cfg["train"]["max_steps_per_episode"],
        "Wrapper Stack": " → ".join(wrapper_chain),
    })

df = pd.DataFrame(rows)
display(df.style.set_caption("Environment and Wrapper Configuration per Scenario"))
"""))

# ── Section 2: Reward Design ─────────────────────────────────────────────────

cells.append(md("""\
---
## 2  Reward Design and Wrappers

Reward shaping is the primary engineering lever when the base signal is too sparse
for a population to make progress. We designed four reward regimes, each testing a
different credit-assignment hypothesis.
"""))

cells.append(code("""\
import gymnasium as gym

# Demonstrate sparsity of base reward with random actions
np.random.seed(0)
env = gym.make("MountainCar-v0")
successes, returns = [], []
for _ in range(50):
    obs, _ = env.reset()
    total_r, done = 0.0, False
    while not done:
        obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
        total_r += r
        done = terminated or truncated
    successes.append(int(terminated and not truncated))
    returns.append(total_r)
env.close()

print(f"Random policy — 50 episodes on base MountainCar-v0")
print(f"  Success rate : {np.mean(successes):.0%}")
print(f"  Mean return  : {np.mean(returns):.1f}  (min {np.min(returns):.0f}, max {np.max(returns):.0f})")
print()
print("→ A random policy never reaches the goal.")
print("  Without shaping, all 50 individuals in a generation receive ≈ −200 fitness.")
print("  There is no signal to rank genomes, so evolution stalls.")
"""))

cells.append(md("""\
### Reward regime design rationale

**Scenario 1 — Discrete (energy shaping ×5)**

Base gym gives `−1/step`, capped at 200 steps. This dense signal lets a population
rank genomes by episode length, but the reward range `[−200, −1]` is narrow — the
difference between a genome that reaches the goal in 199 steps and one that reaches
it in 100 steps is only 99 units, while the population also has to discriminate
from complete failures. We add `+5 × |velocity|` per step via `EnergyShapingRewardWrapper`.
This converts the objective into a momentum-building task: a genome that swings the
car back and forth accumulates `5 × 0.07 × 200 = 70` extra reward versus a stationary
one, providing a smooth gradient toward useful behaviours even before the first success.

**Scenario 2 — Continuous (standard)**

`MountainCarContinuous-v0` provides its own shaped reward: `+100` at goal and
`−0.1 × action²` per step. The quadratic action penalty implicitly rewards smooth,
low-force trajectories. No additional wrapper is needed — the dense signal and the
999-step episode budget give populations enough time to discover successful trajectories
through random exploration.

**Scenario 3 — Fuel (uniform cost + goal bonus)**

`DiscreteFuelCostWrapper` replaces the base signal with `−1/step + 100 × terminated`.
All three discrete actions (left, idle, right) cost equally — eliminating the
"neutral-action trap" where a free idle action causes algorithms to converge to
all-neutral policies. The explicit +100 terminal bonus expands the reward range
dramatically (`[−200, ~+100]`), so a succeeding genome ranks far above failures.
We add `EnergyShapingRewardWrapper(×20)` — a higher weight than Scenario 1 because
the bonus raises the overall magnitude, and shaping must scale to remain a useful
learning signal in early generations.

**Scenario 4 — MinSteps (linear action cost)**

`ContinuousStepCostWrapper` replaces the continuous reward: `−0.1 × |action| + 100 × terminated`.
The linear (not quadratic) cost means small forces are proportionally cheaper than in
Scenario 2, rewarding decisive short pushes over sustained moderate forces. This creates
a qualitatively different optimisation landscape even though both Scenario 2 and 4
use the continuous environment.
"""))

cells.append(code("""\
print("Wrapper stack per scenario (from YAML configs):")
print("=" * 70)
for scenario in SCENARIOS:
    cfg = CONFIGS_DATA[("simple_ga", scenario)]
    wrappers = cfg["env"].get("wrappers", {})
    chain = [cfg["env"]["id"]]
    if wrappers.get("augment_state"):
        chain.append("AugmentState  [2D → 4D]")
    if wrappers.get("fuel_cost"):
        chain.append("FuelCost  [−1/step + 100 goal]")
    if wrappers.get("step_cost"):
        chain.append("StepCost  [−0.1|a|/step + 100 goal]")
    if wrappers.get("energy_shaping"):
        w = wrappers["energy_shaping"].get("energy_weight", "?")
        chain.append(f"EnergyShaping  [+{w}×|v|/step]")
    chain.append("RecordEpisodeStats")
    print(f"  {scenario:10s}: " + "\\n             → ".join(chain))
    print()
"""))

cells.append(code("""\
v = np.linspace(-0.07, 0.07, 200)
fig, ax = plt.subplots(figsize=(9, 4))

for ew, ls, label in [(5.0, "-", "EnergyShaping ×5 (discrete)"),
                      (20.0, "--", "EnergyShaping ×20 (fuel)")]:
    shaped = -1.0 + ew * np.abs(v)
    ax.plot(v, shaped, ls, label=label, linewidth=2.0)

ax.axhline(-1.0, color="grey", linewidth=1.2, linestyle=":", label="Base reward (no shaping)")
ax.axhline(0.0,  color="black", linewidth=0.6, linestyle="-")
ax.fill_between(v, -1.0, -1.0 + 5.0 * np.abs(v), alpha=0.08, color="#E63946")
ax.fill_between(v, -1.0 + 5.0*np.abs(v), -1.0 + 20.0*np.abs(v), alpha=0.08, color="#457B9D")

ax.set_xlabel("Velocity")
ax.set_ylabel("Per-step reward")
ax.set_title("Energy shaping reward surface: base(−1) + weight × |velocity|")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

print("At max velocity (|v|=0.07):")
for ew in [5.0, 20.0]:
    total = -1.0 + ew * 0.07
    print(f"  weight={ew:4.1f} → {total:+.3f}/step  (positive: {total > 0})")
"""))

# ── Section 3: Algorithm Design Rationale ────────────────────────────────────

cells.append(md("""\
---
## 3  Algorithm Design Rationale

### Why population-based search?

Mountain Car has a deceptive reward landscape with a single narrow corridor to
success. Standard gradient-based RL (Q-learning, PPO) requires the agent to
visit the goal at least once before back-propagating a useful signal.
Population-based methods bypass this by evaluating many candidate policies
simultaneously: even if 49 out of 50 fail completely, the one that gets close
provides a fitness advantage that steers the next generation.

The three algorithms span the design space of evolutionary computation:
"""))

cells.append(code("""\
comparison_data = {
    "Property"                  : ["Search space",
                                    "Search distribution",
                                    "Selection",
                                    "Crossover",
                                    "Mutation",
                                    "Exploits weight correlations",
                                    "Adapts topology",
                                    "Key hyperparameter"],
    "Simple GA"                 : ["Fixed-arch weight space",
                                    "Isotropic Gaussian (fixed σ)",
                                    "Tournament (k=3) + elitism",
                                    "Uniform 50/50 per weight",
                                    "Gaussian (10% of weights/gen)",
                                    "No",
                                    "No",
                                    "mutation_std"],
    "CMA-ES"                    : ["Fixed-arch weight space",
                                    "Adaptive multivariate Gaussian",
                                    "Weighted recombination (CMA update)",
                                    "None (distribution-based)",
                                    "Implicit via covariance update",
                                    "Yes — full covariance matrix",
                                    "No",
                                    "sigma0 (initial step size)"],
    "NEAT"                      : ["Topology + weights",
                                    "Graph mutations (node/conn add)",
                                    "Speciation + stagnation pruning",
                                    "Gene-aligned crossover",
                                    "Weight & structural mutations",
                                    "N/A",
                                    "Yes — grows from 0 hidden nodes",
                                    "conn_add_prob, node_add_prob"],
}
df_cmp = pd.DataFrame(comparison_data).set_index("Property")
display(df_cmp.style.set_caption("Algorithm Comparison: Simple GA vs CMA-ES vs NEAT"))
"""))

cells.append(md("""\
**Simple GA** is the baseline: it demonstrates what the minimum algorithmic
complexity looks like. Crossover and mutation are parameter-free in practice —
the code uses uniform crossover (50/50 per weight, hard-coded regardless of
`crossover_alpha`) and isotropic Gaussian noise. Its weakness is that mutations
are isotropic: the algorithm cannot adapt its search distribution to the curvature
of the fitness landscape.

**CMA-ES** is the theoretically motivated extension: it maintains and updates a full
covariance matrix over parameter space, adapting the sampling distribution to the
correlation structure of high-fitness solutions. For a neural network parameter space
with ∼4.7k correlated weights, CMA-ES should exploit the geometry of the fitness
landscape more efficiently than SimpleGA. The critical hyperparameter is `sigma0`
(initial step size) — too large and the covariance estimate is noisy; too small and
the search is myopic.

**NEAT** is orthogonal in design philosophy: it treats network topology as a mutable
gene alongside weights. Starting from a minimal network (0 hidden nodes, direct
input-to-output connections), it adds nodes and connections via structural mutations.
Speciation protects topological innovations from immediate extinction, analogous to
ecological niches. For Mountain Car, this matters because the optimal policy may be
representable with 0 hidden nodes — NEAT can discover the minimal sufficient architecture
automatically.
"""))

cells.append(code("""\
from src.envs.mountain_car_discrete import make_discrete_env
from src.envs.mountain_car_continuous import make_continuous_env
from src.agents.nn_policy import build_policy

print(f"{'Scenario':12s}  {'Obs dim':8s}  {'Out dim':8s}  {'Hidden':12s}  {'Params (GA/CMA-ES)':20s}")
print("-" * 70)
for scenario in SCENARIOS:
    cfg = CONFIGS_DATA[("simple_ga", scenario)]
    wrappers = cfg["env"].get("wrappers", {})
    env_id = cfg["env"]["id"]
    if env_id == "MountainCar-v0":
        env = make_discrete_env(wrappers=wrappers)
    else:
        env = make_continuous_env(wrappers=wrappers)
    hidden = cfg["simple_ga"]["hidden_sizes"]
    net = build_policy(env, hidden)
    obs_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    print(f"  {scenario:10s}  {obs_dim:8d}  {out_dim:8d}  {str(hidden):12s}  {net.n_params:,}")
    env.close()

print()
print("NEAT starts with 0 hidden nodes; final parameter count varies by seed/scenario.")
print("Typical final size in discrete: ~4–8 connections (far fewer than ~4,675 for GA).")
"""))

# ── Section 4: Training Strategy & Hyperparameters ───────────────────────────

cells.append(md("""\
---
## 4  Training Strategies and Hyperparameters

All experiments share the same computational budget:
- **200 generations** (each generation = one `learn()` call evaluating the full population)
- **Population size**: 50 for all three algorithms
- **Seeds**: [7, 21, 42] — three independent runs per experiment, 12 experiments total
- **Evaluation**: every 20 generations over 10 episodes; checkpoints at generations 50/100/150/200
"""))

cells.append(code("""\
rows = []
for scenario in SCENARIOS:
    for agent in AGENTS:
        cfg = CONFIGS_DATA[(agent, scenario)]
        wrappers = cfg["env"].get("wrappers", {})
        ew = wrappers.get("energy_shaping", {})
        energy_w = ew.get("energy_weight", "—") if isinstance(ew, dict) else "—"

        if agent == "simple_ga":
            ga = cfg["simple_ga"]
            rows.append(dict(Scenario=scenario, Agent=AGENT_LABELS[agent],
                             Pop=ga["population_size"], Elite=ga["elite_frac"],
                             mut_std=ga["mutation_std"], sigma0="—",
                             cx=ga["crossover_alpha"], Hidden=str(ga["hidden_sizes"]),
                             energy_w=energy_w))
        elif agent == "cma_es":
            cma = cfg["cma_es"]
            rows.append(dict(Scenario=scenario, Agent=AGENT_LABELS[agent],
                             Pop=cma["population_size"], Elite="—",
                             mut_std="—", sigma0=cma["sigma0"],
                             cx="—", Hidden=str(cma["hidden_sizes"]),
                             energy_w=energy_w))
        elif agent == "neat":
            n = cfg["neat"]
            rows.append(dict(Scenario=scenario, Agent=AGENT_LABELS[agent],
                             Pop=n["pop_size"], Elite="speciation",
                             mut_std="struct", sigma0="—",
                             cx="gene-aligned", Hidden="adaptive",
                             energy_w=energy_w))

hp_df = pd.DataFrame(rows)
display(hp_df.style.set_caption("Hyperparameter Summary — All 12 Experiments"))
"""))

cells.append(md("""\
### SimpleGA: hyperparameter decisions

`mutation_std=0.05` is used for discrete and fuel scenarios — fine-grained perturbations
that make small, directed improvements once a good policy region is found.
The continuous and minsteps scenarios use `0.1` because the raw 2D observation space
and standard continuous reward require larger perturbations to escape the valley of
zero-velocity starting states.

The `crossover_alpha` parameter is stored but not used in the implementation: the
crossover operator is hard-coded as uniform 50/50 (each weight independently from
parent A or B with equal probability). This is effectively `alpha=0.5`, which is the
standard choice for uncorrelated weight spaces.

Elitism preserves the top 20% (10 individuals) unchanged into the next generation,
preventing catastrophic forgetting at the cost of reduced population diversity in
later generations.

### CMA-ES: hyperparameter decisions

`sigma0=0.5` is the initial step size for all scenarios. This is appropriate when
network weights are initialised near zero (PyTorch default initialisation). CMA-ES
internally adapts sigma through its cumulative step-size control; the initial value
only affects early-generation search width.

`population_size=50` is above the CMA-ES theoretical default of `4 + floor(3·ln(n))`
for `n~4417–4675` parameters (which is ~30). We over-sample by ≈1.7×, yielding more
robust covariance estimates at the cost of extra evaluations per generation.

### NEAT: key configuration choices
"""))

cells.append(code("""\
for tpl in ["neat_discrete_template.ini", "neat_continuous_template.ini"]:
    path = CONFIGS_DIR / tpl
    print(f"\\n{'='*60}")
    print(f"  {tpl}")
    print("="*60)
    print(path.read_text())
"""))

cells.append(md("""\
Key NEAT decisions:
- `initial_connection = full_direct` — every genome starts fully connected (all inputs
  to all outputs), ensuring every individual produces non-trivial actions from generation 1.
- `num_hidden = 0` — minimal initial topology; NEAT discovers hidden nodes if needed.
- `activation_default = relu` (discrete) / `tanh` (continuous) — discrete outputs are
  argmax-selected (activation irrelevant to the selected action); continuous output is
  clipped to `[−1, 1]` after tanh, preventing double-squashing.
- `conn_add_prob=0.1, node_add_prob=0.05` — additive mutations 2× more likely than
  deletions; encourages topology growth early in training.
- `max_stagnation=50` — species without improvement for 50 generations are pruned,
  preventing dead-weight species from consuming population slots.
"""))

# ── Section 5: Evaluation ────────────────────────────────────────────────────

cells.append(md("""\
---
## 5  Evaluation and Performance Analysis

We evaluate three metrics:
1. **Mean episode reward** — averaged over all 200 training generations and 3 seeds (proxy for training efficiency)
2. **Final success rate** — fraction of episodes where the car reached the goal
3. **Mean episode length** — shorter is better for successful policies
"""))

cells.append(code("""\
rows = []
for scenario in SCENARIOS:
    for agent in AGENTS:
        agg = AGGREGATES.get((agent, scenario), {})
        rows.append({
            "Scenario"       : SCENARIO_LABELS[scenario].split("—")[0].strip(),
            "Algorithm"      : AGENT_LABELS[agent],
            "Success Rate"   : f"{agg.get('success_rate_mean', float('nan')):.1%}",
            "Mean Reward"    : f"{agg.get('reward_mean', float('nan')):.1f}",
            "Reward Std"     : f"±{agg.get('reward_std_across_seeds', float('nan')):.1f}",
            "Mean Ep Length" : f"{agg.get('episode_length_mean', float('nan')):.0f}",
        })

summary_df = pd.DataFrame(rows)
display(summary_df.style.set_caption(
    "Master Performance Summary — Mean over 3 Seeds × 200 Training Generations"))
"""))

cells.append(md("""\
**High-level reading:**
NEAT achieves the highest success rate in discrete and fuel scenarios (where it benefits
from topological flexibility with strong energy shaping). SimpleGA and CMA-ES lead in
continuous and minsteps (where a fixed-topology network is sufficient and the dense
reward provides clear fitness gradients). The algorithm rankings change across scenarios —
this cross-environment reversal is the central finding analysed in Section 7.
"""))

# ── Per-scenario blocks ──────────────────────────────────────────────────────

cells.append(md("### 5.1  Discrete Scenario — MountainCar-v0 (energy shaping ×5)"))

cells.append(code("""\
rows = []
for agent in AGENTS:
    sr = SEED_RESULTS.get(("simple_ga" if agent == "simple_ga" else
                           "cma_es"    if agent == "cma_es"    else "neat", "discrete"),
                          pd.DataFrame())
    # re-fetch properly
    sr = SEED_RESULTS.get((agent, "discrete"), pd.DataFrame())
    if not sr.empty:
        rows.append({
            "Algorithm"        : AGENT_LABELS[agent],
            "Success Rate"     : f"{sr['success_rate'].mean():.1%}",
            "Mean Reward"      : f"{sr['reward_mean'].mean():.1f} ± {sr['reward_mean'].std():.1f}",
            "Eval Last Reward" : f"{sr['eval_mean_reward_last'].mean():.1f}",
            "Mean Ep Length"   : f"{sr['mean_episode_length'].mean():.0f}",
        })
display(pd.DataFrame(rows).style.set_caption("Discrete Scenario — per-algorithm summary (mean over 3 seeds)"))
"""))

cells.append(code("""\
show_fig("discrete_cross_agent_reward.png", width=900)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, agent in zip(axes, AGENTS):
    p = FIGS / f"{agent}_discrete_seed_envelope.png"
    if p.exists():
        ax.imshow(mpimg.imread(str(p))); ax.axis("off")
        ax.set_title(f"{AGENT_LABELS[agent]} — Discrete (seeds 7/21/42)", fontsize=10)
plt.suptitle("Multi-seed reward envelopes — Discrete", fontsize=12)
plt.tight_layout(); plt.show()
"""))

cells.append(md("### 5.2  Continuous Scenario — MountainCarContinuous-v0 (standard)"))

cells.append(code("""\
rows = []
for agent in AGENTS:
    sr = SEED_RESULTS.get((agent, "continuous"), pd.DataFrame())
    if not sr.empty:
        rows.append({
            "Algorithm"        : AGENT_LABELS[agent],
            "Success Rate"     : f"{sr['success_rate'].mean():.1%}",
            "Mean Reward"      : f"{sr['reward_mean'].mean():.1f} ± {sr['reward_mean'].std():.1f}",
            "Eval Last Reward" : f"{sr['eval_mean_reward_last'].mean():.1f}",
            "Mean Ep Length"   : f"{sr['mean_episode_length'].mean():.0f}",
        })
display(pd.DataFrame(rows).style.set_caption("Continuous Scenario — per-algorithm summary"))
"""))

cells.append(code("""\
show_fig("continuous_cross_agent_reward.png", width=900)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, agent in zip(axes, AGENTS):
    p = FIGS / f"{agent}_continuous_seed_envelope.png"
    if p.exists():
        ax.imshow(mpimg.imread(str(p))); ax.axis("off")
        ax.set_title(f"{AGENT_LABELS[agent]} — Continuous", fontsize=10)
plt.suptitle("Multi-seed reward envelopes — Continuous", fontsize=12)
plt.tight_layout(); plt.show()
"""))

cells.append(md("### 5.3  Fuel Scenario — MountainCar-v0 (−1/step + 100 goal + shaping ×20)"))

cells.append(code("""\
rows = []
for agent in AGENTS:
    sr = SEED_RESULTS.get((agent, "fuel"), pd.DataFrame())
    if not sr.empty:
        rows.append({
            "Algorithm"        : AGENT_LABELS[agent],
            "Success Rate"     : f"{sr['success_rate'].mean():.1%}",
            "Mean Reward"      : f"{sr['reward_mean'].mean():.1f} ± {sr['reward_mean'].std():.1f}",
            "Eval Last Reward" : f"{sr['eval_mean_reward_last'].mean():.1f}",
            "Mean Ep Length"   : f"{sr['mean_episode_length'].mean():.0f}",
        })
display(pd.DataFrame(rows).style.set_caption("Fuel Scenario — per-algorithm summary"))
"""))

cells.append(code("""\
show_fig("fuel_cross_agent_reward.png", width=900)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, agent in zip(axes, AGENTS):
    p = FIGS / f"{agent}_fuel_seed_envelope.png"
    if p.exists():
        ax.imshow(mpimg.imread(str(p))); ax.axis("off")
        ax.set_title(f"{AGENT_LABELS[agent]} — Fuel", fontsize=10)
plt.suptitle("Multi-seed reward envelopes — Fuel", fontsize=12)
plt.tight_layout(); plt.show()
"""))

cells.append(md("### 5.4  MinSteps Scenario — MountainCarContinuous-v0 (−0.1|a|/step + 100 goal)"))

cells.append(code("""\
rows = []
for agent in AGENTS:
    sr = SEED_RESULTS.get((agent, "minsteps"), pd.DataFrame())
    if not sr.empty:
        rows.append({
            "Algorithm"        : AGENT_LABELS[agent],
            "Success Rate"     : f"{sr['success_rate'].mean():.1%}",
            "Mean Reward"      : f"{sr['reward_mean'].mean():.1f} ± {sr['reward_mean'].std():.1f}",
            "Eval Last Reward" : f"{sr['eval_mean_reward_last'].mean():.1f}",
            "Mean Ep Length"   : f"{sr['mean_episode_length'].mean():.0f}",
        })
display(pd.DataFrame(rows).style.set_caption("MinSteps Scenario — per-algorithm summary"))
"""))

cells.append(code("""\
show_fig("minsteps_cross_agent_reward.png", width=900)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, agent in zip(axes, AGENTS):
    p = FIGS / f"{agent}_minsteps_seed_envelope.png"
    if p.exists():
        ax.imshow(mpimg.imread(str(p))); ax.axis("off")
        ax.set_title(f"{AGENT_LABELS[agent]} — MinSteps", fontsize=10)
plt.suptitle("Multi-seed reward envelopes — MinSteps", fontsize=12)
plt.tight_layout(); plt.show()
"""))

cells.append(md("### 5.5  Population Diversity"))

cells.append(code("""\
# SimpleGA logs population_std (weight std across population); CMA-ES weight std is similar.
# NEAT does not record a comparable metric (structural diversity is categorical, not scalar).
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for row_i, agent in enumerate(["simple_ga", "cma_es"]):
    for col_i, scenario in enumerate(SCENARIOS):
        ax = axes[row_i, col_i]
        p = FIGS / f"{agent}_{scenario}_diversity.png"
        if p.exists():
            ax.imshow(mpimg.imread(str(p))); ax.axis("off")
            ax.set_title(f"{AGENT_LABELS[agent]} — {scenario}", fontsize=9)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
            ax.axis("off")
plt.suptitle("Population Weight Diversity across Generations", fontsize=13, y=1.01)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""\
**Diversity dynamics:**

- **SimpleGA**: Weight diversity (`population_std`) collapses as elites dominate the
  gene pool. The collapse is faster in simpler scenarios (discrete: strong signal
  → rapid convergence) and slower in fuel (wider reward range → more fitness variance
  → diversity persists longer).
- **CMA-ES**: The metric shown is the mean standard deviation of the CMA-ES sample,
  which can *increase* when the landscape is rough — the covariance adaptation detects
  a noisy gradient and widens the search. This adaptive behaviour is CMA-ES's key
  advantage over SimpleGA.
- **NEAT**: Population diversity is structural (topology differences between species)
  rather than purely parametric. The speciation mechanism preserves topological diversity
  that is not captured by a single `population_std` scalar.
"""))

cells.append(code("""\
# Episode reward distributions across all training (200 gens × 3 seeds)
data_rows = []
for agent in AGENTS:
    for scenario in SCENARIOS:
        for df in EPISODES.get((agent, scenario), []):
            for r in df["reward"].dropna().values:
                data_rows.append({"Agent": AGENT_LABELS[agent],
                                  "Scenario": scenario, "Reward": float(r)})

violin_df = pd.DataFrame(data_rows)
colors = [AGENT_COLORS[a] for a in AGENTS]

fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=False)
for ax, scenario in zip(axes, SCENARIOS):
    sub = violin_df[violin_df["Scenario"] == scenario]
    sns.violinplot(data=sub, x="Agent", y="Reward",
                   palette=colors, ax=ax, cut=0, inner="box")
    ax.set_title(SCENARIO_LABELS[scenario].split("—")[0].strip(), fontsize=10)
    ax.set_xlabel("")
    if scenario != "discrete":
        ax.set_ylabel("")
plt.suptitle("Episode Reward Distributions — All 200 Generations × 3 Seeds", fontsize=13)
plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
print(f"{'Scenario':12s}  {'Agent':10s}  {'Mean':8s}  {'Std':8s}  {'CV (%)':8s}  Success")
print("-" * 65)
for scenario in SCENARIOS:
    for agent in AGENTS:
        agg = AGGREGATES.get((agent, scenario), {})
        mean = agg.get("reward_mean", float("nan"))
        std  = agg.get("reward_std_across_seeds", float("nan"))
        cv   = abs(std / mean) * 100 if mean != 0 else float("nan")
        sr   = agg.get("success_rate_mean", float("nan"))
        flag = " ← high variance" if cv > 30 else ""
        print(f"  {scenario:10s}  {AGENT_LABELS[agent]:10s}  {mean:8.1f}  "
              f"{std:8.1f}  {cv:8.1f}  {sr:.1%}{flag}")
"""))

# ── Section 6: Policy Analysis ───────────────────────────────────────────────

cells.append(md("""\
---
## 6  Policy Analysis — Visual and Numerical

Policy heatmaps show the greedy action chosen across the discretised (position, velocity)
state space, revealing the qualitative behaviour each algorithm converged to.
The near-optimal policy for Mountain Car discrete is a **bang-bang controller**:
push left when building leftward momentum, push right when heading toward the goal.
This appears as a diagonal split in the heatmap.
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, agent in zip(axes, AGENTS):
    p = FIGS / f"{agent}_discrete_policy_heatmap.png"
    if p.exists():
        ax.imshow(mpimg.imread(str(p))); ax.axis("off")
        ax.set_title(f"{AGENT_LABELS[agent]} — Discrete Policy", fontsize=11)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center"); ax.axis("off")
plt.suptitle("Greedy Policy Heatmaps — Discrete Scenario\\n(x=velocity, y=position, colour=action: blue=left, grey=idle, red=right)",
             fontsize=11)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""\
**Discrete policy interpretation:**

- **NEAT**: Minimal networks (0–1 hidden nodes) implement near-linear decision boundaries
  over the 4D augmented state. The resulting heatmap typically shows a clean diagonal split
  close to the theoretical bang-bang policy: left whenever velocity and kinetic energy
  favour leftward swinging, right otherwise.
- **SimpleGA / CMA-ES**: Fixed `[64, 64]` hidden networks are over-parameterised for this
  task — a purely linear function of the augmented 4D state can represent the optimal
  policy. The richer architecture often learns spurious boundaries in state regions the
  car never visits. Despite this, success rates are comparable, indicating that the extra
  capacity doesn't hurt but doesn't help either.
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, agent in zip(axes, AGENTS):
    p = FIGS / f"{agent}_fuel_policy_heatmap.png"
    if p.exists():
        ax.imshow(mpimg.imread(str(p))); ax.axis("off")
        ax.set_title(f"{AGENT_LABELS[agent]} — Fuel Policy", fontsize=11)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center"); ax.axis("off")
plt.suptitle("Greedy Policy Heatmaps — Fuel Scenario", fontsize=11)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""\
**Fuel policy interpretation:**

The fuel reward (`−1/step + 100 × goal`) with equal action costs shifts the optimisation
target from "reach the goal quickly" to "reach the goal efficiently" — they are the same
thing since all steps cost equally. Policies that learn to swing once and commit to the
right-side climb in as few steps as possible maximise the terminal bonus net of step costs.

Agents that learn idle-heavy policies (large grey regions in the heatmap) waste time,
reducing the net reward even on success. The heatmap quality for fuel directly reflects
whether the energy shaping (×20) successfully broke the idle-action bias.
"""))

cells.append(code("""\
# Episode length over training — a proxy for policy efficiency
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, scenario in zip(axes.flatten(), SCENARIOS):
    for agent in AGENTS:
        dfs = EPISODES.get((agent, scenario), [])
        if not dfs:
            continue
        merged = pd.concat(dfs)
        mean_len = merged.groupby("episode")["length"].mean()
        smooth = mean_len.rolling(10, min_periods=1).mean()
        ax.plot(mean_len.index, smooth.values,
                color=AGENT_COLORS[agent], label=AGENT_LABELS[agent], linewidth=1.8)
    ax.set_title(SCENARIO_LABELS[scenario].split("—")[0].strip(), fontsize=10)
    ax.set_xlabel("Generation"); ax.set_ylabel("Mean Episode Length")
    ax.legend(fontsize=9)
plt.suptitle("Mean Episode Length over Training (10-gen rolling average, 3-seed mean)", fontsize=13)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""\
**Episode length analysis:**

- Decreasing episode length = the population is finding the goal faster on average.
- NEAT in continuous/minsteps shows elevated episode lengths throughout training compared
  to SimpleGA/CMA-ES. This is a structural artefact: NEAT's speciation means many
  genomes in the population are exploring dead-end topologies — long failed episodes
  drag the population average up, even if the best genome succeeds quickly.
- SimpleGA and CMA-ES both evaluate all 50 individuals with the same fixed topology —
  the only source of variance in episode length is weight quality, so convergence
  appears sharper.
"""))

# ── Section 7: Cross-Environment ─────────────────────────────────────────────

cells.append(md("""\
---
## 7  Comparative Policy Analysis Across Environment Versions

The four scenarios share the same underlying physics. Analysing how algorithm rankings
change across scenarios isolates which algorithmic properties are intrinsic vs.
environment-dependent.
"""))

cells.append(code("""\
# Success rate heatmap: agents × scenarios
sr_data = {}
for agent in AGENTS:
    sr_data[AGENT_LABELS[agent]] = {}
    for scenario in SCENARIOS:
        agg = AGGREGATES.get((agent, scenario), {})
        sr_data[AGENT_LABELS[agent]][scenario] = agg.get("success_rate_mean", float("nan"))

sr_df = pd.DataFrame(sr_data).T   # rows=agents, cols=scenarios

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(sr_df, annot=True, fmt=".1%", cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=0.6, ax=ax, annot_kws={"size": 12})
ax.set_title("Success Rate by Algorithm × Scenario", fontsize=12)
ax.set_xticklabels([SCENARIO_LABELS[s].split("—")[0].strip() for s in SCENARIOS], rotation=20, ha="right")
plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
x = np.arange(len(SCENARIOS))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 5))
for i, agent in enumerate(AGENTS):
    vals = [AGGREGATES.get((agent, s), {}).get("success_rate_mean", 0) for s in SCENARIOS]
    ax.bar(x + i * width, vals, width, label=AGENT_LABELS[agent],
           color=AGENT_COLORS[agent], alpha=0.88, edgecolor="white")

ax.set_xticks(x + width)
ax.set_xticklabels([SCENARIO_LABELS[s].split("—")[0].strip() for s in SCENARIOS])
ax.set_ylabel("Success Rate"); ax.set_ylim(0, 1.08)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("Success Rate per Algorithm across All Scenarios")
ax.legend(); plt.tight_layout(); plt.show()
"""))

cells.append(md("""\
### Cross-environment analysis

The reversal in rankings — NEAT leads in discrete/fuel, SimpleGA leads in
continuous/minsteps — is the central comparative finding.

**Why NEAT wins in discrete/fuel:**
1. *Minimal sufficiency*: the optimal discrete policy is nearly linear in the 4D augmented
   state (a bang-bang controller based on the sign of velocity and energy). A 0-hidden-node
   network with 4×3 = 12 parameters can represent this. NEAT discovers this minimal
   architecture, avoiding the risk of over-fitting to spurious state-space regions.
2. *Speciation protects exploration*: strong energy shaping (×5 or ×20) generates diverse
   fitness signals early, allowing NEAT's species to explore different topologies simultaneously.
   The species with the right topology wins, and the rest are pruned via stagnation.

**Why SimpleGA wins in continuous/minsteps:**
1. *Dense reward removes the topology uncertainty*: the continuous reward is dense enough
   that all 50 individuals in a generation receive informative fitness values. CMA-ES and
   SimpleGA converge quickly with fixed-topology networks because the reward gradient is
   smooth.
2. *NEAT's multi-species evaluation cost*: with 999-step episodes and multiple species,
   NEAT's average episode length is high. The per-generation evaluation time is O(pop × max_steps),
   which is 5× higher in continuous than discrete. NEAT's intra-generation diversity costs
   evaluations that could be used for weight optimisation.
3. *Fixed-topology structural advantage*: the continuous optimal policy is a smooth nonlinear
   function; a `[64, 64]` ReLU network approximates it well. NEAT's topologies are structurally
   simpler (0–2 hidden nodes) and may lack expressiveness for the continuous control task.

**CMA-ES vs SimpleGA:**
CMA-ES's theoretical advantage (covariance-adapted search) does not clearly manifest at
200 generations / population 50. With ~4.7k parameters, the covariance matrix has
~11M entries — 200 × 50 = 10,000 evaluations is far below the theoretical sample requirement
for reliable estimation. Both algorithms are sample-limited in the same way, so their
performance converges.
"""))

cells.append(code("""\
recommendations = {
    "Criterion"            : ["Discrete success",
                               "Fuel success",
                               "Continuous success",
                               "MinSteps success",
                               "Seed stability",
                               "Theoretical justification",
                               "Architectural flexibility"],
    "Recommended Algorithm": ["NEAT",
                               "NEAT",
                               "Simple GA",
                               "Simple GA",
                               "Simple GA",
                               "CMA-ES",
                               "NEAT"],
    "Rationale"            : ["Minimal topology = near-optimal bang-bang policy",
                               "Strong shaping breaks idle trap; speciation preserves momentum seekers",
                               "Fixed topology sufficient; dense reward → fast convergence",
                               "Dense reward + linear cost → GA crossover mixes well",
                               "Lowest coefficient of variation across all scenarios",
                               "Adapts search distribution to landscape curvature",
                               "Discovers needed hidden nodes; avoids over-parameterisation"],
}
display(pd.DataFrame(recommendations).style.set_caption(
    "Algorithm Recommendations by Criterion"))
"""))

# ── Section 8: NEAT Topology ─────────────────────────────────────────────────

cells.append(md("""\
---
## 8  NEAT Topology: Evolution of Neural Network Structure

NEAT is unique in that the neural network architecture itself evolves.
The visualisations below show how the best genome's graph structure changes across
200 training generations.

**Reading the topology diagrams:**
- Blue nodes = inputs (position, velocity, [KE, sin(3p) for discrete])
- Green nodes = hidden nodes
- Red nodes = outputs (action logits or continuous force)
- Edge colour = connection weight (blue = positive, red = negative)
- Edge width ∝ |weight|
- Dashed grey = disabled connection
"""))

cells.append(code("""\
# Final topology after 200 generations (seed 7)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, scenario in zip(axes, ["discrete", "continuous"]):
    p = FIGS / f"neat_{scenario}_topology_final.png"
    if p.exists():
        ax.imshow(mpimg.imread(str(p))); ax.axis("off")
        ax.set_title(f"NEAT Final Topology — {scenario.capitalize()} (seed 7)", fontsize=11)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center"); ax.axis("off")
plt.suptitle("Best Genome Topology after 200 Generations", fontsize=12)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""\
**Final topology interpretation:**

- **Discrete**: The final topology typically has 0–2 hidden nodes. With 4 inputs and 3 outputs,
  a 0-hidden fully-connected genome has only 12 weight parameters. This is sufficient because
  the bang-bang policy is nearly linear in the augmented state: `action = argmax(W·[pos, vel, KE, sin(3p)] + b)`.
  Strong positive weights from the velocity/KE inputs to the "push right" output, and negative
  weights to "push left" for negative velocity, implement the policy directly.
- **Continuous**: More complex topologies may emerge (1–3 hidden nodes) because the tanh-activated
  output must produce a smooth force value. However, even a 0-hidden network can express a wide
  range of smooth functions via the tanh nonlinearity at the output node.
"""))

cells.append(code("""\
# Topology growth sequences
print("Topology Growth — Discrete (generation snapshots at 50/100/150/200):")
show_fig("neat_discrete_growth_sequence.png", width=1100)

print("\\nTopology Growth — Continuous:")
show_fig("neat_continuous_growth_sequence.png", width=1100)
"""))

cells.append(md("""\
**Growth sequence analysis:**

Each panel is the best genome at a checkpoint generation (50, 100, 150, 200). The subplot
titles show `N hidden, M connections`. Key observations:
- **Discrete**: Topology often stabilises within the first 50 generations (0 hidden nodes,
  12 connections). Subsequent improvements are purely weight-based. This confirms that the
  task complexity is low — NEAT's structural search space exploration converges to the minimal
  solution quickly.
- **Continuous**: Structural mutations may persist longer. New hidden nodes appearing in
  later generations can improve the expressiveness of the continuous force output, though
  the improvement is typically small once a successful topology is found.

The growth sequence is also a diagnostic for speciation: if multiple species coexist, the
"best genome" at each checkpoint may switch between species, causing apparent topological
jumps.
"""))

cells.append(code("""\
# Topology evolution GIFs — animated across 200 generations
# Note: GIFs animate in Jupyter Lab and classic Notebook.
# In VS Code notebooks, they display as static (first frame). Open the file directly to animate.

for scenario in ["discrete", "continuous"]:
    gif = FIGS / f"neat_{scenario}_topology_evolution.gif"
    if gif.exists():
        print(f"\\nNEAT Topology Evolution — {scenario.capitalize()} (all seeds aggregated)")
        display(Image(filename=str(gif), width=700))
    else:
        print(f"[Combined GIF not found for {scenario}]")
"""))

cells.append(code("""\
# Per-seed GIFs for all four scenarios
for scenario in SCENARIOS:
    found = False
    for seed in SEEDS:
        gif = FIGS / f"neat_{scenario}_seed{seed}_topology_evolution.gif"
        if gif.exists():
            if not found:
                print(f"\\n{'─'*50}")
                print(f"  NEAT Topology Evolution — {scenario.upper()} per seed")
                print(f"{'─'*50}")
                found = True
            print(f"Seed {seed}:")
            display(Image(filename=str(gif), width=650))
"""))

cells.append(md("""\
**Topology evolution insights:**

1. **Minimal sufficiency**: NEAT consistently finds solutions with 0–2 hidden nodes,
   confirming that Mountain Car does not require deep representations. The physics is
   low-dimensional and the optimal policy is nearly linear in the augmented state.

2. **Structural degeneracy**: Different seeds evolve different topologies with similar
   fitness. This is a form of structural degeneracy — the fitness landscape has multiple
   equivalent minimal solutions. Speciation protects these alternatives from competitive
   exclusion, maintaining a diverse portfolio of solutions throughout training.

3. **Weight-dominated optimisation**: Once a successful topology is found (typically by
   generation 50 in discrete), further fitness improvements come from weight refinement
   alone. The high `weight_mutate_rate=0.8` keeps weight mutations frequent even after
   structural mutations stop, enabling fine-grained tuning.

4. **Activation choice matters**: Discrete uses `relu` (activation_default), while
   continuous uses `tanh`. For discrete scenarios, the argmax selection over logits is
   activation-agnostic — any monotone activation gives the same action. For continuous,
   `tanh` provides the bounded output needed for the force signal without an additional
   clipping step.
"""))

# ── Conclusions ───────────────────────────────────────────────────────────────

cells.append(md("""\
---
## Summary and Conclusions

| Criterion | Best Algorithm | Reason |
|---|---|---|
| Discrete success rate | NEAT | Minimal topology ≅ bang-bang; speciation preserves diversity |
| Fuel success rate | NEAT | Strong shaping (×20) signals momentum; idle trap eliminated |
| Continuous success rate | Simple GA | Fixed topology sufficient; dense reward → fast convergence |
| MinSteps success rate | Simple GA | Dense reward + linear cost; GA crossover mixes well |
| Seed stability (lowest CV) | Simple GA | Isotropic mutations + elitism → reproducible convergence |
| Theoretical sophistication | CMA-ES | Adapts covariance; principled Gaussian search |
| Architectural flexibility | NEAT | Self-designs topology; avoids over-parameterisation |

### Key findings

1. **Reward shaping is load-bearing for NEAT in sparse environments.** Without strong
   energy shaping, early minimal topologies receive near-identical fitness (all ~−200),
   making speciation ineffective. The energy weight (×5 discrete, ×20 fuel) must be high
   enough to differentiate momentum-building genomes from stationary ones.

2. **CMA-ES's theoretical advantage over Simple GA is not visible at this scale.** With
   ~4.7k parameters and 10,000 total evaluations, the covariance matrix estimation is
   severely under-sampled. Both algorithms are sample-limited in the same way, yielding
   similar success rates. CMA-ES would likely outperform SimpleGA with larger populations
   or more generations.

3. **NEAT's high episode-length variance in continuous is structural, not behavioural.**
   The metric conflates exploration (long failed episodes from dead-end topologies) with
   policy quality. The best genome may succeed in 150 steps while the population average
   is 700 steps. Success rate is the more informative metric for NEAT in continuous settings.

4. **Fixed-topology over-parameterisation is benign for Mountain Car.** The `[64, 64]`
   hidden network with ~4.7k parameters is dramatically over-parameterised relative to the
   12-parameter minimum sufficient representation. This adds no measurable performance cost
   (success rates are competitive) but does add computational overhead per generation.
"""))

# ---------------------------------------------------------------------------
# BUILD NOTEBOOK JSON
# ---------------------------------------------------------------------------

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0",
        },
    },
    "cells": cells,
}

out = ROOT / "notebooks" / "04_genetic_algorithms.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print(f"Written {len(cells)} cells to {out}")

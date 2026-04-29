# Q-learning and SARSA — Implementation & Usage (Detailed)

This document explains the tabular Q-learning and SARSA implementations in this repository, how they differ, how to run them, tune them, visualize trained policies, and interpret the results produced by the training scripts.

## High-level summary

- Algorithms:
  - Q-learning — off-policy TD control using the max over next-state action values as the bootstrap target.
  - SARSA — on-policy TD control using the next executed action as the bootstrap target.
- Representation: tabular 2D grid obtained by `DiscretizeStateWrapper` that maps continuous `(position, velocity)` to an `n_bins x n_bins` index grid. Policies are stored as a 2D (state) × actions table (for bookkeeping we save greedy action maps and full Q-tables).

## Where the code lives

- Q-learning agent and trainer: `src/q_learning/agent.py`, `src/q_learning/train.py`
- SARSA agent and trainer: `src/sarsa/agent.py`, `src/sarsa/train.py`
- Discretization and wrappers: `src/envs/wrappers.py`, `src/envs/state_utils.py`
- Rollout/visualization: `src/visualization/mountain_car_rollout.py`
- Example configs: `configs/q_learning_separate.yaml`, `configs/q_learning_tuned.yaml`, `configs/sarsa_discrete.yaml`, `configs/sarsa_tuned.yaml`

When referring to these files in commands or examples below, use the same relative paths from the repository root.

## Algorithm details (as implemented)

- Q-learning update (off-policy):

  Q(s,a) := Q(s,a) + alpha _ (r + gamma _ max_a' Q(s',a') - Q(s,a))

  In the code: `QLearningAgent.update(state, action, reward, next_state, done)` computes the $
  r + \gamma \max_{a'} Q(s',a')$ target and performs an in-place Bellman update.

- SARSA update (on-policy):

  Q(s,a) := Q(s,a) + alpha _ (r + gamma _ Q(s',a') - Q(s,a))

  In the code: `SarsaAgent.update(state, action, reward, next_state, next_action, done)` accepts the next action actually selected by the agent and uses that for the bootstrap.

Key practical difference: Q-learning uses the greedy max target and therefore is off-policy — it can learn about a greedy policy while still exploring — whereas SARSA's updates reflect the actual behavior policy (exploration included). This typically makes SARSA safer under highly exploratory policies but can slow learning or change asymptotic performance depending on the exploration schedule.

## Saved artifacts and formats

- Policy files: saved as NumPy arrays in `outputs/models/`.
  - Tabular greedy action maps: `*_policy.npy` (shape `(n_bins, n_bins)` with action indices).
  - Checkpoints (optional): `.npz` archives containing full Q-tables and training metadata.
- Logs and aggregates: `outputs/logs/` contains per-seed CSVs and aggregated JSONs with fields like `mean_reward`, `mean_success_rate`, `n_seeds`.
- Figures: rollout GIFs and plots saved under `outputs/figures/`.

## How to run training (examples)

From the repository root, in an activated virtual environment:

```bash
# Standalone Q-learning (baseline)
python -m src.q_learning.train --config configs/q_learning_separate.yaml --project-root .

# Tuned standalone Q-learning
python -m src.q_learning.train --config configs/q_learning_tuned.yaml --project-root .

# Standalone SARSA (baseline)
python -m src.sarsa.train --config configs/sarsa_discrete.yaml --project-root .

# Tuned standalone SARSA
python -m src.sarsa.train --config configs/sarsa_tuned.yaml --project-root .
```

Each trainer supports a YAML config with keys for environment wrappers and agent hyperparameters. The trainer scripts perform multi-seed runs, save per-seed checkpoints, and aggregate results into `outputs/logs/{experiment}_aggregate.json`.

## Recommended config keys (what to tune)

- `n_bins` (DiscretizeStateWrapper): coarse → fine grids. Typical values: 16–64. More bins increase representational power but increase sample complexity.
- `learning_rate` (alpha): typical ranges 0.01–0.5. For tabular methods, 0.1 is a common start.
- `gamma`: discount factor. Default for MountainCar works well at 0.99.
- `epsilon_start`, `epsilon_end`, `epsilon_decay` (epsilon-greedy): controls exploration. Decay schedule matters a lot:
  - Fast decay trains less under exploration; slow decay keeps exploring longer.
  - A typical schedule: `epsilon_start=1.0`, `epsilon_end=0.01`, decay such that epsilon reaches the floor over ~30–60% of training episodes.
- `episodes`: number of training episodes. Tabular methods often need many episodes (thousands) to converge on higher-resolution grids.
- `seeds`: run multiple random seeds (≥3, preferably 5–10) and aggregate metrics (mean ± std).

## Tuning guidance (practical tips)

- Start with a coarse discretization (e.g. `n_bins=18`) and confirm improvement trends, then increase grid resolution if performance plateaus.
- If final policies are unstable or success rates low:
  - Increase `episodes`.
  - Reduce `learning_rate` if the Q-table oscillates.
  - Slow the epsilon decay (keep exploring a while, then anneal).
- Q-learning vs SARSA: copy-safe hyperparameters rarely transfer perfectly. If one algorithm underperforms after copying tuned settings, sweep:
  - `learning_rate` ∈ {0.05, 0.1, 0.2}
  - `epsilon_decay` slower/faster by factor 2
  - `n_bins` {24, 36, 48}

## Visualize trained policies

Use the rollout renderer (live window or GIF). Ensure you use the same config that was used for training so discretization aligns with the saved policy indices.

Live playback example (opens a window; requires `pygame`):

```bash
python -m src.visualization.mountain_car_rollout \
  --config configs/q_learning_separate.yaml \
  --policy outputs/models/q_learning_discrete_separate_seed21_policy.npy \
  --live --seed 21 --episodes 1 --max-steps 200 --label q_learning
```

GIF export example:

```bash
python -m src.visualization.mountain_car_rollout \
  --config configs/sarsa_discrete.yaml \
  --policy outputs/models/sarsa_discrete_seed21_policy.npy \
  --output outputs/figures/sarsa_discrete_seed21_rollout.gif \
  --seed 21 --episodes 1 --max-steps 200 --fps 20 --label sarsa
```

If `pygame` is missing the renderer will fail — install it with `pip install pygame` in your environment.

## Interpreting training outputs

- `mean_reward`: average episode return across evaluation episodes (higher is better). Note MountainCar returns -1 per time-step until solved, so failing to reach the goal in a 200-step episode yields -200.
- `mean_success_rate`: fraction of episodes where the agent reached the goal within the episode limit.
- Per-seed CSVs include episode-level reward and length; aggregated JSONs include mean ± std across seeds.

Common observation: early smoke tests show `mean_reward ≈ -200` for untrained agents (they never reach the goal). After training you should see mean_reward increase (less negative) and success rate improve.

## Troubleshooting

- Render errors: install `pygame` (`pip install pygame`) if live rendering fails with `ModuleNotFoundError`.
- Discretization mismatch: always visualize using the same `--config` used for training. A different `n_bins` will make the policy indices meaningless.
- File not found: policies are saved under `outputs/models/`. Confirm the exact filename (it includes experiment name and seed) and supply that path to the renderer.

## Reproducibility tips

- Use the `--seed` arguments in training and visualization to reproduce runs exactly.
- Trainers save experiment metadata with hyperparameters and seed — preserve these `outputs/models/*.npz` files for exact replay.

## Observed results (examples from our experiments)

- Baseline runs (3 seeds) produced aggregate mean rewards near -196 to -198 and modest success rates (~10–20%).
- Tuned Q-learning improved to mean reward ≈ -194 and success rate ≈ 23% in our tuned runs.
- Tuned SARSA (copying Q-learning tuned settings) performed worse in our run; this illustrates the need for algorithm-specific sweeps.

Refer to `outputs/logs/` for the exact JSON aggregate files and to `outputs/figures/` for rollout GIFs used as examples.

## Next steps (suggested experiments)

1. Run a grid sweep over `n_bins` and `epsilon_decay` for SARSA specifically.
2. Compare Q-learning and SARSA learning curves side-by-side (use `notebooks/04_q_learning_vs_sarsa_comparison.ipynb`).
3. Try function approximation (tile-coding or a small neural network) if tabular methods become infeasible at high resolutions.

---

If you want, I can also add a short README section with quick run examples (or automatically generate the notebook comparison plots). Which would you prefer next?

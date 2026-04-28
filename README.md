# Mountain Car RL Testbed

A clean, reproducible RL framework for comparing algorithms, representations, and reward formulations on Gymnasium Mountain Car environments.

## What this project is for

This repository is designed for assignment-style benchmarking and analysis, not just single-agent training. It supports:

- MountainCar-v0 (discrete actions) and MountainCarContinuous-v0 (continuous actions)
- raw vs transformed state representations (discretized and augmented)
- optional explicit reward shaping wrappers
- multi-seed training, evaluation, logging, and visualization

## Verified environment behavior (Gymnasium 0.29.1)

The code enforces and checks the following assumptions:

- Observation/state is position and velocity for both environments.
- `MountainCar-v0` action space is `Discrete(3)` with actions left/no-op/right.
- `MountainCarContinuous-v0` action space is `Box([-1.0], [1.0])`.
- Step API is Gymnasium-native:
  - `obs, info = env.reset(seed=...)`
  - `obs, reward, terminated, truncated, info = env.step(action)`
- Episode termination is handled by `terminated or truncated`.

## Project structure

```text
./
  README.md                                  # project documentation and run guide
  requirements.txt                           # pinned dependencies for reproducibility
  .gitignore                                 # ignores caches, venv, and generated outputs
  configs/
    q_learning_discrete.yaml                 # tabular baseline config
    q_learning_separate.yaml                 # standalone Q-learning config
    sarsa_discrete.yaml                      # standalone SARSA config
    dqn_discrete.yaml                        # DQN baseline config
    sac_continuous.yaml                      # SAC baseline config
  src/
    q_learning/
      agent.py                               # dedicated tabular Q-learning agent
      train.py                               # dedicated Q-learning trainer
    sarsa/
      agent.py                               # dedicated tabular SARSA agent
      train.py                               # dedicated SARSA trainer
    envs/
      mountain_car_discrete.py               # MountainCar-v0 constructor and spec info
      mountain_car_continuous.py             # MountainCarContinuous-v0 constructor and spec info
      wrappers.py                            # explicit configurable wrappers
      state_utils.py                         # discretization and feature engineering utilities
    agents/
      q_learning.py                          # tabular Q-learning baseline
      dqn.py                                 # DQN baseline (Stable-Baselines3)
      sac.py                                 # SAC baseline (Stable-Baselines3)
    training/
      train.py                               # multi-seed training loop + checkpointing
      evaluate.py                            # checkpoint evaluation script
      experiments.py                         # run multiple configs and aggregate
    visualization/
      plots.py                               # reward/success and seed-variability plots
      trajectories.py                        # trajectory and phase-space plotting
      policy_maps.py                         # policy and state-visitation heatmaps
    utils/
      seeding.py                             # consistent RNG seeding
      logging.py                             # TensorBoard and JSON logging helpers
      config.py                              # config load and output path handling
  notebooks/
    01_environment_sanity_check.ipynb        # API checks, random rollouts, seed checks
    02_training_and_evaluation.ipynb         # training and evaluation workflow
    03_policy_analysis.ipynb                 # policy, visitation, and trajectory analysis
  outputs/
    logs/                                    # TensorBoard runs + metrics csv/json
    models/                                  # saved checkpoints
    figures/                                 # generated plots
```

## Wrappers and why they exist

All wrappers are explicit and optional via config.

- `DiscretizeStateWrapper`: maps continuous `(position, velocity)` to an `n_bins x n_bins` grid for tabular methods.
- `AugmentStateWrapper`: extends state with kinetic and potential-energy proxy features.
- `EnergyShapingRewardWrapper`: adds potential-energy-difference shaping term.
- `RecordEpisodeStatsWrapper`: collects episode-level reward/length/success stats.

## Installation (clean virtual environment)

1. Clone the repo and enter the repository root:

- `cd mountain-pendulum`

2. Create virtual environment:
   - `python -m venv venv`
3. Activate it:
   - Windows PowerShell: `venv\Scripts\Activate.ps1`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   - `pip install -r requirements.txt`
5. Optional sanity import check:
   - `python -c "import gymnasium as gym; import torch; print(gym.__version__)"`

## Run environment sanity checks

Use Notebook 1 (`notebooks/01_environment_sanity_check.ipynb`) to verify:

- spaces for both envs
- reward and termination behavior from random rollouts
- reproducibility under fixed seeds
- optional frame capture from `render_mode='rgb_array'`

## Train baselines

From the repository root:

- Q-learning (discrete):
  - `python -m src.training.train --config configs/q_learning_discrete.yaml --project-root .`
- Standalone Q-learning package (separate directory):
  - `python -m src.q_learning.train --config configs/q_learning_separate.yaml --project-root .`
- Standalone SARSA package (separate directory):
  - `python -m src.sarsa.train --config configs/sarsa_discrete.yaml --project-root .`
- DQN (discrete):
  - `python -m src.training.train --config configs/dqn_discrete.yaml --project-root .`
- SAC (continuous):
  - `python -m src.training.train --config configs/sac_continuous.yaml --project-root .`

## Evaluate checkpoints

Example:

- `python -m src.training.evaluate --config configs/dqn_discrete.yaml --checkpoint outputs/models/dqn_discrete_seed42_final.zip --seed 42 --episodes 30 --project-root .`

## Run experiment suite

Run multiple configs and aggregate:

- `python -m src.training.experiments --configs configs/q_learning_discrete.yaml configs/dqn_discrete.yaml configs/sac_continuous.yaml --project-root .`

## Visualizations produced

The framework and notebooks support:

- reward curves over episodes
- success-rate curves
- seed variability summaries (mean ± std)
- state visitation heatmaps
- policy heatmaps (for tabular Q-learning)
- trajectories in position-velocity space
- phase-portrait-style interpretation plots

Figures are saved in `outputs/figures/`.

## Render trained MountainCar policies

Use the rollout renderer to watch a trained tabular policy in a live window or export a GIF. The command loads the policy table saved by the standalone Q-learning or SARSA trainers.

Live window mode:

- Q-learning:
  - `python -m src.visualization.mountain_car_rollout --config configs/q_learning_separate.yaml --policy outputs/models/q_learning_discrete_separate_seed21_policy.npy --live --seed 21 --episodes 1 --max-steps 200 --label q_learning`
- SARSA:
  - `python -m src.visualization.mountain_car_rollout --config configs/sarsa_discrete.yaml --policy outputs/models/sarsa_discrete_seed21_policy.npy --live --seed 21 --episodes 1 --max-steps 200 --label sarsa`

GIF export mode:

- Q-learning:
  - `python -m src.visualization.mountain_car_rollout --config configs/q_learning_separate.yaml --policy outputs/models/q_learning_discrete_separate_seed21_policy.npy --output outputs/figures/q_learning_discrete_separate_seed21_rollout.gif --seed 21 --episodes 1 --max-steps 200 --fps 20 --label q_learning`
- SARSA:
  - `python -m src.visualization.mountain_car_rollout --config configs/sarsa_discrete.yaml --policy outputs/models/sarsa_discrete_seed21_policy.npy --output outputs/figures/sarsa_discrete_seed21_rollout.gif --seed 21 --episodes 1 --max-steps 200 --fps 20 --label sarsa`

If you trained with a different seed, replace the policy path with your own file from `outputs/models/`. The rollout renderer reads the saved `.npy` policy table, reconstructs the MountainCar environment using the matching config file, and then steps the policy in real time.

## Procedural sensor-car visualization

This repository now includes a separate visualization demo that is intentionally closer to the "car with probes" style you described.

- Assets are loaded from `assets/maps/` and `assets/sprites/`.
- If no compatible map or sprite is present, the demo falls back to procedural generation.
- The demo renders a top-down car, wall collisions, goal marker, and ray probes.
- The current default demo uses the copied reference art from `RLI_17_A0` (`race_track_ie.png` and `car.png`) when available.

How it works:

- `src/envs/sensor_car_env.py` builds a small navigation world from a PNG map if available, otherwise it procedurally creates walls and corridors.
- The car observes probe distances and simple motion features.
- `src/visualization/sensor_demo.py` can render either a random policy or a trained DQN checkpoint and exports a GIF.
- `src/training/train_sensor_visual.py` trains a DQN agent on the sensor environment so you can compare random vs trained behavior visually.

Training and running the demo:

- Train the visualization agent:
  - `python -m src.training.train_sensor_visual --asset-root assets --timesteps 80000 --output-model outputs/models/sensor_dqn --seed 42`
- Run the live window demo:
  - `python -m src.visualization.live_demo --asset-root assets --steps 10 --fps 10 --seed 42`
- Run the procedural or asset-backed demo:
  - `python -m src.visualization.sensor_demo --asset-root assets --output outputs/figures/sensor_car_demo.gif --episodes 2 --max-steps 120 --fps 15 --seed 42`

If you later add your own map image, place it in `assets/maps/`. If you add a sprite, place it in `assets/sprites/`. The demo will automatically use them when present.

## Notes for extending later

- Add new wrappers in `src/envs/wrappers.py` and wire them in env builders.
- Add new agent adapters in `src/agents/`.
- Keep algorithm hyperparameters in YAML files under `configs/`.
- Reuse `src/training/experiments.py` for fair multi-seed comparisons.

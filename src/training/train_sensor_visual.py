"""Train a DQN agent for the sensor-car visualization environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN

from src.envs.sensor_car_env import SensorCarEnv


def train_sensor_agent(
    asset_root: str,
    total_timesteps: int,
    output_model: str,
    seed: int,
    max_steps: int,
) -> Path:
    """Train DQN on SensorCarEnv and save a model checkpoint."""
    env = SensorCarEnv(
        asset_root=asset_root, render_mode="rgb_array", max_steps=max_steps
    )
    env.reset(seed=seed)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,
        buffer_size=50000,
        batch_size=64,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=250,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.3,
        verbose=1,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps)

    output_path = Path(output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    env.close()

    print(f"Saved sensor visualization model to: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for sensor-car training."""
    parser = argparse.ArgumentParser(description="Train sensor-car DQN baseline.")
    parser.add_argument("--asset-root", default="assets", help="Assets root folder.")
    parser.add_argument(
        "--timesteps", type=int, default=80000, help="Total training timesteps."
    )
    parser.add_argument(
        "--output-model",
        default="outputs/models/sensor_dqn",
        help="Output model path prefix.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-steps", type=int, default=220, help="Max steps per episode."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_sensor_agent(
        asset_root=args.asset_root,
        total_timesteps=args.timesteps,
        output_model=args.output_model,
        seed=args.seed,
        max_steps=args.max_steps,
    )

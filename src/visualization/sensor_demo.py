"""Run and export a sensor-car visualization demo with asset fallbacks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio
import numpy as np

from src.envs.sensor_car_env import SensorCarEnv


def _safe_load_dqn(model_path: str | None) -> Any:
    """Load Stable-Baselines3 DQN model if path is provided and available."""
    if not model_path:
        return None

    try:
        from stable_baselines3 import DQN
    except Exception:
        return None

    path = Path(model_path)
    if not path.exists():
        return None

    return DQN.load(str(path))


def run_demo(
    asset_root: str,
    output_path: str,
    episodes: int,
    max_steps: int,
    fps: int,
    model_path: str | None = None,
    seed: int = 42,
) -> Path:
    """Generate a GIF demo using random control or a trained DQN policy."""
    env = SensorCarEnv(
        asset_root=asset_root, render_mode="rgb_array", max_steps=max_steps
    )
    model = _safe_load_dqn(model_path)

    print(f"Map source: {env.assets.map_source}")
    print(f"Sprite source: {env.assets.sprite_source}")
    print(f"Policy source: {'trained model' if model is not None else 'random policy'}")

    frames: list[np.ndarray] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, _, terminated, truncated, _ = env.step(int(action))
            frames.append(env.render())
            done = bool(terminated or truncated)

    env.close()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=fps)
    print(f"Saved demo GIF to: {out}")
    return out


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the visualization demo."""
    parser = argparse.ArgumentParser(description="Sensor-car visualization demo.")
    parser.add_argument("--asset-root", default="assets", help="Assets root folder.")
    parser.add_argument(
        "--output",
        default="outputs/figures/sensor_car_demo.gif",
        help="Output GIF path.",
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of demo episodes."
    )
    parser.add_argument(
        "--max-steps", type=int, default=220, help="Max steps per episode."
    )
    parser.add_argument("--fps", type=int, default=20, help="GIF frame rate.")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional trained DQN checkpoint path for policy playback.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(
        asset_root=args.asset_root,
        output_path=args.output,
        episodes=args.episodes,
        max_steps=args.max_steps,
        fps=args.fps,
        model_path=args.model,
        seed=args.seed,
    )

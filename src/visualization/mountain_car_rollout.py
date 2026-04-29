"""Render a trained tabular MountainCar policy in a live window or as a GIF."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio
import numpy as np

from src.envs.mountain_car_discrete import make_discrete_env
from src.utils.config import load_config


def _load_policy(policy_path: str | Path) -> np.ndarray:
    """Load a greedy policy table saved during tabular training."""
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    loaded = np.load(path)
    if loaded.ndim != 2:
        raise ValueError("Expected a 2D policy table with shape (n_bins, n_bins).")
    return loaded


def _build_env(config: dict[str, Any], render_mode: str) -> Any:
    """Create the MountainCar environment with the configured wrappers."""
    wrappers = config.get("env", {}).get("wrappers", {})
    return make_discrete_env(render_mode=render_mode, wrappers=wrappers)


def _select_action(policy: np.ndarray, obs: np.ndarray) -> int:
    """Choose the greedy action from a discretized observation."""
    state = tuple(obs)
    return int(policy[state[0], state[1]])


def _compute_success(info: dict[str, Any], terminated: bool) -> bool:
    """Derive a success flag from the environment info and termination state."""
    if "goal_reached" in info:
        return bool(info["goal_reached"])
    return bool(terminated)


def run_live_window(
    config_path: str,
    policy_path: str,
    seed: int,
    episodes: int,
    max_steps: int,
    label: str,
) -> None:
    """Open a real-time Gymnasium window and play the trained policy."""
    config = load_config(config_path)
    policy = _load_policy(policy_path)
    env = _build_env(config, render_mode="human")

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            episode_reward = 0.0
            step = 0
            info: dict[str, Any] = {}
            terminated = False

            while not done and step < max_steps:
                action = _select_action(policy, obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                done = bool(terminated or truncated)
                step += 1

            print(
                f"{label} episode {episode + 1}: reward={episode_reward:.2f}, steps={step}, "
                f"success={_compute_success(info, terminated)}"
            )
    finally:
        env.close()


def run_gif_export(
    config_path: str,
    policy_path: str,
    output_path: str,
    seed: int,
    episodes: int,
    max_steps: int,
    fps: int,
    label: str,
) -> Path:
    """Export one or more episodes as an animated GIF."""
    config = load_config(config_path)
    policy = _load_policy(policy_path)
    env = _build_env(config, render_mode="rgb_array")

    frames: list[np.ndarray] = []

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            episode_reward = 0.0
            step = 0
            info: dict[str, Any] = {}
            terminated = False

            while not done and step < max_steps:
                action = _select_action(policy, obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                done = bool(terminated or truncated)
                frames.append(env.render())
                step += 1

            print(
                f"{label} episode {episode + 1}: reward={episode_reward:.2f}, steps={step}, "
                f"success={_compute_success(info, terminated)}"
            )
    finally:
        env.close()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_file, frames, fps=fps)
    return output_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the rollout renderer."""
    parser = argparse.ArgumentParser(description="Render a trained MountainCar policy.")
    parser.add_argument("--config", required=True, help="Path to the training config.")
    parser.add_argument(
        "--policy", required=True, help="Path to the saved policy .npy file."
    )
    parser.add_argument("--output", default=None, help="Optional output GIF path.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Open a live Gymnasium window instead of exporting a GIF.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--episodes", type=int, default=1, help="How many episodes to render."
    )
    parser.add_argument(
        "--max-steps", type=int, default=200, help="Max steps per episode."
    )
    parser.add_argument("--fps", type=int, default=20, help="GIF frame rate.")
    parser.add_argument(
        "--label", default="policy", help="Label printed in the console."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.live:
        run_live_window(
            config_path=args.config,
            policy_path=args.policy,
            seed=args.seed,
            episodes=args.episodes,
            max_steps=args.max_steps,
            label=args.label,
        )
    else:
        if not args.output:
            raise ValueError("Provide --live for a window or --output to export a GIF.")
        gif_path = run_gif_export(
            config_path=args.config,
            policy_path=args.policy,
            output_path=args.output,
            seed=args.seed,
            episodes=args.episodes,
            max_steps=args.max_steps,
            fps=args.fps,
            label=args.label,
        )
        print(f"Saved rollout GIF to: {gif_path}")

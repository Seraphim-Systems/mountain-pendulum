"""Inspect a trained tabular policy: run a deterministic rollout and report
positions, velocities, actions taken, and whether goal was reached.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.envs.tabular_env import make_tabular_env
from src.utils.config import load_config


def main(config_path: str, model_path: str) -> None:
    config = load_config(config_path)
    env = make_tabular_env(
        env_id=config["env"]["id"],
        wrappers=config["env"]["wrappers"],
    )

    data = np.load(model_path)
    q_table = data["q_table"]
    print(f"Loaded q_table shape={q_table.shape}, min={q_table.min():.3f} max={q_table.max():.3f}")

    obs, _ = env.reset(seed=7)
    state = tuple(obs)
    print(f"Start state (bins): {state}")

    # Inspect policy at each (pos, vel) bin
    policy = np.argmax(q_table, axis=2)
    print(f"Policy shape: {policy.shape}")
    print("Policy at high-position bins (rightmost columns):")
    for i in range(policy.shape[0] - 5, policy.shape[0]):
        row = policy[i].tolist()
        print(f"  pos_bin={i}: {row}")

    # Run a deterministic rollout
    raw_env = env
    obs_history = []
    action_history = []
    raw_position = []
    base = env
    while hasattr(base, "env"):
        base = base.env
    print("Raw base env type:", type(base).__name__)

    for step in range(500):
        i, j = state
        action = int(np.argmax(q_table[i, j]))
        next_obs, reward, terminated, truncated, info = env.step(action)
        action_history.append(action)
        obs_history.append(next_obs)
        # Try fetching raw position from underlying env
        try:
            raw = base.unwrapped.state if hasattr(base, "unwrapped") else None
        except Exception:
            raw = None
        if raw is None:
            try:
                raw = env.unwrapped.state
            except Exception:
                raw = None
        raw_position.append(raw[0] if raw is not None else None)
        state = tuple(next_obs)
        if terminated or truncated:
            print(f"Step {step}: terminated={terminated} truncated={truncated}")
            break

    print(f"Total steps: {len(action_history)}")
    print(f"Action histogram: {np.bincount(action_history, minlength=3).tolist()}")
    raws = [r for r in raw_position if r is not None]
    if raws:
        print(f"Raw position: min={min(raws):.3f} max={max(raws):.3f}, last={raws[-1]:.3f}")
        print(f"Reached goal threshold (>=0.5): {max(raws) >= 0.5}")
    print(f"Bin trajectory sample (last 20): {obs_history[-20:]}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.model)

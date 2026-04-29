"""Evaluation utilities for trained Mountain Car agents."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.cma_es import CMAESAgent
from src.agents.dqn import DQNBaseline
from src.agents.neat_agent import NeatAgent
from src.agents.q_learning import QLearningAgent
from src.agents.reinforce import REINFORCEBaseline
from src.agents.sac import SACBaseline
from src.agents.simple_ga import SimpleGAAgent
from src.training.train import build_agent, build_env, rollout_episode
from src.utils.config import ensure_paths, load_config
from src.utils.logging import dump_json
from src.utils.seeding import seed_env, seed_everything


def load_agent_from_checkpoint(
    config: dict[str, Any], env: Any, checkpoint_path: str
) -> Any:
    """Instantiate and restore agent based on config and checkpoint file."""
    agent = build_agent(config, env)
    if isinstance(agent, QLearningAgent):
        agent.load(checkpoint_path)
    elif isinstance(agent, (DQNBaseline, REINFORCEBaseline, SACBaseline)):
        agent.load(checkpoint_path, env)
    elif isinstance(agent, (SimpleGAAgent, CMAESAgent, NeatAgent)):
        agent.load(checkpoint_path, env)
    else:
        raise ValueError("Unsupported agent type during checkpoint load.")
    return agent


def evaluate_checkpoint(
    config_path: str,
    checkpoint_path: str,
    seed: int,
    n_episodes: int,
    project_root: str,
) -> dict[str, Any]:
    """Evaluate a saved model checkpoint and return summary metrics."""
    config = load_config(config_path)
    root = Path(project_root)
    output_paths = ensure_paths(config, root)

    seed_everything(seed)
    env = build_env(config["env"])
    seed_env(env, seed)
    agent = load_agent_from_checkpoint(config, env, checkpoint_path)

    rewards = []
    lengths = []
    successes = []

    for _ in range(n_episodes):
        reward, length, success = rollout_episode(env, agent, deterministic=True)
        rewards.append(reward)
        lengths.append(length)
        successes.append(float(success))

    env.close()

    result = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "episodes": n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
    }

    file_name = f"eval_{config['experiment_name']}_seed{seed}.json"
    dump_json(result, output_paths["logs_dir"] / file_name)
    print(result)
    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Mountain Car checkpoint.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path.")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        n_episodes=args.episodes,
        project_root=args.project_root,
    )

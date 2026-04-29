"""Train a tabular SARSA policy on MountainCar-v0."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.envs.mountain_car_discrete import make_discrete_env
from src.sarsa.agent import SarsaAgent
from src.utils.config import ensure_paths, load_config
from src.utils.logging import dump_json, make_writer
from src.utils.seeding import seed_env, seed_everything


def _build_env(config: dict[str, Any]) -> Any:
    wrappers = config.get("env", {}).get("wrappers", {})
    if "discretize_state" not in wrappers:
        raise ValueError("SARSA requires env.wrappers.discretize_state.n_bins")
    return make_discrete_env(wrappers=wrappers)


def _build_agent(config: dict[str, Any], env: Any) -> SarsaAgent:
    s_cfg = config["sarsa"]
    wrappers = config["env"]["wrappers"]
    n_bins = int(wrappers["discretize_state"]["n_bins"])

    return SarsaAgent(
        n_bins=n_bins,
        n_actions=int(env.action_space.n),
        learning_rate=float(s_cfg["learning_rate"]),
        gamma=float(s_cfg["gamma"]),
        epsilon_start=float(s_cfg["epsilon_start"]),
        epsilon_end=float(s_cfg["epsilon_end"]),
        epsilon_decay=float(s_cfg["epsilon_decay"]),
    )


def train_single_seed(
    config: dict[str, Any],
    project_root: Path,
    seed: int,
) -> dict[str, Any]:
    paths = ensure_paths(config, project_root)
    run_name = f"{config['experiment_name']}_seed{seed}"

    seed_everything(seed)
    env = _build_env(config)
    seed_env(env, seed)
    agent = _build_agent(config, env)

    train_cfg = config["train"]
    episodes = int(train_cfg["episodes"])
    max_steps = int(train_cfg["max_steps_per_episode"])
    eval_every = int(train_cfg.get("eval_every", 100))
    eval_episodes = int(train_cfg.get("eval_episodes", 20))

    writer = make_writer(paths["logs_dir"], run_name)

    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    eval_rewards: list[float] = []

    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        state = tuple(obs)
        action = agent.select_action(state)

        done = False
        total_reward = 0.0
        steps = 0
        success = False

        while not done and steps < max_steps:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple(next_obs)
            done = bool(terminated or truncated)
            next_action = 0 if done else agent.select_action(next_state)

            agent.update(
                state=state,
                action=int(action),
                reward=float(reward),
                next_state=next_state,
                next_action=int(next_action),
                done=done,
            )

            state = next_state
            action = next_action
            total_reward += float(reward)
            steps += 1
            success = bool(terminated and not truncated)

        agent.on_episode_end()

        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(float(success))

        writer.add_scalar("train/reward", total_reward, episode)
        writer.add_scalar("train/length", steps, episode)
        writer.add_scalar("train/success", float(success), episode)
        writer.add_scalar("train/epsilon", agent.epsilon, episode)

        if episode % eval_every == 0:
            batch_rewards: list[float] = []
            for _ in range(eval_episodes):
                eval_obs, _ = env.reset()
                eval_state = tuple(eval_obs)
                eval_done = False
                eval_total = 0.0
                eval_steps = 0

                while not eval_done and eval_steps < max_steps:
                    eval_action = agent.select_action(eval_state, deterministic=True)
                    eval_next_obs, eval_reward, eval_terminated, eval_truncated, _ = (
                        env.step(eval_action)
                    )
                    eval_state = tuple(eval_next_obs)
                    eval_done = bool(eval_terminated or eval_truncated)
                    eval_total += float(eval_reward)
                    eval_steps += 1

                batch_rewards.append(eval_total)

            mean_eval_reward = float(np.mean(batch_rewards))
            eval_rewards.append(mean_eval_reward)
            writer.add_scalar("eval/mean_reward", mean_eval_reward, episode)

    model_path = paths["models_dir"] / f"{run_name}_final"
    policy_path = paths["models_dir"] / f"{run_name}_policy.npy"
    agent.save(model_path)
    np.save(policy_path, agent.policy_table())

    metrics = {
        "experiment_name": config["experiment_name"],
        "seed": int(seed),
        "algorithm": "sarsa",
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_episode_length": float(np.mean(lengths)),
        "last_eval_mean_reward": float(
            eval_rewards[-1] if eval_rewards else np.mean(rewards)
        ),
        "model_path": str(model_path),
        "policy_path": str(policy_path),
    }

    episode_df = pd.DataFrame(
        {
            "episode": np.arange(1, episodes + 1),
            "reward": rewards,
            "length": lengths,
            "success": successes,
            "seed": seed,
        }
    )
    episode_df.to_csv(paths["logs_dir"] / f"{run_name}_episodes.csv", index=False)
    dump_json(metrics, paths["logs_dir"] / f"{run_name}_summary.json")

    writer.flush()
    writer.close()
    env.close()
    return metrics


def train(config_path: str, project_root: str) -> None:
    config = load_config(config_path)
    root = Path(project_root)

    summaries = []
    for seed in config["train"]["seeds"]:
        summaries.append(
            train_single_seed(config=config, project_root=root, seed=int(seed))
        )

    df = pd.DataFrame(summaries)
    aggregate = {
        "experiment_name": config["experiment_name"],
        "algorithm": "sarsa",
        "n_seeds": int(len(df)),
        "mean_reward": float(df["mean_reward"].mean()),
        "std_reward_across_seeds": float(df["mean_reward"].std(ddof=0)),
        "mean_success_rate": float(df["success_rate"].mean()),
    }

    output_paths = ensure_paths(config, root)
    df.to_csv(
        output_paths["logs_dir"] / f"{config['experiment_name']}_seed_results.csv",
        index=False,
    )
    dump_json(
        aggregate,
        output_paths["logs_dir"] / f"{config['experiment_name']}_aggregate.json",
    )
    print("SARSA training complete.")
    print(aggregate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SARSA on MountainCar-v0")
    parser.add_argument(
        "--config", default="configs/sarsa_discrete.yaml", help="Path to YAML config"
    )
    parser.add_argument("--project-root", default=".", help="Path to project root")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(config_path=args.config, project_root=args.project_root)

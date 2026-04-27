"""Training entrypoint for Mountain Car baselines across multiple seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agents.cma_es import CMAESAgent
from src.agents.dqn import DQNBaseline
from src.agents.neat_agent import NeatAgent
from src.agents.q_learning import QLearningAgent
from src.agents.sac import SACBaseline
from src.agents.simple_ga import SimpleGAAgent
from src.envs.mountain_car_continuous import make_continuous_env
from src.envs.mountain_car_discrete import make_discrete_env
from src.utils.config import ensure_paths, load_config
from src.utils.logging import dump_json, make_writer
from src.utils.seeding import seed_env, seed_everything


def build_env(env_cfg: dict[str, Any], render_mode: str | None = None) -> Any:
    """Create configured environment from config section."""
    env_id = env_cfg["id"]
    wrappers = env_cfg.get("wrappers", {})
    if env_id == "MountainCar-v0":
        return make_discrete_env(render_mode=render_mode, wrappers=wrappers)
    if env_id == "MountainCarContinuous-v0":
        return make_continuous_env(render_mode=render_mode, wrappers=wrappers)
    raise ValueError(f"Unsupported environment id: {env_id}")


def build_agent(config: dict[str, Any], env: Any) -> Any:
    """Build an agent based on top-level config 'agent' field."""
    agent_name = config["agent"]
    if agent_name == "q_learning":
        q_cfg = config["q_learning"]
        wrappers = config["env"].get("wrappers", {})
        n_bins = wrappers.get("discretize_state", {}).get("n_bins")
        if n_bins is None:
            raise ValueError("Q-learning requires env.wrappers.discretize_state.n_bins")
        return QLearningAgent(
            n_bins=int(n_bins),
            n_actions=int(env.action_space.n),
            learning_rate=float(q_cfg["learning_rate"]),
            gamma=float(q_cfg["gamma"]),
            epsilon_start=float(q_cfg["epsilon_start"]),
            epsilon_end=float(q_cfg["epsilon_end"]),
            epsilon_decay=float(q_cfg["epsilon_decay"]),
        )

    if agent_name == "dqn":
        dqn_cfg = config["dqn"]
        return DQNBaseline(
            env=env,
            learning_rate=float(dqn_cfg["learning_rate"]),
            gamma=float(dqn_cfg["gamma"]),
            epsilon_start=float(dqn_cfg["epsilon_start"]),
            epsilon_end=float(dqn_cfg["epsilon_end"]),
            epsilon_decay=float(dqn_cfg["epsilon_decay"]),
            batch_size=int(dqn_cfg["batch_size"]),
            replay_buffer_size=int(dqn_cfg["replay_buffer_size"]),
            target_update_freq=int(dqn_cfg["target_update_freq"]),
            hidden_sizes=list(dqn_cfg["hidden_sizes"]),
        )

    if agent_name == "sac":
        sac_cfg = config["sac"]
        return SACBaseline(
            env=env,
            learning_rate=float(sac_cfg["learning_rate"]),
            gamma=float(sac_cfg["gamma"]),
            tau=float(sac_cfg["tau"]),
            alpha=sac_cfg["alpha"],
            batch_size=int(sac_cfg["batch_size"]),
            replay_buffer_size=int(sac_cfg["replay_buffer_size"]),
            learning_starts=int(sac_cfg["learning_starts"]),
            hidden_sizes=list(sac_cfg["hidden_sizes"]),
        )

    if agent_name == "simple_ga":
        ga_cfg = config["simple_ga"]
        return SimpleGAAgent(
            env=env,
            population_size=int(ga_cfg["population_size"]),
            elite_frac=float(ga_cfg["elite_frac"]),
            mutation_std=float(ga_cfg["mutation_std"]),
            crossover_alpha=float(ga_cfg.get("crossover_alpha", 0.5)),
            hidden_sizes=list(ga_cfg.get("hidden_sizes", [64, 64])),
        )

    if agent_name == "cma_es":
        cma_cfg = config["cma_es"]
        return CMAESAgent(
            env=env,
            population_size=int(cma_cfg["population_size"]),
            sigma0=float(cma_cfg["sigma0"]),
            hidden_sizes=list(cma_cfg.get("hidden_sizes", [64, 64])),
        )

    if agent_name == "neat":
        neat_cfg = config["neat"]
        return NeatAgent(
            env=env,
            pop_size=int(neat_cfg["pop_size"]),
            fitness_threshold=float(neat_cfg.get("fitness_threshold", 90.0)),
        )

    raise ValueError(f"Unsupported agent type: {agent_name}")


def rollout_episode(
    env: Any, agent: Any, deterministic: bool = True
) -> tuple[float, int, bool]:
    """Run one full episode for evaluation."""
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    success = False

    while not done:
        if isinstance(agent, QLearningAgent):
            action = agent.select_action(tuple(obs), deterministic=deterministic)
        else:
            action = agent.predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        done = bool(terminated or truncated)
        success = bool(terminated and not truncated)

    return total_reward, steps, success


def train_single_seed(
    config: dict[str, Any], seed: int, project_root: Path
) -> dict[str, Any]:
    """Train one seed and persist metrics/checkpoints."""
    output_paths = ensure_paths(config, project_root)
    seed_everything(seed)

    env = build_env(config["env"])
    seed_env(env, seed)
    agent = build_agent(config, env)

    run_name = f"{config['experiment_name']}_seed{seed}"
    writer = make_writer(output_paths["logs_dir"], run_name)

    episodes = int(config["train"]["episodes"])
    max_steps = int(config["train"]["max_steps_per_episode"])
    eval_every = int(config["train"]["eval_every"])
    eval_episodes = int(config["train"]["eval_episodes"])
    save_every = int(config["train"]["save_every"])

    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    eval_mean_rewards: list[float] = []
    pop_stds: list[float] = []

    for episode in range(1, episodes + 1):
        if isinstance(agent, QLearningAgent):
            obs, _ = env.reset()
            state = tuple(obs)
            done = False
            ep_reward = 0.0
            ep_len = 0
            ep_success = False

            while not done and ep_len < max_steps:
                action = agent.select_action(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = tuple(next_obs)
                done = bool(terminated or truncated)
                agent.update(state, action, float(reward), next_state, done)
                state = next_state
                ep_reward += float(reward)
                ep_len += 1
                ep_success = bool(terminated and not truncated)

            agent.on_episode_end()
        else:
            agent.learn(total_timesteps=max_steps)
            ep_reward, ep_len, ep_success = rollout_episode(
                env, agent, deterministic=True
            )

        rewards.append(ep_reward)
        lengths.append(ep_len)
        successes.append(float(ep_success))

        writer.add_scalar("train/reward", ep_reward, episode)
        writer.add_scalar("train/episode_length", ep_len, episode)
        writer.add_scalar("train/success", float(ep_success), episode)

        if isinstance(agent, QLearningAgent):
            writer.add_scalar("train/epsilon", agent.epsilon, episode)
        elif hasattr(agent, "exploration_rate"):
            writer.add_scalar("train/exploration_rate", agent.exploration_rate, episode)

        if hasattr(agent, "_pop"):
            pop_std = float(np.mean(np.std(agent._pop, axis=0)))
            writer.add_scalar("train/population_std", pop_std, episode)
            pop_stds.append(pop_std)
        else:
            pop_stds.append(float("nan"))

        if episode % eval_every == 0:
            eval_rewards = []
            eval_success = []
            for _ in range(eval_episodes):
                r, _, s = rollout_episode(env, agent, deterministic=True)
                eval_rewards.append(r)
                eval_success.append(float(s))
            mean_eval_reward = float(np.mean(eval_rewards))
            mean_eval_success = float(np.mean(eval_success))
            eval_mean_rewards.append(mean_eval_reward)
            writer.add_scalar("eval/mean_reward", mean_eval_reward, episode)
            writer.add_scalar("eval/success_rate", mean_eval_success, episode)

        if episode % save_every == 0:
            checkpoint = output_paths["models_dir"] / f"{run_name}_ep{episode}"
            agent.save(checkpoint)

    final_model_path = output_paths["models_dir"] / f"{run_name}_final"
    agent.save(final_model_path)

    metrics = {
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "agent": config["agent"],
        "env_id": config["env"]["id"],
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_episode_length": float(np.mean(lengths)),
        "eval_mean_reward_last": float(
            eval_mean_rewards[-1] if eval_mean_rewards else np.mean(rewards)
        ),
        "final_model_path": str(final_model_path),
    }

    per_episode_df = pd.DataFrame(
        {
            "episode": np.arange(1, episodes + 1),
            "reward": rewards,
            "length": lengths,
            "success": successes,
            "population_std": pop_stds,
            "seed": seed,
        }
    )
    csv_path = output_paths["logs_dir"] / f"{run_name}_episodes.csv"
    per_episode_df.to_csv(csv_path, index=False)
    dump_json(metrics, output_paths["logs_dir"] / f"{run_name}_summary.json")

    writer.flush()
    writer.close()

    if hasattr(agent, "_genome_history") and agent._genome_history:
        from src.visualization.neat_viz import draw_neat_animation
        anim_path = output_paths["figures_dir"] / f"{run_name}_topology_evolution"
        output_paths["figures_dir"].mkdir(parents=True, exist_ok=True)
        draw_neat_animation(agent._genome_history, agent._config, anim_path, fps=6)

    env.close()
    return metrics


def train_experiment(config_path: str, project_root: str) -> None:
    """Train one experiment config across all configured seeds."""
    config = load_config(config_path)
    root = Path(project_root)
    seed_results = []
    for seed in config["train"]["seeds"]:
        result = train_single_seed(config, int(seed), root)
        seed_results.append(result)

    df = pd.DataFrame(seed_results)
    aggregate = {
        "experiment_name": config["experiment_name"],
        "agent": config["agent"],
        "env_id": config["env"]["id"],
        "n_seeds": int(len(df)),
        "reward_mean": float(df["reward_mean"].mean()),
        "reward_std_across_seeds": float(df["reward_mean"].std(ddof=0)),
        "success_rate_mean": float(df["success_rate"].mean()),
        "success_rate_std": float(df["success_rate"].std(ddof=0)),
        "episode_length_mean": float(df["mean_episode_length"].mean()),
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
    print("Training complete.")
    print(aggregate)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train Mountain Car experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Path to project root containing outputs/ and configs/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_experiment(config_path=args.config, project_root=args.project_root)

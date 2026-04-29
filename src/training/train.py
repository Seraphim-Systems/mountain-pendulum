"""Training entrypoint for Mountain Car baselines across multiple seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.live import Live
from rich.table import Table
from rich import box

from src.agents.cma_es import CMAESAgent
from src.agents.dqn import DQNBaseline
from src.agents.neat_agent import NeatAgent
from src.agents.q_learning import QLearningAgent
from src.agents.reinforce import REINFORCEBaseline
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
            learning_starts=int(dqn_cfg.get("min_buffer_size", 1000)),
            target_update_freq=int(dqn_cfg["target_update_freq"]),
            hidden_sizes=list(dqn_cfg["hidden_sizes"]),
            exploration_fraction=(
                float(dqn_cfg["exploration_fraction"])
                if "exploration_fraction" in dqn_cfg
                else None
            ),
        )

    if agent_name == "reinforce":
        reinforce_cfg = config["reinforce"]
        return REINFORCEBaseline(
            env=env,
            learning_rate=float(reinforce_cfg["learning_rate"]),
            gamma=float(reinforce_cfg["gamma"]),
            entropy_coef=float(reinforce_cfg.get("entropy_coef", 0.01)),
            value_fn_coef=float(reinforce_cfg.get("value_fn_coef", 0.5)),
            hidden_sizes=list(reinforce_cfg.get("hidden_sizes", [128, 128])),
            batch_episodes=int(reinforce_cfg.get("batch_episodes", 8)),
            teacher_checkpoint=reinforce_cfg.get("teacher_checkpoint"),
            teacher_pretrain_episodes=int(reinforce_cfg.get("teacher_pretrain_episodes", 64)),
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
    env: Any, agent: Any, deterministic: bool = True, episode_idx: int | None = None, custom_renderer: Any = None
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

        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += float(reward)
        steps += 1
        done = bool(terminated or truncated)
        success = bool(terminated and not truncated)

        if custom_renderer:
            custom_renderer.render(obs, agent.__class__.__name__, episode_idx, steps, total_reward)

    return total_reward, steps, success


def train_single_seed(
    config: dict[str, Any], seed: int, project_root: Path, render: bool = False
) -> dict[str, Any]:
    """Train one seed and persist metrics/checkpoints."""
    output_paths = ensure_paths(config, project_root)
    seed_everything(seed)

    env = build_env(config["env"])
    seed_env(env, seed)
    
    # Custom Renderer takes over visually
    eval_env = build_env(config["env"], render_mode=None) if render else None
    if eval_env:
        seed_env(eval_env, seed)
        
    custom_renderer = None
    if render:
        from src.visualization.custom_renderer import NeonMountainCarRenderer
        custom_renderer = NeonMountainCarRenderer()
        
    agent = build_agent(config, env)

    run_name = f"{config['experiment_name']}_seed{seed}"
    writer = make_writer(output_paths["logs_dir"], run_name)

    episodes = int(config["train"]["episodes"])
    max_steps = int(config["train"]["max_steps_per_episode"])
    eval_every = int(config["train"]["eval_every"])
    eval_episodes = int(config["train"]["eval_episodes"])
    save_every = int(config["train"]["save_every"])

    pop_str = ""
    device_str = ""
    if hasattr(agent, "device"):
        device_str = f" | Device: {str(agent.device).upper()}"
        
    if hasattr(agent, "_population_size"):
        pop_size = agent._population_size
        pop_str = f" | Pop: {pop_size}"
        if hasattr(agent, "_elite_frac"):
            n_elites = max(1, int(pop_size * agent._elite_frac))
            pop_str += f" | Elites: {n_elites}"
    elif hasattr(agent, "_es"):
        pop_str = f" | Pop: {agent._es.popsize}"

    table = Table(
        title=f"[bold cyan]{run_name}[/bold cyan]\nAgent: {config['agent']} | Env: {config['env']['id']}{device_str}{pop_str}", 
        box=box.SIMPLE,
        header_style="bold magenta",
    )
    table.add_column("Gen", justify="right")
    table.add_column("Train Reward", justify="right", style="cyan")
    table.add_column("Eval Reward", justify="right", style="bold green")
    table.add_column("Success", justify="right")
    table.add_column("Best Ever", justify="right", style="yellow")

    is_q = isinstance(agent, QLearningAgent)
    is_ga = config["agent"] == "simple_ga"
    is_cma = config["agent"] == "cma_es"
    is_neat = config["agent"] == "neat"

    if is_q:
        table.add_column("Epsilon", justify="right")
    elif is_ga:
        table.add_column("Pop Std", justify="right")
    elif is_cma:
        table.add_column("Sigma (σ)", justify="right")
    elif is_neat:
        table.add_column("Species", justify="right")
    
    table.add_column("ETA", justify="right", style="dim")

    import time as _time
    _t0 = _time.perf_counter()

    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    eval_mean_rewards: list[float] = []
    pop_stds: list[float] = []

    with Live(table, refresh_per_second=4) as live:
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
                    env, agent, deterministic=True, episode_idx=episode, custom_renderer=custom_renderer
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
                pop_arr = agent._pop.cpu().numpy() if hasattr(agent._pop, 'numpy') else agent._pop
                pop_std = float(np.mean(np.std(pop_arr, axis=0)))
                writer.add_scalar("train/population_std", pop_std, episode)
                pop_stds.append(pop_std)
            else:
                pop_stds.append(float("nan"))

            if episode % eval_every == 0:
                eval_rewards = []
                eval_success = []
                current_eval_env = eval_env if eval_env else env
                for _ in range(eval_episodes):
                    r, _, s = rollout_episode(current_eval_env, agent, deterministic=True)
                    eval_rewards.append(r)
                    eval_success.append(float(s))
                mean_eval_reward = float(np.mean(eval_rewards))
                mean_eval_success = float(np.mean(eval_success))
                eval_mean_rewards.append(mean_eval_reward)
                writer.add_scalar("eval/mean_reward", mean_eval_reward, episode)
                writer.add_scalar("eval/success_rate", mean_eval_success, episode)

                recent_reward = float(np.mean(rewards[-eval_every:]))
                best_ever = getattr(agent, "_best_fitness", float("nan"))
                cur_pop_std = pop_stds[-1] if pop_stds and not np.isnan(pop_stds[-1]) else float("nan")
                elapsed = _time.perf_counter() - _t0
                eta = (elapsed / episode) * (episodes - episode)

                success_color = "bold green" if mean_eval_success > 0 else "red"
                success_str = f"[{success_color}]{mean_eval_success:.1%}[/{success_color}]"
        
                row = [
                    f"{episode}/{episodes}",
                    f"{recent_reward:.1f}",
                    f"{mean_eval_reward:.1f}",
                    success_str,
                    f"{best_ever:.1f}" if not np.isnan(best_ever) else "n/a",
                ]
            
                if is_q:
                    row.append(f"{agent.epsilon:.3f}")
                elif is_ga:
                    row.append(f"{cur_pop_std:.4f}" if not np.isnan(cur_pop_std) else "n/a")
                elif is_cma:
                    row.append(f"{agent._es.sigma:.3f}" if hasattr(agent, "_es") else "n/a")
                elif is_neat:
                    n_species = len(agent._population.species.species) if hasattr(agent, "_population") and hasattr(agent._population, "species") else 0
                    row.append(str(n_species))
                
                row.append(f"{int(eta//60)}m{int(eta%60):02d}s")
                table.add_row(*row)

            if episode % save_every == 0:
                checkpoint = output_paths["models_dir"] / f"{run_name}_ep{episode}"
                agent.save(checkpoint)

    total_time = _time.perf_counter() - _t0
    best_ever = getattr(agent, "_best_fitness", float("nan"))
    print(f"\n  Done in {int(total_time//60)}m{int(total_time%60):02d}s"
          f"  |  best_fitness={best_ever:.1f}"
          f"  |  final_eval={eval_mean_rewards[-1] if eval_mean_rewards else float('nan'):.1f}"
          f"  |  success_rate={float(np.mean(successes)):.1%}", flush=True)

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

    if eval_env:
        eval_env.close()
    if custom_renderer:
        custom_renderer.close()
    env.close()
    return metrics


def train_experiment(config_path: str, project_root: str, render: bool = False) -> None:
    """Train one experiment config across all configured seeds."""
    config = load_config(config_path)
    root = Path(project_root)
    seed_results = []
    for seed in config["train"]["seeds"]:
        result = train_single_seed(config, int(seed), root, render=render)
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
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the Mountain Car live during evaluations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_experiment(config_path=args.config, project_root=args.project_root, render=args.render)

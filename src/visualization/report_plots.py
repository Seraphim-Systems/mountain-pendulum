"""Generate full report figure set from completed training outputs.

Usage:
    python -m src.visualization.report_plots --project-root .
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.visualization.ga_plots import (
    plot_cross_agent_comparison,
    plot_multiseed_envelope,
    plot_population_diversity,
)
from src.visualization.policy_maps import plot_policy_heatmap
from src.visualization.neat_viz import draw_neat_genome, draw_neat_growth_sequence, draw_neat_animation


_AGENTS = [
    "q_learning", "sarsa", "dqn",
    "reinforce", "ppo", "sac",
    "simple_ga", "cma_es", "neat",
]
_AGENT_FAMILY: dict[str, str] = {
    "q_learning": "value_based",
    "sarsa": "value_based",
    "dqn": "value_based",
    "reinforce": "policy_gradient",
    "ppo": "policy_gradient",
    "sac": "policy_gradient",
    "simple_ga": "evolutionary",
    "cma_es": "evolutionary",
    "neat": "evolutionary",
}
_SCENARIOS = ["discrete", "continuous", "fuel", "minsteps"]
_SCENARIO_LABELS = {
    "discrete": "MountainCar-v0 (min steps)",
    "continuous": "MountainCarContinuous-v0 (standard / min energy)",
    "fuel": "MountainCar-v0 (fuel cost)",
    "minsteps": "MountainCarContinuous-v0 (linear step cost)",
}
_AGENT_LABELS = {
    "q_learning": "Q-Learning",
    "sarsa": "SARSA",
    "dqn": "DQN",
    "reinforce": "REINFORCE",
    "ppo": "PPO",
    "sac": "SAC",
    "simple_ga": "Simple GA",
    "cma_es": "CMA-ES",
    "neat": "NEAT",
}
_SEEDS = [7, 21, 42]

_EXPERIMENT_ALIASES: dict[tuple[str, str], list[str]] = {
    ("q_learning", "discrete"): ["q_learning_discrete"],
    ("q_learning", "continuous"): ["q_learning_continuous_minfuel"],
    ("q_learning", "fuel"): ["q_learning_discrete_minfuel"],
    ("q_learning", "minsteps"): ["q_learning_continuous_mintime"],
    ("sarsa", "discrete"): ["sarsa_discrete"],
    ("sarsa", "continuous"): ["sarsa_continuous_minfuel"],
    ("sarsa", "fuel"): ["sarsa_discrete_minfuel"],
    ("sarsa", "minsteps"): ["sarsa_continuous_mintime"],
    ("ppo", "discrete"): ["s1_ppo"],
    ("ppo", "continuous"): ["ppo"],
    ("ppo", "fuel"): ["s2_ppo"],
    ("ppo", "minsteps"): ["ppo_ns"],
    ("sac", "continuous"): ["sac"],
    ("sac", "minsteps"): ["sac_ns"],
}

_EVAL_LOGS: dict[tuple[str, str], str] = {
    ("ppo", "continuous"): "ppo_eval_logs/evaluations.npz",
    ("ppo", "minsteps"): "ppo_ns_eval_logs/evaluations.npz",
    ("sac", "continuous"): "sac_eval_logs/evaluations.npz",
    ("sac", "minsteps"): "sac_ns_eval_logs/evaluations.npz",
}

# Combinations that are deliberately not run (rendered as "Not applicable").
ALGORITHM_APPLICABILITY: dict[tuple[str, str], str] = {
    ("sac", "discrete"): "n/a",
    ("sac", "fuel"): "n/a",
}


def experiment_names(agent: str, scenario: str) -> list[str]:
    """Return concrete log/config experiment names for a canonical report cell."""
    return _EXPERIMENT_ALIASES.get((agent, scenario), [f"{agent}_{scenario}"])


def _models_dir_from_logs(logs_dir: Path) -> Path:
    return Path(logs_dir).parent / "models"


def _eval_log_path(logs_dir: Path, agent: str, scenario: str) -> Path | None:
    rel = _EVAL_LOGS.get((agent, scenario))
    if rel is None:
        return None
    return _models_dir_from_logs(logs_dir) / rel


def status(agent: str, scenario: str, logs_dir: Path) -> str:
    """Return 'done', 'eval', 'partial', 'pending', or 'n/a' for an (agent, scenario) pair."""
    if ALGORITHM_APPLICABILITY.get((agent, scenario)) == "n/a":
        return "n/a"
    logs_dir = Path(logs_dir)
    for experiment_name in experiment_names(agent, scenario):
        if any(logs_dir.glob(f"{experiment_name}_seed*_summary.json")):
            return "done"
    eval_log = _eval_log_path(logs_dir, agent, scenario)
    if eval_log is not None and eval_log.exists():
        return "eval"
    for experiment_name in experiment_names(agent, scenario):
        if any(logs_dir.glob(f"{experiment_name}_seed*")):
            return "partial"
        if any(_models_dir_from_logs(logs_dir).glob(f"{experiment_name}*")):
            return "partial"
    return "pending"


def discover_completed_runs(logs_dir: Path) -> dict[tuple[str, str], str]:
    """Return {(agent, scenario): 'done' | 'eval' | 'partial' | 'pending' | 'n/a'} for the full matrix."""
    return {
        (agent, scenario): status(agent, scenario, logs_dir)
        for agent in _AGENTS
        for scenario in _SCENARIOS
    }


def _load_seed_csvs(logs_dir: Path, experiment_name: str) -> list[pd.DataFrame]:
    dfs = []
    for seed in _SEEDS:
        csv = logs_dir / f"{experiment_name}_seed{seed}_episodes.csv"
        if csv.exists():
            dfs.append(pd.read_csv(csv))
    return dfs


def _load_seed_csvs_for(logs_dir: Path, agent: str, scenario: str) -> list[pd.DataFrame]:
    dfs: list[pd.DataFrame] = []
    for experiment_name in experiment_names(agent, scenario):
        dfs.extend(_load_seed_csvs(logs_dir, experiment_name))
    return dfs


def _read_summary(path: Path, agent: str, scenario: str, experiment_name: str) -> dict:
    data = json.loads(path.read_text())
    return {
        "agent": agent,
        "agent_label": _AGENT_LABELS[agent],
        "family": _AGENT_FAMILY[agent],
        "scenario": scenario,
        "scenario_label": _SCENARIO_LABELS[scenario],
        "experiment_name": experiment_name,
        "evidence_type": "summary_json",
        "seed": data.get("seed"),
        "eval_episodes": None,
        "reward_mean": data.get("reward_mean", data.get("mean_reward")),
        "reward_std": data.get("reward_std", data.get("std_reward")),
        "eval_mean_reward_last": data.get("eval_mean_reward_last", data.get("last_eval_mean_reward")),
        "eval_mean_reward_best": None,
        "success_rate": data.get("success_rate"),
        "mean_episode_length": data.get("mean_episode_length"),
    }


def _read_eval_log(path: Path, agent: str, scenario: str, experiment_name: str) -> dict:
    data = np.load(path)
    results = data["results"]
    ep_lengths = data["ep_lengths"]
    timesteps = data["timesteps"]
    mean_rewards = results.mean(axis=1)
    best_idx = int(np.argmax(mean_rewards))
    final_rewards = results[-1]
    best_rewards = results[best_idx]
    final_lengths = ep_lengths[-1]
    best_lengths = ep_lengths[best_idx]
    return {
        "agent": agent,
        "agent_label": _AGENT_LABELS[agent],
        "family": _AGENT_FAMILY[agent],
        "scenario": scenario,
        "scenario_label": _SCENARIO_LABELS[scenario],
        "experiment_name": experiment_name,
        "evidence_type": "eval_npz",
        "seed": 42,
        "eval_episodes": int(results.shape[1]),
        "reward_mean": float(best_rewards.mean()),
        "reward_std": float(best_rewards.std(ddof=0)),
        "eval_mean_reward_last": float(final_rewards.mean()),
        "eval_mean_reward_best": float(best_rewards.mean()),
        "success_rate": float(np.mean(best_lengths < 999)),
        "mean_episode_length": float(best_lengths.mean()),
        "best_timestep": int(timesteps[best_idx]),
    }


def _plot_eval_curve(agent: str, scenario: str, logs_dir: Path, figs_dir: Path) -> None:
    path = _eval_log_path(logs_dir, agent, scenario)
    if path is None or not path.exists():
        return
    data = np.load(path)
    timesteps = data["timesteps"]
    results = data["results"]
    mean = results.mean(axis=1)
    std = results.std(axis=1, ddof=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(timesteps, mean, linewidth=1.8, color={"ppo": "#1982C4", "sac": "#06A77D"}[agent])
    ax.fill_between(timesteps, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Eval reward")
    ax.set_title(f"{_AGENT_LABELS[agent]} eval reward - {_SCENARIO_LABELS[scenario]}")
    fig.tight_layout()
    fig.savefig(Path(figs_dir) / f"{agent}_{scenario}_eval_curve.png", dpi=150)
    plt.close(fig)


def summary_records(logs_dir: Path) -> list[dict]:
    """Return normalized per-seed summary rows for canonical report experiments."""
    logs_dir = Path(logs_dir)
    rows: list[dict] = []
    for agent in _AGENTS:
        for scenario in _SCENARIOS:
            if status(agent, scenario, logs_dir) not in {"done", "eval"}:
                continue
            for experiment_name in experiment_names(agent, scenario):
                for path in sorted(logs_dir.glob(f"{experiment_name}_seed*_summary.json")):
                    rows.append(_read_summary(path, agent, scenario, experiment_name))
            eval_log = _eval_log_path(logs_dir, agent, scenario)
            if eval_log is not None and eval_log.exists():
                rows.append(_read_eval_log(eval_log, agent, scenario, experiment_names(agent, scenario)[0]))
    return rows


def partial_artifacts(logs_dir: Path) -> list[dict]:
    """Return rows for report cells with artifacts but no comparable summaries."""
    logs_dir = Path(logs_dir)
    rows: list[dict] = []
    for agent in _AGENTS:
        for scenario in _SCENARIOS:
            if status(agent, scenario, logs_dir) != "partial":
                continue
            for experiment_name in experiment_names(agent, scenario):
                artifacts = sorted(logs_dir.glob(f"{experiment_name}_seed*"))
                artifacts.extend(sorted(_models_dir_from_logs(logs_dir).glob(f"{experiment_name}*")))
                if not artifacts:
                    continue
                rows.append(
                    {
                        "agent": agent,
                        "agent_label": _AGENT_LABELS[agent],
                        "family": _AGENT_FAMILY[agent],
                        "scenario": scenario,
                        "scenario_label": _SCENARIO_LABELS[scenario],
                        "experiment_name": experiment_name,
                        "artifact_count": len(artifacts),
                        "artifact_examples": ", ".join(path.name for path in artifacts[:3]),
                    }
                )
    return rows


def summary_dataframe(logs_dir: Path) -> pd.DataFrame:
    """Return normalized per-seed summary rows as a DataFrame."""
    return pd.DataFrame(summary_records(logs_dir))


def aggregate_summary(logs_dir: Path) -> pd.DataFrame:
    """Return one row per completed canonical (agent, scenario) pair."""
    df = summary_dataframe(logs_dir)
    if df.empty:
        return df
    grouped = (
        df.groupby(["family", "agent", "agent_label", "scenario", "scenario_label"], dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            reward_mean=("reward_mean", "mean"),
            reward_std=("reward_mean", "std"),
            eval_mean_reward_last=("eval_mean_reward_last", "mean"),
            eval_mean_reward_best=("eval_mean_reward_best", "mean"),
            success_rate=("success_rate", "mean"),
            mean_episode_length=("mean_episode_length", "mean"),
            eval_episodes=("eval_episodes", "max"),
            evidence_type=("evidence_type", lambda values: "+".join(sorted(set(map(str, values))))),
        )
        .reset_index()
    )
    return grouped


def _load_neat_agent(models_dir: Path, experiment_name: str, seed: int):
    """Load a NeatAgent from its final checkpoint, return (agent, config) or None."""
    try:
        from src.utils.config import load_config
        from src.training.train import build_env, build_agent
        from src.utils.seeding import seed_everything, seed_env

        scenario = experiment_name.replace("neat_", "")
        config = load_config(f"configs/neat_{scenario}.yaml")
        seed_everything(seed)
        env = build_env(config["env"])
        seed_env(env, seed)
        agent = build_agent(config, env)
        ckpt = models_dir / f"{experiment_name}_seed{seed}_final"
        agent.load(ckpt)
        env.close()
        return agent
    except Exception:
        return None


def _load_ga_agent(agent_key: str, scenario: str, models_dir: Path, seed: int):
    """Load a SimpleGA or CMA-ES agent from its final checkpoint, or None."""
    try:
        from src.utils.config import load_config
        from src.training.train import build_env, build_agent
        from src.utils.seeding import seed_everything, seed_env

        exp = f"{agent_key}_{scenario}"
        config = load_config(f"configs/{exp}.yaml")
        seed_everything(seed)
        env = build_env(config["env"])
        seed_env(env, seed)
        agent = build_agent(config, env)
        ckpt = models_dir / f"{exp}_seed{seed}_final"
        agent.load(ckpt)
        env.close()
        return agent
    except Exception:
        return None


def plot_for_family(
    agent: str,
    scenario: str,
    logs_dir: Path,
    figs_dir: Path,
) -> None:
    """Render the appropriate plot(s) for an (agent, scenario) pair, or skip with note."""
    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    s = status(agent, scenario, logs_dir)
    if s == "n/a":
        print(f"[{agent}/{scenario}] not applicable - skipping")
        return
    if s == "eval":
        print(f"[{agent}/{scenario}] eval log only - rendering eval curve")
        _plot_eval_curve(agent, scenario, logs_dir, figs_dir)
        return
    if s == "partial":
        print(f"[{agent}/{scenario}] partial artifacts only - skipping comparable plot")
        return
    if s == "pending":
        print(f"[{agent}/{scenario}] pending - skipping")
        return

    family = _AGENT_FAMILY[agent]
    if family == "evolutionary":
        _plot_evolutionary(agent, scenario, logs_dir, figs_dir)
    elif family == "value_based":
        _plot_value_based(agent, scenario, logs_dir, figs_dir)
    elif family == "policy_gradient":
        _plot_policy_gradient(agent, scenario, logs_dir, figs_dir)


def _plot_evolutionary(agent: str, scenario: str, logs_dir: Path, figs_dir: Path) -> None:
    exp = f"{agent}_{scenario}"
    dfs = _load_seed_csvs_for(logs_dir, agent, scenario)
    if not dfs:
        return
    color = {"simple_ga": "#E63946", "cma_es": "#457B9D", "neat": "#2A9D8F"}[agent]
    plot_multiseed_envelope(
        dfs, "reward",
        figs_dir / f"{exp}_seed_envelope.png",
        title=f"{_AGENT_LABELS[agent]} - {_SCENARIO_LABELS[scenario]}",
        window=10,
        color=color,
    )


def _plot_value_based(agent: str, scenario: str, logs_dir: Path, figs_dir: Path) -> None:
    exp = f"{agent}_{scenario}"
    dfs = _load_seed_csvs_for(logs_dir, agent, scenario)
    if not dfs:
        return
    color = {"q_learning": "#FFB703", "sarsa": "#FB8500", "dqn": "#8338EC"}[agent]
    plot_multiseed_envelope(
        dfs, "reward",
        figs_dir / f"{exp}_seed_envelope.png",
        title=f"{_AGENT_LABELS[agent]} - {_SCENARIO_LABELS[scenario]}",
        window=10,
        color=color,
    )


def _plot_policy_gradient(agent: str, scenario: str, logs_dir: Path, figs_dir: Path) -> None:
    exp = f"{agent}_{scenario}"
    dfs = _load_seed_csvs_for(logs_dir, agent, scenario)
    if not dfs:
        return
    color = {"reinforce": "#6A4C93", "ppo": "#1982C4", "sac": "#06A77D"}[agent]
    plot_multiseed_envelope(
        dfs, "reward",
        figs_dir / f"{exp}_seed_envelope.png",
        title=f"{_AGENT_LABELS[agent]} - {_SCENARIO_LABELS[scenario]}",
        window=10,
        color=color,
    )


def plot_per_scenario_comparison(
    scenario: str,
    logs_dir: Path,
    figs_dir: Path,
) -> None:
    """Cross-family reward comparison for one scenario - whichever agents have logs."""
    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[pd.DataFrame]] = {}
    for agent in _AGENTS:
        if status(agent, scenario, logs_dir) != "done":
            continue
        dfs = _load_seed_csvs_for(logs_dir, agent, scenario)
        if dfs:
            results[agent] = dfs

    if not results:
        print(f"[{scenario}] no completed runs - skipping cross-agent comparison")
        return

    plot_cross_agent_comparison(
        results, "reward",
        figs_dir / f"{scenario}_cross_agent_reward.png",
        title=f"Cross-agent reward - {_SCENARIO_LABELS[scenario]}",
        window=10,
    )


def _try_load_agent(agent: str, scenario: str, models_dir: Path, seed: int):
    """Return the loaded agent or None. Routes to the right loader by family."""
    if _AGENT_FAMILY[agent] == "evolutionary":
        if agent == "neat":
            return _load_neat_agent(models_dir, f"neat_{scenario}", seed)
        return _load_ga_agent(agent, scenario, models_dir, seed)
    return None


def generate_all(project_root: Path) -> None:
    logs_dir = project_root / "outputs" / "logs"
    models_dir = project_root / "outputs" / "models"
    figs_dir = project_root / "outputs" / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-agent x per-scenario reward envelopes (auto-skips pending/n/a).
    for agent in _AGENTS:
        for scenario in _SCENARIOS:
            plot_for_family(agent, scenario, logs_dir, figs_dir)

    # 2. Cross-agent comparison per scenario.
    for scenario in _SCENARIOS:
        plot_per_scenario_comparison(scenario, logs_dir, figs_dir)

    # 3. Population diversity plots for SimpleGA and CMA-ES.
    for agent in ["simple_ga", "cma_es"]:
        for scenario in _SCENARIOS:
            if status(agent, scenario, logs_dir) != "done":
                continue
            exp = f"{agent}_{scenario}"
            dfs = _load_seed_csvs_for(logs_dir, agent, scenario)
            if not dfs:
                continue
            merged = pd.concat(dfs, ignore_index=True)
            if "population_std" not in merged.columns:
                continue
            mean_div = merged.groupby("episode")["population_std"].mean().to_numpy()
            plot_population_diversity(
                mean_div.tolist(),
                figs_dir / f"{exp}_diversity.png",
                title=f"{_AGENT_LABELS[agent]} diversity - {_SCENARIO_LABELS[scenario]}",
            )

    # 4. Policy heatmaps for tabular-friendly scenarios.
    for scenario in ["discrete", "fuel"]:
        for agent in _AGENTS:
            if status(agent, scenario, logs_dir) != "done":
                continue
            a = _try_load_agent(agent, scenario, models_dir, _SEEDS[0])
            if a is None or getattr(a, "policy_table", None) is None:
                continue
            plot_policy_heatmap(
                a.policy_table,
                figs_dir / f"{agent}_{scenario}_policy_heatmap.png",
                title=f"{_AGENT_LABELS[agent]} policy - {_SCENARIO_LABELS[scenario]}",
            )

    # 5. NEAT topology snapshots.
    for scenario in ["discrete", "continuous"]:
        if status("neat", scenario, logs_dir) != "done":
            continue
        agent = _load_neat_agent(models_dir, f"neat_{scenario}", _SEEDS[0])
        if agent is None or agent._best_genome is None:
            continue

        draw_neat_genome(
            agent._best_genome, agent._config,
            figs_dir / f"neat_{scenario}_topology_final.png",
            title=f"NEAT final genome - {_SCENARIO_LABELS[scenario]}",
        )

        checkpoints = sorted(models_dir.glob(f"neat_{scenario}_seed{_SEEDS[0]}_ep*.json"))
        snapshots = []
        for ckpt in checkpoints:
            try:
                gen = int(ckpt.stem.split("_ep")[-1])
            except ValueError:
                continue
            snap_agent = _load_neat_agent(models_dir, f"neat_{scenario}", _SEEDS[0])
            if snap_agent is None:
                continue
            snap_agent.load(ckpt.with_suffix(""))
            if snap_agent._best_genome is not None:
                snapshots.append((gen, copy.deepcopy(snap_agent._best_genome)))

        if snapshots:
            draw_neat_growth_sequence(
                snapshots, agent._config,
                figs_dir / f"neat_{scenario}_growth_sequence.png",
            )
            draw_neat_animation(
                snapshots, agent._config,
                figs_dir / f"neat_{scenario}_topology_evolution.gif",
                fps=4,
            )

    print(f"Figures written to {figs_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report figures from training outputs.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all(Path(args.project_root))

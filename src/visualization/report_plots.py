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

from src.visualization.ga_plots import (
    plot_cross_agent_comparison,
    plot_ga_fitness_curve,
    plot_multiseed_envelope,
    plot_population_diversity,
)
from src.visualization.policy_maps import plot_policy_heatmap
from src.visualization.neat_viz import draw_neat_genome, draw_neat_growth_sequence, draw_neat_animation


_AGENTS = ["simple_ga", "cma_es", "neat"]
_SCENARIOS = ["discrete", "continuous", "fuel", "minsteps"]
_SCENARIO_LABELS = {
    "discrete": "MountainCar-v0 (min steps)",
    "continuous": "MountainCarContinuous-v0 (min fuel)",
    "fuel": "MountainCar-v0 (min fuel)",
    "minsteps": "MountainCarContinuous-v0 (min steps)",
}
_AGENT_LABELS = {
    "simple_ga": "Simple GA",
    "cma_es": "CMA-ES",
    "neat": "NEAT",
}
_SEEDS = [7, 21, 42]


def _load_seed_csvs(logs_dir: Path, experiment_name: str) -> list[pd.DataFrame]:
    dfs = []
    for seed in _SEEDS:
        csv = logs_dir / f"{experiment_name}_seed{seed}_episodes.csv"
        if csv.exists():
            dfs.append(pd.read_csv(csv))
    return dfs


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


def generate_all(project_root: Path) -> None:
    logs_dir = project_root / "outputs" / "logs"
    models_dir = project_root / "outputs" / "models"
    figs_dir = project_root / "outputs" / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Per-agent, per-scenario multi-seed envelope plots (12 plots)
    # ------------------------------------------------------------------ #
    for agent in _AGENTS:
        for scenario in _SCENARIOS:
            exp = f"{agent}_{scenario}"
            dfs = _load_seed_csvs(logs_dir, exp)
            if not dfs:
                continue
            label = f"{_AGENT_LABELS[agent]} — {_SCENARIO_LABELS[scenario]}"
            color = {"simple_ga": "#E63946", "cma_es": "#457B9D", "neat": "#2A9D8F"}[agent]
            plot_multiseed_envelope(
                dfs, "reward",
                figs_dir / f"{exp}_seed_envelope.png",
                title=label,
                window=10,
                color=color,
            )

    # ------------------------------------------------------------------ #
    # 2. Cross-agent reward comparison per scenario (4 plots)
    # ------------------------------------------------------------------ #
    for scenario in _SCENARIOS:
        results = {}
        for agent in _AGENTS:
            dfs = _load_seed_csvs(logs_dir, f"{agent}_{scenario}")
            if dfs:
                results[agent] = dfs
        if results:
            plot_cross_agent_comparison(
                results, "reward",
                figs_dir / f"{scenario}_cross_agent_reward.png",
                title=f"Cross-agent reward — {_SCENARIO_LABELS[scenario]}",
                window=10,
            )

    # ------------------------------------------------------------------ #
    # 3. Population diversity plots for SimpleGA and CMA-ES (8 plots)
    # ------------------------------------------------------------------ #
    for agent in ["simple_ga", "cma_es"]:
        for scenario in _SCENARIOS:
            exp = f"{agent}_{scenario}"
            dfs = _load_seed_csvs(logs_dir, exp)
            if not dfs:
                continue
            merged = pd.concat(dfs, ignore_index=True)
            if "population_std" not in merged.columns:
                continue
            mean_div = merged.groupby("episode")["population_std"].mean().to_numpy()
            plot_population_diversity(
                mean_div.tolist(),
                figs_dir / f"{exp}_diversity.png",
                title=f"{_AGENT_LABELS[agent]} diversity — {_SCENARIO_LABELS[scenario]}",
            )

    # ------------------------------------------------------------------ #
    # 4. Policy heatmaps for discrete scenarios (6 plots)
    # ------------------------------------------------------------------ #
    for scenario in ["discrete", "fuel"]:
        for agent in _AGENTS:
            a = (
                _load_neat_agent(models_dir, f"neat_{scenario}", _SEEDS[0])
                if agent == "neat"
                else _load_ga_agent(agent, scenario, models_dir, _SEEDS[0])
            )
            if a is None:
                continue
            table = a.policy_table
            if table is None:
                continue
            plot_policy_heatmap(
                table,
                figs_dir / f"{agent}_{scenario}_policy_heatmap.png",
                title=f"{_AGENT_LABELS[agent]} policy — {_SCENARIO_LABELS[scenario]}",
            )

    # ------------------------------------------------------------------ #
    # 5. NEAT topology: final genome + growth sequence (2+2 plots)
    # ------------------------------------------------------------------ #
    for scenario in ["discrete", "continuous"]:
        agent = _load_neat_agent(models_dir, f"neat_{scenario}", _SEEDS[0])
        if agent is None or agent._best_genome is None:
            continue

        draw_neat_genome(
            agent._best_genome, agent._config,
            figs_dir / f"neat_{scenario}_topology_final.png",
            title=f"NEAT final genome — {_SCENARIO_LABELS[scenario]}",
        )

        # growth sequence: look for checkpoint snapshots at save_every intervals
        checkpoints = sorted(models_dir.glob(f"neat_{scenario}_seed{_SEEDS[0]}_ep*.json"))
        snapshots = []
        for ckpt in checkpoints:
            gen_str = ckpt.stem.split("_ep")[-1]
            try:
                gen = int(gen_str)
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

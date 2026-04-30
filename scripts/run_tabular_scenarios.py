"""Train tabular Q-learning and SARSA on the four PDF scenarios.

The four scenarios (from the assignment PDF, "Possible Variation: Scenarios
to be analyzed" matrix) are:

    (1) discrete  / min-steps  -> MountainCar-v0           (default reward)
    (2) continuous / min-fuel  -> MountainCarContinuous-v0 (default -0.1*a^2)
    (3) discrete  / min-fuel   -> MountainCar-v0 + DiscreteFuelOnlyCostWrapper
    (4) continuous / min-time  -> MountainCarContinuous-v0 + ContinuousStepCostWrapper

For each scenario the corresponding YAML config under `configs/` is loaded
unchanged and used to drive the existing per-seed trainers. Outputs land in
`outputs/logs/` and `outputs/models/`.

Examples:
    # Dry-run (print plan only):
    python scripts/run_tabular_scenarios.py --dry-run

    # Full run (long, ~tens of minutes):
    python scripts/run_tabular_scenarios.py

    # Subset:
    python scripts/run_tabular_scenarios.py --agents q_learning --scenarios discrete_minfuel

    # Quick smoke test (overrides config episodes/seeds for fast validation):
    python scripts/run_tabular_scenarios.py --smoke --episodes 50 --seeds 7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import ensure_paths, load_config
from src.utils.logging import dump_json

PROJECT_ROOT = Path(".").resolve()

SCENARIO_CONFIGS: dict[str, dict[str, Path]] = {
    "discrete_minsteps": {
        "q_learning": PROJECT_ROOT / "configs" / "q_learning_discrete.yaml",
        "sarsa": PROJECT_ROOT / "configs" / "sarsa_discrete.yaml",
    },
    "continuous_minfuel": {
        "q_learning": PROJECT_ROOT / "configs" / "q_learning_continuous_minfuel.yaml",
        "sarsa": PROJECT_ROOT / "configs" / "sarsa_continuous_minfuel.yaml",
    },
    "discrete_minfuel": {
        "q_learning": PROJECT_ROOT / "configs" / "q_learning_discrete_minfuel.yaml",
        "sarsa": PROJECT_ROOT / "configs" / "sarsa_discrete_minfuel.yaml",
    },
    "continuous_mintime": {
        "q_learning": PROJECT_ROOT / "configs" / "q_learning_continuous_mintime.yaml",
        "sarsa": PROJECT_ROOT / "configs" / "sarsa_continuous_mintime.yaml",
    },
}

DEFAULT_AGENTS = ["q_learning", "sarsa"]
DEFAULT_SCENARIOS = list(SCENARIO_CONFIGS.keys())


def _import_trainer(agent: str):
    if agent == "q_learning":
        from src.q_learning.train import train_single_seed
        return train_single_seed
    if agent == "sarsa":
        from src.sarsa.train import train_single_seed
        return train_single_seed
    raise ValueError(f"No tabular trainer available for agent '{agent}'.")


def _apply_smoke_overrides(
    config: dict[str, Any],
    episodes: int | None,
    seeds: list[int] | None,
) -> dict[str, Any]:
    """Optional fast-path overrides for quick validation."""
    if episodes is not None:
        config["train"]["episodes"] = int(episodes)
        config["train"]["eval_every"] = max(1, int(episodes) // 2)
    if seeds:
        config["train"]["seeds"] = list(seeds)
    return config


def _aggregate(summaries: list[dict[str, Any]], config: dict[str, Any], algorithm: str) -> dict[str, Any]:
    if not summaries:
        return {}
    df = pd.DataFrame(summaries)
    return {
        "experiment_name": config["experiment_name"],
        "algorithm": algorithm,
        "n_seeds": int(len(df)),
        "mean_reward": float(df["mean_reward"].mean()),
        "std_reward_across_seeds": float(df["mean_reward"].std(ddof=0)),
        "mean_success_rate": float(df["success_rate"].mean()),
        "mean_episode_length": float(df["mean_episode_length"].mean()),
    }


def main(
    dry_run: bool,
    agents: list[str],
    scenarios: list[str],
    smoke_episodes: int | None,
    smoke_seeds: list[int] | None,
) -> None:
    for scenario in scenarios:
        if scenario not in SCENARIO_CONFIGS:
            print(f"Unknown scenario '{scenario}', skipping.")
            continue

        for agent in agents:
            cfg_path = SCENARIO_CONFIGS[scenario].get(agent)
            if cfg_path is None:
                print(f"No config for agent={agent} scenario={scenario}, skipping.")
                continue
            if not cfg_path.exists():
                print(f"Config file not found: {cfg_path}, skipping.")
                continue

            config = load_config(cfg_path)
            config = _apply_smoke_overrides(config, smoke_episodes, smoke_seeds)

            plan = {
                "agent": agent,
                "scenario": scenario,
                "experiment_name": config["experiment_name"],
                "env_id": config["env"]["id"],
                "wrappers": config["env"].get("wrappers", {}),
                "episodes": config["train"]["episodes"],
                "seeds": config["train"]["seeds"],
            }
            print("PLAN:", json.dumps(plan, indent=2, default=str))

            if dry_run:
                continue

            train_fn = _import_trainer(agent)
            summaries: list[dict[str, Any]] = []
            for seed in config["train"]["seeds"]:
                print(f"-- Running {config['experiment_name']} seed={seed}")
                try:
                    metrics = train_fn(config=config, project_root=PROJECT_ROOT, seed=int(seed))
                    summaries.append(metrics)
                    print(f"   done: mean_reward={metrics.get('mean_reward'):.2f} "
                          f"success_rate={metrics.get('success_rate'):.2f}")
                except Exception as exc:
                    print(f"   FAILED ({type(exc).__name__}): {exc}")

            if summaries:
                aggregate = _aggregate(summaries, config, algorithm=agent)
                paths = ensure_paths(config, PROJECT_ROOT)
                pd.DataFrame(summaries).to_csv(
                    paths["logs_dir"] / f"{config['experiment_name']}_seed_results.csv",
                    index=False,
                )
                dump_json(
                    aggregate,
                    paths["logs_dir"] / f"{config['experiment_name']}_aggregate.json",
                )
                print("AGGREGATE:", json.dumps(aggregate, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without training.")
    parser.add_argument("--agents", nargs="+", default=DEFAULT_AGENTS, choices=DEFAULT_AGENTS)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        choices=DEFAULT_SCENARIOS,
        help="Scenario keys to run (default: all four).",
    )
    parser.add_argument("--smoke", action="store_true", help="Use smoke-test defaults (50 episodes, single seed).")
    parser.add_argument("--episodes", type=int, default=None, help="Override episodes per run (for fast iteration).")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override seeds list.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    smoke_episodes = args.episodes
    smoke_seeds = args.seeds
    if args.smoke:
        smoke_episodes = smoke_episodes or 50
        smoke_seeds = smoke_seeds or [7]

    main(
        dry_run=args.dry_run,
        agents=args.agents,
        scenarios=args.scenarios,
        smoke_episodes=smoke_episodes,
        smoke_seeds=smoke_seeds,
    )

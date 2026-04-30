"""Re-evaluate teacher-pretrained REINFORCE checkpoints using the *student*.

Background
----------
``REINFORCEBaseline.predict(deterministic=True)`` falls back to the teacher
model when one is configured and the action space is discrete. That means
every per-episode "training reward" and "eval reward" recorded for
``reinforce_fuel`` (the only discrete teacher-using config) actually
reflects the teacher's policy, not the student's. The student weights are
trained correctly, but the JSON summaries report the teacher's numbers.

This script reloads each ``reinforce_*_final.zip`` checkpoint that was
trained with a teacher, disables the teacher fallback, and runs N
deterministic rollouts with the student alone. It then overwrites the
per-seed ``*_summary.json`` (and rewrites ``*_seed_results.csv`` plus
``*_aggregate.json``) with the honest numbers.

Continuous-action REINFORCE configs are untouched: their teacher path is
only used during behaviour-cloning pretraining and never at inference, so
the recorded numbers were always the student's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.reinforce import REINFORCEBaseline  # noqa: E402
from src.training.train import build_env  # noqa: E402
from src.utils.seeding import seed_env, seed_everything  # noqa: E402

DEFAULT_CONFIGS = ["configs/reinforce_fuel.yaml"]


def _coerce_action(action: Any, env: Any) -> Any:
    import gymnasium as gym

    if isinstance(env.action_space, gym.spaces.Discrete):
        return int(np.asarray(action).item())
    if isinstance(env.action_space, gym.spaces.Box):
        arr = np.asarray(action, dtype=np.float32).reshape(env.action_space.shape)
        return np.clip(arr, env.action_space.low, env.action_space.high)
    return action


def _rollout(env: Any, agent: REINFORCEBaseline, max_steps: int = 1000) -> tuple[float, int, bool]:
    obs, _ = env.reset()
    total = 0.0
    steps = 0
    for _ in range(max_steps):
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(_coerce_action(action, env))
        total += float(reward)
        steps += 1
        if terminated or truncated:
            return total, steps, bool(terminated and not truncated)
    return total, steps, False


def reeval_seed(cfg: dict[str, Any], seed: int, n_episodes: int) -> dict[str, Any]:
    seed_everything(seed)
    env = build_env(cfg["env"])
    seed_env(env, seed)

    rcfg = cfg["reinforce"]
    agent = REINFORCEBaseline(
        env=env,
        learning_rate=float(rcfg["learning_rate"]),
        gamma=float(rcfg["gamma"]),
        entropy_coef=float(rcfg.get("entropy_coef", 0.01)),
        value_fn_coef=float(rcfg.get("value_fn_coef", 0.5)),
        hidden_sizes=list(rcfg["hidden_sizes"]),
        batch_episodes=int(rcfg.get("batch_episodes", 4)),
        # NOTE: explicitly no teacher here so predict() uses the student.
        teacher_checkpoint=None,
    )
    final_path = ROOT / "outputs" / "models" / f"{cfg['experiment_name']}_seed{seed}_final.zip"
    if not final_path.exists():
        # ``REINFORCEBaseline.save`` writes a torch file, not a zip; SB3-style
        # paths add the .zip suffix automatically. Try both.
        alt = ROOT / "outputs" / "models" / f"{cfg['experiment_name']}_seed{seed}_final"
        final_path = alt if alt.exists() else final_path
    agent.load(str(final_path), env)

    # Hard-disable the teacher fallback in case load() restored it from the
    # checkpoint payload.
    agent.teacher_model = None

    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    max_steps = int(cfg["train"].get("max_steps_per_episode", 1000))
    for _ in range(n_episodes):
        r, ell, s = _rollout(env, agent, max_steps=max_steps)
        rewards.append(r)
        lengths.append(ell)
        successes.append(float(s))
    env.close()

    return {
        "experiment_name": cfg["experiment_name"],
        "seed": int(seed),
        "agent": "reinforce",
        "env_id": cfg["env"]["id"],
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_episode_length": float(np.mean(lengths)),
        "eval_mean_reward_last": float(np.mean(rewards)),
        "final_model_path": str(final_path),
        "_student_eval": True,
        "_n_eval_episodes": int(n_episodes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Student-only re-eval for teacher-pretrained REINFORCE.")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()

    print(f"Re-evaluating {len(args.configs)} configs with student-only policy", flush=True)
    for cfg_path in args.configs:
        cfg = yaml.safe_load((ROOT / cfg_path).read_text())
        seeds = cfg["train"]["seeds"]
        per_seed_results = []
        for seed in seeds:
            print(f"  [{cfg['experiment_name']}] seed={seed} ...", flush=True)
            try:
                metrics = reeval_seed(cfg, int(seed), n_episodes=args.episodes)
            except FileNotFoundError as exc:
                print(f"    skipped (missing checkpoint): {exc}", flush=True)
                continue
            per_seed_results.append(metrics)
            summary_path = ROOT / "outputs" / "logs" / f"{cfg['experiment_name']}_seed{seed}_summary.json"
            summary_path.write_text(json.dumps(metrics, indent=2))
            print(
                f"    reward_mean={metrics['reward_mean']:.2f} "
                f"success_rate={metrics['success_rate']:.0%} "
                f"length={metrics['mean_episode_length']:.0f}",
                flush=True,
            )

        if not per_seed_results:
            continue

        # Refresh the seed-results CSV.
        import pandas as pd

        seed_df = pd.DataFrame(per_seed_results)
        seed_df.to_csv(
            ROOT / "outputs" / "logs" / f"{cfg['experiment_name']}_seed_results.csv",
            index=False,
        )
        aggregate = {
            "experiment_name": cfg["experiment_name"],
            "agent": "reinforce",
            "env_id": cfg["env"]["id"],
            "n_seeds": int(len(seed_df)),
            "reward_mean": float(seed_df["reward_mean"].mean()),
            "reward_std_across_seeds": float(seed_df["reward_mean"].std(ddof=0)),
            "success_rate_mean": float(seed_df["success_rate"].mean()),
            "success_rate_std": float(seed_df["success_rate"].std(ddof=0)),
            "episode_length_mean": float(seed_df["mean_episode_length"].mean()),
            "_student_eval": True,
        }
        (ROOT / "outputs" / "logs" / f"{cfg['experiment_name']}_aggregate.json").write_text(
            json.dumps(aggregate, indent=2)
        )
        print(
            f"  aggregate: reward_mean={aggregate['reward_mean']:.2f} "
            f"success_rate_mean={aggregate['success_rate_mean']:.0%}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

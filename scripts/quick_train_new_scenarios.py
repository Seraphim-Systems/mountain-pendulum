"""Run a short training pass for each new DQN/REINFORCE scenario config.

Loads each YAML, shrinks the budget (1 seed, ~30 episodes, 1 eval batch),
and invokes the real ``train_single_seed`` pipeline. Catches and reports any
failure per config so all six can be verified in one go without aborting
on the first error.
"""

from __future__ import annotations

import copy
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.train import train_single_seed  # noqa: E402
from src.utils.config import load_config  # noqa: E402


CONFIGS = [
    "configs/dqn_fuel.yaml",
    "configs/dqn_continuous.yaml",
    "configs/dqn_minsteps.yaml",
    "configs/reinforce_fuel.yaml",
    "configs/reinforce_continuous.yaml",
    "configs/reinforce_minsteps.yaml",
]


def shrink(cfg: dict) -> dict:
    """Return a copy of cfg with a tiny training budget for fast verification."""
    cfg = copy.deepcopy(cfg)
    train = cfg.setdefault("train", {})
    train["episodes"] = 30
    train["eval_every"] = 15
    train["eval_episodes"] = 3
    train["save_every"] = 30
    train["seeds"] = [7]
    # REINFORCE batches multiple env episodes per outer "episode"; on the
    # 999-step continuous variants this dominates wall-clock time, so cap it
    # to 2 for the smoke test (real training keeps the YAML default).
    if cfg.get("agent") == "reinforce":
        cfg.setdefault("reinforce", {})["batch_episodes"] = 2
    cfg["experiment_name"] = f"{cfg['experiment_name']}_smoke"
    return cfg


def run_one(cfg_path: str) -> tuple[bool, str, float]:
    cfg = shrink(load_config(ROOT / cfg_path))
    t0 = time.perf_counter()
    try:
        metrics = train_single_seed(cfg, seed=7, project_root=ROOT, render=False)
        elapsed = time.perf_counter() - t0
        msg = (
            f"reward_mean={metrics['reward_mean']:.2f}  "
            f"success_rate={metrics['success_rate']:.0%}  "
            f"eval_last={metrics['eval_mean_reward_last']:.2f}  "
            f"len_mean={metrics['mean_episode_length']:.0f}"
        )
        return True, msg, elapsed
    except Exception as exc:  # pragma: no cover - reporting only
        elapsed = time.perf_counter() - t0
        return False, f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc()}", elapsed


def main() -> int:
    results = []
    for cfg_path in CONFIGS:
        print(f"\n=== {cfg_path} (1 seed, 30 episodes) ===", flush=True)
        ok, msg, elapsed = run_one(cfg_path)
        status = "PASS" if ok else "FAIL"
        results.append((cfg_path, ok, elapsed))
        print(f"[{status}] {cfg_path} ({elapsed:.1f}s) — {msg}", flush=True)

    print("\n--- Summary ---")
    for cfg_path, ok, elapsed in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {elapsed:6.1f}s  {cfg_path}")
    fails = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - fails}/{len(results)} configs ran cleanly")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

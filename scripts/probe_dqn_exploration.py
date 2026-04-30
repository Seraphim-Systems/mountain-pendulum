"""Verify that DQN exploration_rate decays across the *full* training horizon.

Without the fix, ``model.exploration_rate`` plummets to ``exploration_final_eps``
after the first ``learn(max_steps_per_episode)`` call. With the fix in
``DQNBaseline``, it should decay linearly across the full ``episodes *
max_steps_per_episode`` budget instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.train import build_agent, build_env, build_vec_env  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402

CFG_PATH = ROOT / "configs" / "dqn_fuel.yaml"


def main() -> int:
    cfg = load_config(CFG_PATH)
    seed_everything(0)
    n_envs = int(cfg["dqn"].get("n_envs", 1))
    if n_envs > 1:
        env = build_vec_env(cfg["env"], n_envs=n_envs, base_seed=0)
    else:
        env = build_env(cfg["env"])

    agent = build_agent(cfg, env)
    max_steps = int(cfg["train"]["max_steps_per_episode"])
    episodes = int(cfg["train"]["episodes"])
    print(f"Total budget = {episodes} episodes * {max_steps} steps = {episodes*max_steps}")
    print(f"exploration_fraction = {agent.model.exploration_fraction}")
    print()

    print(f"{'call':>5} {'num_steps':>10} {'_total_ts':>10} {'eps':>8}")
    for call_idx in range(1, 11):
        agent.learn(total_timesteps=max_steps)
        print(
            f"{call_idx:>5} "
            f"{agent.model.num_timesteps:>10} "
            f"{agent.model._total_timesteps:>10} "
            f"{agent.model.exploration_rate:>8.4f}"
        )

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

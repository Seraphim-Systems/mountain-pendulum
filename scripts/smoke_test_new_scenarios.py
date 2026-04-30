"""Smoke-test the new DQN/REINFORCE scenario configs.

Builds env + agent, runs a tiny ``learn``/training step, then a deterministic
rollout for each config. Prints a one-line PASS/FAIL per config. Intended to
be fast (a few seconds total) and surface obvious shape/type errors only.

This script imports only what the DQN / REINFORCE paths need so it does not
pull in optional CMA-ES / NEAT dependencies.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.dqn import DQNBaseline  # noqa: E402
from src.agents.reinforce import REINFORCEBaseline  # noqa: E402
from src.envs.mountain_car_continuous import make_continuous_env  # noqa: E402
from src.envs.mountain_car_discrete import make_discrete_env  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seeding import seed_env, seed_everything  # noqa: E402


CONFIGS = [
    "configs/dqn_fuel.yaml",
    "configs/dqn_continuous.yaml",
    "configs/dqn_minsteps.yaml",
    "configs/reinforce_fuel.yaml",
    "configs/reinforce_continuous.yaml",
    "configs/reinforce_minsteps.yaml",
]


def build_env_local(env_cfg: dict[str, Any]) -> Any:
    env_id = env_cfg["id"]
    wrappers = env_cfg.get("wrappers", {})
    if env_id == "MountainCar-v0":
        return make_discrete_env(wrappers=wrappers)
    if env_id == "MountainCarContinuous-v0":
        return make_continuous_env(wrappers=wrappers)
    raise ValueError(f"Unsupported env id: {env_id}")


def build_agent_local(config: dict[str, Any], env: Any) -> Any:
    name = config["agent"]
    if name == "dqn":
        cfg = config["dqn"]
        return DQNBaseline(
            env=env,
            learning_rate=float(cfg["learning_rate"]),
            gamma=float(cfg["gamma"]),
            epsilon_start=float(cfg["epsilon_start"]),
            epsilon_end=float(cfg["epsilon_end"]),
            epsilon_decay=float(cfg["epsilon_decay"]),
            batch_size=int(cfg["batch_size"]),
            replay_buffer_size=int(cfg["replay_buffer_size"]),
            learning_starts=int(cfg.get("min_buffer_size", 1000)),
            target_update_freq=int(cfg["target_update_freq"]),
            hidden_sizes=list(cfg["hidden_sizes"]),
            exploration_fraction=(
                float(cfg["exploration_fraction"]) if "exploration_fraction" in cfg else None
            ),
        )
    if name == "reinforce":
        cfg = config["reinforce"]
        return REINFORCEBaseline(
            env=env,
            learning_rate=float(cfg["learning_rate"]),
            gamma=float(cfg["gamma"]),
            entropy_coef=float(cfg.get("entropy_coef", 0.0)),
            value_fn_coef=float(cfg.get("value_fn_coef", 0.5)),
            hidden_sizes=list(cfg.get("hidden_sizes", [128, 128])),
            batch_episodes=int(cfg.get("batch_episodes", 4)),
        )
    raise ValueError(f"Smoke test only handles dqn / reinforce; got {name}")


def coerce_action(action: Any, env: Any) -> Any:
    if isinstance(env.action_space, gym.spaces.Discrete):
        return int(np.asarray(action).item())
    if isinstance(env.action_space, gym.spaces.Box):
        arr = np.asarray(action, dtype=np.float32).reshape(env.action_space.shape)
        return np.clip(arr, env.action_space.low, env.action_space.high)
    return action


def rollout(env: Any, agent: Any) -> tuple[float, int, bool]:
    obs, _ = env.reset()
    total = 0.0
    steps = 0
    while True:
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(coerce_action(action, env))
        total += float(reward)
        steps += 1
        if terminated or truncated:
            return total, steps, bool(terminated and not truncated)


def smoke_one(cfg_path: str) -> tuple[bool, str]:
    cfg = load_config(ROOT / cfg_path)
    seed_everything(0)
    env = build_env_local(cfg["env"])
    seed_env(env, 0)
    agent = build_agent_local(cfg, env)

    obs_shape = env.observation_space.shape
    act_space = type(env.action_space).__name__
    info = f"obs={obs_shape} act={act_space}"

    try:
        agent.learn(total_timesteps=64)
        reward, length, success = rollout(env, agent)
        env.close()
        return True, f"{info} | learn ok | rollout reward={reward:.2f} len={length} success={success}"
    except Exception as exc:  # pragma: no cover - smoke test reporting
        env.close()
        return False, f"{info} | {exc.__class__.__name__}: {exc}\n{traceback.format_exc()}"


def main() -> int:
    failures = 0
    for cfg in CONFIGS:
        ok, msg = smoke_one(cfg)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {cfg}: {msg}")
        if not ok:
            failures += 1
    print()
    print(f"Smoke test complete: {len(CONFIGS) - failures}/{len(CONFIGS)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

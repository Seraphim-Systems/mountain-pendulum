"""Environment constructor and metadata for MountainCarContinuous-v0."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from .wrappers import ContinuousStepCostWrapper, EnergyShapingRewardWrapper, RecordEpisodeStatsWrapper


def make_continuous_env(
    render_mode: str | None = None,
    wrappers: dict[str, Any] | None = None,
) -> gym.Env:
    """Create MountainCarContinuous-v0 with optional explicit wrappers."""
    env = gym.make("MountainCarContinuous-v0", render_mode=render_mode)
    wrappers = wrappers or {}

    if wrappers.get("energy_shaping") and wrappers.get("step_cost"):
        raise ValueError(
            "energy_shaping and step_cost are mutually exclusive reward wrappers."
        )

    if wrappers.get("energy_shaping"):
        kwargs = (
            wrappers["energy_shaping"]
            if isinstance(wrappers["energy_shaping"], dict)
            else {}
        )
        env = EnergyShapingRewardWrapper(env, **kwargs)

    if wrappers.get("step_cost"):
        env = ContinuousStepCostWrapper(env)

    if wrappers.get("record_episode_stats"):
        env = RecordEpisodeStatsWrapper(env)

    return env


def get_continuous_env_spec() -> dict[str, Any]:
    """Return key MountainCarContinuous-v0 behavior used in assignment checks."""
    env = gym.make("MountainCarContinuous-v0")
    spec = {
        "id": "MountainCarContinuous-v0",
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "action_range": (
            float(env.action_space.low[0]),
            float(env.action_space.high[0]),
        ),
        "max_episode_steps": int(env.spec.max_episode_steps),
    }
    env.close()
    return spec

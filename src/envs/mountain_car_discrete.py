"""Environment constructor and metadata for MountainCar-v0."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from .wrappers import (
    AugmentStateWrapper,
    DiscreteFuelCostWrapper,
    DiscretizeStateWrapper,
    RecordEpisodeStatsWrapper,
    EnergyShapingRewardWrapper,
)


def make_discrete_env(
    render_mode: str | None = None,
    wrappers: dict[str, Any] | None = None,
) -> gym.Env:
    """Create MountainCar-v0 with optional explicit wrappers."""
    env = gym.make("MountainCar-v0", render_mode=render_mode)
    wrappers = wrappers or {}

    if wrappers.get("discretize_state") and wrappers.get("augment_state"):
        raise ValueError(
            "discretize_state and augment_state are mutually exclusive: "
            "DiscretizeStateWrapper produces MultiDiscrete, which AugmentStateWrapper cannot consume."
        )

    if wrappers.get("discretize_state"):
        kwargs = (
            wrappers["discretize_state"]
            if isinstance(wrappers["discretize_state"], dict)
            else {}
        )
        env = DiscretizeStateWrapper(env, **kwargs)

    if wrappers.get("augment_state"):
        env = AugmentStateWrapper(env)

    if wrappers.get("fuel_cost"):
        env = DiscreteFuelCostWrapper(env)
        
    if wrappers.get("energy_shaping"):
        kwargs = (
            wrappers["energy_shaping"]
            if isinstance(wrappers["energy_shaping"], dict)
            else {}
        )
        env = EnergyShapingRewardWrapper(env, **kwargs)

    if wrappers.get("record_episode_stats"):
        env = RecordEpisodeStatsWrapper(env)

    return env


def get_discrete_env_spec() -> dict[str, Any]:
    """Return key MountainCar-v0 behavior used in assignment checks."""
    env = gym.make("MountainCar-v0")
    spec = {
        "id": "MountainCar-v0",
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "n_actions": int(env.action_space.n),
        "action_meaning": {0: "push_left", 1: "no_push", 2: "push_right"},
        "max_episode_steps": int(env.spec.max_episode_steps),
    }
    env.close()
    return spec

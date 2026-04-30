"""Unified env builder for tabular methods (Q-learning, SARSA).

Supports both `MountainCar-v0` (native discrete actions) and
`MountainCarContinuous-v0` (continuous actions, discretized via
`DiscretizeActionWrapper`). All four assignment scenarios are expressible
through wrapper combinations:

- Scenario 1 (discrete, min-steps):   env_id=MountainCar-v0,            no extra reward wrapper
- Scenario 2 (continuous, min-fuel):  env_id=MountainCarContinuous-v0,  default reward (-0.1*a^2)
- Scenario 3 (discrete, min-fuel):    env_id=MountainCar-v0,            wrappers.discrete_fuel_only=true
- Scenario 4 (continuous, min-time):  env_id=MountainCarContinuous-v0,  wrappers.step_cost=true

Tabular methods always need `discretize_state` and (for continuous envs) a
`discretize_action.n_actions` configuration.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from .wrappers import (
    ContinuousFuelCostWrapper,
    ContinuousStepCostWrapper,
    DiscreteFuelCostWrapper,
    DiscreteFuelOnlyCostWrapper,
    DiscretizeActionWrapper,
    DiscretizeStateWrapper,
    EnergyShapingRewardWrapper,
    PotentialShapingWrapper,
    RecordEpisodeStatsWrapper,
)

DISCRETE_ENV_IDS = {"MountainCar-v0"}
CONTINUOUS_ENV_IDS = {"MountainCarContinuous-v0"}
SUPPORTED_ENV_IDS = DISCRETE_ENV_IDS | CONTINUOUS_ENV_IDS


def make_tabular_env(
    env_id: str,
    wrappers: dict[str, Any] | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Build a Mountain Car env wrapped to support tabular Q-learning / SARSA.

    Wrapper order (applied outside-in, so reward wrappers see the raw env):

      base env
       -> reward shaping (one of: discrete_fuel_only / discrete_fuel_cost /
                                  step_cost / energy_shaping)
       -> DiscretizeActionWrapper        (continuous envs only)
       -> DiscretizeStateWrapper         (always required for tabular)
       -> RecordEpisodeStatsWrapper      (optional, for monitoring)
    """
    if env_id not in SUPPORTED_ENV_IDS:
        raise ValueError(
            f"Unsupported env_id '{env_id}'. Supported: {sorted(SUPPORTED_ENV_IDS)}"
        )

    wrappers = wrappers or {}
    if "discretize_state" not in wrappers:
        raise ValueError(
            "Tabular envs require env.wrappers.discretize_state.n_bins."
        )

    extra: dict[str, Any] = {}
    max_steps_override = wrappers.get("max_episode_steps")
    if max_steps_override is not None:
        extra["max_episode_steps"] = int(max_steps_override)
    env = gym.make(env_id, render_mode=render_mode, **extra)
    is_continuous = env_id in CONTINUOUS_ENV_IDS

    fuel_only_cfg = wrappers.get("discrete_fuel_only")
    if fuel_only_cfg:
        if is_continuous:
            raise ValueError("'discrete_fuel_only' only applies to MountainCar-v0.")
        kwargs = fuel_only_cfg if isinstance(fuel_only_cfg, dict) else {}
        env = DiscreteFuelOnlyCostWrapper(env, **kwargs)
    elif wrappers.get("fuel_cost"):
        if is_continuous:
            raise ValueError("'fuel_cost' only applies to MountainCar-v0.")
        env = DiscreteFuelCostWrapper(env)

    step_cost_cfg = wrappers.get("step_cost")
    if step_cost_cfg:
        if not is_continuous:
            raise ValueError("'step_cost' only applies to MountainCarContinuous-v0.")
        kwargs = step_cost_cfg if isinstance(step_cost_cfg, dict) else {}
        env = ContinuousStepCostWrapper(env, **kwargs)

    fuel_cont_cfg = wrappers.get("continuous_fuel_cost")
    if fuel_cont_cfg:
        if not is_continuous:
            raise ValueError("'continuous_fuel_cost' only applies to MountainCarContinuous-v0.")
        kwargs = fuel_cont_cfg if isinstance(fuel_cont_cfg, dict) else {}
        env = ContinuousFuelCostWrapper(env, **kwargs)

    if wrappers.get("energy_shaping"):
        kwargs = (
            wrappers["energy_shaping"]
            if isinstance(wrappers["energy_shaping"], dict)
            else {}
        )
        env = EnergyShapingRewardWrapper(env, **kwargs)

    if wrappers.get("potential_shaping"):
        kwargs = (
            wrappers["potential_shaping"]
            if isinstance(wrappers["potential_shaping"], dict)
            else {}
        )
        env = PotentialShapingWrapper(env, **kwargs)

    if is_continuous:
        action_cfg = wrappers.get("discretize_action", {"n_actions": 5})
        n_actions = int(
            action_cfg["n_actions"] if isinstance(action_cfg, dict) else action_cfg
        )
        env = DiscretizeActionWrapper(env, n_actions=n_actions)

    state_cfg = wrappers["discretize_state"]
    state_kwargs = state_cfg if isinstance(state_cfg, dict) else {}
    env = DiscretizeStateWrapper(env, **state_kwargs)

    if wrappers.get("record_episode_stats"):
        env = RecordEpisodeStatsWrapper(env)

    return env

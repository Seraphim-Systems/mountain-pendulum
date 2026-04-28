"""Custom wrappers for state representation, reward shaping, and statistics."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .state_utils import StateBinner, augment_state


class DiscretizeStateWrapper(gym.ObservationWrapper):
    """Discretize [position, velocity] into a configurable integer grid."""

    def __init__(self, env: gym.Env, n_bins: int = 20) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("DiscretizeStateWrapper requires Box observation space.")

        self.n_bins = n_bins
        low = env.observation_space.low[:2].astype(np.float32)
        high = env.observation_space.high[:2].astype(np.float32)
        self.binner = StateBinner(low=low, high=high, n_bins=n_bins)
        self.observation_space = gym.spaces.MultiDiscrete(np.array([n_bins, n_bins]))

    def observation(self, observation: np.ndarray) -> tuple[int, int]:
        """Map continuous observation to integer coordinates."""
        return self.binner.discretize(observation)

    @property
    def state_shape(self) -> tuple[int, int]:
        """Return discretized state shape."""
        return (self.n_bins, self.n_bins)

    def obs_to_index(self, obs: tuple[int, int]) -> int:
        """Convert discretized coordinates to flat index."""
        return self.binner.to_flat_index(obs)


class AugmentStateWrapper(gym.ObservationWrapper):
    """Expand state representation from 2D to 4D engineered features."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("AugmentStateWrapper requires Box observation space.")

        base_low = env.observation_space.low[:2].astype(np.float32)
        base_high = env.observation_space.high[:2].astype(np.float32)
        v_abs = float(max(abs(base_low[1]), abs(base_high[1])))
        kinetic_max = 0.5 * (v_abs**2)

        self.observation_space = gym.spaces.Box(
            low=np.array([base_low[0], base_low[1], 0.0, -1.0], dtype=np.float32),
            high=np.array(
                [base_high[0], base_high[1], kinetic_max, 1.0], dtype=np.float32
            ),
            shape=(4,),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Return augmented state features."""
        return augment_state(observation)


class EnergyShapingRewardWrapper(gym.Wrapper):
    """Add potential-energy-difference reward shaping term explicitly."""

    def __init__(self, env: gym.Env, energy_weight: float = 100.0) -> None:
        super().__init__(env)
        self.energy_weight = energy_weight

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment."""
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply additive shaping reward based on absolute kinetic momentum."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        velocity = float(obs[1])
        # Reward raw speed (momentum) to encourage swinging
        shaped_reward = float(reward + self.energy_weight * abs(velocity))
        return obs, shaped_reward, terminated, truncated, info


class DiscreteFuelCostWrapper(gym.Wrapper):
    """Scenario 3: engine costs -1/step; idle costs -0.5/step (idle burn); +100 at goal."""

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, _, terminated, truncated, info = self.env.step(action)
        fuel_cost = -0.5 if int(action) == 1 else -1.0
        reward = fuel_cost + (100.0 if terminated else 0.0)
        return obs, reward, terminated, truncated, info


class ContinuousStepCostWrapper(gym.Wrapper):
    """Scenario 4: linear cost -0.1*|action| per step (vs. quadratic standard); +100 at goal."""

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, _, terminated, truncated, info = self.env.step(action)
        linear_cost = -0.1 * float(np.abs(np.asarray(action)).sum())
        reward = linear_cost + (100.0 if terminated else 0.0)
        return obs, reward, terminated, truncated, info


class RecordEpisodeStatsWrapper(gym.Wrapper):
    """Track per-episode reward, length, and success outcomes."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_successes: list[bool] = []
        self._episode_reward = 0.0
        self._episode_length = 0

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset counters for the next episode."""
        self._episode_reward = 0.0
        self._episode_length = 0
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Track episode statistics while forwarding environment outputs."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_reward += float(reward)
        self._episode_length += 1

        if terminated or truncated:
            self.episode_rewards.append(self._episode_reward)
            self.episode_lengths.append(self._episode_length)
            self.episode_successes.append(bool(terminated and not truncated))

        return obs, reward, terminated, truncated, info

    def get_stats(self) -> dict[str, float]:
        """Compute aggregate statistics across completed episodes."""
        if not self.episode_rewards:
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "mean_length": 0.0,
                "success_rate": 0.0,
            }

        return {
            "mean_reward": float(np.mean(self.episode_rewards)),
            "std_reward": float(np.std(self.episode_rewards)),
            "mean_length": float(np.mean(self.episode_lengths)),
            "success_rate": float(np.mean(self.episode_successes)),
        }

    def reset_stats(self) -> None:
        """Clear recorded episode-level history."""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.episode_successes.clear()

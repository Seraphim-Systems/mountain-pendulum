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


class PotentialShapingWrapper(gym.Wrapper):
    """Policy-invariant potential-based shaping (Ng et al., 1999).

    Adds F(s, a, s') = gamma * Phi(s') - Phi(s) to the reward, where Phi is
    the mechanical-energy proxy

        Phi(s) = 0.5 * v^2 + height_weight * sin(3 * x)

    Mountain-Car height is proportional to sin(3 * x), so Phi grows with
    both kinetic and potential energy. Rewarding gamma * Phi(s') - Phi(s)
    per step is theoretically equivalent (in terms of optimal policies) to
    the unshaped problem, but propagates "is the agent gaining energy?"
    information densely, which dramatically accelerates tabular learning
    on the sparse-reward fuel/min-time scenarios.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        weight: float = 1.0,
        height_weight: float = 1.0,
    ) -> None:
        super().__init__(env)
        self.gamma = float(gamma)
        self.weight = float(weight)
        self.height_weight = float(height_weight)
        self._prev_phi = 0.0

    @staticmethod
    def _phi(obs: np.ndarray, height_weight: float) -> float:
        position = float(obs[0])
        velocity = float(obs[1])
        return 0.5 * velocity * velocity + height_weight * float(np.sin(3.0 * position))

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = self._phi(obs, self.height_weight)
        return obs, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        next_phi = self._phi(obs, self.height_weight)
        shaping = self.gamma * next_phi - self._prev_phi
        self._prev_phi = next_phi
        shaped_reward = float(reward + self.weight * shaping)
        return obs, shaped_reward, terminated, truncated, info


class ProgressAndGoalRewardWrapper(gym.Wrapper):
    """Add progress reward and a terminal goal bonus to sparse MountainCar reward."""

    def __init__(self, env: gym.Env, progress_weight: float = 2.0, goal_bonus: float = 100.0) -> None:
        super().__init__(env)
        self.progress_weight = float(progress_weight)
        self.goal_bonus = float(goal_bonus)
        self._prev_position = 0.0

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._prev_position = float(obs[0])
        return obs, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        position = float(obs[0])
        progress = position - self._prev_position
        shaped_reward = float(reward + self.progress_weight * progress)
        if terminated and not truncated:
            shaped_reward += self.goal_bonus
        self._prev_position = position
        return obs, shaped_reward, terminated, truncated, info


class DiscreteFuelCostWrapper(gym.Wrapper):
    """Scenario 3: uniform -1/step cost + explicit +100 goal bonus.

    All actions cost equally so no neutral-action trap; the fuel efficiency
    concept is expressed through the explicit terminal bonus (vs. the base
    discrete variant which has no bonus and relies on episode-length alone).
    """

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = -1.0 + (100.0 if terminated else 0.0)
        return obs, reward, terminated, truncated, info


class DiscreteFuelOnlyCostWrapper(gym.Wrapper):
    """Scenario 3 (true 'min-fuel' for the discrete env).

    Implements the PDF requirement "Costs proportional to number of actions
    [right, left] taken" for the discrete variant: only non-null actions
    [left, right] consume `fuel_cost` (default 1.0) per step. The no-op
    action (action == 1) costs `idle_cost` (default 0.05) per step.

    Why a small idle cost rather than zero?  With a strictly free no-op the
    MDP has a trivial optimum (do nothing forever, accumulate zero), which
    is unavoidable for tabular Q-learning unless the agent stochastically
    discovers the goal during exploration. A small idle cost preserves the
    fuel-minimisation character (non-null actions cost ~20x more than
    idling) while breaking the degenerate fixed point and giving the agent
    a reason to seek the +`goal_bonus` terminal reward.
    """

    def __init__(
        self,
        env: gym.Env,
        goal_bonus: float = 500.0,
        fuel_cost: float = 1.0,
        idle_cost: float = 0.05,
    ) -> None:
        super().__init__(env)
        self.goal_bonus = float(goal_bonus)
        self.fuel_cost = float(fuel_cost)
        self.idle_cost = float(idle_cost)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, _, terminated, truncated, info = self.env.step(action)
        per_step = -self.fuel_cost if int(action) != 1 else -self.idle_cost
        reward = per_step + (self.goal_bonus if terminated else 0.0)
        return obs, reward, terminated, truncated, info


class ContinuousStepCostWrapper(gym.Wrapper):
    """Scenario 4: linear cost -0.1*|action| per step (vs. quadratic standard); +bonus at goal."""

    def __init__(self, env: gym.Env, goal_bonus: float = 100.0) -> None:
        super().__init__(env)
        self.goal_bonus = float(goal_bonus)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, _, terminated, truncated, info = self.env.step(action)
        linear_cost = -0.1 * float(np.abs(np.asarray(action)).sum())
        reward = linear_cost + (self.goal_bonus if terminated else 0.0)
        return obs, reward, terminated, truncated, info


class ContinuousFuelCostWrapper(gym.Wrapper):
    """Scenario 2: keep PDF/Gym semantics (cost ~ a^2) with a configurable goal bonus.

    Mountain-Car-Continuous default reward is `-0.1 * a^2 + 100 * I[goal]`.
    For tabular methods the +100 bonus is often too small to overcome the
    no-op trap (force=0 -> zero cost), so this wrapper substitutes the
    reward with a configurable bonus while preserving the quadratic fuel
    cost shape demanded by the assignment.
    """

    def __init__(self, env: gym.Env, goal_bonus: float = 200.0) -> None:
        super().__init__(env)
        self.goal_bonus = float(goal_bonus)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, _, terminated, truncated, info = self.env.step(action)
        a2 = float(np.square(np.asarray(action, dtype=np.float32)).sum())
        reward = -0.1 * a2 + (self.goal_bonus if terminated else 0.0)
        return obs, reward, terminated, truncated, info


class DqnReinforceActionWrapper(gym.ActionWrapper):
    """Expose a Box(1,) continuous action space as Discrete(n_actions).

    Maps each integer action index to an evenly spaced force value in
    ``[low, high]``. The default ``n_actions=3`` reproduces the classic
    MountainCar-v0 semantics (full-left / no-op / full-right) on top of
    MountainCarContinuous-v0, allowing discrete-only agents (e.g., DQN) to
    operate on the continuous environment with all its reward variants.
    """

    def __init__(self, env: gym.Env, n_actions: int = 3) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError(
                "DiscretizeActionWrapper requires a Box action space."
            )
        if int(np.prod(env.action_space.shape)) != 1:
            raise ValueError(
                "DiscretizeActionWrapper currently supports 1-D Box actions only."
            )
        if int(n_actions) < 2:
            raise ValueError("n_actions must be at least 2.")

        self.n_actions = int(n_actions)
        low = float(env.action_space.low[0])
        high = float(env.action_space.high[0])
        self._action_table = np.linspace(low, high, self.n_actions, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.n_actions)

    def action(self, action: int) -> np.ndarray:
        """Convert discrete index to continuous force vector."""
        idx = int(action)
        if idx < 0 or idx >= self.n_actions:
            raise ValueError(
                f"Discrete action {idx} outside [0, {self.n_actions - 1}]."
            )
        return np.array([self._action_table[idx]], dtype=np.float32)
      
class DiscretizeActionWrapper(gym.ActionWrapper):
    """Expose a Discrete(N) action space over a continuous Box(-1, 1) base env.

    Tabular Q-learning / SARSA require a finite, indexable action space.
    For `MountainCarContinuous-v0` we map a discrete action index to one of
    `n_actions` evenly-spaced force values in [low, high].
    """

    def __init__(self, env: gym.Env, n_actions: int = 5) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("DiscretizeActionWrapper requires a Box action space.")
        if n_actions < 2:
            raise ValueError("n_actions must be >= 2 for a meaningful discretization.")

        low = float(env.action_space.low[0])
        high = float(env.action_space.high[0])
        self.n_actions = int(n_actions)
        self._action_values = np.linspace(low, high, self.n_actions, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.n_actions)

    def action(self, action: int) -> np.ndarray:
        idx = int(action)
        if idx < 0 or idx >= self.n_actions:
            raise ValueError(f"Discrete action {idx} out of range [0, {self.n_actions - 1}].")
        return np.array([self._action_values[idx]], dtype=np.float32)

    @property
    def action_values(self) -> np.ndarray:
        """Continuous action values mapped from discrete indices."""
        return self._action_values.copy()


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

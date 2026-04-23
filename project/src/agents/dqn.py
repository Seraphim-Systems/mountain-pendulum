"""Stable-Baselines3 DQN baseline adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import DQN


class DQNBaseline:
    """DQN baseline for discrete-action MountainCar with function approximation."""

    def __init__(
        self,
        env: Any,
        learning_rate: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        batch_size: int,
        replay_buffer_size: int,
        target_update_freq: int,
        hidden_sizes: list[int],
    ) -> None:
        exploration_fraction = min(1.0, max(1e-4, 1.0 - epsilon_decay))
        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            gamma=gamma,
            exploration_initial_eps=epsilon_start,
            exploration_final_eps=epsilon_end,
            exploration_fraction=exploration_fraction,
            batch_size=batch_size,
            buffer_size=replay_buffer_size,
            target_update_interval=target_update_freq,
            policy_kwargs={"net_arch": hidden_sizes},
            verbose=0,
        )

    def learn(self, total_timesteps: int) -> None:
        """Run DQN updates for the provided number of steps."""
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

    def predict(self, state: np.ndarray, deterministic: bool = True) -> int:
        """Predict an action from current policy network."""
        action, _ = self.model.predict(state, deterministic=deterministic)
        return int(action)

    def save(self, model_path: str | Path) -> None:
        """Save DQN model checkpoint."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, model_path: str | Path, env: Any) -> None:
        """Load DQN model checkpoint bound to an environment."""
        self.model = DQN.load(str(model_path), env=env)

    @property
    def exploration_rate(self) -> float:
        """Return the current epsilon-like exploration rate."""
        return float(self.model.exploration_rate)

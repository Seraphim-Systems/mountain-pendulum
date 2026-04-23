"""Stable-Baselines3 SAC baseline adapter for continuous MountainCar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC


class SACBaseline:
    """SAC baseline for MountainCarContinuous-v0."""

    def __init__(
        self,
        env: Any,
        learning_rate: float,
        gamma: float,
        tau: float,
        alpha: str | float,
        batch_size: int,
        replay_buffer_size: int,
        learning_starts: int,
        hidden_sizes: list[int],
    ) -> None:
        self.model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            ent_coef=alpha,
            batch_size=batch_size,
            buffer_size=replay_buffer_size,
            learning_starts=learning_starts,
            policy_kwargs={"net_arch": hidden_sizes},
            verbose=0,
        )

    def learn(self, total_timesteps: int) -> None:
        """Run SAC updates for the provided number of steps."""
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict continuous action in [-1, 1]."""
        action, _ = self.model.predict(state, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    def save(self, model_path: str | Path) -> None:
        """Save SAC model checkpoint."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, model_path: str | Path, env: Any) -> None:
        """Load SAC model checkpoint bound to an environment."""
        self.model = SAC.load(str(model_path), env=env)

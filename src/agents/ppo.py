"""Stable-Baselines3 PPO baseline adapter for continuous MountainCar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO


class PPOBaseline:
    """PPO baseline for MountainCarContinuous-v0."""

    def __init__(
        self,
        env: Any,
        learning_rate: float,
        n_steps: int,
        batch_size: int,
        gamma: float,
        clip_range: float,
        ent_coef: float,
        hidden_sizes: list[int],
    ) -> None:
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs={"net_arch": hidden_sizes},
            verbose=0,
        )

    def learn(self, total_timesteps: int) -> None:
        """Run PPO updates for the provided number of steps."""
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict continuous action in [-1, 1]."""
        action, _ = self.model.predict(state, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    def save(self, model_path: str | Path) -> None:
        """Save PPO model checkpoint."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, model_path: str | Path, env: Any) -> None:
        """Load PPO model checkpoint bound to an environment."""
        self.model = PPO.load(str(model_path), env=env)

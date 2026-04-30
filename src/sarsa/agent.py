"""Tabular SARSA implementation for discretized MountainCar-v0."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class SarsaAgent:
    """Epsilon-greedy tabular SARSA agent."""

    def __init__(
        self,
        n_bins: int,
        n_actions: int,
        learning_rate: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        optimistic_init: float | Sequence[float] = 0.0,
    ) -> None:
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.optimistic_init = optimistic_init

        init_array = np.broadcast_to(
            np.asarray(optimistic_init, dtype=np.float32), (n_actions,)
        )
        self.q_table = np.broadcast_to(
            init_array, (n_bins, n_bins, n_actions)
        ).astype(np.float32).copy()

    def select_action(self, state: tuple[int, int], deterministic: bool = False) -> int:
        """Select an action using epsilon-greedy exploration."""
        if not deterministic and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_actions))

        i, j = state
        return int(np.argmax(self.q_table[i, j]))

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
        next_action: int,
        done: bool,
    ) -> None:
        """Apply one-step on-policy SARSA update."""
        i, j = state
        ni, nj = next_state
        next_value = 0.0 if done else float(self.q_table[ni, nj, next_action])
        target = reward + self.gamma * next_value
        td_error = target - self.q_table[i, j, action]
        self.q_table[i, j, action] += self.learning_rate * td_error

    def on_episode_end(self) -> None:
        """Decay epsilon once per episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def policy_table(self) -> np.ndarray:
        """Return the greedy policy induced by the current Q-table."""
        return np.argmax(self.q_table, axis=2)

    def save(self, model_path: str | Path) -> None:
        """Save Q-table and epsilon."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, q_table=self.q_table, epsilon=self.epsilon)

    def load(self, model_path: str | Path) -> None:
        """Load Q-table and epsilon."""
        data = np.load(model_path)
        self.q_table = data["q_table"]
        self.epsilon = float(data["epsilon"])

"""Tabular Q-learning implementation for discretized MountainCar-v0."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class QLearningAgent:
    """Epsilon-greedy tabular Q-learning agent."""

    def __init__(
        self,
        n_bins: int,
        n_actions: int,
        learning_rate: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
    ) -> None:
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_bins, n_bins, n_actions), dtype=np.float32)

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
        done: bool,
    ) -> None:
        """Apply one-step off-policy Q-learning update."""
        i, j = state
        ni, nj = next_state
        next_value = 0.0 if done else float(np.max(self.q_table[ni, nj]))
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

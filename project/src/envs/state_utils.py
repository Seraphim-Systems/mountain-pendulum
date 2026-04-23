"""State preprocessing utilities for Mountain Car environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateBinner:
    """Discretize a 2D continuous state (position, velocity) onto a fixed grid."""

    low: np.ndarray
    high: np.ndarray
    n_bins: int

    def discretize(self, observation: np.ndarray) -> tuple[int, int]:
        """Convert a continuous state to integer bin coordinates."""
        obs = np.asarray(observation[:2], dtype=np.float32)
        ratio = (obs - self.low) / (self.high - self.low + 1e-12)
        clipped = np.clip(ratio, 0.0, 1.0)
        bins = np.floor(clipped * self.n_bins).astype(np.int64)
        bins = np.clip(bins, 0, self.n_bins - 1)
        return int(bins[0]), int(bins[1])

    def to_flat_index(self, state: tuple[int, int]) -> int:
        """Convert 2D integer bin coordinates to a flat state index."""
        i, j = state
        return i * self.n_bins + j


def augment_state(observation: np.ndarray) -> np.ndarray:
    """Append physically motivated features to [position, velocity]."""
    position = float(observation[0])
    velocity = float(observation[1])
    kinetic_energy = 0.5 * velocity * velocity
    potential_proxy = np.sin(3.0 * position)
    return np.array([position, velocity, kinetic_energy, potential_proxy], dtype=np.float32)

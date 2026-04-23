"""Trajectory plotting for position-velocity analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_phase_space(
    positions: np.ndarray,
    velocities: np.ndarray,
    output_path: str | Path,
    title: str = "Trajectory in Position-Velocity Space",
) -> None:
    """Plot one episode trajectory in phase-space style."""
    plt.figure(figsize=(6, 5))
    plt.plot(positions, velocities, linewidth=1.5)
    plt.scatter([positions[0]], [velocities[0]], label="start", marker="o")
    plt.scatter([positions[-1]], [velocities[-1]], label="end", marker="x")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trajectory_time_series(
    positions: np.ndarray,
    velocities: np.ndarray,
    output_path: str | Path,
) -> None:
    """Plot position and velocity evolution during one episode."""
    steps = np.arange(len(positions))
    plt.figure(figsize=(9, 4))
    plt.plot(steps, positions, label="Position")
    plt.plot(steps, velocities, label="Velocity")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Episode Trajectory Time Series")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

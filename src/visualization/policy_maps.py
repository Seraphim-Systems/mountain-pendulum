"""Policy and visitation map visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_theme(style="white")


def plot_visitation_heatmap(
    visitation_counts: np.ndarray,
    output_path: str | Path,
    title: str = "State Visitation Heatmap",
) -> None:
    """Visualize state visitation count matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(visitation_counts.T, cmap="mako")
    plt.title(title)
    plt.xlabel("Position Bin")
    plt.ylabel("Velocity Bin")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_policy_heatmap(
    action_map: np.ndarray,
    output_path: str | Path,
    title: str = "Greedy Policy Action Map",
) -> None:
    """Visualize discrete action chosen per discretized state."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(action_map.T, cmap="viridis")
    plt.title(title)
    plt.xlabel("Position Bin")
    plt.ylabel("Velocity Bin")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

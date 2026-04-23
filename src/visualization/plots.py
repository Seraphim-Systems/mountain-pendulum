"""Training and cross-seed plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def plot_reward_curve(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot per-episode reward curve for one run."""
    plt.figure(figsize=(9, 4))
    plt.plot(df["episode"], df["reward"], linewidth=1.3)
    plt.title("Reward over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_success_curve(
    df: pd.DataFrame, output_path: str | Path, window: int = 50
) -> None:
    """Plot moving-average success rate over episodes."""
    rolling = df["success"].rolling(window=window, min_periods=1).mean()
    plt.figure(figsize=(9, 4))
    plt.plot(df["episode"], rolling, linewidth=1.3)
    plt.ylim(0.0, 1.0)
    plt.title(f"Success Rate (Moving Average, window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_seed_comparison(
    seed_episode_frames: list[pd.DataFrame],
    metric: str,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot mean and variability across seeds for selected metric."""
    merged = pd.concat(seed_episode_frames, ignore_index=True)
    grouped = merged.groupby("episode")[metric]
    mean = grouped.mean().to_numpy()
    std = grouped.std(ddof=0).fillna(0.0).to_numpy()
    episodes = np.sort(merged["episode"].unique())

    plt.figure(figsize=(9, 4))
    plt.plot(episodes, mean, label="Mean")
    plt.fill_between(episodes, mean - std, mean + std, alpha=0.25, label="±1 std")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

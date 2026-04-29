"""GA-specific training plots: multi-seed envelopes and cross-agent comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)

_AGENT_COLORS = {
    "simple_ga": "#E63946",
    "cma_es": "#457B9D",
    "neat": "#2A9D8F",
}
_AGENT_LABELS = {
    "simple_ga": "Simple GA",
    "cma_es": "CMA-ES",
    "neat": "NEAT",
}


def _smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def plot_ga_fitness_curve(
    df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Reward over Generations",
    window: int = 5,
) -> None:
    """Single-seed reward curve with cumulative-best overlay."""
    fig, ax = plt.subplots(figsize=(9, 4))
    episodes = df["episode"].to_numpy()
    reward = _smooth(df["reward"], window).to_numpy()
    best_so_far = np.maximum.accumulate(df["reward"].to_numpy())

    ax.plot(episodes, reward, linewidth=1.4, color="#457B9D", label="Reward (smoothed)")
    ax.plot(episodes, best_so_far, linewidth=1.2, color="#E63946", linestyle="--", label="Best so far")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.legend(framealpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_multiseed_envelope(
    dfs: list[pd.DataFrame],
    metric: str,
    output_path: str | Path,
    title: str,
    window: int = 1,
    color: str = "#457B9D",
) -> None:
    """Mean ± 1 std shaded envelope across multiple seed DataFrames."""
    merged = pd.concat(dfs, ignore_index=True)
    grouped = merged.groupby("episode")[metric]
    episodes = np.sort(merged["episode"].unique())

    if window > 1:
        smoothed_per_seed = [
            _smooth(df.set_index("episode")[metric].reindex(episodes), window)
            for df in dfs
        ]
        stacked = np.stack([s.to_numpy() for s in smoothed_per_seed])
    else:
        stacked = np.stack([
            df.set_index("episode")[metric].reindex(episodes).to_numpy()
            for df in dfs
        ])

    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0, ddof=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(episodes, mean, linewidth=1.6, color=color, label="Mean")
    ax.fill_between(episodes, mean - std, mean + std, alpha=0.22, color=color, label="±1 std")
    ax.set_xlabel("Generation")
    ax.set_ylabel(metric.replace("_", " ").capitalize())
    ax.set_title(title)
    ax.legend(framealpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cross_agent_comparison(
    results: dict[str, list[pd.DataFrame]],
    metric: str,
    output_path: str | Path,
    title: str,
    window: int = 5,
) -> None:
    """One shaded envelope per agent on the same axes."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for agent_key, dfs in results.items():
        if not dfs:
            continue
        merged = pd.concat(dfs, ignore_index=True)
        episodes = np.sort(merged["episode"].unique())
        stacked = np.stack([
            _smooth(df.set_index("episode")[metric].reindex(episodes), window)
            for df in dfs
        ])
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0, ddof=0)
        color = _AGENT_COLORS.get(agent_key, "#888888")
        label = _AGENT_LABELS.get(agent_key, agent_key)
        ax.plot(episodes, mean, linewidth=1.8, color=color, label=label)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.18, color=color)

    ax.set_xlabel("Generation")
    ax.set_ylabel(metric.replace("_", " ").capitalize())
    ax.set_title(title)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_population_diversity(
    diversity_series: list[float],
    output_path: str | Path,
    title: str = "Population Weight Diversity over Generations",
) -> None:
    """Mean pairwise std of population weight vectors across generations."""
    fig, ax = plt.subplots(figsize=(9, 4))
    gens = np.arange(1, len(diversity_series) + 1)
    ax.plot(gens, diversity_series, linewidth=1.5, color="#2A9D8F")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Param Std")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

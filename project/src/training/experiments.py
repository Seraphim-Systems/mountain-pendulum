"""Run multiple experiment configurations and collate summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.training.train import train_experiment


def run_experiment_suite(config_paths: list[str], project_root: str) -> Path:
    """Train all provided configs and merge aggregate outputs."""
    root = Path(project_root)
    log_dir = root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for config_path in config_paths:
        train_experiment(config_path=config_path, project_root=project_root)

    aggregate_files = sorted(log_dir.glob("*_aggregate.json"))
    rows = []
    for path in aggregate_files:
        rows.append(pd.read_json(path, typ="series").to_dict())

    summary_df = pd.DataFrame(rows)
    output_path = log_dir / "experiment_suite_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"Saved experiment suite summary to: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for experiment suite execution."""
    parser = argparse.ArgumentParser(description="Run multiple Mountain Car experiments.")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="List of YAML experiment config paths.",
    )
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment_suite(config_paths=args.configs, project_root=args.project_root)

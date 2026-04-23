"""Configuration loading and validation for experiment runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")
    return config


def ensure_paths(config: dict[str, Any], project_root: str | Path) -> dict[str, Path]:
    """Resolve and create output paths from config."""
    root = Path(project_root)
    paths = config.get("paths", {})
    logs_dir = root / paths.get("logs_dir", "outputs/logs")
    models_dir = root / paths.get("models_dir", "outputs/models")
    figures_dir = root / paths.get("figures_dir", "outputs/figures")

    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    return {
        "logs_dir": logs_dir,
        "models_dir": models_dir,
        "figures_dir": figures_dir,
    }

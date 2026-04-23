"""Logging helpers for TensorBoard and JSON result persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


def make_writer(log_root: str | Path, run_name: str) -> SummaryWriter:
    """Create a TensorBoard SummaryWriter with timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(log_root) / f"{run_name}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def dump_json(data: dict[str, Any], path: str | Path) -> None:
    """Write a JSON file with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

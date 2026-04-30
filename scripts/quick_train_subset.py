"""Smoke a subset of the fixed configs."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scripts.quick_train_new_scenarios as base  # noqa: E402

base.CONFIGS = [
    "configs/dqn_fuel.yaml",
    "configs/dqn_continuous.yaml",
    "configs/reinforce_fuel.yaml",
    "configs/reinforce_continuous.yaml",
]

if __name__ == "__main__":
    raise SystemExit(base.main())

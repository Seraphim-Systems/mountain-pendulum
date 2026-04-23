"""Utilities for deterministic seeding across Python, NumPy, Torch, and Gymnasium."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed common random number generators and hash seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_env(env: Any, seed: int) -> None:
    """Seed a Gymnasium environment and its action/observation spaces."""
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)

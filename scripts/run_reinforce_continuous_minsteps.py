"""Re-run only reinforce_continuous + reinforce_minsteps with lower
energy_weight (0.5 instead of 5.0) so the +100 goal bonus dominates the
swinging-shaping bonus.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_all_new_configs as base  # noqa: E402

base.CONFIGS = [
    "configs/reinforce_continuous.yaml",
    "configs/reinforce_minsteps.yaml",
]

if __name__ == "__main__":
    raise SystemExit(base.main())

"""Resume after dqn_fuel + dqn_minsteps already finished.

Runs ``dqn_continuous`` (with the new ``n_actions=5`` setting) followed by
the 3 REINFORCE configs that bootstrap from DQN teachers. ``dqn_continuous``
is re-run first so that its fresh checkpoint matches the ``teacher_env``
declared in ``configs/reinforce_continuous.yaml``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_all_new_configs as base  # noqa: E402

base.CONFIGS = base.RESUME_AFTER_DQN_FUEL_AND_MINSTEPS

if __name__ == "__main__":
    raise SystemExit(base.main())

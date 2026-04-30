"""Run the 6 new DQN/REINFORCE configs sequentially with full budgets.

Each config goes through the real ``train_experiment`` pipeline (all seeds in
the YAML, full episode count). Per-config log files are written under
``outputs/run_logs/`` and a master ``run_summary.txt`` records start/end
times, durations, and outcomes. Failures in one config do not abort the rest.
"""

from __future__ import annotations

import io
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.train import train_experiment  # noqa: E402

CONFIGS = [
    "configs/dqn_fuel.yaml",
    "configs/dqn_continuous.yaml",
    "configs/dqn_minsteps.yaml",
    "configs/reinforce_fuel.yaml",
    "configs/reinforce_continuous.yaml",
    "configs/reinforce_minsteps.yaml",
]

# The 3 REINFORCE configs now bootstrap from DQN teachers via behaviour
# cloning. ``reinforce_continuous`` in particular requires ``dqn_continuous``
# to have been re-trained with ``n_actions=5`` so its checkpoint matches the
# teacher_env declared in ``configs/reinforce_continuous.yaml``. When you only
# need to produce the missing pieces (after a partial run), point ``CONFIGS``
# at this shorter list instead.
RESUME_AFTER_DQN_FUEL_AND_MINSTEPS = [
    "configs/dqn_continuous.yaml",      # re-run with n_actions=5 first
    "configs/reinforce_fuel.yaml",      # teacher = dqn_discrete_solve_seed7
    "configs/reinforce_continuous.yaml",  # teacher = fresh dqn_continuous (n=5)
    "configs/reinforce_minsteps.yaml",  # teacher = dqn_minsteps (n=3, already trained)
]


def main() -> int:
    log_dir = ROOT / "outputs" / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "run_summary.txt"

    total_start = time.perf_counter()
    with summary_path.open("w", encoding="utf-8") as summary:
        summary.write(f"Started at {datetime.now().isoformat(timespec='seconds')}\n")
        summary.write(f"Configs: {len(CONFIGS)}\n\n")
        summary.flush()

        results: list[tuple[str, bool, float, str]] = []
        for cfg_path in CONFIGS:
            tag = Path(cfg_path).stem
            cfg_log_path = log_dir / f"{tag}.log"
            line_header = (
                f"\n=== {cfg_path}  start={datetime.now().isoformat(timespec='seconds')} ==="
            )
            print(line_header, flush=True)
            summary.write(line_header + "\n")
            summary.flush()

            t0 = time.perf_counter()
            ok = True
            err_msg = ""
            try:
                with cfg_log_path.open("w", encoding="utf-8") as cfg_log:
                    buf = io.StringIO()
                    # tee stdout/stderr to per-config log; we still want the
                    # master process to know progress, so we periodically
                    # flush the buffer to the cfg_log.
                    with redirect_stdout(buf), redirect_stderr(buf):
                        train_experiment(
                            config_path=str(ROOT / cfg_path),
                            project_root=str(ROOT),
                            render=False,
                        )
                    cfg_log.write(buf.getvalue())
                    cfg_log.flush()
            except Exception as exc:  # pragma: no cover - reporting only
                ok = False
                err_msg = f"{exc.__class__.__name__}: {exc}"
                with cfg_log_path.open("a", encoding="utf-8") as cfg_log:
                    cfg_log.write(f"\nEXCEPTION: {err_msg}\n")
                    cfg_log.write(traceback.format_exc())

            elapsed = time.perf_counter() - t0
            results.append((cfg_path, ok, elapsed, err_msg))
            status = "PASS" if ok else "FAIL"
            line = (
                f"[{status}] {cfg_path}  elapsed={elapsed/60:.1f}min  "
                f"end={datetime.now().isoformat(timespec='seconds')}"
            )
            if err_msg:
                line += f"  msg={err_msg}"
            print(line, flush=True)
            summary.write(line + "\n")
            summary.flush()

        total_elapsed = time.perf_counter() - total_start
        passed = sum(1 for _, ok, _, _ in results if ok)
        footer = (
            f"\nFinished at {datetime.now().isoformat(timespec='seconds')}  "
            f"total={total_elapsed/3600:.2f}h  ({passed}/{len(results)} passed)"
        )
        print(footer, flush=True)
        summary.write(footer + "\n")
        summary.flush()

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

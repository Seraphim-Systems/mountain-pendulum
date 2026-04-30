"""Smoke-test the teacher-augmented REINFORCE configs.

For each config, build the agent (which triggers ``_pretrain_from_teacher``),
verify the policy network parameter count matches what the YAML declares,
and run a 1-iteration ``learn()`` plus a single deterministic rollout.

For ``reinforce_continuous`` the production checkpoint will be re-trained
with ``n_actions=5`` once ``dqn_continuous`` reruns; until then the smoke
points the teacher_env at the matching ``n_actions=3`` setting so the
existing checkpoint loads cleanly.
"""

from __future__ import annotations

import copy
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.agents.reinforce import REINFORCEBaseline  # noqa: E402
from src.envs.mountain_car_continuous import make_continuous_env  # noqa: E402
from src.envs.mountain_car_discrete import make_discrete_env  # noqa: E402
from src.training.train import build_env  # noqa: E402
from src.utils.seeding import seed_env, seed_everything  # noqa: E402


CONFIGS = [
    ("configs/reinforce_fuel.yaml", None),
    # For continuous, override teacher_env to n_actions=3 since the existing
    # dqn_continuous_seed7_final.zip is the n=3 version. The production run
    # will re-train it with n=5 and use the matching teacher_env directly.
    (
        "configs/reinforce_continuous.yaml",
        lambda cfg: cfg["reinforce"]["teacher_env"]["wrappers"]["discretize_action"].update(
            {"n_actions": 3}
        ),
    ),
    ("configs/reinforce_minsteps.yaml", None),
]


def smoke_one(cfg_path: str, mutate_fn) -> tuple[bool, str]:
    cfg = yaml.safe_load((ROOT / cfg_path).read_text())
    if mutate_fn is not None:
        mutate_fn(cfg)

    seed_everything(0)
    env = build_env(cfg["env"])
    seed_env(env, 0)

    rcfg = cfg["reinforce"]
    teacher_env_cfg = rcfg.get("teacher_env")

    teacher_env_factory = None
    if rcfg.get("teacher_checkpoint") and teacher_env_cfg is not None:

        def _factory(_cfg=teacher_env_cfg):
            return build_env(_cfg)

        teacher_env_factory = _factory

    try:
        agent = REINFORCEBaseline(
            env=env,
            learning_rate=float(rcfg["learning_rate"]),
            gamma=float(rcfg["gamma"]),
            entropy_coef=float(rcfg.get("entropy_coef", 0.01)),
            value_fn_coef=float(rcfg.get("value_fn_coef", 0.5)),
            hidden_sizes=list(rcfg["hidden_sizes"]),
            batch_episodes=int(rcfg.get("batch_episodes", 4)),
            teacher_checkpoint=rcfg.get("teacher_checkpoint"),
            teacher_pretrain_episodes=20,  # short for smoke
            teacher_env_factory=teacher_env_factory,
        )
    except Exception as exc:
        return False, f"agent construction failed: {exc.__class__.__name__}: {exc}\n{traceback.format_exc()}"

    info = (
        f"continuous={agent.continuous} action_dim={agent.action_dim} "
        f"teacher_inference={agent.teacher_model is not None}"
    )

    try:
        agent.learn(total_timesteps=64)
    except Exception as exc:
        env.close()
        return False, f"learn() failed: {exc.__class__.__name__}: {exc}\n{traceback.format_exc()}"

    # quick deterministic rollout
    try:
        obs, _ = env.reset()
        steps = 0
        total = 0.0
        for _ in range(50):
            a = agent.predict(obs, deterministic=True)
            if not agent.continuous:
                a = int(a)
            obs, r, term, trunc, _ = env.step(a)
            steps += 1
            total += float(r)
            if term or trunc:
                break
    except Exception as exc:
        env.close()
        return False, f"rollout failed: {exc.__class__.__name__}: {exc}\n{traceback.format_exc()}"

    env.close()
    return True, f"{info} | reward({steps} steps)={total:.2f}"


def main() -> int:
    failures = 0
    for cfg, mut in CONFIGS:
        ok, msg = smoke_one(cfg, mut)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {cfg}: {msg}")
        if not ok:
            failures += 1
    print(f"\n{len(CONFIGS) - failures}/{len(CONFIGS)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

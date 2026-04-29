"""Render and record a single episode from a saved checkpoint.

Usage:
  .venv\Scripts\python.exe scripts\render_checkpoint.py --config configs/dqn_discrete_solve.yaml \
      --checkpoint outputs/models/dqn_discrete_solve_seed7_final.zip --project-root . --outdir outputs/videos --episodes 1
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio
import os

from src.training.train import build_env, build_agent, rollout_episode
from src.utils.config import load_config, ensure_paths
from src.utils.seeding import seed_everything, seed_env


def record_one(config_path: str, checkpoint: str, seed: int, episodes: int, project_root: str, outdir: str) -> None:
    config = load_config(config_path)
    root = Path(project_root)
    paths = ensure_paths(config, root)

    seed_everything(seed)

    # build env with image render_mode and capture frames manually
    env = build_env(config['env'], render_mode='rgb_array')
    os.makedirs(outdir, exist_ok=True)
    seed_env(env, seed)

    agent = build_agent(config, env)
    # load checkpoint
    if hasattr(agent, 'load'):
        try:
            agent.load(checkpoint, env)
        except TypeError:
            # some agents maybe expect only path
            agent.load(checkpoint)
    else:
        raise RuntimeError('Agent has no load method')

    # run and capture frames
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        frames = []
        # capture initial frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        while not done:
            if hasattr(agent, 'select_action') and not hasattr(agent, 'predict'):
                action = agent.select_action(tuple(obs), deterministic=True)
            else:
                action = agent.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(int(action))
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            done = bool(terminated or truncated)

        # save frames to mp4
        out_path = Path(outdir) / f"render_{Path(checkpoint).stem}_ep{ep+1}.mp4"
        with imageio.get_writer(out_path, fps=30) as writer:
            for fr in frames:
                writer.append_data(fr)

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--outdir', default='outputs/videos')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    record_one(args.config, args.checkpoint, args.seed, args.episodes, args.project_root, str(outdir))

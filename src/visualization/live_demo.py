"""Run a live, real-time sensor car demo in a pygame window."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.envs.sensor_car_env import SensorCarEnv


def _safe_load_dqn(model_path: str | None) -> Any:
    """Load a Stable-Baselines3 DQN checkpoint if available."""
    if not model_path:
        return None

    try:
        from stable_baselines3 import DQN
    except Exception:
        return None

    checkpoint = Path(model_path)
    if not checkpoint.exists():
        return None

    return DQN.load(str(checkpoint))


def run_live_demo(
    asset_root: str,
    steps: int,
    fps: int,
    model_path: str | None = None,
    seed: int = 42,
) -> None:
    """Open a pygame window and step the demo in real time."""
    import pygame

    env = SensorCarEnv(asset_root=asset_root, render_mode="rgb_array", max_steps=steps)
    model = _safe_load_dqn(model_path)

    print(f"Map source: {env.assets.map_source}")
    print(f"Sprite source: {env.assets.sprite_source}")
    print(f"Policy source: {'trained model' if model is not None else 'random policy'}")

    pygame.init()
    window = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Sensor Car Live Demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)

    obs, info = env.reset(seed=seed)
    done = False
    step_index = 0
    running = True

    while running and not done and step_index < steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

        if not running:
            break

        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(int(action))
        done = bool(terminated or truncated)
        frame = env.render()

        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        window.blit(surface, (0, 0))

        overlay_lines = [
            f"step: {step_index + 1}/{steps}",
            f"reward: {reward:.2f}",
            f"collision: {info['collision']}",
            f"goal: {info['reached_goal']}",
            f"map: {Path(env.assets.map_source).name if env.assets.map_source != 'procedural' else 'procedural'}",
        ]
        for line_index, text in enumerate(overlay_lines):
            label = font.render(text, True, (20, 20, 20))
            window.blit(label, (12, 12 + line_index * 20))

        pygame.display.flip()
        clock.tick(fps)
        step_index += 1
        time.sleep(0.001)

    env.close()
    pygame.quit()
    print("Live demo finished.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the live demo."""
    parser = argparse.ArgumentParser(description="Live sensor-car visualization demo.")
    parser.add_argument("--asset-root", default="assets", help="Assets root folder.")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to show.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for playback.")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional trained DQN checkpoint path for policy playback.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_live_demo(
        asset_root=args.asset_root,
        steps=args.steps,
        fps=args.fps,
        model_path=args.model,
        seed=args.seed,
    )

"""Top-down sensor car environment with asset-driven or procedural maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v3 as iio
import numpy as np


@dataclass
class MapAssets:
    """Resolved map/sprite metadata for rendering and diagnostics."""

    map_source: str
    sprite_source: str


def _first_image_path(folder: Path) -> Path | None:
    """Return the first supported image file from a folder, if any."""
    if not folder.exists():
        return None

    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        files = sorted(folder.glob(ext))
        if files:
            return files[0]
    return None


def _generate_procedural_map(height: int = 256, width: int = 384) -> np.ndarray:
    """Create a simple procedural occupancy map with walls and obstacles."""
    occ = np.zeros((height, width), dtype=np.uint8)

    # Outer walls.
    occ[:6, :] = 1
    occ[-6:, :] = 1
    occ[:, :6] = 1
    occ[:, -6:] = 1

    # Procedural obstacles (a slalom-like corridor).
    for x in range(70, width - 70, 60):
        if (x // 60) % 2 == 0:
            occ[40:165, x : x + 12] = 1
        else:
            occ[95 : height - 40, x : x + 12] = 1

    return occ


def _load_or_generate_map(map_dir: Path) -> tuple[np.ndarray, str]:
    """Load occupancy map from assets, or generate one procedurally."""
    image_path = _first_image_path(map_dir)
    if image_path is None:
        return _generate_procedural_map(), "procedural"

    image = iio.imread(image_path)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)

    # Dark = wall, bright = free.
    occupancy = (image < 128).astype(np.uint8)
    if occupancy.sum() == 0:
        return _generate_procedural_map(), "procedural"

    # Ensure hard border walls even for custom maps.
    occupancy[:2, :] = 1
    occupancy[-2:, :] = 1
    occupancy[:, :2] = 1
    occupancy[:, -2:] = 1
    return occupancy, str(image_path)


def _load_sprite(sprite_dir: Path, size: int = 18) -> tuple[np.ndarray | None, str]:
    """Load car sprite image from assets or return None for procedural drawing."""
    image_path = _first_image_path(sprite_dir)
    if image_path is None:
        return None, "procedural"

    sprite = iio.imread(image_path)
    if sprite.ndim == 2:
        alpha = np.full_like(sprite, 255)
        sprite = np.stack([sprite, sprite, sprite, alpha], axis=2)
    elif sprite.ndim == 3 and sprite.shape[2] == 3:
        alpha = np.full((sprite.shape[0], sprite.shape[1], 1), 255, dtype=sprite.dtype)
        sprite = np.concatenate([sprite, alpha], axis=2)

    sprite = sprite.astype(np.uint8)
    if min(sprite.shape[:2]) < 6:
        return None, "procedural"

    # Lightweight nearest-neighbor resize without external deps.
    y_idx = np.linspace(0, sprite.shape[0] - 1, size).astype(int)
    x_idx = np.linspace(0, sprite.shape[1] - 1, size).astype(int)
    sprite_resized = sprite[np.ix_(y_idx, x_idx)]
    return sprite_resized, str(image_path)


class SensorCarEnv(gym.Env[np.ndarray, int]):
    """2D navigation environment with ray probes and collision dynamics."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        asset_root: str | Path = "assets",
        render_mode: str | None = None,
        max_steps: int = 350,
        n_probes: int = 5,
        probe_max_distance: float = 80.0,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.n_probes = n_probes
        self.probe_max_distance = float(probe_max_distance)

        root = Path(asset_root)
        self.occupancy, map_source = _load_or_generate_map(root / "maps")
        self.sprite_rgba, sprite_source = _load_sprite(root / "sprites")
        self.assets = MapAssets(map_source=map_source, sprite_source=sprite_source)

        self.height, self.width = self.occupancy.shape

        # Actions: 0=idle, 1=accelerate, 2=brake, 3=turn_left, 4=turn_right.
        self.action_space = gym.spaces.Discrete(5)

        # Observation: probe distances + speed + heading_sin + heading_cos.
        obs_size = self.n_probes + 3
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self._probe_angles = np.linspace(-0.9, 0.9, self.n_probes)
        self._max_speed = 3.5
        self._acceleration = 0.25
        self._turn_rate = 0.18
        self._friction = 0.05

        self._start = np.array([22.0, self.height / 2.0], dtype=np.float32)
        self._goal = np.array([self.width - 24.0, self.height / 2.0], dtype=np.float32)

        if self._is_wall(self._start):
            self._start = self._find_first_free(start_col=8)
        if self._is_wall(self._goal):
            self._goal = self._find_first_free(start_col=self.width - 20)

        self.position = self._start.copy()
        self.heading = 0.0
        self.speed = 0.0
        self.step_count = 0
        self.crashed = False
        self.reached_goal = False

    def _find_first_free(self, start_col: int) -> np.ndarray:
        """Find a fallback free cell near the provided column."""
        x = int(np.clip(start_col, 3, self.width - 4))
        for y in range(3, self.height - 3):
            if self.occupancy[y, x] == 0:
                return np.array([float(x), float(y)], dtype=np.float32)
        return np.array([float(x), self.height / 2.0], dtype=np.float32)

    def _is_wall(self, pos: np.ndarray) -> bool:
        """Return True if position lies on a wall cell or out of bounds."""
        x = int(round(float(pos[0])))
        y = int(round(float(pos[1])))
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return True
        return bool(self.occupancy[y, x] == 1)

    def _ray_distance(self, angle: float) -> float:
        """Cast one ray probe and return normalized free-space distance."""
        ray_angle = self.heading + angle
        direction = np.array([np.cos(ray_angle), np.sin(ray_angle)], dtype=np.float32)
        dist = 0.0
        probe_pos = self.position.copy()

        while dist < self.probe_max_distance:
            probe_pos += direction
            dist += 1.0
            if self._is_wall(probe_pos):
                break

        return float(min(dist, self.probe_max_distance) / self.probe_max_distance)

    def _get_obs(self) -> np.ndarray:
        """Build normalized observation vector from probes and kinematics."""
        probes = [self._ray_distance(a) for a in self._probe_angles]
        speed_norm = float(np.clip(self.speed / self._max_speed, 0.0, 1.0))
        heading_sin = float((np.sin(self.heading) + 1.0) * 0.5)
        heading_cos = float((np.cos(self.heading) + 1.0) * 0.5)
        return np.array(
            probes + [speed_norm, heading_sin, heading_cos], dtype=np.float32
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment state."""
        super().reset(seed=seed)
        _ = options

        self.position = self._start.copy()
        self.heading = 0.0
        self.speed = 0.2
        self.step_count = 0
        self.crashed = False
        self.reached_goal = False

        obs = self._get_obs()
        info = {
            "assets": {
                "map_source": self.assets.map_source,
                "sprite_source": self.assets.sprite_source,
            }
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one simulation step with simple car-like dynamics."""
        steer = 0.0
        accel = 0.0

        if action == 1:
            accel = self._acceleration
        elif action == 2:
            accel = -self._acceleration
        elif action == 3:
            steer = -self._turn_rate
        elif action == 4:
            steer = self._turn_rate

        self.heading += steer
        self.speed = float(
            np.clip(self.speed + accel - self._friction, 0.0, self._max_speed)
        )

        prev_pos = self.position.copy()
        velocity = np.array(
            [np.cos(self.heading), np.sin(self.heading)], dtype=np.float32
        )
        self.position = self.position + velocity * self.speed
        self.step_count += 1

        collision = self._is_wall(self.position)
        if collision:
            self.position = prev_pos
            self.crashed = True

        dist_goal = float(np.linalg.norm(self.position - self._goal))
        self.reached_goal = dist_goal < 10.0

        terminated = bool(self.crashed or self.reached_goal)
        truncated = bool(self.step_count >= self.max_steps and not terminated)

        progress = max(0.0, prev_pos[0] - self._start[0])
        reward = -0.02 + 0.002 * progress + 0.02 * self.speed
        if self.crashed:
            reward -= 5.0
        if self.reached_goal:
            reward += 25.0

        obs = self._get_obs()
        info = {
            "collision": self.crashed,
            "reached_goal": self.reached_goal,
            "distance_to_goal": dist_goal,
            "position": self.position.copy(),
            "heading": float(self.heading),
        }
        return obs, float(reward), terminated, truncated, info

    def _draw_car_procedural(self, frame: np.ndarray) -> None:
        """Draw a triangular car marker when no sprite asset is available."""
        cx = int(round(float(self.position[0])))
        cy = int(round(float(self.position[1])))

        tip = np.array([cx + 8 * np.cos(self.heading), cy + 8 * np.sin(self.heading)])
        left = np.array(
            [
                cx + 5 * np.cos(self.heading + 2.4),
                cy + 5 * np.sin(self.heading + 2.4),
            ]
        )
        right = np.array(
            [
                cx + 5 * np.cos(self.heading - 2.4),
                cy + 5 * np.sin(self.heading - 2.4),
            ]
        )

        points = np.vstack([tip, left, right]).astype(int)
        min_x, max_x = points[:, 0].min(), points[:, 0].max()
        min_y, max_y = points[:, 1].min(), points[:, 1].max()

        for y in range(max(0, min_y), min(self.height, max_y + 1)):
            for x in range(max(0, min_x), min(self.width, max_x + 1)):
                p = np.array([x, y])
                v0 = points[2] - points[0]
                v1 = points[1] - points[0]
                v2 = p - points[0]

                den = v0[0] * v1[1] - v1[0] * v0[1]
                if den == 0:
                    continue
                a = (v2[0] * v1[1] - v1[0] * v2[1]) / den
                b = (v0[0] * v2[1] - v2[0] * v0[1]) / den
                c = 1.0 - a - b
                if a >= 0 and b >= 0 and c >= 0:
                    frame[y, x] = np.array([230, 90, 70], dtype=np.uint8)

    def _overlay_sprite(self, frame: np.ndarray) -> None:
        """Overlay sprite image centered at car position."""
        if self.sprite_rgba is None:
            self._draw_car_procedural(frame)
            return

        sprite = self.sprite_rgba
        h, w = sprite.shape[:2]
        x0 = int(round(float(self.position[0]))) - w // 2
        y0 = int(round(float(self.position[1]))) - h // 2

        for sy in range(h):
            yy = y0 + sy
            if yy < 0 or yy >= self.height:
                continue
            for sx in range(w):
                xx = x0 + sx
                if xx < 0 or xx >= self.width:
                    continue
                alpha = float(sprite[sy, sx, 3]) / 255.0
                if alpha <= 0:
                    continue
                frame[yy, xx] = (
                    alpha * sprite[sy, sx, :3] + (1.0 - alpha) * frame[yy, xx]
                ).astype(np.uint8)

    def _draw_probes(self, frame: np.ndarray) -> None:
        """Draw probe rays for sensor interpretation."""
        origin = self.position.copy()
        for angle in self._probe_angles:
            ray_angle = self.heading + angle
            direction = np.array(
                [np.cos(ray_angle), np.sin(ray_angle)], dtype=np.float32
            )
            probe_pos = origin.copy()
            dist = 0.0
            while dist < self.probe_max_distance:
                probe_pos += direction
                dist += 1.0
                x = int(round(float(probe_pos[0])))
                y = int(round(float(probe_pos[1])))
                if x < 0 or x >= self.width or y < 0 or y >= self.height:
                    break
                if self.occupancy[y, x] == 1:
                    frame[y, x] = np.array([255, 60, 60], dtype=np.uint8)
                    break
                frame[y, x] = np.array([120, 220, 255], dtype=np.uint8)

    def render(self) -> np.ndarray:
        """Render current state to an RGB array frame."""
        free = np.array([240, 240, 240], dtype=np.uint8)
        wall = np.array([28, 30, 34], dtype=np.uint8)
        frame = np.where(self.occupancy[..., None] == 1, wall, free).astype(np.uint8)

        gx = int(round(float(self._goal[0])))
        gy = int(round(float(self._goal[1])))
        frame[
            max(0, gy - 6) : min(self.height, gy + 6),
            max(0, gx - 6) : min(self.width, gx + 6),
        ] = np.array([80, 220, 110], dtype=np.uint8)

        self._draw_probes(frame)
        self._overlay_sprite(frame)

        return frame

    def close(self) -> None:
        """Close environment resources."""
        return None

import math
import numpy as np
import pygame
from typing import Any

class NeonMountainCarRenderer:
    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Antigravity Neon AI Testbed")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Courier", 24, bold=True)
        self.title_font = pygame.font.SysFont("Courier", 32, bold=True)

        # Physics constraints of MountainCar-v0
        self.min_position = -1.2
        self.max_position = 0.6
        self.goal_position = 0.5
        
        # Colors
        self.bg_color = (15, 15, 25)
        self.curve_color = (0, 255, 255) # Neon Cyan
        self.car_color = (255, 0, 255)   # Neon Magenta
        self.goal_color = (0, 255, 100)  # Neon Green
        self.text_color = (220, 220, 220)

        # Precompute mountain curve points
        self.curve_points = []
        for x_pixel in range(self.width):
            x_math = self._pixel_to_math_x(x_pixel)
            y_math = math.sin(3 * x_math) * 0.45 + 0.55
            y_pixel = self._math_to_pixel_y(y_math)
            self.curve_points.append((x_pixel, y_pixel))
            
        # Slider State
        self.fps = 60
        self.min_fps = 5
        self.max_fps = 300
        self.slider_rect = pygame.Rect(self.width - 300, 80, 250, 10)
        self.is_dragging = False

    def _pixel_to_math_x(self, x_pixel: int) -> float:
        return self.min_position + (x_pixel / self.width) * (self.max_position - self.min_position)

    def _math_to_pixel_x(self, x_math: float) -> int:
        return int(((x_math - self.min_position) / (self.max_position - self.min_position)) * self.width)

    def _math_to_pixel_y(self, y_math: float) -> int:
        # Scale to leave some padding at top and bottom
        padding = 100
        usable_height = self.height - 2 * padding
        # y_math is roughly between 0.1 and 1.0. We map it to screen.
        return int(self.height - padding - y_math * usable_height)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    mouse_pos = pygame.mouse.get_pos()
                    # Check if click is near the slider
                    hitbox = self.slider_rect.inflate(20, 40)
                    if hitbox.collidepoint(mouse_pos):
                        self.is_dragging = True
                        self._update_fps_from_mouse(mouse_pos[0])
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.is_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.is_dragging:
                    self._update_fps_from_mouse(event.pos[0])

    def _update_fps_from_mouse(self, mouse_x: int):
        rel_x = max(0, min(mouse_x - self.slider_rect.x, self.slider_rect.width))
        ratio = rel_x / self.slider_rect.width
        self.fps = int(self.min_fps + ratio * (self.max_fps - self.min_fps))

    def render(self, state: np.ndarray, agent_name: str, episode: int, step: int, total_reward: float):
        self.handle_events()

        # Clear screen
        self.screen.fill(self.bg_color)

        # Draw glowing mountain curve with anti-aliasing
        pygame.draw.aalines(self.screen, self.curve_color, False, self.curve_points)
        # Add subtle glow by drawing a thicker, darker aliased line underneath
        pygame.draw.lines(self.screen, (0, 100, 100), False, self.curve_points, 6)
        pygame.draw.aalines(self.screen, self.curve_color, False, self.curve_points)
        
        # Draw Goal Flag
        goal_px = self._math_to_pixel_x(self.goal_position)
        goal_py = self._math_to_pixel_y(math.sin(3 * self.goal_position) * 0.45 + 0.55)
        pygame.draw.line(self.screen, (200, 200, 200), (goal_px, goal_py), (goal_px, goal_py - 60), 3)
        pygame.draw.polygon(self.screen, self.goal_color, [(goal_px, goal_py - 60), (goal_px + 30, goal_py - 45), (goal_px, goal_py - 30)])

        # Draw Car
        car_pos = state[0]
        car_px = self._math_to_pixel_x(car_pos)
        car_py = self._math_to_pixel_y(math.sin(3 * car_pos) * 0.45 + 0.55)
        
        # Draw car body (sleek angled polygon)
        car_width = 40
        car_height = 20
        angle = math.cos(3 * car_pos) # rough slope angle
        
        # Draw simple glowing circle for now for clean neon look
        pygame.draw.circle(self.screen, self.car_color, (car_px, car_py - 15), 15)
        # Add a subtle core glow
        pygame.draw.circle(self.screen, (255, 150, 255), (car_px, car_py - 15), 8)

        # Draw Metrics Dashboard
        texts = [
            f"Agent   : {agent_name}",
            f"Eval Run: {episode}",
            f"Step    : {step}",
            f"Reward  : {total_reward:.1f}"
        ]
        
        for i, t in enumerate(texts):
            img = self.font.render(t, True, self.text_color)
            self.screen.blit(img, (30, 30 + i * 35))

        # Draw FPS Slider UI
        title = self.title_font.render("Speed Control", True, self.curve_color)
        self.screen.blit(title, (self.width - 300, 30))
        
        # Draw slider track
        pygame.draw.rect(self.screen, (60, 60, 80), self.slider_rect, border_radius=5)
        
        # Draw active track segment
        ratio = (self.fps - self.min_fps) / (self.max_fps - self.min_fps)
        active_rect = pygame.Rect(self.slider_rect.x, self.slider_rect.y, int(self.slider_rect.width * ratio), self.slider_rect.height)
        pygame.draw.rect(self.screen, self.curve_color, active_rect, border_radius=5)
        
        # Draw knob
        knob_x = self.slider_rect.x + int(self.slider_rect.width * ratio)
        knob_y = self.slider_rect.centery
        pygame.draw.circle(self.screen, (255, 255, 255), (knob_x, knob_y), 10)
        
        # Draw FPS label
        fps_label = self.font.render(f"{self.fps} FPS", True, self.text_color)
        self.screen.blit(fps_label, (self.width - 150 - fps_label.get_width()//2, 105))

        pygame.display.flip()
        
        # Cap framerate
        if self.fps < self.max_fps:
            self.clock.tick(self.fps)
            
    def close(self):
        pygame.quit()

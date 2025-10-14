"""
Enhanced 2D Drift Environment with improved visualization.

Enhancements over base gym_drift_env:
- Tire smoke particles during drift
- Skid marks on track  
- Motion trail/ghost
- Enhanced vehicle graphics (gradient, wheels)
- Professional racing HUD
- Speed lines and motion blur
- Shadow effects
- Better telemetry display
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import math
import pygame
from collections import deque
from typing import Optional, Tuple

from src.rl.gym_drift_env import GymDriftEnv


class Particle:
    """Smoke/drift particle."""
    def __init__(self, x: float, y: float, vx: float = 0, vy: float = 0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = 1.0  # 1.0 = fully alive, 0.0 = dead
        self.max_life = 1.0
        self.size = np.random.uniform(3, 8)
    
    def update(self, dt: float = 0.05):
        """Update particle."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt * 2  # Fade out over 0.5s
        self.vx *= 0.95  # Drag
        self.vy *= 0.95


class EnhancedDriftEnv(GymDriftEnv):
    """
    Enhanced drift environment with better visualization.
    
    New features:
    - Tire smoke when drifting
    - Skid marks
    - Motion trail
    - Professional HUD
    - Better vehicle graphics
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Visual effects
        self.smoke_particles = []
        self.skid_marks = deque(maxlen=1000)
        self.motion_trail = deque(maxlen=30)
        
        # Colors
        self.VEHICLE_BLUE = (40, 80, 200)
        self.VEHICLE_DARK = (20, 40, 120)
        self.SMOKE_GRAY = (150, 150, 150)
        self.SKID_BLACK = (30, 30, 30)
        self.HUD_GREEN = (50, 255, 100)
        self.HUD_RED = (255, 50, 50)
        self.HUD_YELLOW = (255, 200, 50)
    
    def reset(self, seed=None, options=None):
        """Reset and clear visual effects."""
        obs, info = super().reset(seed=seed, options=options)
        self.smoke_particles.clear()
        self.skid_marks.clear()
        self.motion_trail.clear()
        return obs, info
    
    def step(self, action):
        """Step and update visual effects."""
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update visual effects
        self._update_effects()
        
        return obs, reward, terminated, truncated, info
    
    def _update_effects(self):
        """Update all visual effects."""
        state = self.sim_env.vehicle.get_state()
        
        # Add to motion trail
        self.motion_trail.append((state.x, state.y, state.theta))
        
        # Check if drifting (high slip angle)
        if abs(state.velocity) > 0.5:
            expected_av = state.velocity * math.tan(state.steering_angle) / 0.324
            slip = abs(state.angular_velocity - expected_av)
            
            if slip > 0.3:  # Significant drift
                # Add skid marks
                self.skid_marks.append((state.x, state.y, state.theta))
                
                # Add smoke particles
                for _ in range(2):
                    # Spawn near rear tires
                    offset_x = -0.15 * math.cos(state.theta)
                    offset_y = -0.15 * math.sin(state.theta)
                    lateral = np.random.uniform(-0.2, 0.2)
                    
                    px = state.x + offset_x + lateral * math.sin(state.theta)
                    py = state.y + offset_y - lateral * math.cos(state.theta)
                    
                    # Particle velocity (slightly opposite to motion)
                    vx = -state.velocity * math.cos(state.theta) * 0.3
                    vy = -state.velocity * math.sin(state.theta) * 0.3
                    
                    self.smoke_particles.append(Particle(px, py, vx, vy))
        
        # Update smoke particles
        self.smoke_particles = [
            p for p in self.smoke_particles 
            if p.life > 0
        ]
        for particle in self.smoke_particles:
            particle.update()
    
    def _render_human(self):
        """Enhanced rendering."""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Enhanced Drift Simulator 🏎️")
            self.clock = pygame.time.Clock()
        
        # Clear with slight gradient
        for y in range(self.window_size):
            color_val = int(220 + (y / self.window_size) * 20)
            pygame.draw.line(
                self.window,
                (color_val, color_val, color_val),
                (0, y),
                (self.window_size, y)
            )
        
        # Draw in layers (back to front)
        self._draw_grid()
        self._draw_skid_marks()
        self._draw_obstacles()
        self._draw_gate()
        self._draw_motion_trail()
        self._draw_smoke_particles()
        self._draw_vehicle_shadow()
        self._draw_enhanced_vehicle()
        self._draw_professional_hud()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def _draw_skid_marks(self):
        """Draw tire skid marks."""
        if len(self.skid_marks) < 2:
            return
        
        for i, (x, y, theta) in enumerate(self.skid_marks):
            pos = self._world_to_screen(x, y)
            alpha = int(50 * (i / len(self.skid_marks)))  # Fade older marks
            
            # Draw small rectangle as skid mark
            surf = pygame.Surface((3, 8), pygame.SRCALPHA)
            surf.fill((*self.SKID_BLACK, alpha))
            
            # Rotate
            surf = pygame.transform.rotate(surf, -math.degrees(theta))
            rect = surf.get_rect(center=pos)
            self.window.blit(surf, rect)
    
    def _draw_smoke_particles(self):
        """Draw drift smoke."""
        for particle in self.smoke_particles:
            pos = self._world_to_screen(particle.x, particle.y)
            alpha = int(150 * particle.life)
            size = int(particle.size * (1 + (1 - particle.life)))
            
            if alpha > 0 and 0 <= pos[0] < self.window_size and 0 <= pos[1] < self.window_size:
                surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(
                    surf,
                    (*self.SMOKE_GRAY, alpha),
                    (size, size),
                    size
                )
                self.window.blit(surf, (pos[0]-size, pos[1]-size))
    
    def _draw_motion_trail(self):
        """Draw motion trail (ghost car)."""
        if len(self.motion_trail) < 2:
            return
        
        # Convert deque to list for slicing
        trail_list = list(self.motion_trail)
        for i, (x, y, theta) in enumerate(trail_list[::3]):  # Every 3rd point
            pos = self._world_to_screen(x, y)
            alpha = int(30 * (i / (len(self.motion_trail) / 3)))
            
            # Small ghost
            surf = pygame.Surface((15, 10), pygame.SRCALPHA)
            pygame.draw.ellipse(surf, (*self.VEHICLE_BLUE, alpha), surf.get_rect())
            surf = pygame.transform.rotate(surf, -math.degrees(theta))
            rect = surf.get_rect(center=pos)
            self.window.blit(surf, rect)
    
    def _draw_vehicle_shadow(self):
        """Draw shadow under vehicle."""
        state = self.sim_env.vehicle.get_state()
        pos = self._world_to_screen(state.x, state.y)
        
        # Offset shadow slightly
        shadow_offset = (5, 5)
        shadow_pos = (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1])
        
        length = int(self.sim_env.vehicle.WHEELBASE * self.scale * 0.9)
        width = int(self.sim_env.vehicle.VEHICLE_WIDTH * self.scale * 0.9)
        
        surf = pygame.Surface((length, width), pygame.SRCALPHA)
        pygame.draw.ellipse(surf, (0, 0, 0, 50), surf.get_rect())
        surf = pygame.transform.rotate(surf, -math.degrees(state.theta))
        rect = surf.get_rect(center=shadow_pos)
        self.window.blit(surf, rect)
    
    def _draw_enhanced_vehicle(self):
        """Draw vehicle with gradient, wheels, details."""
        state = self.sim_env.vehicle.get_state()
        pos = self._world_to_screen(state.x, state.y)
        
        length = int(self.sim_env.vehicle.WHEELBASE * self.scale)
        width = int(self.sim_env.vehicle.VEHICLE_WIDTH * self.scale)
        
        # Create vehicle surface
        surf = pygame.Surface((length + 10, width + 10), pygame.SRCALPHA)
        center = (length//2 + 5, width//2 + 5)
        
        # Main body with rounded rect
        body_rect = pygame.Rect(5, 5, length, width)
        pygame.draw.rect(surf, self.VEHICLE_BLUE, body_rect, border_radius=width//3)
        pygame.draw.rect(surf, self.VEHICLE_DARK, body_rect, 2, border_radius=width//3)
        
        # Wheels (4 small rectangles)
        wheel_w, wheel_h = width//4, width//2
        # Front left
        pygame.draw.rect(surf, (30, 30, 30), 
                        (length - wheel_h//2, 2, wheel_h, wheel_w))
        # Front right
        pygame.draw.rect(surf, (30, 30, 30),
                        (length - wheel_h//2, width - wheel_w + 3, wheel_h, wheel_w))
        # Rear left
        pygame.draw.rect(surf, (30, 30, 30),
                        (3, 2, wheel_h, wheel_w))
        # Rear right
        pygame.draw.rect(surf, (30, 30, 30),
                        (3, width - wheel_w + 3, wheel_h, wheel_w))
        
        # Heading indicator (arrow)
        arrow_start = (center[0] + length//3, center[1])
        arrow_end = (center[0] + length//2 + 5, center[1])
        pygame.draw.line(surf, self.HUD_YELLOW, arrow_start, arrow_end, 3)
        
        # Drift angle indicator (if drifting)
        if abs(state.velocity) > 0.5:
            expected_av = state.velocity * math.tan(state.steering_angle) / 0.324
            slip = abs(state.angular_velocity - expected_av)
            if slip > 0.3:
                # Draw drift angle arc
                pygame.draw.circle(surf, self.HUD_RED, center, length//2, 2)
        
        # Rotate and blit
        surf = pygame.transform.rotate(surf, -math.degrees(state.theta))
        rect = surf.get_rect(center=pos)
        self.window.blit(surf, rect)
    
    def _draw_professional_hud(self):
        """Racing game style HUD."""
        state = self.sim_env.vehicle.get_state()
        
        font_large = pygame.font.Font(None, 36)
        font_medium = pygame.font.Font(None, 28)
        font_small = pygame.font.Font(None, 22)
        
        # Semi-transparent panel
        panel_width = 250
        panel_height = 200
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((20, 20, 40, 200))
        self.window.blit(panel, (10, 10))
        
        y_offset = 20
        
        # Speed (large and prominent)
        speed_kmh = abs(state.velocity) * 3.6  # m/s to km/h
        speed_text = font_large.render(f"{speed_kmh:.0f}", True, self.HUD_GREEN)
        self.window.blit(speed_text, (30, y_offset))
        unit_text = font_small.render("km/h", True, (200, 200, 200))
        self.window.blit(unit_text, (120, y_offset + 10))
        y_offset += 50
        
        # Angular velocity
        av_text = font_medium.render(
            f"ω: {state.angular_velocity:.2f} rad/s",
            True,
            (200, 200, 200)
        )
        self.window.blit(av_text, (20, y_offset))
        y_offset += 30
        
        # Position
        pos_text = font_small.render(
            f"Pos: ({state.x:.1f}, {state.y:.1f})",
            True,
            (180, 180, 180)
        )
        self.window.blit(pos_text, (20, y_offset))
        y_offset += 25
        
        # Episode info
        episode_text = font_small.render(
            f"Step: {self.steps}/{self.max_steps}",
            True,
            (180, 180, 180)
        )
        self.window.blit(episode_text, (20, y_offset))
        y_offset += 25
        
        # Reward
        reward_color = self.HUD_GREEN if self.episode_reward > 0 else self.HUD_RED
        reward_text = font_small.render(
            f"Reward: {self.episode_reward:.1f}",
            True,
            reward_color
        )
        self.window.blit(reward_text, (20, y_offset))
        
        # Drift indicator (bottom center)
        if abs(state.velocity) > 0.5:
            expected_av = state.velocity * math.tan(state.steering_angle) / 0.324
            slip = abs(state.angular_velocity - expected_av)
            if slip > 0.3:
                drift_text = font_large.render("DRIFT!", True, self.HUD_YELLOW)
                drift_rect = drift_text.get_rect(
                    center=(self.window_size//2, self.window_size - 50)
                )
                
                # Pulsing effect
                import time
                pulse = abs(math.sin(time.time() * 5))
                alpha = int(150 + 105 * pulse)
                drift_surf = pygame.Surface(drift_text.get_size(), pygame.SRCALPHA)
                drift_text_copy = font_large.render("DRIFT!", True, (*self.HUD_YELLOW, alpha))
                self.window.blit(drift_text_copy, drift_rect)


# Make this the default when imported
DriftEnv = EnhancedDriftEnv

"""
Gymnasium-compliant drift environment with visual rendering.

Fully compatible with Gymnasium API for RL training with real-time visualization.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import pygame
import math

from src.simulator.environment import SimulationEnvironment


class GymDriftEnv(gym.Env):
    """
    Gymnasium environment for autonomous drifting.
    
    Observation Space:
        Box(10): [velocity, angular_velocity, x, y, cos(theta), sin(theta),
                  dist_to_gate, cos(angle_to_gate), sin(angle_to_gate), min_obstacle_dist]
    
    Action Space:
        Box(2): [velocity_command, angular_velocity_command] in [-1, 1]
    
    Reward:
        Dense reward shaping:
        - Progress toward gate
        - Alignment with goal
        - Collision penalty
        - Success bonus
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }
    
    def __init__(
        self,
        scenario: str = "loose",
        max_steps: int = 400,
        render_mode: Optional[str] = None,
        dense_rewards: bool = True
    ):
        """
        Initialize Gymnasium environment.
        
        Args:
            scenario: "loose" or "tight"
            max_steps: Maximum episode length
            render_mode: "human" (pygame window) or "rgb_array" (for video)
            dense_rewards: Use dense vs sparse rewards
        """
        super().__init__()
        
        self.scenario = scenario
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.dense_rewards = dense_rewards
        
        # Action space: [velocity_cmd, angular_velocity_cmd] normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation space: 10-dimensional state
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(10,),
            dtype=np.float32
        )
        
        # Physical limits for action scaling
        self.max_velocity = 4.0
        self.max_angular_velocity = 3.0
        
        # Simulation environment
        self.sim_env = SimulationEnvironment(dt=0.05)
        
        # Episode tracking
        self.steps = 0
        self.episode_reward = 0.0
        self.trajectory = []
        
        # Scenario configuration
        if scenario == "loose":
            self.gate_width = 2.13
            self.gate_pos = np.array([3.0, 1.065])
        else:  # tight
            self.gate_width = 0.81
            self.gate_pos = np.array([3.0, 0.405])
        
        # Pygame rendering
        self.window = None
        self.clock = None
        self.window_size = 800
        self.scale = 60  # pixels per meter
        self.camera_offset = np.array([100, self.window_size // 2])
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset simulation
        self.sim_env.reset()
        
        # Setup scenario
        if self.scenario == "loose":
            self.sim_env.setup_loose_drift_test()
        else:
            self.sim_env.setup_tight_drift_test()
        
        self.steps = 0
        self.episode_reward = 0.0
        self.trajectory = []
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Action array [velocity_cmd, angular_velocity_cmd]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Scale action to physical limits
        velocity_cmd = float(action[0]) * self.max_velocity
        angular_velocity_cmd = float(action[1]) * self.max_angular_velocity
        
        # Execute in simulator
        self.sim_env.set_control(velocity_cmd, angular_velocity_cmd)
        measurements = self.sim_env.step()
        
        # Record trajectory
        state = self.sim_env.vehicle.get_state()
        self.trajectory.append([state.x, state.y, state.theta])
        
        # Get observation
        observation = self._get_obs()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check termination
        terminated, truncated, info = self._check_done()
        
        self.steps += 1
        self.episode_reward += reward
        
        info["episode_reward"] = self.episode_reward
        info["steps"] = self.steps
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render environment based on render_mode."""
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render to pygame window."""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Autonomous Drift Simulator")
            self.clock = pygame.time.Clock()
        
        # Clear screen
        self.window.fill((240, 240, 240))
        
        # Draw grid
        self._draw_grid()
        
        # Draw obstacles
        self._draw_obstacles()
        
        # Draw gate
        self._draw_gate()
        
        # Draw trajectory
        self._draw_trajectory()
        
        # Draw vehicle
        self._draw_vehicle()
        
        # Draw info panel
        self._draw_info_panel()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render to RGB array for video recording."""
        if self.window is None:
            pygame.init()
            self.window = pygame.Surface((self.window_size, self.window_size))
        
        # Same as human rendering but to surface
        self.window.fill((240, 240, 240))
        self._draw_grid()
        self._draw_obstacles()
        self._draw_gate()
        self._draw_trajectory()
        self._draw_vehicle()
        self._draw_info_panel()
        
        # Convert to RGB array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window)),
            axes=(1, 0, 2)
        )
    
    def _draw_grid(self):
        """Draw background grid."""
        for i in range(0, self.window_size, self.scale):
            # Vertical lines
            pygame.draw.line(
                self.window,
                (220, 220, 220),
                (i, 0),
                (i, self.window_size),
                1
            )
            # Horizontal lines
            pygame.draw.line(
                self.window,
                (220, 220, 220),
                (0, i),
                (self.window_size, i),
                1
            )
    
    def _draw_obstacles(self):
        """Draw obstacles."""
        for obs in self.sim_env.obstacles:
            pos = self._world_to_screen(obs.x, obs.y)
            radius = int(obs.radius * self.scale)
            
            # Draw obstacle
            pygame.draw.circle(
                self.window,
                (200, 100, 100),  # Red
                pos,
                radius
            )
            pygame.draw.circle(
                self.window,
                (150, 50, 50),
                pos,
                radius,
                2
            )
    
    def _draw_gate(self):
        """Draw goal gate."""
        # Gate posts (cones)
        y1 = self.gate_pos[1] - self.gate_width / 2
        y2 = self.gate_pos[1] + self.gate_width / 2
        
        pos1 = self._world_to_screen(self.gate_pos[0], y1)
        pos2 = self._world_to_screen(self.gate_pos[0], y2)
        
        # Draw gate line
        pygame.draw.line(
            self.window,
            (50, 200, 50),  # Green
            pos1,
            pos2,
            4
        )
        
        # Draw gate markers
        for pos in [pos1, pos2]:
            pygame.draw.circle(
                self.window,
                (50, 200, 50),
                pos,
                8
            )
    
    def _draw_trajectory(self):
        """Draw vehicle trajectory."""
        if len(self.trajectory) < 2:
            return
        
        points = [
            self._world_to_screen(x, y)
            for x, y, _ in self.trajectory
        ]
        
        # Draw trajectory line
        if len(points) > 1:
            pygame.draw.lines(
                self.window,
                (100, 150, 255),  # Light blue
                False,
                points,
                2
            )
    
    def _draw_vehicle(self):
        """Draw vehicle."""
        state = self.sim_env.vehicle.get_state()
        pos = self._world_to_screen(state.x, state.y)
        
        # Vehicle dimensions
        length = self.sim_env.vehicle.WHEELBASE * self.scale
        width = self.sim_env.vehicle.VEHICLE_WIDTH * self.scale
        
        # Vehicle corners (rectangle)
        corners = [
            (-length/2, -width/2),
            (length/2, -width/2),
            (length/2, width/2),
            (-length/2, width/2)
        ]
        
        # Rotate corners
        cos_theta = math.cos(state.theta)
        sin_theta = math.sin(state.theta)
        
        rotated_corners = []
        for dx, dy in corners:
            rx = dx * cos_theta - dy * sin_theta
            ry = dx * sin_theta + dy * cos_theta
            rotated_corners.append((
                pos[0] + rx,
                pos[1] + ry
            ))
        
        # Draw vehicle body
        pygame.draw.polygon(
            self.window,
            (50, 50, 200),  # Blue
            rotated_corners
        )
        pygame.draw.polygon(
            self.window,
            (30, 30, 150),
            rotated_corners,
            3
        )
        
        # Draw heading indicator
        heading_length = length * 0.7
        heading_end = (
            pos[0] + heading_length * cos_theta,
            pos[1] + heading_length * sin_theta
        )
        pygame.draw.line(
            self.window,
            (255, 200, 50),  # Yellow
            pos,
            heading_end,
            3
        )
    
    def _draw_info_panel(self):
        """Draw info panel with stats."""
        state = self.sim_env.vehicle.get_state()
        
        font = pygame.font.Font(None, 24)
        
        info_lines = [
            f"Step: {self.steps}/{self.max_steps}",
            f"Reward: {self.episode_reward:.2f}",
            f"Velocity: {state.velocity:.2f} m/s",
            f"Ang Vel: {state.angular_velocity:.2f} rad/s",
            f"Position: ({state.x:.2f}, {state.y:.2f})",
            f"Scenario: {self.scenario.upper()}"
        ]
        
        y_offset = 10
        for line in info_lines:
            text = font.render(line, True, (0, 0, 0))
            self.window.blit(text, (10, y_offset))
            y_offset += 25
    
    def _world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int(x * self.scale + self.camera_offset[0])
        screen_y = int(self.window_size - (y * self.scale + self.camera_offset[1]))
        return (screen_x, screen_y)
    
    def _get_obs(self) -> np.ndarray:
        """Get observation from current state."""
        state = self.sim_env.vehicle.get_state()
        
        # Normalize values
        obs = [
            state.velocity / self.max_velocity,
            state.angular_velocity / self.max_angular_velocity,
            state.x / 10.0,
            state.y / 10.0,
            math.cos(state.theta),
            math.sin(state.theta),
        ]
        
        # Distance and angle to gate
        dx = self.gate_pos[0] - state.x
        dy = self.gate_pos[1] - state.y
        dist_to_gate = math.sqrt(dx**2 + dy**2) / 10.0
        angle_to_gate = math.atan2(dy, dx) - state.theta
        
        # Normalize angle
        angle_to_gate = math.atan2(math.sin(angle_to_gate), math.cos(angle_to_gate))
        
        obs.extend([
            dist_to_gate,
            math.cos(angle_to_gate),
            math.sin(angle_to_gate)
        ])
        
        # Distance to nearest obstacle
        min_dist = 10.0
        for obstacle in self.sim_env.obstacles:
            dist = math.sqrt((state.x - obstacle.x)**2 + (state.y - obstacle.y)**2)
            min_dist = min(min_dist, dist)
        
        obs.append(min_dist / 10.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        state = self.sim_env.vehicle.get_state()
        
        if not self.dense_rewards:
            # Sparse reward
            if self._passed_gate():
                return 100.0
            elif self.sim_env.check_collision():
                return -100.0
            else:
                return -0.1
        
        # Dense reward shaping
        reward = 0.0
        
        # Progress toward gate
        dx = self.gate_pos[0] - state.x
        dy = self.gate_pos[1] - state.y
        dist_to_gate = math.sqrt(dx**2 + dy**2)
        
        reward += -0.1 * dist_to_gate
        
        # Penalty for lateral error (distance from gate centerline)
        lateral_error = abs(dy - self.gate_width / 2)
        reward += -0.5 * lateral_error
        
        # Alignment with gate
        angle_to_gate = math.atan2(dy, dx) - state.theta
        angle_error = abs(math.atan2(math.sin(angle_to_gate), math.cos(angle_to_gate)))
        reward += -0.3 * angle_error
        
        # Collision penalty
        if self.sim_env.check_collision():
            reward += -50.0
        
        # Success bonus
        if self._passed_gate():
            reward += 50.0
        
        # Control effort penalty
        reward += -0.01 * (abs(state.velocity) + abs(state.angular_velocity))
        
        return reward
    
    def _check_done(self) -> Tuple[bool, bool, Dict[str, Any]]:
        """
        Check termination conditions.
        
        Returns:
            Tuple of (terminated, truncated, info)
        """
        state = self.sim_env.vehicle.get_state()
        info = {}
        
        # Terminated (success or failure)
        terminated = False
        
        if self.sim_env.check_collision():
            terminated = True
            info["termination_reason"] = "collision"
        elif self._passed_gate():
            terminated = True
            info["termination_reason"] = "success"
        
        # Truncated (timeout or out of bounds)
        truncated = False
        
        if self.steps >= self.max_steps:
            truncated = True
            info["truncation_reason"] = "max_steps"
        elif abs(state.x) > 20 or abs(state.y) > 20:
            truncated = True
            info["truncation_reason"] = "out_of_bounds"
        
        return terminated, truncated, info
    
    def _passed_gate(self) -> bool:
        """Check if vehicle successfully passed through gate."""
        state = self.sim_env.vehicle.get_state()
        
        # Must be past gate in x direction
        if state.x <= self.gate_pos[0] - 0.5:
            return False
        
        if state.x >= self.gate_pos[0] + 0.5:
            # Check y position within gate bounds
            y_min = self.gate_pos[1] - self.gate_width / 2
            y_max = self.gate_pos[1] + self.gate_width / 2
            
            if y_min <= state.y <= y_max:
                return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dict."""
        state = self.sim_env.vehicle.get_state()
        return {
            "x": state.x,
            "y": state.y,
            "theta": state.theta,
            "velocity": state.velocity,
            "angular_velocity": state.angular_velocity,
            "scenario": self.scenario
        }
    
    def close(self):
        """Clean up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

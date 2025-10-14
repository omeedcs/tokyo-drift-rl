"""
Gym-style environment wrapper for drift control.

Provides standard RL interface for training drift policies.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from src.simulator.environment import SimulationEnvironment


class DriftEnvironment:
    """
    Gym-style environment for drift tasks.
    
    State space (10-15 dims):
        - velocity
        - angular_velocity  
        - x, y, theta (position/heading)
        - distance to obstacles (ray casts)
        - distance to gate center
        - angle to gate
        
    Action space (2 dims):
        - velocity_command [-1, 1]
        - angular_velocity_command [-1, 1]
    """
    
    def __init__(
        self,
        gate_width: float = 2.13,
        gate_pos: Tuple[float, float] = (3.0, 1.065),
        dt: float = 0.05,
        max_steps: int = 400,
        dense_rewards: bool = True
    ):
        """
        Initialize environment.
        
        Args:
            gate_width: Width of gate
            gate_pos: Position of gate center
            dt: Simulation timestep
            max_steps: Maximum episode length
            dense_rewards: Use dense (shaped) rewards vs sparse
        """
        self.gate_width = gate_width
        self.gate_pos = gate_pos
        self.max_steps = max_steps
        self.dense_rewards = dense_rewards
        
        # Create simulation environment
        self.sim_env = SimulationEnvironment(dt=dt)
        
        # Action/state dimensions
        self.action_dim = 2  # velocity, angular_velocity
        self.state_dim = 10  # Will be expanded
        
        # Episode tracking
        self.steps = 0
        self.total_reward = 0.0
        
        # Action bounds (will be scaled to physical limits)
        self.max_velocity = 4.0
        self.max_angular_velocity = 3.0
    
    def reset(self, scenario: str = "loose") -> np.ndarray:
        """
        Reset environment to initial state.
        
        Args:
            scenario: "loose" or "tight"
            
        Returns:
            Initial state observation
        """
        self.sim_env.reset()
        
        # Setup scenario
        if scenario == "loose":
            self.sim_env.setup_loose_drift_test()
            self.gate_width = 2.13
            self.gate_pos = (3.0, 1.065)
        else:
            self.sim_env.setup_tight_drift_test()
            self.gate_width = 0.81
            self.gate_pos = (3.0, 0.405)
        
        self.steps = 0
        self.total_reward = 0.0
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take environment step.
        
        Args:
            action: Action array [velocity_cmd, angular_velocity_cmd] in [-1, 1]
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Scale actions to physical limits
        velocity_cmd = action[0] * self.max_velocity
        angular_velocity_cmd = action[1] * self.max_angular_velocity
        
        # Execute in simulator
        self.sim_env.set_control(velocity_cmd, angular_velocity_cmd)
        measurements = self.sim_env.step()
        
        # Get next state
        next_state = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward(measurements)
        
        # Check termination
        done, info = self._check_done(measurements)
        
        self.steps += 1
        self.total_reward += reward
        
        return next_state, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Extract state observation from simulator."""
        state = self.sim_env.vehicle.get_state()
        
        # Core measurements
        obs = [
            state.velocity / self.max_velocity,  # Normalized
            state.angular_velocity / self.max_angular_velocity,
            state.x / 10.0,  # Normalized position
            state.y / 10.0,
            np.cos(state.theta),  # Heading as sin/cos
            np.sin(state.theta),
        ]
        
        # Distance and angle to gate
        dx = self.gate_pos[0] - state.x
        dy = self.gate_pos[1] - state.y
        dist_to_gate = np.sqrt(dx**2 + dy**2) / 10.0  # Normalized
        angle_to_gate = np.arctan2(dy, dx) - state.theta
        
        # Normalize angle to [-pi, pi]
        angle_to_gate = np.arctan2(np.sin(angle_to_gate), np.cos(angle_to_gate))
        
        obs.extend([dist_to_gate, np.cos(angle_to_gate), np.sin(angle_to_gate)])
        
        # Distance to nearest obstacle
        min_dist = 10.0
        for obstacle in self.sim_env.obstacles:
            dist = np.sqrt((state.x - obstacle.x)**2 + (state.y - obstacle.y)**2)
            min_dist = min(min_dist, dist)
        
        obs.append(min_dist / 10.0)  # Normalized
        
        return np.array(obs, dtype=np.float32)
    
    def _compute_reward(self, measurements: Dict) -> float:
        """Compute reward for current state."""
        state = self.sim_env.vehicle.get_state()
        
        if not self.dense_rewards:
            # Sparse reward: only at gate passage or collision
            if self._passed_gate(state):
                return 100.0
            elif self.sim_env.check_collision():
                return -100.0
            else:
                return -0.1  # Small time penalty
        
        # Dense (shaped) reward
        reward = 0.0
        
        # Progress toward gate
        dx = self.gate_pos[0] - state.x
        dy = self.gate_pos[1] - state.y
        dist_to_gate = np.sqrt(dx**2 + dy**2)
        
        # Reward for getting closer
        reward += -0.1 * dist_to_gate
        
        # Penalty for being far from gate centerline
        lateral_error = abs(dy - self.gate_width / 2)
        reward += -0.5 * lateral_error
        
        # Reward for good alignment
        angle_to_gate = np.arctan2(dy, dx) - state.theta
        angle_error = abs(np.arctan2(np.sin(angle_to_gate), np.cos(angle_to_gate)))
        reward += -0.3 * angle_error
        
        # Penalty for collision
        if self.sim_env.check_collision():
            reward += -50.0
        
        # Bonus for passing gate
        if self._passed_gate(state):
            reward += 50.0
        
        # Small penalty for high control effort
        reward += -0.01 * (abs(state.velocity) + abs(state.angular_velocity))
        
        return reward
    
    def _check_done(self, measurements: Dict) -> Tuple[bool, Dict]:
        """Check if episode should terminate."""
        done = False
        info = {}
        
        state = self.sim_env.vehicle.get_state()
        
        # Collision
        if self.sim_env.check_collision():
            done = True
            info["termination"] = "collision"
        
        # Passed gate successfully
        elif self._passed_gate(state):
            done = True
            info["termination"] = "success"
        
        # Timeout
        elif self.steps >= self.max_steps:
            done = True
            info["termination"] = "timeout"
        
        # Out of bounds
        elif abs(state.x) > 20 or abs(state.y) > 20:
            done = True
            info["termination"] = "out_of_bounds"
        
        info["steps"] = self.steps
        info["total_reward"] = self.total_reward
        
        return done, info
    
    def _passed_gate(self, state) -> bool:
        """Check if vehicle passed through gate."""
        # Vehicle is past gate in x direction
        if state.x < self.gate_pos[0] - 0.5:
            return False
        
        if state.x > self.gate_pos[0] + 0.5:
            # Check if y position is within gate bounds
            y_min = self.gate_pos[1] - self.gate_width / 2
            y_max = self.gate_pos[1] + self.gate_width / 2
            
            if y_min <= state.y <= y_max:
                return True
        
        return False
    
    def render(self):
        """Render environment (optional)."""
        pass
    
    def close(self):
        """Close environment."""
        pass

"""
Virtual joystick and control strategies for simulation.
"""

import numpy as np
from typing import Optional, Callable, Tuple
from enum import Enum


class ControlMode(Enum):
    """Control modes for virtual joystick."""
    MANUAL = "manual"
    CIRCLE = "circle"
    DRIFT = "drift"
    TRAJECTORY = "trajectory"
    IKD_CORRECTED = "ikd_corrected"


class VirtualJoystick:
    """
    Virtual joystick for controlling simulated vehicle.
    
    Supports multiple control modes:
    - Manual: Direct velocity/curvature control
    - Circle: Constant velocity and curvature
    - Drift: Teleoperated drift sequence
    - Trajectory: Follow predefined trajectory
    - IKD-corrected: Use IKD model to correct commands
    """
    
    def __init__(self):
        """Initialize virtual joystick."""
        self.mode = ControlMode.MANUAL
        self.velocity = 0.0
        self.angular_velocity = 0.0
        
        # Circle mode parameters
        self.circle_velocity = 2.0
        self.circle_curvature = 0.7
        
        # IKD model (if loaded)
        self.ikd_model = None
        self.use_ikd_correction = False
    
    def set_manual_control(self, velocity: float, angular_velocity: float):
        """
        Set manual control inputs.
        
        Args:
            velocity: Linear velocity (m/s)
            angular_velocity: Angular velocity (rad/s)
        """
        self.mode = ControlMode.MANUAL
        self.velocity = velocity
        self.angular_velocity = angular_velocity
    
    def set_circle_mode(self, velocity: float, curvature: float):
        """
        Set circle navigation mode.
        
        Args:
            velocity: Constant velocity (m/s)
            curvature: Path curvature (1/m)
        """
        self.mode = ControlMode.CIRCLE
        self.circle_velocity = velocity
        self.circle_curvature = curvature
        self.velocity = velocity
        self.angular_velocity = velocity * curvature
    
    def load_ikd_model(self, model):
        """
        Load IKD model for control correction.
        
        Args:
            model: Trained IKD model
        """
        import torch
        self.ikd_model = model
        self.ikd_model.eval()
    
    def enable_ikd_correction(self, enable: bool = True):
        """Enable or disable IKD correction."""
        if self.ikd_model is None and enable:
            raise ValueError("IKD model not loaded. Call load_ikd_model() first.")
        self.use_ikd_correction = enable
    
    def get_control(
        self,
        current_velocity: float,
        current_angular_velocity: float
    ) -> Tuple[float, float]:
        """
        Get control commands based on current mode.
        
        Args:
            current_velocity: Current vehicle velocity (m/s)
            current_angular_velocity: Current angular velocity (rad/s)
            
        Returns:
            Tuple of (velocity_command, angular_velocity_command)
        """
        if self.mode == ControlMode.CIRCLE:
            velocity_cmd = self.circle_velocity
            angular_velocity_cmd = self.circle_velocity * self.circle_curvature
        else:
            velocity_cmd = self.velocity
            angular_velocity_cmd = self.angular_velocity
        
        # Apply IKD correction if enabled
        if self.use_ikd_correction and self.ikd_model is not None:
            angular_velocity_cmd = self._apply_ikd_correction(
                velocity_cmd,
                current_angular_velocity
            )
        
        return velocity_cmd, angular_velocity_cmd
    
    def _apply_ikd_correction(
        self,
        velocity: float,
        true_angular_velocity: float
    ) -> float:
        """
        Apply IKD model correction to angular velocity command.
        
        Args:
            velocity: Commanded velocity
            true_angular_velocity: True angular velocity from IMU
            
        Returns:
            Corrected angular velocity command
        """
        import torch
        
        # Prepare input for model
        model_input = torch.FloatTensor([[velocity, true_angular_velocity]])
        
        # Get prediction
        with torch.no_grad():
            corrected_av = self.ikd_model(model_input).item()
        
        return corrected_av


class DriftController:
    """
    Controller for executing drift maneuvers.
    
    Replicates teleoperated drift sequences from the paper.
    """
    
    def __init__(self, turbo_speed: float = 5.0):
        """
        Initialize drift controller.
        
        Args:
            turbo_speed: Maximum speed for drift approach (m/s)
        """
        self.turbo_speed = turbo_speed
        self.state = "idle"
        self.drift_start_time = 0.0
    
    def execute_drift_ccw(
        self,
        time: float,
        approach_distance: float = 2.0
    ) -> Tuple[float, float]:
        """
        Execute counter-clockwise drift maneuver.
        
        Args:
            time: Current simulation time
            approach_distance: Distance to travel before drift
            
        Returns:
            Tuple of (velocity, angular_velocity) commands
        """
        # State machine for drift execution
        if self.state == "idle":
            # Accelerate to turbo speed
            self.state = "accelerating"
            self.drift_start_time = time
            return self.turbo_speed, 0.0
        
        elif self.state == "accelerating":
            elapsed = time - self.drift_start_time
            
            if elapsed < 1.0:
                # Still accelerating
                return self.turbo_speed, 0.0
            else:
                # Start turning
                self.state = "turning"
                return self.turbo_speed, 2.0  # High angular velocity
        
        elif self.state == "turning":
            elapsed = time - self.drift_start_time
            
            if elapsed < 2.0:
                # Cut throttle, maintain turn
                return 2.0, 2.0
            else:
                # Exit drift
                self.state = "recovering"
                return 1.0, 0.5
        
        elif self.state == "recovering":
            elapsed = time - self.drift_start_time
            
            if elapsed < 3.0:
                return 1.0, 0.0
            else:
                self.state = "idle"
                return 0.0, 0.0
        
        return 0.0, 0.0
    
    def reset(self):
        """Reset drift controller state."""
        self.state = "idle"
        self.drift_start_time = 0.0


class TrajectoryFollower:
    """
    Controller to follow predefined trajectories.
    
    Useful for replaying recorded teleoperation sequences.
    """
    
    def __init__(self, velocity_trajectory: np.ndarray, av_trajectory: np.ndarray, dt: float = 0.05):
        """
        Initialize trajectory follower.
        
        Args:
            velocity_trajectory: Array of velocity commands
            av_trajectory: Array of angular velocity commands
            dt: Timestep between commands
        """
        self.velocity_trajectory = velocity_trajectory
        self.av_trajectory = av_trajectory
        self.dt = dt
        self.index = 0
    
    def get_control(self) -> Tuple[float, float]:
        """
        Get next control command from trajectory.
        
        Returns:
            Tuple of (velocity, angular_velocity)
        """
        if self.index >= len(self.velocity_trajectory):
            return 0.0, 0.0
        
        velocity = self.velocity_trajectory[self.index]
        angular_velocity = self.av_trajectory[self.index]
        self.index += 1
        
        return velocity, angular_velocity
    
    def reset(self):
        """Reset to start of trajectory."""
        self.index = 0
    
    def is_complete(self) -> bool:
        """Check if trajectory is complete."""
        return self.index >= len(self.velocity_trajectory)

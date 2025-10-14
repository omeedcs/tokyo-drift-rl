"""
Virtual joystick and control strategies for simulation.
"""

import numpy as np
from typing import Optional, Callable, Tuple, List
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
    Controller for executing drift maneuvers using trajectory tracking.
    
    Uses trajectory planning + Pure Pursuit for closed-loop control.
    Much better than the old open-loop time-based approach!
    """
    
    def __init__(
        self,
        turbo_speed: float = 3.5,
        drift_speed: float = 2.5,
        controller_type: str = "pure_pursuit",
        use_optimizer: bool = True
    ):
        """
        Initialize drift controller.
        
        Args:
            turbo_speed: Speed for drift approach (m/s)
            drift_speed: Speed during drift maneuver (m/s)
            controller_type: "pure_pursuit" or "stanley"
            use_optimizer: Use optimization-based planning (better for tight spaces)
        """
        from src.simulator.trajectory import DriftTrajectoryPlanner
        from src.simulator.path_tracking import TrajectoryTracker
        
        self.turbo_speed = turbo_speed
        self.drift_speed = drift_speed
        self.use_optimizer = use_optimizer
        
        # Create planner and tracker
        self.planner = DriftTrajectoryPlanner()
        self.tracker = TrajectoryTracker(controller_type=controller_type)
        self.optimizer = None  # Created on-demand
        
        self.trajectory_planned = False
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.current_velocity = 0.0
    
    def plan_trajectory(
        self,
        start_pos: Tuple[float, float],
        gate_center: Tuple[float, float],
        gate_width: float,
        direction: str = "ccw",
        obstacles: Optional[List[Tuple[float, float, float]]] = None
    ):
        """
        Plan drift trajectory through gate.
        
        Args:
            start_pos: Starting position (x, y)
            gate_center: Gate center position (x, y)
            gate_width: Width of gate opening
            direction: "ccw" or "cw"
            obstacles: Optional list of (x, y, radius) obstacles
        """
        if self.use_optimizer and obstacles is not None:
            # Use optimization-based planning
            from src.simulator.trajectory_optimizer import create_adaptive_planner
            
            self.optimizer = create_adaptive_planner(gate_width)
            trajectory = self.optimizer.plan_drift_trajectory(
                start_pos=start_pos,
                gate_pos=gate_center,
                gate_width=gate_width,
                obstacles=obstacles,
                direction=direction
            )
        else:
            # Use heuristic planning
            trajectory = self.planner.plan_drift_through_gate(
                start_pos=start_pos,
                gate_center=gate_center,
                gate_width=gate_width,
                drift_direction=direction,
                approach_speed=self.turbo_speed,
                drift_speed=self.drift_speed
            )
        
        self.tracker.set_trajectory(trajectory)
        self.trajectory_planned = True
        
        print(f"[INFO] Planned drift trajectory with {len(trajectory.waypoints)} waypoints")
    
    def update(
        self,
        x: float,
        y: float,
        theta: float,
        velocity: float
    ) -> Tuple[float, float]:
        """
        Update controller and get commands.
        
        Args:
            x: Current x position
            y: Current y position
            theta: Current heading
            velocity: Current velocity
            
        Returns:
            Tuple of (velocity_cmd, angular_velocity_cmd)
        """
        self.current_x = x
        self.current_y = y
        self.current_theta = theta
        self.current_velocity = velocity
        
        if not self.trajectory_planned:
            return 0.0, 0.0
        
        # Use trajectory tracker for closed-loop control
        return self.tracker.update(x, y, theta, velocity)
    
    def is_complete(self) -> bool:
        """Check if drift maneuver is complete."""
        return self.tracker.is_complete()
    
    def reset(self):
        """Reset drift controller state."""
        self.trajectory_planned = False
        self.tracker.reset()


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

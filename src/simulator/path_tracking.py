"""
Path tracking controllers for trajectory following.

Implements Pure Pursuit and Stanley controllers for closed-loop trajectory tracking.
"""

import numpy as np
from typing import Tuple, Optional
from src.simulator.trajectory import Trajectory, Waypoint


class PurePursuitController:
    """
    Pure Pursuit path tracking controller.
    
    A geometric path tracking algorithm that computes steering commands
    to follow a path by tracking a lookahead point.
    
    Reference: Coulter, R. Craig. "Implementation of the pure pursuit path 
    tracking algorithm." Carnegie Mellon University, 1992.
    """
    
    def __init__(
        self,
        wheelbase: float = 0.324,
        lookahead_distance: float = 0.5,
        lookahead_gain: float = 0.5,
        min_lookahead: float = 0.3,
        max_lookahead: float = 2.0
    ):
        """
        Initialize Pure Pursuit controller.
        
        Args:
            wheelbase: Vehicle wheelbase (m)
            lookahead_distance: Base lookahead distance (m)
            lookahead_gain: Proportional gain for speed-dependent lookahead
            min_lookahead: Minimum lookahead distance (m)
            max_lookahead: Maximum lookahead distance (m)
        """
        self.wheelbase = wheelbase
        self.base_lookahead = lookahead_distance
        self.lookahead_gain = lookahead_gain
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
    
    def compute_control(
        self,
        current_x: float,
        current_y: float,
        current_theta: float,
        current_velocity: float,
        trajectory: Trajectory
    ) -> Tuple[float, float]:
        """
        Compute steering and velocity commands.
        
        Args:
            current_x: Current x position
            current_y: Current y position
            current_theta: Current heading angle (radians)
            current_velocity: Current velocity
            trajectory: Trajectory to follow
            
        Returns:
            Tuple of (steering_angle, velocity_command)
        """
        # Compute adaptive lookahead distance
        lookahead_dist = np.clip(
            self.base_lookahead + self.lookahead_gain * abs(current_velocity),
            self.min_lookahead,
            self.max_lookahead
        )
        
        # Get lookahead point
        lookahead_point = trajectory.get_lookahead_point(
            current_x, current_y, lookahead_dist
        )
        
        if lookahead_point is None:
            # Trajectory complete, stop
            return 0.0, 0.0
        
        # Transform lookahead point to vehicle frame
        dx = lookahead_point.x - current_x
        dy = lookahead_point.y - current_y
        
        # Rotate to vehicle frame
        dx_veh = dx * np.cos(-current_theta) - dy * np.sin(-current_theta)
        dy_veh = dx * np.sin(-current_theta) + dy * np.cos(-current_theta)
        
        # Compute steering angle using pure pursuit law
        # steering_angle = atan(2 * L * sin(alpha) / ld)
        # where alpha is angle to lookahead point, L is wheelbase, ld is lookahead distance
        
        alpha = np.arctan2(dy_veh, dx_veh)
        
        # Pure pursuit steering law
        steering_angle = np.arctan2(
            2.0 * self.wheelbase * np.sin(alpha),
            lookahead_dist
        )
        
        # Velocity command from trajectory
        velocity_cmd = lookahead_point.velocity
        
        # Reduce speed in tight turns
        turn_speed_factor = 1.0 / (1.0 + 0.5 * abs(steering_angle))
        velocity_cmd *= turn_speed_factor
        
        return steering_angle, velocity_cmd
    
    def compute_angular_velocity(
        self,
        current_x: float,
        current_y: float,
        current_theta: float,
        current_velocity: float,
        trajectory: Trajectory
    ) -> Tuple[float, float]:
        """
        Compute velocity and angular velocity commands.
        
        Args:
            current_x: Current x position
            current_y: Current y position  
            current_theta: Current heading angle (radians)
            current_velocity: Current velocity
            trajectory: Trajectory to follow
            
        Returns:
            Tuple of (velocity_command, angular_velocity_command)
        """
        steering_angle, velocity_cmd = self.compute_control(
            current_x, current_y, current_theta, current_velocity, trajectory
        )
        
        # Convert steering angle to angular velocity
        # For Ackerman steering: omega = v * tan(delta) / L
        if abs(velocity_cmd) > 0.01:
            angular_velocity = velocity_cmd * np.tan(steering_angle) / self.wheelbase
        else:
            angular_velocity = 0.0
        
        return velocity_cmd, angular_velocity


class StanleyController:
    """
    Stanley path tracking controller.
    
    Uses both heading error and cross-track error for more aggressive tracking.
    Better for high-speed scenarios.
    
    Reference: Hoffmann, Gabriel M., et al. "Autonomous automobile trajectory 
    tracking for off-road driving: Controller design, experimental validation 
    and racing." ACC, 2007.
    """
    
    def __init__(
        self,
        wheelbase: float = 0.324,
        k_e: float = 0.5,  # Cross-track error gain
        k_v: float = 1.0,  # Velocity gain for softening
        max_steer: float = 0.45
    ):
        """
        Initialize Stanley controller.
        
        Args:
            wheelbase: Vehicle wheelbase (m)
            k_e: Cross-track error gain
            k_v: Velocity gain for softening
            max_steer: Maximum steering angle (rad)
        """
        self.wheelbase = wheelbase
        self.k_e = k_e
        self.k_v = k_v
        self.max_steer = max_steer
    
    def compute_control(
        self,
        current_x: float,
        current_y: float,
        current_theta: float,
        current_velocity: float,
        trajectory: Trajectory
    ) -> Tuple[float, float]:
        """
        Compute steering and velocity commands.
        
        Args:
            current_x: Current x position
            current_y: Current y position
            current_theta: Current heading angle (radians)
            current_velocity: Current velocity
            trajectory: Trajectory to follow
            
        Returns:
            Tuple of (steering_angle, velocity_command)
        """
        # Find closest point on trajectory
        closest_idx, closest_point = trajectory.get_closest_point(current_x, current_y)
        
        # Get next point for heading calculation
        if closest_idx + 1 < len(trajectory.waypoints):
            next_point = trajectory.waypoints[closest_idx + 1]
            path_heading = np.arctan2(
                next_point.y - closest_point.y,
                next_point.x - closest_point.x
            )
        else:
            path_heading = current_theta
        
        # Heading error
        heading_error = self._normalize_angle(path_heading - current_theta)
        
        # Cross-track error (lateral distance to path)
        dx = current_x - closest_point.x
        dy = current_y - closest_point.y
        
        # Project error onto path normal
        cross_track_error = -dx * np.sin(path_heading) + dy * np.cos(path_heading)
        
        # Stanley steering law
        # delta = heading_error + atan(k * e / (k_v + v))
        velocity_term = self.k_v + abs(current_velocity)
        steering_angle = heading_error + np.arctan2(
            self.k_e * cross_track_error,
            velocity_term
        )
        
        # Clamp steering
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
        
        # Velocity command
        velocity_cmd = closest_point.velocity
        
        # Slow down for large errors
        error_magnitude = abs(heading_error) + abs(cross_track_error)
        speed_factor = 1.0 / (1.0 + error_magnitude)
        velocity_cmd *= max(speed_factor, 0.5)
        
        return steering_angle, velocity_cmd
    
    def compute_angular_velocity(
        self,
        current_x: float,
        current_y: float,
        current_theta: float,
        current_velocity: float,
        trajectory: Trajectory
    ) -> Tuple[float, float]:
        """
        Compute velocity and angular velocity commands.
        
        Args:
            current_x: Current x position
            current_y: Current y position
            current_theta: Current heading angle (radians)
            current_velocity: Current velocity
            trajectory: Trajectory to follow
            
        Returns:
            Tuple of (velocity_command, angular_velocity_command)
        """
        steering_angle, velocity_cmd = self.compute_control(
            current_x, current_y, current_theta, current_velocity, trajectory
        )
        
        # Convert to angular velocity
        if abs(velocity_cmd) > 0.01:
            angular_velocity = velocity_cmd * np.tan(steering_angle) / self.wheelbase
        else:
            angular_velocity = 0.0
        
        return velocity_cmd, angular_velocity
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class TrajectoryTracker:
    """
    High-level trajectory tracking interface.
    
    Wraps Pure Pursuit or Stanley controller with additional features:
    - Trajectory completion detection
    - Emergency stops
    - Obstacle avoidance hooks
    """
    
    def __init__(
        self,
        controller_type: str = "pure_pursuit",
        wheelbase: float = 0.324
    ):
        """
        Initialize trajectory tracker.
        
        Args:
            controller_type: "pure_pursuit" or "stanley"
            wheelbase: Vehicle wheelbase
        """
        if controller_type == "pure_pursuit":
            self.controller = PurePursuitController(wheelbase=wheelbase)
        elif controller_type == "stanley":
            self.controller = StanleyController(wheelbase=wheelbase)
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")
        
        self.trajectory: Optional[Trajectory] = None
        self.completed = False
    
    def set_trajectory(self, trajectory: Trajectory):
        """Set new trajectory to follow."""
        self.trajectory = trajectory
        self.trajectory.reset()
        self.completed = False
    
    def update(
        self,
        current_x: float,
        current_y: float,
        current_theta: float,
        current_velocity: float
    ) -> Tuple[float, float]:
        """
        Update controller and get commands.
        
        Args:
            current_x: Current x position
            current_y: Current y position
            current_theta: Current heading angle
            current_velocity: Current velocity
            
        Returns:
            Tuple of (velocity_cmd, angular_velocity_cmd)
        """
        if self.trajectory is None:
            return 0.0, 0.0
        
        # Check if trajectory complete
        if self.trajectory.is_complete(current_x, current_y):
            self.completed = True
            return 0.0, 0.0
        
        # Get control from underlying controller
        velocity_cmd, angular_velocity_cmd = self.controller.compute_angular_velocity(
            current_x, current_y, current_theta, current_velocity, self.trajectory
        )
        
        return velocity_cmd, angular_velocity_cmd
    
    def is_complete(self) -> bool:
        """Check if trajectory tracking is complete."""
        return self.completed
    
    def reset(self):
        """Reset tracker."""
        if self.trajectory is not None:
            self.trajectory.reset()
        self.completed = False

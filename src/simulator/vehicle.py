"""
F1/10 Scale Vehicle Simulation
Based on UT AUTOmata specifications from:
"Learning Inverse Kinodynamics for Autonomous Vehicle Drifting" (Suvarna & Tehrani, 2024)

Physical Specifications:
- Scale: 1/10 F1 RC car
- Wheelbase: 0.324 meters (12.76 inches)
- Vehicle width: 0.48 meters (19 inches)
- Motor: TRAXXAS Titan 550
- Controller: Flipsky VESC 4.12 50A
- IMU: Vectornav VN-100
- Steering: Ackerman steering system
- Drive: Four-wheel drive
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class VehicleState:
    """Vehicle state representation."""
    x: float = 0.0          # X position (meters)
    y: float = 0.0          # Y position (meters)
    theta: float = 0.0      # Heading angle (radians)
    velocity: float = 0.0   # Linear velocity (m/s)
    angular_velocity: float = 0.0  # Angular velocity (rad/s)
    steering_angle: float = 0.0    # Steering angle (radians)


class F110Vehicle:
    """
    High-fidelity F1/10 vehicle simulator based on UT AUTOmata specifications.
    
    Includes:
    - Ackerman steering kinematics
    - Motor dynamics with ERPM limits
    - Servo constraints
    - Acceleration limits
    - Slip dynamics for drifting simulation
    """
    
    # Physical parameters from paper
    WHEELBASE = 0.324  # meters
    VEHICLE_WIDTH = 0.48  # meters (19 inches)
    
    # Motor and controller parameters
    SPEED_TO_ERPM_GAIN = 5356
    SPEED_TO_ERPM_OFFSET = 180.0
    ERPM_SPEED_LIMIT = 22000
    MAX_SPEED = 4.219  # m/s (ERPM-limited, from paper)
    
    # Servo parameters
    STEERING_TO_SERVO_GAIN = -0.9015
    STEERING_TO_SERVO_OFFSET = 0.57
    SERVO_MIN = 0.05
    SERVO_MAX = 0.95
    MAX_TURN_RATE = 0.25  # Maximum steering angle (radians)
    
    # Dynamics parameters
    ACCEL_LIMIT = 6.0  # m/s^2
    COMMAND_INTERVAL = 1.0 / 20.0  # 20 Hz command rate
    
    # Tire friction parameters (for slip modeling)
    MU_STATIC = 0.9   # Static friction coefficient
    MU_KINETIC = 0.7  # Kinetic friction coefficient
    SLIP_THRESHOLD = 3.0  # rad/s threshold for slip
    
    def __init__(self, dt: float = 0.05, enable_slip: bool = True):
        """
        Initialize F1/10 vehicle simulator.
        
        Args:
            dt: Simulation timestep (seconds)
            enable_slip: Whether to simulate tire slip/drifting
        """
        self.dt = dt
        self.enable_slip = enable_slip
        
        # State
        self.state = VehicleState()
        self.last_velocity = 0.0
        
        # Control inputs
        self.commanded_velocity = 0.0
        self.commanded_steering_angle = 0.0
        
    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """Reset vehicle to initial state."""
        self.state = VehicleState(x=x, y=y, theta=theta)
        self.last_velocity = 0.0
        self.commanded_velocity = 0.0
        self.commanded_steering_angle = 0.0
    
    def set_control(self, velocity: float, steering_angle: float):
        """
        Set control inputs (as would come from joystick).
        
        Args:
            velocity: Desired velocity (m/s)
            steering_angle: Desired steering angle (radians)
        """
        self.commanded_velocity = velocity
        self.commanded_steering_angle = steering_angle
    
    def _apply_motor_constraints(self, desired_velocity: float) -> float:
        """
        Apply motor ERPM limits and acceleration constraints.
        
        This replicates the behavior of the VESC motor controller.
        """
        # Acceleration limiting (smooth speed transitions)
        max_accel = self.ACCEL_LIMIT * self.COMMAND_INTERVAL
        smooth_velocity = np.clip(
            desired_velocity,
            self.last_velocity - max_accel,
            self.last_velocity + max_accel
        )
        
        # ERPM conversion and limiting
        erpm = self.SPEED_TO_ERPM_GAIN * smooth_velocity + self.SPEED_TO_ERPM_OFFSET
        erpm_clipped = np.clip(erpm, -self.ERPM_SPEED_LIMIT, self.ERPM_SPEED_LIMIT)
        
        # Convert back to velocity
        actual_velocity = (erpm_clipped - self.SPEED_TO_ERPM_OFFSET) / self.SPEED_TO_ERPM_GAIN
        
        self.last_velocity = actual_velocity
        return actual_velocity
    
    def _apply_servo_constraints(self, desired_angle: float) -> float:
        """
        Apply servo constraints to steering angle.
        
        This replicates the servo limits on the physical vehicle.
        """
        # Convert to servo value
        servo = self.STEERING_TO_SERVO_GAIN * desired_angle + self.STEERING_TO_SERVO_OFFSET
        
        # Clip to servo range
        servo_clipped = np.clip(servo, self.SERVO_MIN, self.SERVO_MAX)
        
        # Convert back to angle
        actual_angle = (servo_clipped - self.STEERING_TO_SERVO_OFFSET) / self.STEERING_TO_SERVO_GAIN
        
        return actual_angle
    
    def _compute_slip_factor(self, angular_velocity: float) -> float:
        """
        Compute tire slip factor based on angular velocity.
        
        High angular velocities cause tire slip (drifting).
        Returns a value between 0 (no slip) and 1 (full slip).
        """
        if not self.enable_slip:
            return 0.0
        
        # Slip increases with angular velocity magnitude
        slip_ratio = min(abs(angular_velocity) / self.SLIP_THRESHOLD, 1.0)
        
        # Friction coefficient decreases with slip
        mu = self.MU_STATIC - (self.MU_STATIC - self.MU_KINETIC) * slip_ratio
        
        return 1.0 - mu / self.MU_STATIC
    
    def step(self) -> VehicleState:
        """
        Simulate one timestep of vehicle dynamics.
        
        Uses Ackerman steering kinematics with slip dynamics.
        
        Returns:
            Updated vehicle state
        """
        # Apply constraints to control inputs
        actual_velocity = self._apply_motor_constraints(self.commanded_velocity)
        actual_steering_angle = self._apply_servo_constraints(self.commanded_steering_angle)
        
        # Ackerman steering kinematics
        # Angular velocity from geometry
        if abs(actual_velocity) > 1e-6:
            ideal_angular_velocity = (actual_velocity / self.WHEELBASE) * np.tan(actual_steering_angle)
        else:
            ideal_angular_velocity = 0.0
        
        # Apply slip dynamics
        slip_factor = self._compute_slip_factor(ideal_angular_velocity)
        
        # During slip, actual angular velocity is higher (oversteer)
        # This creates the drift behavior
        actual_angular_velocity = ideal_angular_velocity * (1.0 + slip_factor * 0.3)
        
        # Update state using kinematic bicycle model
        self.state.velocity = actual_velocity
        self.state.angular_velocity = actual_angular_velocity
        self.state.steering_angle = actual_steering_angle
        
        # Integrate position and heading
        self.state.x += actual_velocity * np.cos(self.state.theta) * self.dt
        self.state.y += actual_velocity * np.sin(self.state.theta) * self.dt
        self.state.theta += actual_angular_velocity * self.dt
        
        # Normalize heading to [-pi, pi]
        self.state.theta = np.arctan2(np.sin(self.state.theta), np.cos(self.state.theta))
        
        return self.state
    
    def get_state(self) -> VehicleState:
        """Get current vehicle state."""
        return self.state
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position (x, y)."""
        return (self.state.x, self.state.y)
    
    def get_velocity(self) -> float:
        """Get current linear velocity."""
        return self.state.velocity
    
    def get_angular_velocity(self) -> float:
        """Get current angular velocity (as measured by IMU)."""
        return self.state.angular_velocity
    
    def compute_curvature(self) -> float:
        """Compute current path curvature."""
        if abs(self.state.velocity) > 1e-6:
            return self.state.angular_velocity / self.state.velocity
        return 0.0

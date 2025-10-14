"""
Simulation environment for testing IKD models.
Replicates testing scenarios from the paper.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

from src.simulator.vehicle import F110Vehicle, VehicleState
from src.simulator.sensors import IMUSensor, VelocitySensor, OdometrySensor


@dataclass
class Obstacle:
    """Simple obstacle representation."""
    x: float
    y: float
    radius: float = 0.15  # meters


class SimulationEnvironment:
    """
    Main simulation environment for IKD testing.
    
    Supports multiple testing scenarios from the paper:
    - Circle navigation (Section IV-C)
    - Loose drifting (Section IV-D1)
    - Tight drifting (Section IV-D2)
    """
    
    def __init__(
        self,
        dt: float = 0.05,
        enable_slip: bool = True,
        add_sensor_noise: bool = True
    ):
        """
        Initialize simulation environment.
        
        Args:
            dt: Simulation timestep (seconds)
            enable_slip: Enable tire slip dynamics
            add_sensor_noise: Add realistic sensor noise
        """
        self.dt = dt
        self.enable_slip = enable_slip
        self.add_sensor_noise = add_sensor_noise
        
        # Create vehicle and sensors
        self.vehicle = F110Vehicle(dt=dt, enable_slip=enable_slip)
        self.imu = IMUSensor()
        self.velocity_sensor = VelocitySensor()
        self.odometry = OdometrySensor()
        
        # Environment state
        self.time = 0.0
        self.obstacles: List[Obstacle] = []
        
        # Data recording
        self.recording = False
        self.recorded_data = {
            'time': [],
            'commanded_velocity': [],
            'commanded_angular_velocity': [],
            'measured_velocity': [],
            'measured_angular_velocity': [],
            'true_velocity': [],
            'true_angular_velocity': [],
            'x': [],
            'y': [],
            'theta': [],
            'curvature': []
        }
    
    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """Reset environment to initial state."""
        self.vehicle.reset(x, y, theta)
        self.imu.reset()
        self.time = 0.0
        self.recorded_data = {key: [] for key in self.recorded_data.keys()}
    
    def add_obstacle(self, x: float, y: float, radius: float = 0.15):
        """Add an obstacle to the environment."""
        self.obstacles.append(Obstacle(x, y, radius))
    
    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles = []
    
    def setup_circle_test(self, velocity: float = 2.0, curvature: float = 0.7):
        """
        Setup circle navigation test from Section IV-C.
        
        Args:
            velocity: Test velocity (m/s)
            curvature: Commanded curvature (1/m)
        """
        self.reset()
        self.clear_obstacles()
        
        # Calculate angular velocity from curvature
        angular_velocity = velocity * curvature
        
        return velocity, angular_velocity
    
    def setup_loose_drift_test(self):
        """
        Setup loose drifting test from Section IV-D1.
        
        Setup: 2 cones, 2 boxes, 2.13 meters apart.
        """
        self.reset()
        self.clear_obstacles()
        
        # Cone positions
        self.add_obstacle(x=3.0, y=0.0, radius=0.1)  # Cone 1
        self.add_obstacle(x=3.0, y=2.13, radius=0.1)  # Cone 2
        
        # Box positions (boundaries)
        self.add_obstacle(x=5.0, y=0.0, radius=0.2)  # Box 1
        self.add_obstacle(x=5.0, y=2.13, radius=0.2)  # Box 2
    
    def setup_tight_drift_test(self):
        """
        Setup tight drifting test from Section IV-D2.
        
        Setup: 2 cones, 2 boxes, 0.81 meters apart (32 inches).
        Vehicle width: 0.48 meters (19 inches), so clearance is only 0.33m.
        """
        self.reset()
        self.clear_obstacles()
        
        # Cone positions (tight spacing)
        self.add_obstacle(x=3.0, y=0.0, radius=0.1)  # Cone
        self.add_obstacle(x=3.0, y=0.81, radius=0.1)  # Cone
        
        # Box positions (boundaries)
        self.add_obstacle(x=4.5, y=0.0, radius=0.2)  # Box
        self.add_obstacle(x=4.5, y=0.81, radius=0.2)  # Box
    
    def set_control(self, velocity: float, angular_velocity: float):
        """
        Set control inputs to vehicle.
        
        Args:
            velocity: Commanded linear velocity (m/s)
            angular_velocity: Commanded angular velocity (rad/s)
        """
        # Convert angular velocity to steering angle
        if abs(velocity) > 1e-6:
            curvature = angular_velocity / velocity
            steering_angle = np.arctan(curvature * self.vehicle.WHEELBASE)
        else:
            steering_angle = 0.0
        
        self.vehicle.set_control(velocity, steering_angle)
    
    def step(self) -> Dict:
        """
        Step simulation forward by one timestep.
        
        Returns:
            Dictionary of sensor measurements
        """
        # Step vehicle dynamics
        state = self.vehicle.step()
        
        # Get sensor measurements
        if self.add_sensor_noise:
            measured_velocity = self.velocity_sensor.measure(state.velocity)
            measured_angular_velocity = self.imu.measure(state.angular_velocity)
            measured_x, measured_y = self.odometry.measure_position(state.x, state.y)
            measured_theta = self.odometry.measure_heading(state.theta)
        else:
            measured_velocity = state.velocity
            measured_angular_velocity = state.angular_velocity
            measured_x, measured_y = state.x, state.y
            measured_theta = state.theta
        
        # Record data if recording
        if self.recording:
            self.recorded_data['time'].append(self.time)
            self.recorded_data['commanded_velocity'].append(self.vehicle.commanded_velocity)
            # Angular velocity from commanded steering
            if abs(self.vehicle.commanded_velocity) > 1e-6:
                commanded_av = (self.vehicle.commanded_velocity / self.vehicle.WHEELBASE) * \
                              np.tan(self.vehicle.commanded_steering_angle)
            else:
                commanded_av = 0.0
            self.recorded_data['commanded_angular_velocity'].append(commanded_av)
            self.recorded_data['measured_velocity'].append(measured_velocity)
            self.recorded_data['measured_angular_velocity'].append(measured_angular_velocity)
            self.recorded_data['true_velocity'].append(state.velocity)
            self.recorded_data['true_angular_velocity'].append(state.angular_velocity)
            self.recorded_data['x'].append(state.x)
            self.recorded_data['y'].append(state.y)
            self.recorded_data['theta'].append(state.theta)
            self.recorded_data['curvature'].append(self.vehicle.compute_curvature())
        
        self.time += self.dt
        
        return {
            'time': self.time,
            'velocity': measured_velocity,
            'angular_velocity': measured_angular_velocity,
            'x': measured_x,
            'y': measured_y,
            'theta': measured_theta,
            'true_angular_velocity': state.angular_velocity,  # For debugging
            'curvature': self.vehicle.compute_curvature()
        }
    
    def start_recording(self):
        """Start recording simulation data."""
        self.recording = True
        self.recorded_data = {key: [] for key in self.recorded_data.keys()}
    
    def stop_recording(self):
        """Stop recording simulation data."""
        self.recording = False
    
    def get_recorded_data(self) -> Dict:
        """Get recorded simulation data."""
        return self.recorded_data
    
    def save_recorded_data(self, filepath: str):
        """Save recorded data to file."""
        import pandas as pd
        
        # Convert to DataFrame format similar to real data
        df = pd.DataFrame({
            'joystick': [f"[{v}, {av}]" for v, av in zip(
                self.recorded_data['commanded_velocity'],
                self.recorded_data['commanded_angular_velocity']
            )],
            'executed': [f"[{av}]" for av in self.recorded_data['measured_angular_velocity']]
        })
        
        df.to_csv(filepath, index=False)
        print(f"[INFO] Saved simulation data to {filepath}")
    
    def check_collision(self) -> bool:
        """Check if vehicle collides with any obstacle."""
        state = self.vehicle.get_state()
        vehicle_radius = self.vehicle.VEHICLE_WIDTH / 2
        
        for obstacle in self.obstacles:
            distance = np.sqrt((state.x - obstacle.x)**2 + (state.y - obstacle.y)**2)
            if distance < (vehicle_radius + obstacle.radius):
                return True
        
        return False
    
    def get_trajectory(self) -> np.ndarray:
        """Get vehicle trajectory as array of (x, y) points."""
        if not self.recorded_data['x']:
            return np.array([])
        
        return np.array(list(zip(self.recorded_data['x'], self.recorded_data['y'])))
    
    def measure_circle_radius(self) -> float:
        """
        Measure radius of circular trajectory (for circle tests).
        
        Returns:
            Estimated radius in meters
        """
        trajectory = self.get_trajectory()
        if len(trajectory) < 10:
            return 0.0
        
        # Fit circle to trajectory points
        # Simple method: use mean distance from center
        center_x = np.mean(trajectory[:, 0])
        center_y = np.mean(trajectory[:, 1])
        
        distances = np.sqrt((trajectory[:, 0] - center_x)**2 + (trajectory[:, 1] - center_y)**2)
        radius = np.mean(distances)
        
        return radius

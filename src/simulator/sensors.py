"""
Sensor simulation for F1/10 vehicle.
Replicates IMU, velocity sensors, and odometry.
"""

import numpy as np
from typing import Tuple, Optional


class IMUSensor:
    """
    Vectornav VN-100 IMU simulator.
    
    Provides angular velocity measurements with realistic noise and delay.
    Based on specifications from the paper.
    """
    
    def __init__(
        self,
        noise_std: float = 0.01,
        bias: float = 0.0,
        delay: float = 0.18,  # Average IMU delay from paper (0.18-0.20s)
        sample_rate: float = 40.0  # Hz
    ):
        """
        Initialize IMU sensor.
        
        Args:
            noise_std: Standard deviation of measurement noise (rad/s)
            bias: Constant bias in measurements (rad/s)
            delay: Time delay in measurements (seconds)
            sample_rate: Sampling rate (Hz)
        """
        self.noise_std = noise_std
        self.bias = bias
        self.delay = delay
        self.sample_rate = sample_rate
        
        # Circular buffer for delay simulation
        self.buffer_size = int(delay * sample_rate) + 1
        self.measurement_buffer = [0.0] * self.buffer_size
        self.buffer_index = 0
    
    def measure(self, true_angular_velocity: float) -> float:
        """
        Measure angular velocity with noise and delay.
        
        Args:
            true_angular_velocity: True angular velocity (rad/s)
            
        Returns:
            Measured angular velocity with noise and delay
        """
        # Add noise and bias
        noisy_measurement = true_angular_velocity + np.random.normal(0, self.noise_std) + self.bias
        
        # Add to buffer
        self.measurement_buffer[self.buffer_index] = noisy_measurement
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        
        # Return delayed measurement
        delayed_index = (self.buffer_index - self.buffer_size + 1) % self.buffer_size
        return self.measurement_buffer[delayed_index]
    
    def reset(self):
        """Reset sensor buffer."""
        self.measurement_buffer = [0.0] * self.buffer_size
        self.buffer_index = 0


class VelocitySensor:
    """
    Velocity sensor (from motor encoder).
    
    Provides linear velocity measurements with minimal noise.
    """
    
    def __init__(self, noise_std: float = 0.001):
        """
        Initialize velocity sensor.
        
        Args:
            noise_std: Standard deviation of measurement noise (m/s)
        """
        self.noise_std = noise_std
    
    def measure(self, true_velocity: float) -> float:
        """
        Measure linear velocity with minimal noise.
        
        Args:
            true_velocity: True linear velocity (m/s)
            
        Returns:
            Measured velocity
        """
        return true_velocity + np.random.normal(0, self.noise_std)


class OdometrySensor:
    """
    Odometry sensor for position and heading estimation.
    
    Note: In the paper, position is not used for IKD training,
    but is useful for visualization and trajectory analysis.
    """
    
    def __init__(
        self,
        position_noise_std: float = 0.01,
        heading_noise_std: float = 0.01
    ):
        """
        Initialize odometry sensor.
        
        Args:
            position_noise_std: Standard deviation of position noise (meters)
            heading_noise_std: Standard deviation of heading noise (radians)
        """
        self.position_noise_std = position_noise_std
        self.heading_noise_std = heading_noise_std
    
    def measure_position(self, true_x: float, true_y: float) -> Tuple[float, float]:
        """
        Measure position with noise.
        
        Args:
            true_x: True x position (meters)
            true_y: True y position (meters)
            
        Returns:
            Measured (x, y) position
        """
        x = true_x + np.random.normal(0, self.position_noise_std)
        y = true_y + np.random.normal(0, self.position_noise_std)
        return (x, y)
    
    def measure_heading(self, true_heading: float) -> float:
        """
        Measure heading with noise.
        
        Args:
            true_heading: True heading angle (radians)
            
        Returns:
            Measured heading
        """
        return true_heading + np.random.normal(0, self.heading_noise_std)

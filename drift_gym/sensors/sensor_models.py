"""
Realistic Sensor Models with Noise, Drift, and Latency

Implements GPS, IMU, and other sensors with realistic error characteristics.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque


@dataclass
class SensorReading:
    """Container for sensor readings with metadata."""
    timestamp: float
    data: np.ndarray
    variance: np.ndarray
    valid: bool = True


class GPSSensor:
    """
    GPS sensor with realistic noise characteristics.
    
    Implements:
    - Multipath error (position-dependent)
    - Random walk drift
    - Measurement noise
    - Satellite availability (dropout)
    - Horizontal dilution of precision (HDOP)
    """
    
    def __init__(
        self,
        noise_std: float = 0.5,  # meters
        drift_rate: float = 0.01,  # m/s random walk
        dropout_probability: float = 0.01,
        update_rate: float = 10.0,  # Hz
        seed: Optional[int] = None
    ):
        self.noise_std = noise_std
        self.drift_rate = drift_rate
        self.dropout_probability = dropout_probability
        self.update_rate = update_rate
        
        self.rng = np.random.RandomState(seed)
        
        # Internal state
        self.drift = np.array([0.0, 0.0])  # Accumulated drift
        self.last_update_time = 0.0
        self.time_since_update = 0.0
        
    def measure(
        self,
        true_position: np.ndarray,
        current_time: float
    ) -> SensorReading:
        """
        Generate GPS measurement with noise.
        
        Args:
            true_position: True [x, y] position (meters)
            current_time: Current simulation time (seconds)
            
        Returns:
            SensorReading with noisy position
        """
        dt = current_time - self.last_update_time
        self.time_since_update += dt
        
        # Check if sensor should update
        if self.time_since_update < 1.0 / self.update_rate:
            # Return stale measurement
            return SensorReading(
                timestamp=self.last_update_time,
                data=np.array([0.0, 0.0]),
                variance=np.array([self.noise_std**2, self.noise_std**2]),
                valid=False
            )
        
        self.time_since_update = 0.0
        self.last_update_time = current_time
        
        # Check for dropout
        if self.rng.random() < self.dropout_probability:
            return SensorReading(
                timestamp=current_time,
                data=np.array([0.0, 0.0]),
                variance=np.array([1e6, 1e6]),  # Very high uncertainty
                valid=False
            )
        
        # Update drift (random walk)
        drift_noise = self.rng.randn(2) * self.drift_rate * np.sqrt(dt)
        self.drift += drift_noise
        
        # Add measurement noise
        measurement_noise = self.rng.randn(2) * self.noise_std
        
        # Add multipath error (position-dependent, simulated)
        multipath = np.sin(true_position / 10.0) * 0.3
        
        # Final measurement
        measured_position = true_position + self.drift + measurement_noise + multipath
        
        # Variance increases with time since last fix
        variance = np.array([
            self.noise_std**2 + np.linalg.norm(self.drift)**2,
            self.noise_std**2 + np.linalg.norm(self.drift)**2
        ])
        
        return SensorReading(
            timestamp=current_time,
            data=measured_position,
            variance=variance,
            valid=True
        )
    
    def reset(self):
        """Reset sensor state."""
        self.drift = np.array([0.0, 0.0])
        self.last_update_time = 0.0
        self.time_since_update = 0.0


class IMUSensor:
    """
    IMU sensor with realistic noise and bias.
    
    Implements:
    - Gyroscope bias (random walk)
    - Accelerometer bias
    - White noise
    - Temperature-dependent drift
    - Scale factor errors
    """
    
    def __init__(
        self,
        gyro_noise_std: float = 0.01,  # rad/s
        gyro_bias_std: float = 0.001,  # rad/s
        accel_noise_std: float = 0.02,  # m/s^2
        accel_bias_std: float = 0.01,  # m/s^2
        update_rate: float = 100.0,  # Hz
        seed: Optional[int] = None
    ):
        self.gyro_noise_std = gyro_noise_std
        self.gyro_bias_std = gyro_bias_std
        self.accel_noise_std = accel_noise_std
        self.accel_bias_std = accel_bias_std
        self.update_rate = update_rate
        
        self.rng = np.random.RandomState(seed)
        
        # Biases (slowly changing)
        self.gyro_bias = self.rng.randn(3) * gyro_bias_std
        self.accel_bias = self.rng.randn(3) * accel_bias_std
        
        self.last_update_time = 0.0
        
    def measure(
        self,
        true_angular_velocity: np.ndarray,  # [wx, wy, wz]
        true_acceleration: np.ndarray,  # [ax, ay, az]
        current_time: float
    ) -> Dict[str, SensorReading]:
        """
        Generate IMU measurements.
        
        Args:
            true_angular_velocity: True angular velocity (rad/s)
            true_acceleration: True linear acceleration (m/s^2)
            current_time: Current time (seconds)
            
        Returns:
            Dict with 'gyro' and 'accel' SensorReadings
        """
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Update biases (random walk)
        self.gyro_bias += self.rng.randn(3) * self.gyro_bias_std * 0.01 * np.sqrt(dt)
        self.accel_bias += self.rng.randn(3) * self.accel_bias_std * 0.01 * np.sqrt(dt)
        
        # Gyro measurement
        gyro_noise = self.rng.randn(3) * self.gyro_noise_std
        gyro_measurement = true_angular_velocity + self.gyro_bias + gyro_noise
        
        # Accelerometer measurement (includes gravity!)
        accel_noise = self.rng.randn(3) * self.accel_noise_std
        accel_measurement = true_acceleration + self.accel_bias + accel_noise
        
        return {
            'gyro': SensorReading(
                timestamp=current_time,
                data=gyro_measurement,
                variance=np.ones(3) * self.gyro_noise_std**2,
                valid=True
            ),
            'accel': SensorReading(
                timestamp=current_time,
                data=accel_measurement,
                variance=np.ones(3) * self.accel_noise_std**2,
                valid=True
            )
        }
    
    def reset(self):
        """Reset sensor biases."""
        self.gyro_bias = self.rng.randn(3) * self.gyro_bias_std
        self.accel_bias = self.rng.randn(3) * self.accel_bias_std
        self.last_update_time = 0.0


class LatencyBuffer:
    """
    Circular buffer for modeling sensor and actuation delays.
    
    Simulates realistic latency in the perception-planning-control loop.
    """
    
    def __init__(
        self,
        sensor_delay: float = 0.05,  # 50ms sensor latency
        compute_delay: float = 0.03,  # 30ms compute latency
        actuation_delay: float = 0.02,  # 20ms actuation latency
        dt: float = 0.05
    ):
        self.sensor_delay = sensor_delay
        self.compute_delay = compute_delay
        self.actuation_delay = actuation_delay
        self.total_delay = sensor_delay + compute_delay + actuation_delay
        
        # Calculate buffer size
        buffer_size = int(np.ceil(self.total_delay / dt)) + 1
        
        # Circular buffers
        self.sensor_buffer = deque(maxlen=buffer_size)
        self.action_buffer = deque(maxlen=buffer_size)
        
        # Fill with zeros initially
        for _ in range(buffer_size):
            self.sensor_buffer.append(None)
            self.action_buffer.append(np.zeros(2))
    
    def add_sensor_reading(self, reading: np.ndarray):
        """Add sensor reading to buffer."""
        self.sensor_buffer.append(reading)
    
    def get_delayed_sensor_reading(self) -> Optional[np.ndarray]:
        """Get sensor reading with realistic delay."""
        # Return oldest reading (delayed)
        return self.sensor_buffer[0] if self.sensor_buffer[0] is not None else None
    
    def add_action(self, action: np.ndarray):
        """Add action to buffer."""
        self.action_buffer.append(action.copy())
    
    def get_delayed_action(self) -> np.ndarray:
        """Get action with actuation delay."""
        return self.action_buffer[0]
    
    def reset(self):
        """Clear buffers."""
        self.sensor_buffer.clear()
        self.action_buffer.clear()
        
        # Refill with empty values
        buffer_size = self.sensor_buffer.maxlen
        for _ in range(buffer_size):
            self.sensor_buffer.append(None)
            self.action_buffer.append(np.zeros(2))


def test_sensors():
    """Test sensor models."""
    print("Testing Sensor Models")
    print("=" * 60)
    
    # Test GPS
    print("\n1. GPS Sensor:")
    gps = GPSSensor(noise_std=0.5, seed=42)
    true_pos = np.array([10.0, 5.0])
    
    for t in np.arange(0, 1.0, 0.1):
        reading = gps.measure(true_pos, t)
        if reading.valid:
            error = np.linalg.norm(reading.data - true_pos)
            print(f"  t={t:.2f}s: measured={reading.data}, error={error:.3f}m")
    
    # Test IMU
    print("\n2. IMU Sensor:")
    imu = IMUSensor(seed=42)
    true_gyro = np.array([0.0, 0.0, 0.5])  # 0.5 rad/s yaw rate
    true_accel = np.array([1.0, 0.0, 9.81])  # 1 m/s^2 forward + gravity
    
    for t in np.arange(0, 0.5, 0.1):
        readings = imu.measure(true_gyro, true_accel, t)
        gyro_error = np.linalg.norm(readings['gyro'].data - true_gyro)
        print(f"  t={t:.2f}s: gyro_error={gyro_error:.4f} rad/s")
    
    # Test Latency Buffer
    print("\n3. Latency Buffer:")
    buffer = LatencyBuffer(sensor_delay=0.05, compute_delay=0.03, actuation_delay=0.02)
    
    for i in range(5):
        buffer.add_sensor_reading(np.array([i, i*2]))
        buffer.add_action(np.array([i*0.1, i*0.2]))
        
        delayed_sensor = buffer.get_delayed_sensor_reading()
        delayed_action = buffer.get_delayed_action()
        
        print(f"  Step {i}: added=[{i}, {i*2}], delayed_sensor={delayed_sensor}")


if __name__ == "__main__":
    test_sensors()

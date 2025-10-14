"""
Unit tests for sensor models.

Tests verify:
- Noise characteristics match specifications
- Drift behavior is correct
- Variance estimates are reasonable
- Edge cases handled properly
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_gym.sensors import GPSSensor, IMUSensor


class TestGPSSensor:
    """Test GPS sensor model."""
    
    def test_initialization(self):
        """Test sensor initializes correctly."""
        sensor = GPSSensor(noise_std=0.3, seed=42)
        assert sensor.noise_std == 0.3
        assert sensor.drift[0] == 0.0
        assert sensor.drift[1] == 0.0
    
    def test_measurement_noise(self):
        """Test measurement noise statistics."""
        sensor = GPSSensor(noise_std=0.3, seed=42)
        true_position = np.array([10.0, 5.0])
        
        # Collect many measurements
        errors = []
        for t in np.arange(0, 10.0, 0.1):
            reading = sensor.measure(true_position, t)
            if reading.valid:
                error = reading.data - true_position
                errors.append(np.linalg.norm(error))
        
        # Check statistics
        errors = np.array(errors)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Mean should be small (unbiased)
        assert mean_error < 0.5, f"Mean error too large: {mean_error}"
        
        # Std should be close to specified noise_std
        assert 0.2 < std_error < 0.6, f"Noise std mismatch: {std_error}"
    
    def test_dropout(self):
        """Test dropout behavior."""
        sensor = GPSSensor(dropout_probability=0.5, seed=42)
        true_position = np.array([10.0, 5.0])
        
        invalid_count = 0
        total_count = 0
        
        for t in np.arange(0, 10.0, 0.1):
            reading = sensor.measure(true_position, t)
            if not reading.valid:
                invalid_count += 1
            total_count += 1
        
        dropout_rate = invalid_count / total_count
        
        # Should have some dropouts
        assert dropout_rate > 0.3, f"Dropout rate too low: {dropout_rate}"
    
    def test_update_rate(self):
        """Test sensor update rate."""
        sensor = GPSSensor(update_rate=10.0, seed=42)
        true_position = np.array([10.0, 5.0])
        
        # Measurements should only update at specified rate
        reading1 = sensor.measure(true_position, 0.0)
        reading2 = sensor.measure(true_position, 0.05)  # Too soon
        reading3 = sensor.measure(true_position, 0.1)   # Should update
        
        assert reading1.valid
        assert not reading2.valid  # Too soon
        assert reading3.valid
    
    def test_reset(self):
        """Test sensor reset."""
        sensor = GPSSensor(seed=42)
        true_position = np.array([10.0, 5.0])
        
        # Accumulate some drift
        for t in np.arange(0, 5.0, 0.1):
            sensor.measure(true_position, t)
        
        drift_before = np.linalg.norm(sensor.drift)
        
        # Reset
        sensor.reset()
        drift_after = np.linalg.norm(sensor.drift)
        
        assert drift_before > 0.0
        assert drift_after == 0.0


class TestIMUSensor:
    """Test IMU sensor model."""
    
    def test_initialization(self):
        """Test sensor initializes correctly."""
        sensor = IMUSensor(gyro_noise_std=0.0087, seed=42)
        assert sensor.gyro_noise_std == 0.0087
        assert len(sensor.gyro_bias) == 3
    
    def test_gyro_measurement(self):
        """Test gyroscope measurement."""
        sensor = IMUSensor(gyro_noise_std=0.01, seed=42)
        true_gyro = np.array([0.0, 0.0, 0.5])  # 0.5 rad/s yaw rate
        true_accel = np.array([0.0, 0.0, 9.81])
        
        readings = sensor.measure(true_gyro, true_accel, 0.0)
        gyro_reading = readings['gyro']
        
        assert gyro_reading.valid
        assert gyro_reading.data.shape == (3,)
        
        # Measurement should be close to true value
        error = np.abs(gyro_reading.data[2] - true_gyro[2])
        assert error < 0.1, f"Gyro error too large: {error}"
    
    def test_accel_measurement(self):
        """Test accelerometer measurement."""
        sensor = IMUSensor(accel_noise_std=0.015, seed=42)
        true_gyro = np.array([0.0, 0.0, 0.0])
        true_accel = np.array([1.0, 0.0, 9.81])
        
        readings = sensor.measure(true_gyro, true_accel, 0.0)
        accel_reading = readings['accel']
        
        assert accel_reading.valid
        assert accel_reading.data.shape == (3,)
    
    def test_bias_evolution(self):
        """Test that bias evolves over time."""
        sensor = IMUSensor(seed=42)
        true_gyro = np.array([0.0, 0.0, 0.0])
        true_accel = np.array([0.0, 0.0, 9.81])
        
        bias_initial = sensor.gyro_bias.copy()
        
        # Run for some time
        for t in np.arange(0, 10.0, 0.1):
            sensor.measure(true_gyro, true_accel, t)
        
        bias_final = sensor.gyro_bias.copy()
        
        # Bias should have changed
        bias_change = np.linalg.norm(bias_final - bias_initial)
        assert bias_change > 0.0, "Bias should evolve over time"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

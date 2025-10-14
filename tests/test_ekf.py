"""
Unit tests for Extended Kalman Filter.

Tests verify:
- State estimation accuracy
- Covariance consistency
- Update steps work correctly
- Numerical stability
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_gym.estimation import ExtendedKalmanFilter, EKFState


class TestEKF:
    """Test Extended Kalman Filter."""
    
    def test_initialization(self):
        """Test EKF initializes correctly."""
        ekf = ExtendedKalmanFilter(dt=0.05)
        
        assert ekf.x.shape == (6,)
        assert ekf.P.shape == (6, 6)
        assert ekf.dt == 0.05
    
    def test_prediction_step(self):
        """Test prediction step updates state."""
        ekf = ExtendedKalmanFilter(dt=0.05)
        ekf.x = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.5])  # Moving forward, turning
        
        initial_x = ekf.x[0]
        initial_y = ekf.x[1]
        
        ekf.predict()
        
        # Position should have updated
        assert ekf.x[0] != initial_x
        assert ekf.x[1] != initial_y
        
        # Covariance should increase (uncertainty grows)
        assert ekf.P[0, 0] > 1.0
    
    def test_gps_update(self):
        """Test GPS measurement update."""
        ekf = ExtendedKalmanFilter(dt=0.05)
        ekf.x = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        
        # Perfect GPS measurement
        gps_meas = np.array([1.5, 1.2])
        gps_var = np.array([0.1, 0.1])
        
        ekf.update_gps(gps_meas, gps_var)
        
        # State should move toward measurement
        assert ekf.x[0] > 1.0 and ekf.x[0] < 1.5
        assert ekf.x[1] > 1.0 and ekf.x[1] < 1.2
    
    def test_imu_update(self):
        """Test IMU measurement update."""
        ekf = ExtendedKalmanFilter(dt=0.05)
        ekf.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.3])  # omega = 0.3
        
        # IMU measures omega = 0.5
        imu_omega = 0.5
        imu_accel = np.array([0.0, 0.0])
        imu_var = np.array([0.01, 0.01, 0.01])
        
        ekf.update_imu(imu_omega, imu_accel, imu_var)
        
        # Omega estimate should move toward measurement
        assert ekf.x[5] > 0.3 and ekf.x[5] < 0.5
    
    def test_get_state(self):
        """Test get_state returns EKFState."""
        ekf = ExtendedKalmanFilter(dt=0.05)
        ekf.x = np.array([1.0, 2.0, 0.5, 1.5, 0.1, 0.3])
        
        state = ekf.get_state()
        
        assert isinstance(state, EKFState)
        assert state.x == 1.0
        assert state.y == 2.0
        assert state.theta == 0.5
        assert state.vx == 1.5
        assert state.vy == 0.1
        assert state.omega == 0.3
    
    def test_covariance_positive_definite(self):
        """Test covariance stays positive definite."""
        ekf = ExtendedKalmanFilter(dt=0.05)
        
        # Run many iterations
        for _ in range(100):
            ekf.predict()
            
            # Add some measurements
            gps_meas = np.array([0.0, 0.0]) + np.random.randn(2) * 0.1
            ekf.update_gps(gps_meas, np.array([0.1, 0.1]))
        
        # Covariance should still be positive definite
        eigenvalues = np.linalg.eigvals(ekf.P)
        assert np.all(eigenvalues > 0), "Covariance not positive definite"
    
    def test_estimation_accuracy(self):
        """Test EKF accurately estimates ground truth."""
        ekf = ExtendedKalmanFilter(dt=0.05)
        
        # Simulate vehicle motion
        true_state = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.5])
        ekf.x = true_state.copy()
        
        errors = []
        
        for t in np.arange(0, 2.0, 0.05):
            # Predict
            ekf.predict()
            
            # Update true state
            dt = 0.05
            true_state[0] += true_state[3] * np.cos(true_state[2]) * dt
            true_state[1] += true_state[3] * np.sin(true_state[2]) * dt
            true_state[2] += true_state[5] * dt
            
            # Noisy GPS measurement
            gps_noise = np.random.randn(2) * 0.3
            gps_meas = true_state[:2] + gps_noise
            ekf.update_gps(gps_meas, np.array([0.3**2, 0.3**2]))
            
            # Noisy IMU measurement
            imu_noise = np.random.randn() * 0.01
            imu_meas = true_state[5] + imu_noise
            ekf.update_imu(imu_meas, np.zeros(2), np.array([0.01**2, 0.01**2, 0.01**2]))
            
            # Compute error
            pos_error = np.linalg.norm(ekf.x[:2] - true_state[:2])
            errors.append(pos_error)
        
        # Average error should be reasonable
        avg_error = np.mean(errors)
        assert avg_error < 0.5, f"Average estimation error too large: {avg_error}"
    
    def test_reset(self):
        """Test reset restores initial conditions."""
        ekf = ExtendedKalmanFilter(dt=0.05)
        
        # Run some iterations
        for _ in range(10):
            ekf.predict()
        
        # State should have changed
        assert np.any(ekf.x != 0)
        
        # Reset
        ekf.reset()
        
        # Should be back to zero
        assert np.allclose(ekf.x, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

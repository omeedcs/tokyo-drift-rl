"""
Extended Kalman Filter for Vehicle State Estimation

Fuses GPS and IMU measurements to estimate vehicle state with uncertainty.

State vector: [x, y, theta, vx, vy, omega]
- x, y: Position (m)
- theta: Heading (rad)
- vx, vy: Velocity in body frame (m/s)
- omega: Yaw rate (rad/s)

Measurements:
- GPS: [x, y] with covariance
- IMU: [omega, ax, ay] with covariance
- Odometry: [vx, omega] (from control inputs)

References:
- Thrun, S., Burgard, W., & Fox, D. (2005). "Probabilistic Robotics"
- Farrell, J. A. (2008). "Aided Navigation: GPS with High Rate Sensors"
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class EKFState:
    """EKF state estimate with uncertainty."""
    x: float  # Position x (m)
    y: float  # Position y (m)
    theta: float  # Heading (rad)
    vx: float  # Forward velocity (m/s)
    vy: float  # Lateral velocity (m/s)
    omega: float  # Yaw rate (rad/s)
    
    # Covariance diagonal (full covariance stored separately)
    position_var: float  # Position uncertainty (m^2)
    heading_var: float  # Heading uncertainty (rad^2)
    velocity_var: float  # Velocity uncertainty (m^2/s^2)
    omega_var: float  # Yaw rate uncertainty (rad^2/s^2)
    
    @property
    def state_vector(self) -> np.ndarray:
        """Return state as numpy array."""
        return np.array([self.x, self.y, self.theta, self.vx, self.vy, self.omega])
    
    @property
    def uncertainty_norms(self) -> Tuple[float, float, float]:
        """Return uncertainty norms for observation space."""
        pos_std = np.sqrt(self.position_var)
        vel_std = np.sqrt(self.velocity_var)
        omega_std = np.sqrt(self.omega_var)
        return pos_std, vel_std, omega_std


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for 2D vehicle state estimation.
    
    Implements prediction-update cycle with nonlinear motion model
    and linear measurement models.
    """
    
    def __init__(
        self,
        dt: float = 0.05,
        process_noise_std: float = 0.1,
        initial_state: Optional[np.ndarray] = None,
        initial_covariance: Optional[np.ndarray] = None
    ):
        """
        Initialize EKF.
        
        Args:
            dt: Time step (seconds)
            process_noise_std: Process noise standard deviation
            initial_state: Initial state [x, y, theta, vx, vy, omega]
            initial_covariance: Initial covariance matrix (6x6)
        """
        self.dt = dt
        
        # State vector: [x, y, theta, vx, vy, omega]
        if initial_state is not None:
            self.x = initial_state.copy()
        else:
            self.x = np.zeros(6)
        
        # Covariance matrix (6x6)
        if initial_covariance is not None:
            self.P = initial_covariance.copy()
        else:
            # Initial uncertainty
            self.P = np.diag([1.0, 1.0, 0.1, 0.5, 0.5, 0.1])
        
        # Process noise covariance (models uncertainty in dynamics)
        self.Q = np.diag([
            process_noise_std**2,  # x
            process_noise_std**2,  # y
            (0.05)**2,  # theta
            (0.2)**2,  # vx
            (0.2)**2,  # vy
            (0.05)**2   # omega
        ])
        
    def predict(self, control_input: Optional[np.ndarray] = None):
        """
        Prediction step using motion model.
        
        Args:
            control_input: Optional [v_cmd, omega_cmd] control inputs
        """
        # Extract state
        x, y, theta, vx, vy, omega = self.x
        
        # Motion model: constant velocity with nonholonomic constraints
        # Position update: integrate velocity in global frame
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Transform body velocities to global frame
        vx_global = vx * cos_theta - vy * sin_theta
        vy_global = vx * sin_theta + vy * cos_theta
        
        # Predicted state
        x_pred = x + vx_global * self.dt
        y_pred = y + vy_global * self.dt
        theta_pred = theta + omega * self.dt
        
        # Normalize angle to [-pi, pi]
        theta_pred = np.arctan2(np.sin(theta_pred), np.cos(theta_pred))
        
        # Velocity assumed constant (will be updated by measurements)
        vx_pred = vx
        vy_pred = vy
        omega_pred = omega
        
        # Update state
        self.x = np.array([x_pred, y_pred, theta_pred, vx_pred, vy_pred, omega_pred])
        
        # Jacobian of motion model
        F = np.eye(6)
        F[0, 2] = -vx * sin_theta - vy * cos_theta  # dx/dtheta
        F[0, 3] = cos_theta * self.dt  # dx/dvx
        F[0, 4] = -sin_theta * self.dt  # dx/dvy
        F[1, 2] = vx * cos_theta - vy * sin_theta  # dy/dtheta
        F[1, 3] = sin_theta * self.dt  # dy/dvx
        F[1, 4] = cos_theta * self.dt  # dy/dvy
        F[2, 5] = self.dt  # dtheta/domega
        
        # Covariance prediction: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q
        
    def update_gps(self, gps_measurement: np.ndarray, gps_variance: np.ndarray):
        """
        Update with GPS measurement.
        
        Args:
            gps_measurement: [x, y] position (m)
            gps_variance: [var_x, var_y] measurement variance
        """
        # Measurement model: H maps state to GPS measurement
        # GPS measures [x, y] directly
        H = np.zeros((2, 6))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        
        # Measurement noise covariance
        R = np.diag(gps_variance)
        
        # Innovation (measurement residual)
        z = gps_measurement
        z_pred = H @ self.x
        y = z - z_pred
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Normalize angle
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))
        
        # Covariance update
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form for numerical stability
        
    def update_imu(self, imu_gyro: float, imu_accel: np.ndarray, imu_variance: np.ndarray):
        """
        Update with IMU measurement.
        
        Args:
            imu_gyro: Yaw rate measurement (rad/s)
            imu_accel: [ax, ay] acceleration in body frame (m/s^2)
            imu_variance: [var_omega, var_ax, var_ay]
        """
        # IMU measures omega directly
        H_gyro = np.zeros((1, 6))
        H_gyro[0, 5] = 1.0  # omega
        
        R_gyro = np.array([[imu_variance[0]]])
        
        # Innovation
        y = imu_gyro - self.x[5]
        
        # Innovation covariance
        S = H_gyro @ self.P @ H_gyro.T + R_gyro
        
        # Kalman gain
        K = self.P @ H_gyro.T / S
        
        # Update
        self.x = self.x + K.flatten() * y
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))
        
        # Covariance update
        self.P = (np.eye(6) - np.outer(K, H_gyro)) @ self.P
        
        # Note: We could also use accelerometer to update velocities,
        # but this requires integrating acceleration which accumulates drift
        # For now, we rely on GPS for velocity corrections
        
    def update_odometry(self, v_measured: float, omega_measured: float, odom_variance: np.ndarray):
        """
        Update with wheel odometry.
        
        Args:
            v_measured: Forward velocity from encoders (m/s)
            omega_measured: Yaw rate from encoders (rad/s)
            odom_variance: [var_v, var_omega]
        """
        # Odometry measures [vx, omega]
        H = np.zeros((2, 6))
        H[0, 3] = 1.0  # vx
        H[1, 5] = 1.0  # omega
        
        R = np.diag(odom_variance)
        
        # Innovation
        z = np.array([v_measured, omega_measured])
        z_pred = H @ self.x
        y = z - z_pred
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update
        self.x = self.x + K @ y
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))
        
        # Covariance update
        self.P = (np.eye(6) - K @ H) @ self.P
        
    def get_state(self) -> EKFState:
        """
        Get current state estimate with uncertainties.
        
        Returns:
            EKFState with mean and variance estimates
        """
        return EKFState(
            x=float(self.x[0]),
            y=float(self.x[1]),
            theta=float(self.x[2]),
            vx=float(self.x[3]),
            vy=float(self.x[4]),
            omega=float(self.x[5]),
            position_var=float((self.P[0, 0] + self.P[1, 1]) / 2),
            heading_var=float(self.P[2, 2]),
            velocity_var=float((self.P[3, 3] + self.P[4, 4]) / 2),
            omega_var=float(self.P[5, 5])
        )
    
    def reset(self, initial_state: Optional[np.ndarray] = None):
        """Reset filter to initial conditions."""
        if initial_state is not None:
            self.x = initial_state.copy()
        else:
            self.x = np.zeros(6)
        
        self.P = np.diag([1.0, 1.0, 0.1, 0.5, 0.5, 0.1])


def test_ekf():
    """Test EKF with simulated data."""
    print("Testing Extended Kalman Filter")
    print("=" * 60)
    
    ekf = ExtendedKalmanFilter(dt=0.05)
    
    # Simulate vehicle motion
    true_state = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.5])  # Moving forward, turning
    
    for t in np.arange(0, 5.0, 0.05):
        # Predict
        ekf.predict()
        
        # Simulated GPS measurement (every 0.1s)
        if t % 0.1 < 0.05:
            gps_noise = np.random.randn(2) * 0.3
            gps_meas = true_state[:2] + gps_noise
            ekf.update_gps(gps_meas, np.array([0.3**2, 0.3**2]))
        
        # Simulated IMU measurement (every step)
        imu_noise = np.random.randn() * 0.01
        imu_meas = true_state[5] + imu_noise
        ekf.update_imu(imu_meas, np.zeros(2), np.array([0.01**2, 0.01**2, 0.01**2]))
        
        # Update true state (simple motion model)
        true_state[0] += true_state[3] * np.cos(true_state[2]) * 0.05
        true_state[1] += true_state[3] * np.sin(true_state[2]) * 0.05
        true_state[2] += true_state[5] * 0.05
        
        if t % 1.0 < 0.05:  # Print every second
            state = ekf.get_state()
            pos_error = np.sqrt((state.x - true_state[0])**2 + (state.y - true_state[1])**2)
            print(f"t={t:.1f}s: pos_error={pos_error:.3f}m, pos_std={np.sqrt(state.position_var):.3f}m")
    
    print("\n✅ EKF test complete")


if __name__ == "__main__":
    test_ekf()

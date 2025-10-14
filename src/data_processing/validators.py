"""
Data validation utilities for ensuring data quality.
Detects anomalies, corrupted files, and data quality issues.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, float]


class DataValidator:
    """Validates training and test data for IKD models."""
    
    def __init__(
        self,
        max_velocity: float = 4.5,
        max_angular_velocity: float = 4.0,
        min_data_points: int = 100,
        expected_imu_delay_range: Tuple[float, float] = (0.15, 0.25)
    ):
        """
        Initialize data validator.
        
        Args:
            max_velocity: Maximum expected velocity (m/s)
            max_angular_velocity: Maximum expected angular velocity (rad/s)
            min_data_points: Minimum required data points
            expected_imu_delay_range: Expected range for IMU delay (seconds)
        """
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity
        self.min_data_points = min_data_points
        self.expected_imu_delay_range = expected_imu_delay_range
    
    def validate_csv(self, csv_path: str) -> ValidationResult:
        """
        Validate a CSV file containing training data.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []
        statistics = {}
        
        try:
            data = pd.read_csv(csv_path)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Failed to read CSV: {str(e)}"],
                warnings=[],
                statistics={}
            )
        
        # Check required columns
        required_columns = ['joystick', 'executed']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return ValidationResult(False, errors, warnings, statistics)
        
        # Parse data
        try:
            joystick = np.array([eval(i) for i in data["joystick"]])
            executed = np.array([eval(i) for i in data["executed"]])
        except Exception as e:
            errors.append(f"Failed to parse data: {str(e)}")
            return ValidationResult(False, errors, warnings, statistics)
        
        # Check data dimensions
        if len(joystick.shape) != 2 or joystick.shape[1] != 2:
            errors.append(f"Invalid joystick data shape: {joystick.shape}. Expected (N, 2)")
        
        if len(executed.shape) != 2 or executed.shape[1] != 1:
            errors.append(f"Invalid executed data shape: {executed.shape}. Expected (N, 1)")
        
        if len(joystick) != len(executed):
            errors.append(f"Mismatched lengths: joystick={len(joystick)}, executed={len(executed)}")
        
        # Check minimum data points
        if len(joystick) < self.min_data_points:
            warnings.append(
                f"Low data count: {len(joystick)} points (minimum recommended: {self.min_data_points})"
            )
        
        # Extract components
        velocities = joystick[:, 0]
        angular_velocities = joystick[:, 1]
        true_angular_velocities = executed[:, 0]
        
        # Validate velocity ranges
        if np.any(velocities > self.max_velocity):
            errors.append(
                f"Velocity exceeds maximum: {np.max(velocities):.2f} > {self.max_velocity}"
            )
        
        if np.any(np.abs(angular_velocities) > self.max_angular_velocity):
            errors.append(
                f"Angular velocity exceeds maximum: {np.max(np.abs(angular_velocities)):.2f} > {self.max_angular_velocity}"
            )
        
        # Check for NaN or Inf values
        if np.any(np.isnan(joystick)) or np.any(np.isinf(joystick)):
            errors.append("Found NaN or Inf values in joystick data")
        
        if np.any(np.isnan(executed)) or np.any(np.isinf(executed)):
            errors.append("Found NaN or Inf values in executed data")
        
        # Check for constant values (potential data collection issue)
        if np.std(velocities) < 1e-6:
            warnings.append("Velocities are nearly constant - potential data issue")
        
        if np.std(angular_velocities) < 1e-6:
            warnings.append("Angular velocities are nearly constant - potential data issue")
        
        # Compute statistics
        statistics = {
            'num_samples': len(joystick),
            'velocity_mean': float(np.mean(velocities)),
            'velocity_std': float(np.std(velocities)),
            'velocity_min': float(np.min(velocities)),
            'velocity_max': float(np.max(velocities)),
            'angular_velocity_mean': float(np.mean(angular_velocities)),
            'angular_velocity_std': float(np.std(angular_velocities)),
            'angular_velocity_min': float(np.min(angular_velocities)),
            'angular_velocity_max': float(np.max(angular_velocities)),
            'true_av_mean': float(np.mean(true_angular_velocities)),
            'true_av_std': float(np.std(true_angular_velocities)),
            'avg_deviation': float(np.mean(np.abs(angular_velocities - true_angular_velocities)))
        }
        
        # Compute curvature distribution
        valid_velocity_mask = np.abs(velocities) > 1e-3
        if np.any(valid_velocity_mask):
            curvatures = angular_velocities[valid_velocity_mask] / velocities[valid_velocity_mask]
            statistics['curvature_mean'] = float(np.mean(curvatures))
            statistics['curvature_std'] = float(np.std(curvatures))
            statistics['zero_curvature_ratio'] = float(np.sum(np.abs(curvatures) < 1e-3) / len(curvatures))
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, statistics)
    
    def validate_imu_delay(self, imu_delay: float) -> Tuple[bool, List[str]]:
        """
        Validate IMU delay value.
        
        Args:
            imu_delay: Computed IMU delay
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = []
        
        min_delay, max_delay = self.expected_imu_delay_range
        
        if imu_delay < min_delay or imu_delay > max_delay:
            warnings.append(
                f"Unusual IMU delay: {imu_delay:.4f}s. Expected range: [{min_delay}, {max_delay}]. "
                "This may indicate corrupted data or incorrect alignment."
            )
            return False, warnings
        
        return True, warnings
    
    def check_data_diversity(self, data: np.ndarray, feature_name: str = "data") -> List[str]:
        """
        Check data diversity and distribution.
        
        Args:
            data: Data array to check
            feature_name: Name of the feature for reporting
            
        Returns:
            List of warnings
        """
        warnings = []
        
        # Check for skewness
        from scipy.stats import skew
        data_skew = skew(data)
        if abs(data_skew) > 2.0:
            warnings.append(
                f"{feature_name} is highly skewed ({data_skew:.2f}). "
                "Consider collecting more diverse data."
            )
        
        # Check for outliers (using IQR method)
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        outlier_mask = (data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)
        outlier_ratio = np.sum(outlier_mask) / len(data)
        
        if outlier_ratio > 0.1:
            warnings.append(
                f"{feature_name} has {outlier_ratio*100:.1f}% outliers. "
                "Consider reviewing data collection process."
            )
        
        return warnings

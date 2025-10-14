"""
Evaluation metrics for Inverse Kinodynamic Model.
Implements metrics specific to autonomous vehicle drifting tasks.
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class IKDMetrics:
    """Collection of metrics for evaluating IKD model performance."""
    
    @staticmethod
    def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Mean Squared Error."""
        return mean_squared_error(targets, predictions)
    
    @staticmethod
    def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Mean Absolute Error."""
        return mean_absolute_error(targets, predictions)
    
    @staticmethod
    def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(targets, predictions))
    
    @staticmethod
    def r2(predictions: np.ndarray, targets: np.ndarray) -> float:
        """R² Score (Coefficient of Determination)."""
        return r2_score(targets, predictions)
    
    @staticmethod
    def curvature_error(
        predicted_av: np.ndarray,
        target_av: np.ndarray,
        velocities: np.ndarray,
        eps: float = 1e-6
    ) -> float:
        """
        Compute average curvature error.
        
        Curvature is calculated as: c = av / v
        
        Args:
            predicted_av: Predicted angular velocities
            target_av: Target angular velocities
            velocities: Linear velocities
            eps: Small value to avoid division by zero
            
        Returns:
            Mean absolute curvature error
        """
        # Avoid division by zero
        valid_mask = np.abs(velocities) > eps
        
        if not np.any(valid_mask):
            return 0.0
        
        pred_curvature = predicted_av[valid_mask] / velocities[valid_mask]
        target_curvature = target_av[valid_mask] / velocities[valid_mask]
        
        return np.mean(np.abs(pred_curvature - target_curvature))
    
    @staticmethod
    def percentage_error(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-6) -> float:
        """
        Mean Absolute Percentage Error.
        
        Args:
            predictions: Predicted values
            targets: Target values
            eps: Small value to avoid division by zero
            
        Returns:
            MAPE as percentage
        """
        mask = np.abs(targets) > eps
        if not np.any(mask):
            return 0.0
        
        return 100 * np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask]))
    
    @staticmethod
    def angular_velocity_deviation(
        predicted_av: np.ndarray,
        commanded_av: np.ndarray,
        true_av: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute deviation metrics between predicted, commanded, and true angular velocities.
        
        This metric is specific to the IKD problem where we want predicted AV to
        better match true AV compared to commanded AV.
        
        Args:
            predicted_av: Model predictions
            commanded_av: Original commanded angular velocities
            true_av: Ground truth IMU angular velocities
            
        Returns:
            Dictionary with deviation metrics
        """
        # Error between commanded and true (baseline)
        baseline_error = np.mean(np.abs(commanded_av - true_av))
        
        # Error between predicted and true (our model)
        model_error = np.mean(np.abs(predicted_av - true_av))
        
        # Improvement percentage
        improvement = ((baseline_error - model_error) / baseline_error) * 100
        
        return {
            'baseline_mae': baseline_error,
            'model_mae': model_error,
            'improvement_percentage': improvement
        }
    
    @classmethod
    def compute_all_metrics(
        cls,
        predictions: np.ndarray,
        targets: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        commanded_av: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            velocities: Linear velocities (optional, for curvature error)
            commanded_av: Commanded angular velocities (optional, for deviation metrics)
            
        Returns:
            Dictionary of all computed metrics
        """
        metrics = {
            'mse': cls.mse(predictions, targets),
            'mae': cls.mae(predictions, targets),
            'rmse': cls.rmse(predictions, targets),
            'r2': cls.r2(predictions, targets),
            'mape': cls.percentage_error(predictions, targets)
        }
        
        # Add curvature error if velocities provided
        if velocities is not None:
            metrics['curvature_error'] = cls.curvature_error(predictions, targets, velocities)
        
        # Add deviation metrics if commanded_av provided
        if commanded_av is not None:
            deviation_metrics = cls.angular_velocity_deviation(predictions, commanded_av, targets)
            metrics.update(deviation_metrics)
        
        return metrics


class CircleMetrics:
    """
    Metrics for evaluating circle navigation performance.
    Based on Section IV-C of the paper.
    """
    
    @staticmethod
    def compute_curvature_from_radius(radius: float) -> float:
        """Compute curvature from radius: c = 1/r"""
        return 1.0 / radius if radius > 0 else 0.0
    
    @staticmethod
    def compute_radius_from_curvature(curvature: float) -> float:
        """Compute radius from curvature: r = 1/c"""
        return 1.0 / curvature if abs(curvature) > 1e-6 else float('inf')
    
    @staticmethod
    def curvature_deviation_percentage(
        commanded_curvature: float,
        measured_radius: float
    ) -> float:
        """
        Compute deviation between commanded and executed curvature.
        
        Args:
            commanded_curvature: Commanded curvature value
            measured_radius: Measured radius from actual trajectory
            
        Returns:
            Percentage deviation
        """
        executed_curvature = CircleMetrics.compute_curvature_from_radius(measured_radius)
        deviation = abs(commanded_curvature - executed_curvature)
        percentage = (deviation / commanded_curvature) * 100 if commanded_curvature > 0 else 0.0
        return percentage
    
    @staticmethod
    def compare_trajectories(
        baseline_radius: float,
        ikd_radius: float,
        commanded_curvature: float
    ) -> Dict[str, float]:
        """
        Compare baseline and IKD-corrected trajectories.
        
        Args:
            baseline_radius: Measured radius without IKD
            ikd_radius: Measured radius with IKD correction
            commanded_curvature: Commanded curvature
            
        Returns:
            Dictionary with comparison metrics
        """
        baseline_curvature = CircleMetrics.compute_curvature_from_radius(baseline_radius)
        ikd_curvature = CircleMetrics.compute_curvature_from_radius(ikd_radius)
        
        baseline_error = abs(commanded_curvature - baseline_curvature)
        ikd_error = abs(commanded_curvature - ikd_curvature)
        
        improvement = ((baseline_error - ikd_error) / baseline_error) * 100
        
        return {
            'commanded_curvature': commanded_curvature,
            'baseline_curvature': baseline_curvature,
            'ikd_curvature': ikd_curvature,
            'baseline_error': baseline_error,
            'ikd_error': ikd_error,
            'improvement_percentage': improvement,
            'baseline_deviation_pct': CircleMetrics.curvature_deviation_percentage(
                commanded_curvature, baseline_radius
            ),
            'ikd_deviation_pct': CircleMetrics.curvature_deviation_percentage(
                commanded_curvature, ikd_radius
            )
        }

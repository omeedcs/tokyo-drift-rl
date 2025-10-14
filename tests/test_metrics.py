"""
Unit tests for evaluation metrics.
"""
import unittest
import numpy as np
from src.evaluation.metrics import IKDMetrics, CircleMetrics


class TestIKDMetrics(unittest.TestCase):
    """Test cases for IKD metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.targets = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    
    def test_mse(self):
        """Test MSE calculation."""
        mse = IKDMetrics.mse(self.predictions, self.targets)
        expected = np.mean((self.predictions - self.targets) ** 2)
        self.assertAlmostEqual(mse, expected, places=6)
    
    def test_mae(self):
        """Test MAE calculation."""
        mae = IKDMetrics.mae(self.predictions, self.targets)
        expected = np.mean(np.abs(self.predictions - self.targets))
        self.assertAlmostEqual(mae, expected, places=6)
    
    def test_rmse(self):
        """Test RMSE calculation."""
        rmse = IKDMetrics.rmse(self.predictions, self.targets)
        expected = np.sqrt(np.mean((self.predictions - self.targets) ** 2))
        self.assertAlmostEqual(rmse, expected, places=6)
    
    def test_r2(self):
        """Test R² score calculation."""
        r2 = IKDMetrics.r2(self.predictions, self.targets)
        self.assertGreaterEqual(r2, 0.0)
        self.assertLessEqual(r2, 1.0)
    
    def test_curvature_error(self):
        """Test curvature error calculation."""
        predicted_av = np.array([0.5, 1.0, 1.5])
        target_av = np.array([0.6, 1.1, 1.4])
        velocities = np.array([2.0, 2.5, 3.0])
        
        error = IKDMetrics.curvature_error(predicted_av, target_av, velocities)
        self.assertGreaterEqual(error, 0.0)
    
    def test_curvature_error_zero_velocity(self):
        """Test curvature error handles zero velocity."""
        predicted_av = np.array([0.5, 1.0])
        target_av = np.array([0.6, 1.1])
        velocities = np.array([0.0, 2.0])  # One zero velocity
        
        error = IKDMetrics.curvature_error(predicted_av, target_av, velocities)
        self.assertGreaterEqual(error, 0.0)
    
    def test_angular_velocity_deviation(self):
        """Test angular velocity deviation metrics."""
        predicted_av = np.array([1.0, 2.0, 3.0])
        commanded_av = np.array([1.2, 2.3, 3.1])
        true_av = np.array([1.1, 2.1, 2.9])
        
        metrics = IKDMetrics.angular_velocity_deviation(predicted_av, commanded_av, true_av)
        
        self.assertIn('baseline_mae', metrics)
        self.assertIn('model_mae', metrics)
        self.assertIn('improvement_percentage', metrics)
        self.assertGreaterEqual(metrics['baseline_mae'], 0.0)
        self.assertGreaterEqual(metrics['model_mae'], 0.0)
    
    def test_compute_all_metrics(self):
        """Test computing all metrics at once."""
        velocities = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
        commanded_av = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        
        metrics = IKDMetrics.compute_all_metrics(
            predictions=self.predictions,
            targets=self.targets,
            velocities=velocities,
            commanded_av=commanded_av
        )
        
        # Check all expected keys exist
        expected_keys = ['mse', 'mae', 'rmse', 'r2', 'mape', 'curvature_error',
                        'baseline_mae', 'model_mae', 'improvement_percentage']
        for key in expected_keys:
            self.assertIn(key, metrics)


class TestCircleMetrics(unittest.TestCase):
    """Test cases for circle metrics."""
    
    def test_compute_curvature_from_radius(self):
        """Test curvature calculation from radius."""
        radius = 2.0
        curvature = CircleMetrics.compute_curvature_from_radius(radius)
        self.assertAlmostEqual(curvature, 0.5, places=6)
    
    def test_compute_radius_from_curvature(self):
        """Test radius calculation from curvature."""
        curvature = 0.5
        radius = CircleMetrics.compute_radius_from_curvature(curvature)
        self.assertAlmostEqual(radius, 2.0, places=6)
    
    def test_curvature_radius_roundtrip(self):
        """Test converting between curvature and radius."""
        original_radius = 1.5
        curvature = CircleMetrics.compute_curvature_from_radius(original_radius)
        recovered_radius = CircleMetrics.compute_radius_from_curvature(curvature)
        self.assertAlmostEqual(original_radius, recovered_radius, places=6)
    
    def test_curvature_deviation_percentage(self):
        """Test curvature deviation percentage calculation."""
        commanded_curvature = 0.7
        measured_radius = 1.49
        
        deviation = CircleMetrics.curvature_deviation_percentage(
            commanded_curvature, measured_radius
        )
        self.assertGreaterEqual(deviation, 0.0)
        self.assertLess(deviation, 100.0)
    
    def test_compare_trajectories(self):
        """Test trajectory comparison."""
        baseline_radius = 1.49
        ikd_radius = 1.42
        commanded_curvature = 0.7
        
        comparison = CircleMetrics.compare_trajectories(
            baseline_radius, ikd_radius, commanded_curvature
        )
        
        # Check all expected keys
        expected_keys = ['commanded_curvature', 'baseline_curvature', 'ikd_curvature',
                        'baseline_error', 'ikd_error', 'improvement_percentage',
                        'baseline_deviation_pct', 'ikd_deviation_pct']
        for key in expected_keys:
            self.assertIn(key, comparison)
        
        # IKD should improve (reduce error)
        self.assertLess(comparison['ikd_error'], comparison['baseline_error'])
        self.assertGreater(comparison['improvement_percentage'], 0.0)


if __name__ == '__main__':
    unittest.main()

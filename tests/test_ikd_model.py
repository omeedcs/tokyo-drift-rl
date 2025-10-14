"""
Unit tests for IKD model.
"""
import unittest
import torch
import numpy as np
from src.models.ikd_model import IKDModel


class TestIKDModel(unittest.TestCase):
    """Test cases for IKD model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = IKDModel(dim_input=2, dim_output=1)
        self.model.eval()
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        self.assertIsInstance(self.model, IKDModel)
        self.assertEqual(self.model.l1.in_features, 2)
        self.assertEqual(self.model.correction.out_features, 1)
    
    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape."""
        batch_size = 32
        inputs = torch.randn(batch_size, 2)
        
        with torch.no_grad():
            outputs = self.model(inputs)
        
        self.assertEqual(outputs.shape, (batch_size, 1))
    
    def test_forward_pass_single_sample(self):
        """Test forward pass with single sample."""
        input_tensor = torch.FloatTensor([[2.0, 0.5]])  # velocity, angular_velocity
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        self.assertEqual(output.shape, (1, 1))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_model_deterministic(self):
        """Test model produces deterministic outputs."""
        input_tensor = torch.FloatTensor([[2.0, 0.5]])
        
        with torch.no_grad():
            output1 = self.model(input_tensor)
            output2 = self.model(input_tensor)
        
        self.assertTrue(torch.allclose(output1, output2))
    
    def test_model_gradients(self):
        """Test model computes gradients correctly."""
        self.model.train()
        inputs = torch.randn(10, 2, requires_grad=True)
        outputs = self.model(inputs)
        loss = outputs.sum()
        loss.backward()
        
        # Check all parameters have gradients
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
            self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient")
    
    def test_model_different_architectures(self):
        """Test model with different input/output dimensions."""
        # Test with different dimensions (though IKD uses 2->1)
        model = IKDModel(dim_input=3, dim_output=2)
        inputs = torch.randn(5, 3)
        
        with torch.no_grad():
            outputs = model(inputs)
        
        self.assertEqual(outputs.shape, (5, 2))
    
    def test_model_save_load(self):
        """Test model can be saved and loaded."""
        import tempfile
        import os
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model.pt")
            torch.save(self.model.state_dict(), save_path)
            
            # Load model
            new_model = IKDModel(dim_input=2, dim_output=1)
            new_model.load_state_dict(torch.load(save_path))
            new_model.eval()
            
            # Compare outputs
            test_input = torch.randn(5, 2)
            with torch.no_grad():
                output1 = self.model(test_input)
                output2 = new_model(test_input)
            
            self.assertTrue(torch.allclose(output1, output2))
    
    def test_model_realistic_inputs(self):
        """Test model with realistic vehicle inputs."""
        # Velocity range: 0-4.2 m/s, Angular velocity: -4 to 4 rad/s
        velocities = torch.FloatTensor([1.0, 2.0, 3.0, 4.0])
        angular_velocities = torch.FloatTensor([0.5, -0.5, 1.0, -1.0])
        inputs = torch.stack([velocities, angular_velocities], dim=1)
        
        with torch.no_grad():
            outputs = self.model(inputs)
        
        self.assertEqual(outputs.shape, (4, 1))
        # Check outputs are in reasonable range
        self.assertTrue((outputs.abs() < 10).all(), "Outputs outside expected range")


if __name__ == '__main__':
    unittest.main()

"""
Inverse Kinodynamic Model for autonomous vehicle drifting.

This module defines a simple feedforward neural network that learns to predict
the corrected joystick angular velocity given the current velocity and IMU measurements.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IKDModel(nn.Module):
    """
    Inverse Kinodynamic Model for drifting control.
    
    Architecture:
        - Input Layer: 2 features (velocity, true_angular_velocity)
        - Hidden Layer 1: 32 units with ReLU activation
        - Hidden Layer 2: 32 units with ReLU activation
        - Output Layer: 1 feature (predicted_joystick_angular_velocity)
    
    Inputs:
        - velocity (v): Linear velocity from joystick controller
        - true_angular_velocity (av'): Ground truth angular velocity from IMU
    
    Output:
        - predicted_angular_velocity (av): Predicted joystick angular velocity
        
    The model learns the inverse kinodynamic function f_θ⁺ that maps
    onboard observations to corrected control inputs.
    """
    
    def __init__(self, dim_input=2, dim_output=1):
        """
        Initialize the IKD model.
        
        Args:
            dim_input (int): Input dimension (default: 2)
            dim_output (int): Output dimension (default: 1)
        """
        super(IKDModel, self).__init__()
        self.l1 = nn.Linear(dim_input, 32)
        self.l2 = nn.Linear(32, 32)
        self.correction = nn.Linear(32, dim_output)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim_input)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, dim_output)
        """
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.correction(x)
        return x
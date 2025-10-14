"""
Improved Inverse Kinodynamic Model with temporal dynamics.

Version 2: Addresses limitations of the original model:
- Adds LSTM for temporal modeling (captures delay dynamics)
- Expands input features (position, obstacles, slip indicators)
- Deeper network (more capacity)
- Multi-objective loss function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class IKDModelV2(nn.Module):
    """
    Improved Inverse Kinodynamic Model with temporal awareness.
    
    Architecture:
        - Input: Multi-feature observations over time window
        - LSTM: Captures temporal dynamics and delay compensation
        - MLP: Processes LSTM output to control commands
    
    Key Improvements over V1:
        1. Temporal modeling (LSTM) - captures dynamics over time
        2. More input features (10-15 vs 2) - better state awareness
        3. Deeper network (128+ units vs 32) - more capacity
        4. Residual connections - easier training
    
    Input Features (per timestep):
        - velocity: Linear velocity
        - angular_velocity: From IMU (delayed)
        - acceleration: Estimated from velocity diff
        - steering_angle: Current servo position
        - x, y, theta: Localization (optional)
        - slip_indicator: |av_measured - av_expected|
        - distance_to_obstacles: Ray casts (8 directions)
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        output_dim: int = 2,
        dropout: float = 0.2,
        use_attention: bool = False
    ):
        """
        Initialize improved IKD model.
        
        Args:
            input_dim: Number of input features per timestep
            hidden_dim: LSTM hidden dimension
            num_lstm_layers: Number of LSTM layers
            output_dim: Number of outputs (velocity, angular_velocity corrections)
            dropout: Dropout rate for regularization
            use_attention: Whether to use attention over LSTM outputs
        """
        super(IKDModelV2, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal modeling with LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Optional attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        
        # Output MLP (deeper than V1)
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        # Learnable initial hidden state
        self.h0 = nn.Parameter(torch.zeros(num_lstm_layers, 1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(num_lstm_layers, 1, hidden_dim))
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
               or (batch_size, input_dim) for single timestep
            hidden: Optional tuple of (h, c) hidden states for LSTM
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Handle single timestep input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        batch_size, seq_len, _ = x.shape
        
        # Embed inputs
        x = self.input_embedding(x)  # (batch, seq, hidden)
        
        # LSTM temporal processing
        if hidden is None:
            # Initialize hidden states
            h0 = self.h0.expand(-1, batch_size, -1).contiguous()
            c0 = self.c0.expand(-1, batch_size, -1).contiguous()
            hidden = (h0, c0)
        
        lstm_out, hidden = self.lstm(x, hidden)  # (batch, seq, hidden)
        
        # Apply attention if enabled
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = lstm_out + attn_out  # Residual connection
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]  # (batch, hidden)
        
        # Generate control corrections
        output = self.output_network(last_output)  # (batch, output_dim)
        
        return output
    
    def predict_sequence(
        self,
        x: torch.Tensor,
        return_hidden: bool = False
    ) -> torch.Tensor:
        """
        Predict for entire sequence (useful for training).
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            return_hidden: Whether to return final hidden state
            
        Returns:
            Output tensor (batch, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Embed inputs
        x = self.input_embedding(x)
        
        # LSTM
        h0 = self.h0.expand(-1, batch_size, -1).contiguous()
        c0 = self.c0.expand(-1, batch_size, -1).contiguous()
        lstm_out, hidden = self.lstm(x, (h0, c0))
        
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = lstm_out + attn_out
        
        # Predict for each timestep
        lstm_out_flat = lstm_out.reshape(-1, self.hidden_dim)
        output_flat = self.output_network(lstm_out_flat)
        output = output_flat.reshape(batch_size, seq_len, self.output_dim)
        
        if return_hidden:
            return output, hidden
        return output


class IKDModelSimple(nn.Module):
    """
    Simplified version without LSTM for comparison.
    
    Still has more features than V1, but no temporal modeling.
    Useful for ablation studies.
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: list = [256, 128, 64],
        output_dim: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize simple (non-temporal) model.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of outputs
            dropout: Dropout rate
        """
        super(IKDModelSimple, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:  # Don't add dropout before last layer
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, input_dim)
            
        Returns:
            Output tensor (batch, output_dim)
        """
        return self.network(x)


def create_model(
    model_type: str = "lstm",
    input_dim: int = 10,
    **kwargs
) -> nn.Module:
    """
    Factory function to create IKD model.
    
    Args:
        model_type: "lstm", "simple", or "v1" (original)
        input_dim: Number of input features
        **kwargs: Additional arguments for model constructor
        
    Returns:
        Initialized model
    """
    if model_type == "lstm":
        return IKDModelV2(input_dim=input_dim, **kwargs)
    elif model_type == "simple":
        return IKDModelSimple(input_dim=input_dim, **kwargs)
    elif model_type == "v1":
        # Original model for comparison
        from src.models.ikd_model import IKDModel
        return IKDModel(dim_input=2, dim_output=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Feature extraction utilities
def extract_features(
    velocity: float,
    angular_velocity: float,
    x: float,
    y: float,
    theta: float,
    steering_angle: float,
    obstacles: list,
    previous_velocity: Optional[float] = None
) -> torch.Tensor:
    """
    Extract feature vector for improved model.
    
    Args:
        velocity: Linear velocity
        angular_velocity: From IMU
        x, y, theta: Vehicle pose
        steering_angle: Current servo angle
        obstacles: List of obstacle positions
        previous_velocity: Velocity from previous timestep
        
    Returns:
        Feature tensor of shape (input_dim,)
    """
    import numpy as np
    
    features = []
    
    # Core measurements
    features.append(velocity)
    features.append(angular_velocity)
    
    # Acceleration (if available)
    if previous_velocity is not None:
        acceleration = velocity - previous_velocity
    else:
        acceleration = 0.0
    features.append(acceleration)
    
    # Steering
    features.append(steering_angle)
    
    # Position and heading
    features.append(x)
    features.append(y)
    features.append(np.cos(theta))  # Encode heading as sin/cos
    features.append(np.sin(theta))
    
    # Slip indicator: difference between expected and measured angular velocity
    if abs(velocity) > 0.01:
        expected_av = velocity * np.tan(steering_angle) / 0.324  # wheelbase
        slip_indicator = abs(angular_velocity - expected_av)
    else:
        slip_indicator = 0.0
    features.append(slip_indicator)
    
    # Distance to nearest obstacle (simplified - can add ray casts)
    if obstacles:
        min_dist = min(
            np.sqrt((x - obs[0])**2 + (y - obs[1])**2)
            for obs in obstacles
        )
    else:
        min_dist = 10.0  # Large value if no obstacles
    features.append(min_dist)
    
    return torch.FloatTensor(features)

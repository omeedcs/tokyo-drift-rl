#!/usr/bin/env python3
"""
Simple IKD training script that works with .npz data.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from src.models.ikd_model import IKDModel


def train_ikd(data_path, epochs=100, lr=0.001, model_save_path="trained_models/ikd_model.pt"):
    """Train IKD model on real data."""
    
    print("\n" + "="*60)
    print("Training IKD Model on Real Data")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    
    print(f"  Data shape: X={X.shape}, y={y.shape}")
    print(f"  Training samples: {len(X)}")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    # Create model
    print("\nInitializing IKD model...")
    model = IKDModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {epochs}")
    
    # Train
    print("\n" + "="*60)
    print("Training...")
    print("="*60)
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}: Loss = {loss.item():.6f} (Best: {best_loss:.6f})")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"  Final loss: {loss.item():.6f}")
    print(f"  Best loss: {best_loss:.6f}")
    
    # Save model
    save_path = Path(model_save_path)
    save_path.parent.mkdir(exist_ok=True)
    
    torch.save(model.state_dict(), save_path)
    print(f"\n💾 Model saved to: {save_path}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train IKD model on real data")
    parser.add_argument("--data", type=str, default="data/ikd_real_data.npz",
                        help="Path to training data (.npz file)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--output", type=str, default="trained_models/ikd_model.pt",
                        help="Where to save trained model")
    
    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("#  Train IKD Model")
    print("#" + "#"*60)
    
    train_ikd(
        data_path=args.data,
        epochs=args.epochs,
        lr=args.lr,
        model_save_path=args.output
    )
    
    print("\n" + "="*60)
    print("Next step: Test the model!")
    print("="*60)
    print(f"\nRun:")
    print(f"  python test_ikd_simulation.py --model {args.output}")
    print()

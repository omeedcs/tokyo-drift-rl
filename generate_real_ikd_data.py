#!/usr/bin/env python3
"""
Generate REAL IKD training data from the simulator.

This collects actual (commanded, actual) pairs by running
the simulator with various control inputs.
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from src.simulator.environment import SimulationEnvironment


def collect_training_data(num_samples=10000):
    """Collect real training data from simulator."""
    print("\n" + "="*60)
    print("Collecting REAL Training Data from Simulator")
    print("="*60)
    
    env = SimulationEnvironment(dt=0.05)
    
    commanded_velocities = []
    commanded_angular_velocities = []
    actual_velocities = []
    
    print(f"\nCollecting {num_samples} samples...")
    print("(This will take a few minutes)")
    
    for i in tqdm(range(num_samples)):
        # Reset periodically
        if i % 100 == 0:
            env.reset()
        
        # Random control commands
        vel_cmd = np.random.uniform(-3, 3)
        av_cmd = np.random.uniform(-2, 2)
        
        # Apply to simulator
        env.set_control(vel_cmd, av_cmd)
        env.step()
        
        # Record actual response
        state = env.vehicle.get_state()
        actual_vel = state.velocity
        
        commanded_velocities.append(vel_cmd)
        commanded_angular_velocities.append(av_cmd)
        actual_velocities.append(actual_vel)
    
    # Convert to arrays
    commanded_vel = np.array(commanded_velocities)
    commanded_av = np.array(commanded_angular_velocities)
    actual_vel = np.array(actual_velocities)
    
    # Calculate what correction is needed
    corrections = actual_vel - commanded_vel
    
    # Prepare features and labels
    X = np.column_stack([commanded_vel, commanded_av])
    y = corrections
    
    print(f"\n✅ Collected {num_samples} samples")
    print(f"  Commanded velocity range: [{commanded_vel.min():.2f}, {commanded_vel.max():.2f}]")
    print(f"  Actual velocity range: [{actual_vel.min():.2f}, {actual_vel.max():.2f}]")
    print(f"  Correction range: [{corrections.min():.2f}, {corrections.max():.2f}]")
    print(f"  Mean correction: {corrections.mean():.3f} m/s")
    print(f"  Std correction: {corrections.std():.3f} m/s")
    
    return X, y


def save_data(X, y, filepath="data/ikd_real_data.npz"):
    """Save training data."""
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True)
    
    np.savez(filepath, X=X, y=y)
    print(f"\n💾 Saved training data: {filepath}")
    print(f"  Shape: X={X.shape}, y={y.shape}")


def main():
    print("\n" + "#"*60)
    print("#  Generate REAL IKD Training Data")
    print("#" + "#"*60)
    print("\nThis will:")
    print("  1. Run the simulator with random commands")
    print("  2. Record actual vehicle responses")
    print("  3. Learn the real system dynamics")
    print("  4. Save data for IKD training")
    
    # Collect data
    X, y = collect_training_data(num_samples=10000)
    
    # Save
    save_data(X, y, "data/ikd_real_data.npz")
    
    # Show statistics
    print("\n" + "="*60)
    print("Statistics")
    print("="*60)
    
    # Analyze relationship
    correlation = np.corrcoef(X[:, 0], y)[0, 1]
    print(f"\nCorrelation (cmd_vel vs correction): {correlation:.3f}")
    
    # What did we learn about the system?
    mean_actual_to_cmd_ratio = (X[:, 0] + y) / (X[:, 0] + 1e-6)
    mean_actual_to_cmd_ratio = mean_actual_to_cmd_ratio[np.abs(X[:, 0]) > 0.1]  # Filter near-zero
    
    print(f"Actual/Commanded ratio: {mean_actual_to_cmd_ratio.mean():.3f}")
    print(f"  (e.g., if cmd=1.0, actual≈{mean_actual_to_cmd_ratio.mean():.3f})")
    
    print("\n" + "="*60)
    print("✅ Data collection complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Train IKD model:")
    print("     python train.py --data data/ikd_real_data.npz")
    print("\n  2. Test it:")
    print("     python test_ikd_simulation.py --model trained_models/ikd_model.pt")


if __name__ == "__main__":
    main()

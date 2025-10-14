#!/usr/bin/env python3
"""
Collect IKD training data from REAL trajectory tracking episodes.

This collects (commanded, actual) velocity pairs during actual
drift maneuvers, which gives realistic correction magnitudes.
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from src.simulator.environment import SimulationEnvironment
from src.simulator.controller import DriftController


def collect_tracking_data(num_episodes=100, scenario="loose"):
    """Collect data from real trajectory tracking."""
    print("\n" + "="*60)
    print("Collecting IKD Data from Real Trajectory Tracking")
    print("="*60)
    
    commanded_velocities = []
    commanded_angular_velocities = []
    actual_velocities = []
    
    print(f"\nRunning {num_episodes} drift episodes...")
    print("(This will take a few minutes)")
    
    for episode in tqdm(range(num_episodes)):
        env = SimulationEnvironment(dt=0.05)
        
        if scenario == "loose":
            env.setup_loose_drift_test()
            gate_center = (3.0, 1.065)
            gate_width = 2.13
        else:
            env.setup_tight_drift_test()
            gate_center = (3.0, 0.405)
            gate_width = 0.81
        
        # Create controller
        controller = DriftController(use_optimizer=False)
        obstacles_list = [(obs.x, obs.y, obs.radius) for obs in env.obstacles]
        
        start_pos = (0.0, 0.0)
        controller.plan_trajectory(
            start_pos=start_pos,
            gate_center=gate_center,
            gate_width=gate_width,
            direction="ccw",
            obstacles=obstacles_list
        )
        
        # Run episode and collect data
        for step in range(200):
            state = env.vehicle.get_state()
            
            # Get control commands from controller
            vel_cmd, av_cmd = controller.update(
                state.x, state.y, state.theta, state.velocity
            )
            
            # Record data
            commanded_velocities.append(vel_cmd)
            commanded_angular_velocities.append(av_cmd)
            actual_velocities.append(state.velocity)
            
            # Apply command and step
            env.set_control(vel_cmd, av_cmd)
            env.step()
            
            if env.check_collision() or controller.is_complete():
                break
    
    # Convert to arrays
    commanded_vel = np.array(commanded_velocities)
    commanded_av = np.array(commanded_angular_velocities)
    actual_vel = np.array(actual_velocities)
    
    # Calculate corrections needed
    corrections = actual_vel - commanded_vel
    
    # Prepare features and labels
    X = np.column_stack([commanded_vel, commanded_av])
    y = corrections
    
    print(f"\n✅ Collected {len(X)} samples from {num_episodes} episodes")
    print(f"  Commanded velocity range: [{commanded_vel.min():.2f}, {commanded_vel.max():.2f}]")
    print(f"  Actual velocity range: [{actual_vel.min():.2f}, {actual_vel.max():.2f}]")
    print(f"  Correction range: [{corrections.min():.2f}, {corrections.max():.2f}]")
    print(f"  Mean correction: {corrections.mean():.3f} m/s")
    print(f"  Std correction: {corrections.std():.3f} m/s")
    
    return X, y


def save_data(X, y, filepath="data/ikd_tracking_data.npz"):
    """Save training data."""
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True)
    
    np.savez(filepath, X=X, y=y)
    print(f"\n💾 Saved training data: {filepath}")
    print(f"  Shape: X={X.shape}, y={y.shape}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect IKD data from tracking")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes to run")
    parser.add_argument("--scenario", type=str, default="loose",
                        choices=["loose", "tight"],
                        help="Scenario to use")
    parser.add_argument("--output", type=str, default="data/ikd_tracking_data.npz",
                        help="Output file path")
    
    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("#  Collect IKD Data from Trajectory Tracking")
    print("#" + "#"*60)
    print("\nThis will:")
    print("  1. Run drift episodes with the controller")
    print("  2. Record commanded vs actual velocities")
    print("  3. Learn realistic correction magnitudes")
    print("  4. Save data for IKD training")
    
    # Collect data
    X, y = collect_tracking_data(
        num_episodes=args.episodes,
        scenario=args.scenario
    )
    
    # Save
    save_data(X, y, args.output)
    
    # Show statistics
    print("\n" + "="*60)
    print("Statistics")
    print("="*60)
    
    # Analyze relationship
    print(f"\nData characteristics:")
    print(f"  Total samples: {len(X)}")
    print(f"  Mean absolute correction: {np.mean(np.abs(y)):.3f} m/s")
    print(f"  95th percentile correction: {np.percentile(np.abs(y), 95):.3f} m/s")
    
    print("\n" + "="*60)
    print("✅ Data collection complete!")
    print("="*60)
    print("\nNext steps:")
    print(f"  1. Train IKD model:")
    print(f"     python train_ikd_simple.py --data {args.output} --epochs 100")
    print("\n  2. Test it:")
    print("     python test_ikd_simulation.py --model trained_models/ikd_model.pt")


if __name__ == "__main__":
    main()

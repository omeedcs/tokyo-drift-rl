#!/usr/bin/env python3
"""
Collect IKD data with CORRECT labels.

The key insight: IKD should learn to predict the INVERSE correction.
If actual < commanded, IKD should output POSITIVE correction to compensate.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.simulator.environment import SimulationEnvironment
from src.simulator.controller import DriftController


def collect_corrected_data(num_episodes=100, scenario="loose"):
    """Collect data with correct IKD labels."""
    print("\n" + "="*60)
    print("Collecting IKD Data with CORRECTED Labels")
    print("="*60)
    
    commanded_velocities = []
    commanded_angular_velocities = []
    corrections_needed = []
    
    print(f"\nRunning {num_episodes} drift episodes...")
    
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
        
        controller = DriftController(use_optimizer=False)
        obstacles_list = [(obs.x, obs.y, obs.radius) for obs in env.obstacles]
        
        controller.plan_trajectory(
            start_pos=(0.0, 0.0),
            gate_center=gate_center,
            gate_width=gate_width,
            direction="ccw",
            obstacles=obstacles_list
        )
        
        # Run episode
        for step in range(200):
            state = env.vehicle.get_state()
            
            # Get control command
            vel_cmd, av_cmd = controller.update(
                state.x, state.y, state.theta, state.velocity
            )
            
            # Apply and step
            env.set_control(vel_cmd, av_cmd)
            env.step()
            
            # After stepping, see what happened
            next_state = env.vehicle.get_state()
            actual_vel = next_state.velocity
            
            # Calculate what correction would have been needed
            # If we commanded 1.0 but got 0.7, we need +0.3 correction
            correction_needed = vel_cmd - actual_vel
            
            # Record
            commanded_velocities.append(vel_cmd)
            commanded_angular_velocities.append(av_cmd)
            corrections_needed.append(correction_needed)
            
            if env.check_collision() or controller.is_complete():
                break
    
    # Convert to arrays
    commanded_vel = np.array(commanded_velocities)
    commanded_av = np.array(commanded_angular_velocities)
    corrections = np.array(corrections_needed)
    
    # Prepare features and labels
    X = np.column_stack([commanded_vel, commanded_av])
    y = corrections
    
    print(f"\n✅ Collected {len(X)} samples from {num_episodes} episodes")
    print(f"  Correction range: [{corrections.min():.2f}, {corrections.max():.2f}]")
    print(f"  Mean correction: {corrections.mean():.3f} m/s")
    print(f"  Std correction: {corrections.std():.3f} m/s")
    print(f"  Mean absolute correction: {np.mean(np.abs(corrections)):.3f} m/s")
    
    return X, y


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--scenario", type=str, default="loose")
    parser.add_argument("--output", type=str, default="data/ikd_corrected_data.npz")
    
    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("#  Collect IKD Data with CORRECT Labels")
    print("#" + "#"*60)
    print("\nKey insight:")
    print("  If commanded=1.0, actual=0.7 → correction=+0.3")
    print("  IKD learns to ADD this correction to compensate!")
    
    X, y = collect_corrected_data(args.episodes, args.scenario)
    
    filepath = Path(args.output)
    filepath.parent.mkdir(exist_ok=True)
    np.savez(filepath, X=X, y=y)
    
    print(f"\n💾 Saved: {filepath}")
    print("\nNext:")
    print(f"  python train_ikd_simple.py --data {args.output} --epochs 100 --output trained_models/ikd_corrected.pt")


if __name__ == "__main__":
    main()

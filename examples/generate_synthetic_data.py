#!/usr/bin/env python3
"""
Example: Generate synthetic training data using the simulator.

This is useful for:
- Augmenting real training data
- Testing data diversity effects
- Pre-training before real data collection
"""

import numpy as np
from pathlib import Path

from src.simulator import SimulationEnvironment


def generate_random_trajectories(
    num_episodes: int = 100,
    steps_per_episode: int = 200,
    velocity_range: tuple = (1.0, 4.0),
    curvature_range: tuple = (-0.8, 0.8),
    output_file: str = "synthetic_data.csv"
):
    """
    Generate random driving trajectories.
    
    Args:
        num_episodes: Number of trajectory episodes
        steps_per_episode: Steps per episode
        velocity_range: Min/max velocity (m/s)
        curvature_range: Min/max curvature (1/m)
        output_file: Output CSV file
    """
    print(f"Generating {num_episodes} random trajectories...")
    
    env = SimulationEnvironment(dt=0.05, enable_slip=True)
    env.start_recording()
    
    for episode in range(num_episodes):
        env.reset()
        
        # Random episode parameters
        base_velocity = np.random.uniform(*velocity_range)
        base_curvature = np.random.uniform(*curvature_range)
        
        for step in range(steps_per_episode):
            # Add variation
            velocity = base_velocity + np.random.normal(0, 0.2)
            curvature = base_curvature + np.random.normal(0, 0.05)
            
            # Clip to safe ranges
            velocity = np.clip(velocity, 0.5, 4.2)
            curvature = np.clip(curvature, -1.0, 1.0)
            
            angular_velocity = velocity * curvature
            
            env.set_control(velocity, angular_velocity)
            env.step()
        
        if (episode + 1) % 10 == 0:
            print(f"  Generated {episode + 1}/{num_episodes} episodes")
    
    env.stop_recording()
    
    # Save data
    env.save_recorded_data(output_file)
    print(f"\n✓ Saved {len(env.recorded_data['time'])} data points to {output_file}")


def generate_circle_dataset(
    velocities: list = [1.0, 2.0, 3.0, 4.0],
    curvatures: list = [0.2, 0.4, 0.6, 0.8],
    duration: float = 10.0,
    output_file: str = "circle_data.csv"
):
    """
    Generate circle navigation dataset.
    
    Args:
        velocities: List of test velocities
        curvatures: List of test curvatures
        duration: Duration per test (seconds)
        output_file: Output CSV file
    """
    print(f"Generating circle dataset...")
    print(f"  Velocities: {velocities}")
    print(f"  Curvatures: {curvatures}")
    
    env = SimulationEnvironment(dt=0.05, enable_slip=True)
    env.start_recording()
    
    total_tests = len(velocities) * len(curvatures)
    test_count = 0
    
    for velocity in velocities:
        for curvature in curvatures:
            test_count += 1
            print(f"  Test {test_count}/{total_tests}: v={velocity}, c={curvature}")
            
            env.reset()
            env.setup_circle_test(velocity, curvature)
            
            steps = int(duration / env.dt)
            for _ in range(steps):
                env.set_control(velocity, velocity * curvature)
                env.step()
    
    env.stop_recording()
    env.save_recorded_data(output_file)
    print(f"\n✓ Saved circle dataset to {output_file}")


def generate_drift_dataset(
    num_drifts: int = 50,
    output_file: str = "drift_data.csv"
):
    """
    Generate drift maneuver dataset.
    
    Args:
        num_drifts: Number of drift maneuvers
        output_file: Output CSV file
    """
    print(f"Generating {num_drifts} drift maneuvers...")
    
    from src.simulator.controller import DriftController
    
    env = SimulationEnvironment(dt=0.05, enable_slip=True)
    env.start_recording()
    
    for drift_num in range(num_drifts):
        env.reset()
        env.setup_loose_drift_test()
        
        drift_controller = DriftController(turbo_speed=5.0)
        
        # Execute drift
        for _ in range(300):  # 15 seconds
            velocity, av = drift_controller.execute_drift_ccw(env.time)
            env.set_control(velocity, av)
            env.step()
        
        if (drift_num + 1) % 10 == 0:
            print(f"  Generated {drift_num + 1}/{num_drifts} drifts")
    
    env.stop_recording()
    env.save_recorded_data(output_file)
    print(f"\n✓ Saved drift dataset to {output_file}")


def main():
    """Generate multiple synthetic datasets."""
    print("\n" + "="*80)
    print("Synthetic Data Generation")
    print("="*80)
    
    output_dir = Path("./synthetic_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/3] Generating random trajectories...")
    generate_random_trajectories(
        num_episodes=100,
        steps_per_episode=200,
        output_file=str(output_dir / "random_trajectories.csv")
    )
    
    print("\n[2/3] Generating circle dataset...")
    generate_circle_dataset(
        velocities=[1.0, 2.0, 3.0, 4.0],
        curvatures=[0.2, 0.4, 0.6, 0.8],
        duration=10.0,
        output_file=str(output_dir / "circle_dataset.csv")
    )
    
    print("\n[3/3] Generating drift dataset...")
    generate_drift_dataset(
        num_drifts=50,
        output_file=str(output_dir / "drift_dataset.csv")
    )
    
    print("\n" + "="*80)
    print("✓ Synthetic data generation complete!")
    print(f"Files saved to: {output_dir}")
    print("\nYou can now train on this data:")
    print(f"  python train.py --data-path {output_dir}/random_trajectories.csv")
    print("="*80)


if __name__ == "__main__":
    main()

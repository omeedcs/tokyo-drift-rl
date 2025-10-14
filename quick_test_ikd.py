#!/usr/bin/env python3
"""
Quick IKD Test - Generate data, train, and test in one script

This script:
1. Generates synthetic training data
2. Trains a simple IKD model
3. Tests it in simulation
4. Shows you the results

Usage:
    python quick_test_ikd.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

from src.simulator.environment import SimulationEnvironment
from src.simulator.controller import DriftController
from src.models.ikd_model import IKDModel


def generate_synthetic_data(num_samples=5000):
    """Generate synthetic IKD training data."""
    print("\n" + "="*60)
    print("Step 1: Generating Synthetic Training Data")
    print("="*60)
    
    # Simulate various commanded velocities and angular velocities
    np.random.seed(42)
    
    commanded_velocities = np.random.uniform(-3, 3, num_samples)
    commanded_angular_velocities = np.random.uniform(-2, 2, num_samples)
    
    # Simulate system delay/error (this is what IKD should learn)
    # Real system has delays and nonlinearities
    actual_velocities = commanded_velocities * 0.85 + np.random.normal(0, 0.1, num_samples)
    
    # Calculate what correction is needed
    corrections = actual_velocities - commanded_velocities
    
    X = np.column_stack([commanded_velocities, commanded_angular_velocities])
    y = corrections
    
    print(f"  Generated {num_samples} training samples")
    print(f"  Velocity range: [{commanded_velocities.min():.2f}, {commanded_velocities.max():.2f}]")
    print(f"  Correction range: [{corrections.min():.2f}, {corrections.max():.2f}]")
    
    return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)


def train_ikd_model(X, y, epochs=50):
    """Train IKD model."""
    print("\n" + "="*60)
    print("Step 2: Training IKD Model")
    print("="*60)
    
    model = IKDModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    print(f"\n  ✅ Training complete!")
    print(f"  Final loss: {loss.item():.6f}")
    
    return model


def test_in_simulation(model):
    """Test IKD model in simulation."""
    print("\n" + "="*60)
    print("Step 3: Testing in Simulation")
    print("="*60)
    
    results = {}
    
    for use_ikd in [False, True]:
        label = "With IKD" if use_ikd else "Baseline"
        print(f"\n  Running: {label}")
        
        env = SimulationEnvironment(dt=0.05)
        env.setup_loose_drift_test()
        
        controller = DriftController(use_optimizer=False)
        obstacles_list = [(obs.x, obs.y, obs.radius) for obs in env.obstacles]
        
        controller.plan_trajectory(
            start_pos=(0.0, 0.0),
            gate_center=(3.0, 1.065),
            gate_width=2.13,
            direction="ccw",
            obstacles=obstacles_list
        )
        
        trajectory = []
        vel_commanded = []
        vel_actual = []
        corrections = []
        
        model.eval()
        
        for i in range(200):
            state = env.vehicle.get_state()
            trajectory.append([state.x, state.y])
            
            vel_cmd, av_cmd = controller.update(
                state.x, state.y, state.theta, state.velocity
            )
            
            vel_commanded.append(vel_cmd)
            vel_actual.append(state.velocity)
            
            if use_ikd:
                with torch.no_grad():
                    model_input = torch.FloatTensor([vel_cmd, av_cmd]).unsqueeze(0)
                    correction = model(model_input).item()
                    corrected_vel = vel_cmd + correction
                    corrections.append(correction)
                env.set_control(corrected_vel, av_cmd)
            else:
                corrections.append(0.0)
                env.set_control(vel_cmd, av_cmd)
            
            env.step()
            
            if env.check_collision() or controller.is_complete():
                break
        
        results[label] = {
            'trajectory': np.array(trajectory),
            'vel_commanded': np.array(vel_commanded),
            'vel_actual': np.array(vel_actual),
            'corrections': np.array(corrections),
            'collision': env.check_collision(),
            'success': controller.is_complete()
        }
        
        print(f"    Result: {'✅ Success' if results[label]['success'] else '❌ Collision'}")
        print(f"    Steps: {i+1}")
        print(f"    Avg velocity error: {np.mean(np.abs(results[label]['vel_commanded'] - results[label]['vel_actual'])):.3f} m/s")
    
    return results


def plot_results(results):
    """Create visualization plots."""
    print("\n" + "="*60)
    print("Step 4: Generating Plots")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Trajectories
    ax = axes[0, 0]
    ax.plot(results['Baseline']['trajectory'][:, 0], 
            results['Baseline']['trajectory'][:, 1],
            'r--', linewidth=2, label='Baseline (No IKD)', alpha=0.7)
    ax.plot(results['With IKD']['trajectory'][:, 0],
            results['With IKD']['trajectory'][:, 1],
            'b-', linewidth=2, label='With IKD', alpha=0.7)
    ax.plot(0, 0, 'go', markersize=12, label='Start')
    ax.axvline(x=3.0, color='lime', linewidth=4, label='Goal', alpha=0.5)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Trajectory Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 2. Velocity tracking
    ax = axes[0, 1]
    time = np.arange(len(results['Baseline']['vel_commanded'])) * 0.05
    ax.plot(time, results['Baseline']['vel_commanded'], 'k--', 
            linewidth=1.5, label='Commanded', alpha=0.8)
    ax.plot(time, results['Baseline']['vel_actual'], 'r-',
            linewidth=2, label='Actual (No IKD)', alpha=0.7)
    
    time_ikd = np.arange(len(results['With IKD']['vel_actual'])) * 0.05
    ax.plot(time_ikd, results['With IKD']['vel_actual'], 'b-',
            linewidth=2, label='Actual (With IKD)', alpha=0.7)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Velocity (m/s)', fontsize=12)
    ax.set_title('Velocity Tracking', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Velocity errors
    ax = axes[1, 0]
    errors_baseline = results['Baseline']['vel_commanded'] - results['Baseline']['vel_actual']
    errors_ikd = results['With IKD']['vel_commanded'] - results['With IKD']['vel_actual']
    
    ax.plot(time, np.abs(errors_baseline), 'r-', linewidth=2, label='Baseline')
    ax.plot(time_ikd, np.abs(errors_ikd), 'b-', linewidth=2, label='With IKD')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Absolute Velocity Error (m/s)', fontsize=12)
    ax.set_title('Velocity Tracking Error', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. IKD corrections
    ax = axes[1, 1]
    ax.plot(time_ikd, results['With IKD']['corrections'], 'g-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('IKD Correction (m/s)', fontsize=12)
    ax.set_title('IKD Corrections Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path("ikd_test_results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "quick_test_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved plot: {output_path}")
    
    plt.show()


def main():
    print("\n" + "#"*60)
    print("#  Quick IKD Test - Full Pipeline")
    print("#" + "#"*60)
    print("\nThis will:")
    print("  1. Generate synthetic training data")
    print("  2. Train a simple IKD model")
    print("  3. Test it in simulation")
    print("  4. Show you the results")
    
    # Generate data
    X, y = generate_synthetic_data(num_samples=5000)
    
    # Train model
    model = train_ikd_model(X, y, epochs=50)
    
    # Save model
    model_dir = Path("trained_models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "ikd_quick_test.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n  💾 Saved model: {model_path}")
    
    # Test in simulation
    results = test_in_simulation(model)
    
    # Plot results
    plot_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    baseline_error = np.mean(np.abs(
        results['Baseline']['vel_commanded'] - results['Baseline']['vel_actual']
    ))
    ikd_error = np.mean(np.abs(
        results['With IKD']['vel_commanded'] - results['With IKD']['vel_actual']
    ))
    
    improvement = (baseline_error - ikd_error) / baseline_error * 100
    
    print(f"\nBaseline (No IKD):")
    print(f"  Avg Velocity Error: {baseline_error:.3f} m/s")
    print(f"  Success: {results['Baseline']['success']}")
    
    print(f"\nWith IKD:")
    print(f"  Avg Velocity Error: {ikd_error:.3f} m/s")
    print(f"  Success: {results['With IKD']['success']}")
    print(f"  Improvement: {improvement:+.1f}%")
    
    if improvement > 0:
        print(f"\n  ✅ IKD is WORKING! Error reduced by {improvement:.1f}%")
    else:
        print(f"\n  ⚠️  IKD needs more tuning")
    
    print("\n" + "="*60)
    print("✨ Test complete! Check ikd_test_results/ for plots")
    print("="*60)


if __name__ == "__main__":
    main()

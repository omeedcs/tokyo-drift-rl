#!/usr/bin/env python3
"""
Example: Reproduce all experiments from the paper using simulation.

This script runs:
1. Circle navigation tests (Table I)
2. Loose drifting tests (Table II)
3. Baseline vs IKD comparisons

Perfect for testing without physical hardware!
"""

import numpy as np
from pathlib import Path

from src.simulator import SimulationEnvironment, VirtualJoystick
from src.simulator.visualization import (
    plot_trajectory,
    plot_simulation_data,
    compare_baseline_vs_ikd
)
from src.evaluation.metrics import CircleMetrics


def run_circle_tests():
    """Reproduce Table I from the paper."""
    print("\n" + "="*80)
    print("Table I: Circle Navigation Tests")
    print("="*80)
    
    # Test parameters from paper
    velocity = 2.0
    curvatures = [0.12, 0.63, 0.70, 0.80]
    
    results = []
    
    for curvature in curvatures:
        print(f"\nTesting curvature = {curvature} 1/m")
        
        # Setup environment
        env = SimulationEnvironment(dt=0.05)
        joystick = VirtualJoystick()
        
        env.setup_circle_test(velocity, curvature)
        joystick.set_circle_mode(velocity, curvature)
        
        # Run simulation
        env.start_recording()
        for _ in range(200):  # ~10 seconds
            velocity_cmd, av_cmd = joystick.get_control(
                env.vehicle.get_velocity(),
                env.vehicle.get_angular_velocity()
            )
            env.set_control(velocity_cmd, av_cmd)
            env.step()
        env.stop_recording()
        
        # Measure results
        measured_radius = env.measure_circle_radius()
        measured_curvature = 1.0 / measured_radius
        deviation = CircleMetrics.curvature_deviation_percentage(
            curvature, measured_radius
        )
        
        results.append({
            'commanded': curvature,
            'measured': measured_curvature,
            'deviation_pct': deviation
        })
        
        print(f"  Commanded: {curvature:.3f} 1/m")
        print(f"  Measured: {measured_curvature:.3f} 1/m")
        print(f"  Deviation: {deviation:.2f}%")
    
    # Print summary table
    print("\n" + "="*80)
    print("Summary (Compare with Table I):")
    print("-"*80)
    print(f"{'Commanded':>12} | {'Measured':>12} | {'Deviation':>12} | {'Paper':>12}")
    print("-"*80)
    
    paper_deviations = [2.33, 0.11, None, 1.78]  # From paper
    for i, result in enumerate(results):
        paper_val = f"{paper_deviations[i]:.2f}%" if paper_deviations[i] else "N/A"
        print(f"{result['commanded']:>12.3f} | "
              f"{result['measured']:>12.3f} | "
              f"{result['deviation_pct']:>11.2f}% | "
              f"{paper_val:>12}")
    print("="*80)


def run_drift_comparison():
    """Compare baseline vs IKD on drift test."""
    print("\n" + "="*80)
    print("Drift Test: Baseline vs IKD Comparison")
    print("="*80)
    
    # You would load your trained model here
    # For demo, we'll just show the framework
    
    print("\n[INFO] To run this with a real IKD model:")
    print("  1. Train model: python train.py")
    print("  2. Run: python simulate.py --mode drift-loose --compare --model path/to/model.pt")
    
    # Setup
    env = SimulationEnvironment(dt=0.05)
    env.setup_loose_drift_test()
    
    print("\n[INFO] Loose drift environment configured:")
    print(f"  Obstacles: {len(env.obstacles)}")
    print(f"  Spacing: 2.13 meters (84 inches)")
    print(f"  Expected: Vehicle should clear obstacles")


def main():
    """Run all paper experiments."""
    print("\n" + "="*80)
    print("Reproducing Paper Experiments in Simulation")
    print("Paper: Suvarna & Tehrani, 2024 (arXiv:2402.14928)")
    print("="*80)
    
    # Create output directory
    output_dir = Path("./simulation_results/paper_reproduction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    run_circle_tests()
    run_drift_comparison()
    
    print("\n" + "="*80)
    print("Simulation Complete!")
    print(f"Results would be saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

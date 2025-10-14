#!/usr/bin/env python3
"""
Main simulation script for testing IKD models without physical hardware.

Usage:
    python simulate.py --mode circle --velocity 2.0 --curvature 0.7
    python simulate.py --mode drift-loose --use-ikd --model path/to/model.pt
    python simulate.py --mode drift-tight --duration 10.0
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from src.simulator import (
    F110Vehicle,
    SimulationEnvironment,
    VirtualJoystick
)
from src.simulator.visualization import (
    plot_trajectory,
    plot_simulation_data,
    compare_baseline_vs_ikd
)
from src.simulator.controller import DriftController
from src.models.ikd_model import IKDModel
from src.evaluation.metrics import IKDMetrics, CircleMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="IKD Simulation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["circle", "drift-loose", "drift-tight", "manual"],
        default="circle",
        help="Simulation mode"
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=2.0,
        help="Velocity for circle mode (m/s)"
    )
    parser.add_argument(
        "--curvature",
        type=float,
        default=0.7,
        help="Curvature for circle mode (1/m)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Simulation duration (seconds)"
    )
    parser.add_argument(
        "--use-ikd",
        action="store_true",
        help="Use IKD model for control correction"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained IKD model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./simulation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save simulation data to CSV"
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Disable sensor noise"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both baseline and IKD and compare"
    )
    
    return parser.parse_args()


def run_circle_test(args, env, joystick, output_dir):
    """Run circle navigation test (Section IV-C from paper)."""
    print(f"\n{'='*80}")
    print(f"Circle Navigation Test")
    print(f"{'='*80}")
    print(f"Velocity: {args.velocity} m/s")
    print(f"Curvature: {args.curvature} 1/m")
    print(f"Expected radius: {1/args.curvature:.3f} m")
    
    # Setup
    env.setup_circle_test(args.velocity, args.curvature)
    joystick.set_circle_mode(args.velocity, args.curvature)
    
    # Run simulation
    env.start_recording()
    steps = int(args.duration / env.dt)
    
    for i in range(steps):
        # Get control from joystick
        velocity_cmd, av_cmd = joystick.get_control(
            env.vehicle.get_velocity(),
            env.vehicle.get_angular_velocity()
        )
        
        # Set control and step
        env.set_control(velocity_cmd, av_cmd)
        measurements = env.step()
        
        if i % 100 == 0:
            print(f"  t={measurements['time']:.2f}s | "
                  f"v={measurements['velocity']:.2f} | "
                  f"av={measurements['angular_velocity']:.3f}")
    
    env.stop_recording()
    
    # Analyze results
    data = env.get_recorded_data()
    measured_radius = env.measure_circle_radius()
    measured_curvature = 1.0 / measured_radius if measured_radius > 0 else 0
    
    print(f"\nResults:")
    print(f"  Commanded curvature: {args.curvature:.4f} 1/m")
    print(f"  Measured radius: {measured_radius:.3f} m")
    print(f"  Measured curvature: {measured_curvature:.4f} 1/m")
    print(f"  Error: {abs(args.curvature - measured_curvature):.4f} 1/m")
    print(f"  Deviation: {CircleMetrics.curvature_deviation_percentage(args.curvature, measured_radius):.2f}%")
    
    # Save results
    plot_trajectory(
        env.get_trajectory(),
        env.obstacles,
        title=f"Circle Navigation (v={args.velocity} m/s, c={args.curvature} 1/m)",
        save_path=str(output_dir / "circle_trajectory.png")
    )
    
    plot_simulation_data(
        data,
        title="Circle Navigation Results",
        save_path=str(output_dir / "circle_data.png")
    )
    
    if args.save_data:
        env.save_recorded_data(str(output_dir / "circle_data.csv"))
    
    return data


def run_drift_test(args, env, joystick, output_dir, test_type="loose"):
    """Run drift test (Section IV-D from paper)."""
    print(f"\n{'='*80}")
    print(f"{test_type.capitalize()} Drift Test")
    print(f"{'='*80}")
    
    # Setup environment
    if test_type == "loose":
        env.setup_loose_drift_test()
        gate_center = (3.0, 2.13 / 2)  # Midpoint between cones
        gate_width = 2.13
    else:
        env.setup_tight_drift_test()
        gate_center = (3.0, 0.81 / 2)
        gate_width = 0.81
    
    # Create drift controller with trajectory planning
    drift_controller = DriftController(
        turbo_speed=3.5, 
        drift_speed=2.5,
        use_optimizer=True  # Enable optimization for tight scenarios
    )
    
    # Plan trajectory from start to gate
    start_pos = (env.vehicle.state.x, env.vehicle.state.y)
    
    # Convert obstacles to format expected by optimizer
    obstacles_list = [(obs.x, obs.y, obs.radius) for obs in env.obstacles]
    
    drift_controller.plan_trajectory(
        start_pos=start_pos,
        gate_center=gate_center,
        gate_width=gate_width,
        direction="ccw",
        obstacles=obstacles_list
    )
    
    # Run simulation
    env.start_recording()
    steps = int(args.duration / env.dt)
    
    collision_detected = False
    success = False
    
    for i in range(steps):
        # Get vehicle state
        state = env.vehicle.get_state()
        
        # Get drift control from trajectory tracker
        velocity_cmd, av_cmd = drift_controller.update(
            state.x, state.y, state.theta, state.velocity
        )
        
        # Apply IKD correction if enabled
        if args.use_ikd:
            velocity_cmd, av_cmd = joystick.get_control(
                env.vehicle.get_velocity(),
                env.vehicle.get_angular_velocity()
            )
        
        # Set control and step
        env.set_control(velocity_cmd, av_cmd)
        measurements = env.step()
        
        # Check collision
        if env.check_collision() and not collision_detected:
            print(f"  ⚠️  Collision detected at t={measurements['time']:.2f}s!")
            collision_detected = True
        
        # Check if trajectory complete
        if drift_controller.is_complete() and not collision_detected:
            print(f"  ✅ Drift maneuver complete at t={measurements['time']:.2f}s!")
            success = True
            break
        
        if i % 100 == 0:
            print(f"  t={measurements['time']:.2f}s | "
                  f"v={measurements['velocity']:.2f} | "
                  f"av={measurements['angular_velocity']:.3f}")
    
    env.stop_recording()
    
    # Results
    data = env.get_recorded_data()
    
    print(f"\nResults:")
    print(f"  Success: {'Yes ✅' if success else 'No ❌'}")
    print(f"  Collision: {'Yes ❌' if collision_detected else 'No ✅'}")
    print(f"  Final position: ({env.vehicle.state.x:.2f}, {env.vehicle.state.y:.2f})")
    
    # Save results
    plot_trajectory(
        env.get_trajectory(),
        env.obstacles,
        title=f"{test_type.capitalize()} Drift Trajectory",
        save_path=str(output_dir / f"drift_{test_type}_trajectory.png")
    )
    
    plot_simulation_data(
        data,
        title=f"{test_type.capitalize()} Drift Results",
        save_path=str(output_dir / f"drift_{test_type}_data.png")
    )
    
    if args.save_data:
        env.save_recorded_data(str(output_dir / f"drift_{test_type}_data.csv"))
    
    return data


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("IKD Autonomous Vehicle Drifting - Simulator")
    print("Paper: Suvarna & Tehrani, 2024 (arXiv:2402.14928)")
    print("="*80)
    
    # Create environment
    env = SimulationEnvironment(
        dt=0.05,
        enable_slip=True,
        add_sensor_noise=not args.no_noise
    )
    
    # Create joystick
    joystick = VirtualJoystick()
    
    # Load IKD model if requested
    if args.use_ikd or args.compare:
        if args.model is None:
            print("[WARNING] --use-ikd specified but no model path provided.")
            print("           Using default: experiments/ikd_baseline/checkpoints/best_model.pt")
            model_path = "experiments/ikd_baseline/checkpoints/best_model.pt"
        else:
            model_path = args.model
        
        if Path(model_path).exists():
            print(f"\n[INFO] Loading IKD model from {model_path}")
            model = IKDModel(dim_input=2, dim_output=1)
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            joystick.load_ikd_model(model)
            if args.use_ikd:
                joystick.enable_ikd_correction(True)
            print("[INFO] IKD model loaded successfully")
        else:
            print(f"[ERROR] Model file not found: {model_path}")
            return
    
    # Run simulation based on mode
    if args.mode == "circle":
        if args.compare and args.model:
            # Run baseline
            print("\n[INFO] Running baseline (no IKD)...")
            joystick.enable_ikd_correction(False)
            baseline_data = run_circle_test(args, env, joystick, output_dir / "baseline")
            
            # Run IKD
            print("\n[INFO] Running with IKD correction...")
            env.reset()
            joystick.enable_ikd_correction(True)
            ikd_data = run_circle_test(args, env, joystick, output_dir / "ikd")
            
            # Compare
            compare_baseline_vs_ikd(
                baseline_data,
                ikd_data,
                save_path=str(output_dir / "comparison.png")
            )
            print(f"\n[INFO] Comparison plot saved to {output_dir / 'comparison.png'}")
        else:
            run_circle_test(args, env, joystick, output_dir)
    
    elif args.mode == "drift-loose":
        run_drift_test(args, env, joystick, output_dir, test_type="loose")
    
    elif args.mode == "drift-tight":
        run_drift_test(args, env, joystick, output_dir, test_type="tight")
    
    elif args.mode == "manual":
        print("\n[INFO] Manual mode - implement custom control logic")
        # Users can extend this for custom experiments
    
    print(f"\n{'='*80}")
    print(f"Simulation complete! Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

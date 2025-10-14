#!/usr/bin/env python3
"""
Test IKD Models in Simulation with Visualization

This script:
1. Runs simulation with and without IKD correction
2. Compares trajectories
3. Generates plots showing drift correction
4. Benchmarks IKD performance metrics

Usage:
    python test_ikd_simulation.py --model path/to/model.pt
    python test_ikd_simulation.py --compare-all  # Compare all IKD versions
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from pathlib import Path
import time
from collections import defaultdict

from src.simulator.environment import SimulationEnvironment
from src.simulator.controller import DriftController
from src.models.ikd_model import IKDModel
from src.models.ikd_model_v2 import IKDModelV2, IKDModelSimple


class IKDSimulationTester:
    """Test IKD models in simulation environment."""
    
    def __init__(self, output_dir="ikd_test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = defaultdict(list)
    
    def run_without_ikd(self, scenario="loose", duration=10.0):
        """Run simulation without IKD correction (baseline)."""
        print("\n" + "="*60)
        print("Running BASELINE (No IKD Correction)")
        print("="*60)
        
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
        
        start_pos = (env.vehicle.state.x, env.vehicle.state.y)
        controller.plan_trajectory(
            start_pos=start_pos,
            gate_center=gate_center,
            gate_width=gate_width,
            direction="ccw",
            obstacles=obstacles_list
        )
        
        # Run simulation
        steps = int(duration / env.dt)
        trajectory = []
        velocities_commanded = []
        velocities_actual = []
        errors = []
        
        collision = False
        success = False
        
        for i in range(steps):
            state = env.vehicle.get_state()
            trajectory.append([state.x, state.y, state.theta])
            
            # Get control commands
            vel_cmd, av_cmd = controller.update(
                state.x, state.y, state.theta, state.velocity
            )
            
            velocities_commanded.append(vel_cmd)
            velocities_actual.append(state.velocity)
            errors.append(vel_cmd - state.velocity)
            
            # Apply commands directly (no IKD correction)
            env.set_control(vel_cmd, av_cmd)
            env.step()
            
            if env.check_collision():
                collision = True
                break
            
            if controller.is_complete():
                success = True
                break
        
        results = {
            'trajectory': np.array(trajectory),
            'velocities_commanded': np.array(velocities_commanded),
            'velocities_actual': np.array(velocities_actual),
            'errors': np.array(errors),
            'collision': collision,
            'success': success,
            'steps': i + 1,
            'final_error': np.mean(np.abs(errors[-10:])) if len(errors) > 10 else 0
        }
        
        print(f"  Result: {'✅ Success' if success else '❌ Collision' if collision else '⚠️ Timeout'}")
        print(f"  Steps: {i+1}")
        print(f"  Final velocity error: {results['final_error']:.3f} m/s")
        
        return results
    
    def run_with_ikd(self, model, model_name, scenario="loose", duration=10.0):
        """Run simulation with IKD correction."""
        print("\n" + "="*60)
        print(f"Running WITH IKD: {model_name}")
        print("="*60)
        
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
        
        start_pos = (env.vehicle.state.x, env.vehicle.state.y)
        controller.plan_trajectory(
            start_pos=start_pos,
            gate_center=gate_center,
            gate_width=gate_width,
            direction="ccw",
            obstacles=obstacles_list
        )
        
        # Run simulation with IKD
        steps = int(duration / env.dt)
        trajectory = []
        velocities_commanded = []
        velocities_actual = []
        ikd_corrections = []
        errors = []
        
        collision = False
        success = False
        
        model.eval()
        
        for i in range(steps):
            state = env.vehicle.get_state()
            trajectory.append([state.x, state.y, state.theta])
            
            # Get control commands
            vel_cmd, av_cmd = controller.update(
                state.x, state.y, state.theta, state.velocity
            )
            
            velocities_commanded.append(vel_cmd)
            velocities_actual.append(state.velocity)
            
            # Apply IKD correction
            with torch.no_grad():
                # Prepare input based on model type
                if isinstance(model, IKDModel):
                    # V1 model: just velocity and angular velocity
                    model_input = torch.FloatTensor([vel_cmd, av_cmd]).unsqueeze(0)
                    correction = model(model_input).item()
                    corrected_vel = vel_cmd + correction
                    corrected_av = av_cmd
                else:
                    # V2 models: expanded features
                    model_input = self._prepare_v2_input(state, vel_cmd, av_cmd, env)
                    output = model(model_input)
                    if output.shape[-1] == 2:
                        # Multi-output model
                        corrected_vel = output[0, 0].item()
                        corrected_av = output[0, 1].item()
                        correction = corrected_vel - vel_cmd
                    else:
                        # Single output (velocity only)
                        correction = output[0, 0].item()
                        corrected_vel = vel_cmd + correction
                        corrected_av = av_cmd
            
            ikd_corrections.append(correction)
            errors.append(vel_cmd - state.velocity)
            
            # Apply corrected commands
            env.set_control(corrected_vel, corrected_av)
            env.step()
            
            if env.check_collision():
                collision = True
                break
            
            if controller.is_complete():
                success = True
                break
        
        results = {
            'trajectory': np.array(trajectory),
            'velocities_commanded': np.array(velocities_commanded),
            'velocities_actual': np.array(velocities_actual),
            'ikd_corrections': np.array(ikd_corrections),
            'errors': np.array(errors),
            'collision': collision,
            'success': success,
            'steps': i + 1,
            'final_error': np.mean(np.abs(errors[-10:])) if len(errors) > 10 else 0,
            'avg_correction': np.mean(np.abs(ikd_corrections))
        }
        
        print(f"  Result: {'✅ Success' if success else '❌ Collision' if collision else '⚠️ Timeout'}")
        print(f"  Steps: {i+1}")
        print(f"  Final velocity error: {results['final_error']:.3f} m/s")
        print(f"  Avg IKD correction: {results['avg_correction']:.3f} m/s")
        
        return results
    
    def _prepare_v2_input(self, state, vel_cmd, av_cmd, env):
        """Prepare 10-dimensional input for V2 models."""
        # Calculate distance to nearest obstacle
        min_dist = 10.0
        for obs in env.obstacles:
            dist = np.sqrt((state.x - obs.x)**2 + (state.y - obs.y)**2)
            min_dist = min(min_dist, dist)
        
        # Slip indicator (simple approximation)
        expected_av = state.velocity * np.tan(state.steering_angle) / 0.324
        slip = abs(state.angular_velocity - expected_av)
        
        features = [
            state.velocity,
            state.angular_velocity,
            0.0,  # acceleration (would need history)
            state.steering_angle,
            state.x,
            state.y,
            np.cos(state.theta),
            np.sin(state.theta),
            slip,
            min_dist
        ]
        
        return torch.FloatTensor(features).unsqueeze(0)
    
    def compare_results(self, baseline, ikd_results, model_names):
        """Generate comprehensive comparison plots."""
        print("\n" + "="*60)
        print("Generating Comparison Plots")
        print("="*60)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Trajectory comparison
        ax1 = plt.subplot(2, 3, 1)
        self._plot_trajectories(ax1, baseline, ikd_results, model_names)
        
        # 2. Velocity tracking
        ax2 = plt.subplot(2, 3, 2)
        self._plot_velocity_tracking(ax2, baseline, ikd_results, model_names)
        
        # 3. Velocity errors
        ax3 = plt.subplot(2, 3, 3)
        self._plot_velocity_errors(ax3, baseline, ikd_results, model_names)
        
        # 4. IKD corrections over time
        ax4 = plt.subplot(2, 3, 4)
        self._plot_ikd_corrections(ax4, ikd_results, model_names)
        
        # 5. Error distribution
        ax5 = plt.subplot(2, 3, 5)
        self._plot_error_distribution(ax5, baseline, ikd_results, model_names)
        
        # 6. Performance metrics bar chart
        ax6 = plt.subplot(2, 3, 6)
        self._plot_performance_metrics(ax6, baseline, ikd_results, model_names)
        
        plt.tight_layout()
        output_path = self.output_dir / "ikd_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved comparison plot: {output_path}")
        plt.close()
        
        # Create detailed trajectory plot
        self._plot_detailed_trajectories(baseline, ikd_results, model_names)
    
    def _plot_trajectories(self, ax, baseline, ikd_results, model_names):
        """Plot trajectory comparison."""
        ax.plot(baseline['trajectory'][:, 0], baseline['trajectory'][:, 1], 
                'r--', linewidth=2, label='Baseline (No IKD)', alpha=0.7)
        
        colors = ['blue', 'green', 'orange', 'purple']
        for i, (result, name) in enumerate(zip(ikd_results, model_names)):
            ax.plot(result['trajectory'][:, 0], result['trajectory'][:, 1],
                   color=colors[i % len(colors)], linewidth=2, 
                   label=f'With {name}', alpha=0.7)
        
        ax.set_xlabel('X Position (m)', fontsize=11)
        ax.set_ylabel('Y Position (m)', fontsize=11)
        ax.set_title('Trajectory Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_velocity_tracking(self, ax, baseline, ikd_results, model_names):
        """Plot commanded vs actual velocity."""
        time_baseline = np.arange(len(baseline['velocities_commanded'])) * 0.05
        
        ax.plot(time_baseline, baseline['velocities_commanded'], 
                'k--', linewidth=1.5, label='Commanded', alpha=0.8)
        ax.plot(time_baseline, baseline['velocities_actual'],
                'r-', linewidth=2, label='Actual (No IKD)', alpha=0.7)
        
        colors = ['blue', 'green', 'orange', 'purple']
        for i, (result, name) in enumerate(zip(ikd_results, model_names)):
            time_ikd = np.arange(len(result['velocities_actual'])) * 0.05
            ax.plot(time_ikd, result['velocities_actual'],
                   color=colors[i % len(colors)], linewidth=2,
                   label=f'Actual ({name})', alpha=0.7)
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Velocity (m/s)', fontsize=11)
        ax.set_title('Velocity Tracking', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_velocity_errors(self, ax, baseline, ikd_results, model_names):
        """Plot velocity tracking errors over time."""
        time_baseline = np.arange(len(baseline['errors'])) * 0.05
        ax.plot(time_baseline, np.abs(baseline['errors']),
                'r-', linewidth=2, label='Baseline', alpha=0.7)
        
        colors = ['blue', 'green', 'orange', 'purple']
        for i, (result, name) in enumerate(zip(ikd_results, model_names)):
            time_ikd = np.arange(len(result['errors'])) * 0.05
            ax.plot(time_ikd, np.abs(result['errors']),
                   color=colors[i % len(colors)], linewidth=2,
                   label=name, alpha=0.7)
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Absolute Velocity Error (m/s)', fontsize=11)
        ax.set_title('Velocity Tracking Error', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_ikd_corrections(self, ax, ikd_results, model_names):
        """Plot IKD corrections over time."""
        colors = ['blue', 'green', 'orange', 'purple']
        for i, (result, name) in enumerate(zip(ikd_results, model_names)):
            time_ikd = np.arange(len(result['ikd_corrections'])) * 0.05
            ax.plot(time_ikd, result['ikd_corrections'],
                   color=colors[i % len(colors)], linewidth=2,
                   label=name, alpha=0.7)
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('IKD Correction (m/s)', fontsize=11)
        ax.set_title('IKD Corrections Over Time', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_error_distribution(self, ax, baseline, ikd_results, model_names):
        """Plot error distribution as box plots."""
        data = [np.abs(baseline['errors'])]
        labels = ['Baseline']
        
        for result, name in zip(ikd_results, model_names):
            data.append(np.abs(result['errors']))
            labels.append(name)
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('Absolute Velocity Error (m/s)', fontsize=11)
        ax.set_title('Error Distribution', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    def _plot_performance_metrics(self, ax, baseline, ikd_results, model_names):
        """Plot performance metrics bar chart."""
        metrics = ['Final Error', 'Avg Correction', 'Success']
        
        x = np.arange(len(model_names) + 1)
        width = 0.25
        
        # Final errors
        final_errors = [baseline['final_error']]
        for result in ikd_results:
            final_errors.append(result['final_error'])
        
        # Average corrections
        avg_corrections = [0]  # Baseline has no correction
        for result in ikd_results:
            avg_corrections.append(result['avg_correction'])
        
        # Success
        success_values = [1 if baseline['success'] else 0]
        for result in ikd_results:
            success_values.append(1 if result['success'] else 0)
        
        ax.bar(x - width, final_errors, width, label='Final Error (m/s)', alpha=0.8)
        ax.bar(x, avg_corrections, width, label='Avg Correction (m/s)', alpha=0.8)
        ax.bar(x + width, success_values, width, label='Success (0/1)', alpha=0.8)
        
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title('Performance Metrics', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Baseline'] + model_names, rotation=15, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_detailed_trajectories(self, baseline, ikd_results, model_names):
        """Create detailed trajectory plot with obstacles."""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot baseline
        ax.plot(baseline['trajectory'][:, 0], baseline['trajectory'][:, 1],
                'r--', linewidth=3, label='Baseline (No IKD)', alpha=0.7, zorder=2)
        
        # Plot IKD results
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
        for i, (result, name) in enumerate(zip(ikd_results, model_names)):
            ax.plot(result['trajectory'][:, 0], result['trajectory'][:, 1],
                   color=colors[i % len(colors)], linewidth=3,
                   label=f'With {name}', alpha=0.8, zorder=3)
        
        # Add start/goal markers
        ax.plot(0, 0, 'go', markersize=15, label='Start', zorder=4)
        ax.axvline(x=3.0, color='lime', linewidth=4, label='Goal Gate', alpha=0.6, zorder=1)
        
        ax.set_xlabel('X Position (m)', fontsize=13)
        ax.set_ylabel('Y Position (m)', fontsize=13)
        ax.set_title('Detailed Trajectory Comparison\n(Does IKD Correct the Drift?)', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        output_path = self.output_dir / "ikd_trajectories_detailed.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved detailed trajectory plot: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test IKD in Simulation")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained IKD model"
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare all available IKD models"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="loose",
        choices=["loose", "tight"],
        help="Test scenario"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Simulation duration (seconds)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("#  IKD Simulation Testing & Benchmarking")
    print("#" + "#"*60)
    
    tester = IKDSimulationTester()
    
    # Run baseline
    baseline_results = tester.run_without_ikd(args.scenario, args.duration)
    
    # Run with IKD models
    ikd_results = []
    model_names = []
    
    if args.compare_all:
        # Test all available models
        model_paths = [
            ("IKD V1", "trained_models/ikd_model.pt"),
            ("IKD V2 Simple", "trained_models/ikd_v2_simple.pt"),
            ("IKD V2 LSTM", "trained_models/ikd_v2_lstm.pt"),
        ]
        
        for name, path in model_paths:
            model_file = Path(path)
            if model_file.exists():
                print(f"\nLoading {name} from {path}")
                if "v2" in name.lower():
                    if "lstm" in name.lower():
                        model = IKDModelV2(input_dim=10, output_dim=2, use_lstm=True)
                    else:
                        model = IKDModelSimple(input_dim=10, output_dim=2)
                else:
                    model = IKDModel()
                
                model.load_state_dict(torch.load(path))
                result = tester.run_with_ikd(model, name, args.scenario, args.duration)
                ikd_results.append(result)
                model_names.append(name)
            else:
                print(f"⚠️  Model not found: {path}")
    
    elif args.model:
        # Test specific model
        model_path = Path(args.model)
        if model_path.exists():
            print(f"\nLoading model from {args.model}")
            # Try to infer model type from name
            if "v2" in args.model.lower():
                if "lstm" in args.model.lower():
                    model = IKDModelV2(input_dim=10, output_dim=2, use_lstm=True)
                else:
                    model = IKDModelSimple(input_dim=10, output_dim=2)
            else:
                model = IKDModel()
            
            model.load_state_dict(torch.load(args.model))
            result = tester.run_with_ikd(model, "IKD", args.scenario, args.duration)
            ikd_results.append(result)
            model_names.append("IKD")
        else:
            print(f"❌ Model not found: {args.model}")
            return
    else:
        print("\n⚠️  No model specified. Use --model or --compare-all")
        print("Running baseline only.")
    
    # Generate comparison plots
    if ikd_results:
        tester.compare_results(baseline_results, ikd_results, model_names)
        
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        print(f"\nBaseline (No IKD):")
        print(f"  Final Error: {baseline_results['final_error']:.3f} m/s")
        print(f"  Success: {baseline_results['success']}")
        
        for name, result in zip(model_names, ikd_results):
            print(f"\n{name}:")
            print(f"  Final Error: {result['final_error']:.3f} m/s")
            print(f"  Avg Correction: {result['avg_correction']:.3f} m/s")
            print(f"  Success: {result['success']}")
            
            # Calculate improvement
            improvement = (baseline_results['final_error'] - result['final_error']) / baseline_results['final_error'] * 100
            print(f"  Improvement: {improvement:+.1f}%")
    
    print("\n" + "="*60)
    print("Testing complete! Results saved to ikd_test_results/")
    print("="*60)


if __name__ == "__main__":
    main()

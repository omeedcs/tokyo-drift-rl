#!/usr/bin/env python3
"""
Comprehensive benchmarking suite for drift control approaches.

Compares:
1. Baseline heuristic planner
2. Optimization-based planner  
3. IKD Model V1
4. IKD Model V2 (Simple)
5. IKD Model V2 (LSTM)
6. SAC Reinforcement Learning

Generates detailed comparison plots and metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple
import torch

# Simulator imports
from src.simulator.environment import SimulationEnvironment
from src.simulator.controller import DriftController
from src.models.ikd_model import IKDModel
from src.models.ikd_model_v2 import IKDModelV2, IKDModelSimple

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class BenchmarkRunner:
    """Run comprehensive benchmarks across all approaches."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize benchmark runner.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
    
    def benchmark_heuristic_planner(
        self,
        scenario: str = "loose",
        num_trials: int = 20
    ) -> Dict:
        """Benchmark heuristic trajectory planner."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: Heuristic Planner ({scenario})")
        print(f"{'='*60}")
        
        results = {
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "avg_completion_time": 0.0,
            "avg_final_distance": 0.0,
            "trajectories": [],
            "computation_times": []
        }
        
        successes = 0
        collisions = 0
        completion_times = []
        final_distances = []
        
        for trial in range(num_trials):
            env = SimulationEnvironment(dt=0.05)
            
            if scenario == "loose":
                env.setup_loose_drift_test()
                gate_center = (3.0, 1.065)
                gate_width = 2.13
            else:
                env.setup_tight_drift_test()
                gate_center = (3.0, 0.405)
                gate_width = 0.81
            
            # Create controller WITHOUT optimizer (heuristic)
            start_time = time.time()
            controller = DriftController(use_optimizer=False)
            controller.plan_trajectory(
                start_pos=(0.0, 0.0),
                gate_center=gate_center,
                gate_width=gate_width,
                direction="ccw",
                obstacles=None
            )
            comp_time = time.time() - start_time
            results["computation_times"].append(comp_time)
            
            # Run simulation
            collision = False
            trajectory = []
            
            for step in range(500):
                state = env.vehicle.get_state()
                trajectory.append((state.x, state.y))
                
                vel_cmd, av_cmd = controller.update(
                    state.x, state.y, state.theta, state.velocity
                )
                
                env.set_control(vel_cmd, av_cmd)
                env.step()
                
                if env.check_collision():
                    collision = True
                    break
                
                if controller.is_complete():
                    break
            
            # Record results
            if collision:
                collisions += 1
            elif controller.is_complete():
                successes += 1
                completion_times.append(step * env.dt)
            
            final_dist = np.sqrt(
                (state.x - gate_center[0])**2 + 
                (state.y - gate_center[1])**2
            )
            final_distances.append(final_dist)
            results["trajectories"].append(trajectory)
            
            print(f"  Trial {trial+1}/{num_trials}: "
                  f"{'✅ Success' if not collision and controller.is_complete() else '❌ Fail'}")
        
        results["success_rate"] = successes / num_trials
        results["collision_rate"] = collisions / num_trials
        results["avg_completion_time"] = np.mean(completion_times) if completion_times else 0.0
        results["avg_final_distance"] = np.mean(final_distances)
        results["std_final_distance"] = np.std(final_distances)
        results["avg_computation_time"] = np.mean(results["computation_times"])
        
        print(f"\nResults:")
        print(f"  Success Rate: {results['success_rate']*100:.1f}%")
        print(f"  Collision Rate: {results['collision_rate']*100:.1f}%")
        print(f"  Avg Completion Time: {results['avg_completion_time']:.2f}s")
        print(f"  Avg Final Distance: {results['avg_final_distance']:.3f}m")
        print(f"  Avg Computation Time: {results['avg_computation_time']*1000:.2f}ms")
        
        return results
    
    def benchmark_optimized_planner(
        self,
        scenario: str = "loose",
        num_trials: int = 20
    ) -> Dict:
        """Benchmark optimization-based planner."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: Optimized Planner ({scenario})")
        print(f"{'='*60}")
        
        results = {
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "avg_completion_time": 0.0,
            "avg_final_distance": 0.0,
            "trajectories": [],
            "computation_times": []
        }
        
        successes = 0
        collisions = 0
        completion_times = []
        final_distances = []
        
        for trial in range(num_trials):
            env = SimulationEnvironment(dt=0.05)
            
            if scenario == "loose":
                env.setup_loose_drift_test()
                gate_center = (3.0, 1.065)
                gate_width = 2.13
            else:
                env.setup_tight_drift_test()
                gate_center = (3.0, 0.405)
                gate_width = 0.81
            
            # Create controller WITH optimizer
            start_time = time.time()
            controller = DriftController(use_optimizer=True)
            obstacles_list = [(obs.x, obs.y, obs.radius) for obs in env.obstacles]
            controller.plan_trajectory(
                start_pos=(0.0, 0.0),
                gate_center=gate_center,
                gate_width=gate_width,
                direction="ccw",
                obstacles=obstacles_list
            )
            comp_time = time.time() - start_time
            results["computation_times"].append(comp_time)
            
            # Run simulation
            collision = False
            trajectory = []
            
            for step in range(500):
                state = env.vehicle.get_state()
                trajectory.append((state.x, state.y))
                
                vel_cmd, av_cmd = controller.update(
                    state.x, state.y, state.theta, state.velocity
                )
                
                env.set_control(vel_cmd, av_cmd)
                env.step()
                
                if env.check_collision():
                    collision = True
                    break
                
                if controller.is_complete():
                    break
            
            # Record results
            if collision:
                collisions += 1
            elif controller.is_complete():
                successes += 1
                completion_times.append(step * env.dt)
            
            final_dist = np.sqrt(
                (state.x - gate_center[0])**2 + 
                (state.y - gate_center[1])**2
            )
            final_distances.append(final_dist)
            results["trajectories"].append(trajectory)
            
            print(f"  Trial {trial+1}/{num_trials}: "
                  f"{'✅ Success' if not collision and controller.is_complete() else '❌ Fail'}")
        
        results["success_rate"] = successes / num_trials
        results["collision_rate"] = collisions / num_trials
        results["avg_completion_time"] = np.mean(completion_times) if completion_times else 0.0
        results["avg_final_distance"] = np.mean(final_distances)
        results["std_final_distance"] = np.std(final_distances)
        results["avg_computation_time"] = np.mean(results["computation_times"])
        
        print(f"\nResults:")
        print(f"  Success Rate: {results['success_rate']*100:.1f}%")
        print(f"  Collision Rate: {results['collision_rate']*100:.1f}%")
        print(f"  Avg Completion Time: {results['avg_completion_time']:.2f}s")
        print(f"  Avg Final Distance: {results['avg_final_distance']:.3f}m")
        print(f"  Avg Computation Time: {results['avg_computation_time']*1000:.2f}ms")
        
        return results
    
    def run_full_benchmark(self):
        """Run all benchmarks."""
        print(f"\n{'#'*60}")
        print(f"#  COMPREHENSIVE BENCHMARKING SUITE")
        print(f"#{'#'*60}\n")
        
        # Benchmark loose scenarios
        self.results["heuristic_loose"] = self.benchmark_heuristic_planner("loose", num_trials=10)
        self.results["optimized_loose"] = self.benchmark_optimized_planner("loose", num_trials=10)
        
        # Benchmark tight scenarios
        self.results["heuristic_tight"] = self.benchmark_heuristic_planner("tight", num_trials=10)
        self.results["optimized_tight"] = self.benchmark_optimized_planner("tight", num_trials=10)
        
        # Save results
        self.save_results()
        
        # Generate plots
        self.generate_plots()
        
        print(f"\n{'='*60}")
        print(f"Benchmarking Complete!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def save_results(self):
        """Save results to JSON."""
        # Convert trajectories to serializable format
        results_copy = {}
        for key, value in self.results.items():
            results_copy[key] = {
                "success_rate": value["success_rate"],
                "collision_rate": value["collision_rate"],
                "avg_completion_time": value["avg_completion_time"],
                "avg_final_distance": value["avg_final_distance"],
                "std_final_distance": value.get("std_final_distance", 0.0),
                "avg_computation_time": value.get("avg_computation_time", 0.0)
            }
        
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"[INFO] Saved results to {self.output_dir / 'results.json'}")
    
    def generate_plots(self):
        """Generate comparison plots."""
        print(f"\n[INFO] Generating plots...")
        
        # Plot 1: Success rates comparison
        self.plot_success_rates()
        
        # Plot 2: Computation time comparison
        self.plot_computation_times()
        
        # Plot 3: Trajectory visualizations
        self.plot_trajectories()
        
        # Plot 4: Performance heatmap
        self.plot_performance_heatmap()
        
        print(f"[INFO] Plots saved to {self.output_dir}")
    
    def plot_success_rates(self):
        """Plot success rates comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Loose scenario
        methods = ["Heuristic", "Optimized"]
        success_rates_loose = [
            self.results["heuristic_loose"]["success_rate"] * 100,
            self.results["optimized_loose"]["success_rate"] * 100
        ]
        
        ax1.bar(methods, success_rates_loose, color=['#3498db', '#2ecc71'], alpha=0.8)
        ax1.set_ylabel("Success Rate (%)", fontsize=12)
        ax1.set_title("Loose Drift Scenario\n(Gate Width: 2.13m)", fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 105])
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(success_rates_loose):
            ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # Tight scenario
        success_rates_tight = [
            self.results["heuristic_tight"]["success_rate"] * 100,
            self.results["optimized_tight"]["success_rate"] * 100
        ]
        
        ax2.bar(methods, success_rates_tight, color=['#3498db', '#2ecc71'], alpha=0.8)
        ax2.set_ylabel("Success Rate (%)", fontsize=12)
        ax2.set_title("Tight Drift Scenario\n(Gate Width: 0.81m)", fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 105])
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(success_rates_tight):
            ax2.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "success_rates_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_computation_times(self):
        """Plot computation time comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ["Heuristic\n(Loose)", "Optimized\n(Loose)", 
                   "Heuristic\n(Tight)", "Optimized\n(Tight)"]
        times = [
            self.results["heuristic_loose"]["avg_computation_time"] * 1000,
            self.results["optimized_loose"]["avg_computation_time"] * 1000,
            self.results["heuristic_tight"]["avg_computation_time"] * 1000,
            self.results["optimized_tight"]["avg_computation_time"] * 1000
        ]
        
        bars = ax.bar(methods, times, color=['#3498db', '#2ecc71', '#3498db', '#2ecc71'], alpha=0.8)
        ax.set_ylabel("Computation Time (ms)", fontsize=12)
        ax.set_title("Planning Computation Time Comparison", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, v) in enumerate(zip(bars, times)):
            ax.text(bar.get_x() + bar.get_width()/2, v + max(times)*0.02, 
                   f'{v:.1f}ms', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "computation_times.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_trajectories(self):
        """Plot sample trajectories."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
        
        scenarios = [
            ("heuristic_loose", ax1, "Heuristic - Loose (2.13m gate)"),
            ("optimized_loose", ax2, "Optimized - Loose (2.13m gate)"),
            ("heuristic_tight", ax3, "Heuristic - Tight (0.81m gate)"),
            ("optimized_tight", ax4, "Optimized - Tight (0.81m gate)")
        ]
        
        for result_key, ax, title in scenarios:
            # Plot first 3 trajectories
            for i, traj in enumerate(self.results[result_key]["trajectories"][:3]):
                traj = np.array(traj)
                ax.plot(traj[:, 0], traj[:, 1], alpha=0.6, linewidth=2, label=f'Trial {i+1}')
            
            # Plot gate
            if "loose" in result_key:
                gate_y = [0, 2.13]
                gate_x = 3.0
            else:
                gate_y = [0, 0.81]
                gate_x = 3.0
            
            ax.plot([gate_x, gate_x], gate_y, 'r-', linewidth=4, label='Gate')
            ax.plot(0, 0, 'go', markersize=10, label='Start')
            
            ax.set_xlabel("X Position (m)", fontsize=11)
            ax.set_ylabel("Y Position (m)", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "trajectories_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_heatmap(self):
        """Plot performance metrics heatmap."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        methods = ["Heuristic", "Optimized"]
        scenarios = ["Loose\n(Success %)", "Loose\n(Comp Time ms)", 
                    "Tight\n(Success %)", "Tight\n(Comp Time ms)"]
        
        data = np.array([
            [
                self.results["heuristic_loose"]["success_rate"] * 100,
                self.results["heuristic_loose"]["avg_computation_time"] * 1000,
                self.results["heuristic_tight"]["success_rate"] * 100,
                self.results["heuristic_tight"]["avg_computation_time"] * 1000
            ],
            [
                self.results["optimized_loose"]["success_rate"] * 100,
                self.results["optimized_loose"]["avg_computation_time"] * 1000,
                self.results["optimized_tight"]["success_rate"] * 100,
                self.results["optimized_tight"]["avg_computation_time"] * 1000
            ]
        ])
        
        # Normalize for visualization (not for display values)
        data_norm = data.copy()
        data_norm[:, [0, 2]] /= 100  # Success rates
        data_norm[:, [1, 3]] /= data_norm[:, [1, 3]].max()  # Comp times
        
        im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(scenarios, fontsize=11)
        ax.set_yticklabels(methods, fontsize=12)
        
        # Add text annotations with actual values
        for i in range(len(methods)):
            for j in range(len(scenarios)):
                text = ax.text(j, i, f'{data[i, j]:.1f}',
                             ha="center", va="center", color="black", 
                             fontsize=14, fontweight='bold')
        
        ax.set_title("Performance Metrics Heatmap\n(Green=Better)", 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Run benchmarking."""
    runner = BenchmarkRunner(output_dir="benchmark_results")
    runner.run_full_benchmark()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare all drift control methods:
1. Baseline (controller only)
2. IKD (controller + inverse dynamics)
3. SAC (learned policy)

Generates comprehensive comparison plots and statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from pathlib import Path
from collections import defaultdict

from src.simulator.environment import SimulationEnvironment
from src.simulator.controller import DriftController
from src.models.ikd_model import IKDModel


class MethodComparison:
    """Compare different drift control methods."""
    
    def __init__(self, scenario="loose", num_trials=10):
        self.scenario = scenario
        self.num_trials = num_trials
        self.results = {}
        
        # Setup
        if scenario == "loose":
            self.gate_center = (3.0, 1.065)
            self.gate_width = 2.13
        else:
            self.gate_center = (3.0, 0.405)
            self.gate_width = 0.81
    
    def test_baseline(self):
        """Test baseline controller."""
        print("\n" + "="*60)
        print("Testing BASELINE (Controller Only)")
        print("="*60)
        
        metrics = self._run_trials(use_ikd=False, ikd_model=None, use_sac=False)
        self.results['Baseline'] = metrics
        
        print(f"\n  Average reward: {metrics['avg_reward']:.2f}")
        print(f"  Success rate: {metrics['success_rate']:.1%}")
        print(f"  Average steps: {metrics['avg_steps']:.1f}")
    
    def test_ikd(self, model_path="trained_models/ikd_final.pt"):
        """Test IKD controller."""
        print("\n" + "="*60)
        print("Testing IKD (Controller + Inverse Dynamics)")
        print("="*60)
        
        # Load IKD model
        model = IKDModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        metrics = self._run_trials(use_ikd=True, ikd_model=model, use_sac=False)
        self.results['IKD'] = metrics
        
        print(f"\n  Average reward: {metrics['avg_reward']:.2f}")
        print(f"  Success rate: {metrics['success_rate']:.1%}")
        print(f"  Average steps: {metrics['avg_steps']:.1f}")
    
    def test_sac(self, model_path="trained_agents/sac/model.pt"):
        """Test SAC policy."""
        print("\n" + "="*60)
        print("Testing SAC (Learned Policy)")
        print("="*60)
        
        # TODO: Load SAC model and test
        # For now, placeholder
        print("  SAC testing not yet implemented")
        metrics = {
            'avg_reward': 0,
            'success_rate': 0,
            'avg_steps': 0,
            'rewards': [],
            'steps': [],
            'successes': []
        }
        self.results['SAC'] = metrics
    
    def _run_trials(self, use_ikd=False, ikd_model=None, use_sac=False, sac_agent=None):
        """Run multiple trials and collect metrics."""
        rewards = []
        steps_list = []
        successes = []
        
        for trial in range(self.num_trials):
            env = SimulationEnvironment(dt=0.05)
            
            if self.scenario == "loose":
                env.setup_loose_drift_test()
            else:
                env.setup_tight_drift_test()
            
            controller = DriftController(use_optimizer=False)
            obstacles_list = [(obs.x, obs.y, obs.radius) for obs in env.obstacles]
            
            controller.plan_trajectory(
                start_pos=(0.0, 0.0),
                gate_center=self.gate_center,
                gate_width=self.gate_width,
                direction="ccw",
                obstacles=obstacles_list
            )
            
            # Run episode
            total_reward = 0
            steps = 0
            
            for step in range(200):
                state = env.vehicle.get_state()
                
                if use_sac:
                    # Use SAC policy
                    pass  # TODO
                else:
                    # Use controller
                    vel_cmd, av_cmd = controller.update(
                        state.x, state.y, state.theta, state.velocity
                    )
                    
                    if use_ikd and ikd_model:
                        # Apply IKD correction
                        with torch.no_grad():
                            model_input = torch.FloatTensor([vel_cmd, av_cmd]).unsqueeze(0)
                            correction = ikd_model(model_input).item()
                            vel_cmd = vel_cmd + correction
                    
                    env.set_control(vel_cmd, av_cmd)
                
                env.step()
                steps += 1
                
                # Compute reward (distance to goal)
                dist_to_goal = np.sqrt(
                    (state.x - self.gate_center[0])**2 + 
                    (state.y - self.gate_center[1])**2
                )
                reward = -dist_to_goal
                total_reward += reward
                
                if env.check_collision():
                    successes.append(0)
                    break
                
                if controller.is_complete():
                    successes.append(1)
                    break
            
            rewards.append(total_reward)
            steps_list.append(steps)
            
            if len(successes) < trial + 1:
                successes.append(0)  # Timeout
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_steps': np.mean(steps_list),
            'std_steps': np.std(steps_list),
            'success_rate': np.mean(successes),
            'rewards': rewards,
            'steps': steps_list,
            'successes': successes
        }
    
    def generate_comparison_plots(self, save_dir="comparison_results"):
        """Generate comparison plots."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("Generating Comparison Plots")
        print("="*60)
        
        methods = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Average reward comparison
        ax = axes[0, 0]
        rewards = [self.results[m]['avg_reward'] for m in methods]
        errors = [self.results[m]['std_reward'] for m in methods]
        colors = ['red', 'blue', 'green']
        
        ax.bar(methods, rewards, yerr=errors, capsize=5, color=colors, alpha=0.7)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.set_title('Reward Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Success rate comparison
        ax = axes[0, 1]
        success_rates = [self.results[m]['success_rate'] * 100 for m in methods]
        ax.bar(methods, success_rates, color=colors, alpha=0.7)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Average steps comparison
        ax = axes[1, 0]
        steps = [self.results[m]['avg_steps'] for m in methods]
        errors = [self.results[m]['std_steps'] for m in methods]
        ax.bar(methods, steps, yerr=errors, capsize=5, color=colors, alpha=0.7)
        ax.set_ylabel('Average Steps', fontsize=12)
        ax.set_title('Efficiency Comparison (Lower = Better)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Reward distributions
        ax = axes[1, 1]
        reward_data = [self.results[m]['rewards'] for m in methods]
        bp = ax.boxplot(reward_data, labels=methods, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Reward Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = save_dir / "method_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {plot_path}")
        plt.close()
    
    def save_results_table(self, save_dir="comparison_results"):
        """Save results as JSON and markdown table."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Save JSON
        json_path = save_dir / "results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        print(f"  ✅ Saved: {json_path}")
        
        # Create markdown table
        md_path = save_dir / "RESULTS.md"
        with open(md_path, 'w') as f:
            f.write("# Drift Control Method Comparison\n\n")
            f.write(f"Scenario: {self.scenario}\n")
            f.write(f"Trials per method: {self.num_trials}\n\n")
            
            f.write("## Results Summary\n\n")
            f.write("| Method | Avg Reward | Success Rate | Avg Steps | Improvement |\n")
            f.write("|--------|------------|--------------|-----------|-------------|\n")
            
            baseline_reward = self.results['Baseline']['avg_reward']
            
            for method in self.results.keys():
                r = self.results[method]
                improvement = ((r['avg_reward'] - baseline_reward) / abs(baseline_reward)) * 100
                
                f.write(f"| {method:10s} | {r['avg_reward']:10.2f} | "
                       f"{r['success_rate']*100:6.1f}% | "
                       f"{r['avg_steps']:9.1f} | "
                       f"{improvement:+7.1f}% |\n")
            
            f.write("\n## Detailed Statistics\n\n")
            for method, stats in self.results.items():
                f.write(f"### {method}\n\n")
                f.write(f"- Average Reward: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}\n")
                f.write(f"- Success Rate: {stats['success_rate']*100:.1f}%\n")
                f.write(f"- Average Steps: {stats['avg_steps']:.1f} ± {stats['std_steps']:.1f}\n")
                f.write(f"- Successes: {sum(stats['successes'])}/{len(stats['successes'])}\n\n")
        
        print(f"  ✅ Saved: {md_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare all drift control methods")
    parser.add_argument("--scenario", type=str, default="loose",
                        choices=["loose", "tight"])
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of trials per method")
    parser.add_argument("--ikd-model", type=str, default="trained_models/ikd_final.pt")
    parser.add_argument("--sac-model", type=str, default="trained_agents/sac/model.pt")
    parser.add_argument("--save-dir", type=str, default="comparison_results")
    
    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("#  Drift Control Method Comparison")
    print("#" + "#"*60)
    print(f"\nScenario: {args.scenario}")
    print(f"Trials per method: {args.trials}")
    
    # Create comparison
    comparison = MethodComparison(
        scenario=args.scenario,
        num_trials=args.trials
    )
    
    # Test all methods
    comparison.test_baseline()
    
    if Path(args.ikd_model).exists():
        comparison.test_ikd(args.ikd_model)
    else:
        print(f"\n⚠️  IKD model not found: {args.ikd_model}")
    
    if Path(args.sac_model).exists():
        comparison.test_sac(args.sac_model)
    else:
        print(f"\n⚠️  SAC model not found: {args.sac_model} (will train first)")
    
    # Generate comparison
    comparison.generate_comparison_plots(args.save_dir)
    comparison.save_results_table(args.save_dir)
    
    print("\n" + "="*60)
    print("Comparison Complete!")
    print("="*60)
    print(f"\nResults saved to: {args.save_dir}/")
    print("\nView results:")
    print(f"  - Plots: {args.save_dir}/method_comparison.png")
    print(f"  - Table: {args.save_dir}/RESULTS.md")


if __name__ == "__main__":
    main()

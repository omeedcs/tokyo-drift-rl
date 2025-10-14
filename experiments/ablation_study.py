"""
Ablation Study Framework

Systematically evaluates the impact of each "research-grade" feature:
- Noisy sensors (GPS + IMU)
- Perception pipeline (object detection)
- Latency modeling
- 3D dynamics
- Moving agents

Each feature is added incrementally and evaluated to measure its contribution.

Output:
- Learning curves for each configuration
- Final performance comparison
- Quantified impact of each feature
- Recommendations for future work

Usage:
    python experiments/ablation_study.py --algorithm SAC --seeds 3
"""

import argparse
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.benchmark_algorithms import train_algorithm, evaluate_model, ENV_CONFIGS
from experiments.evaluation import EvaluationMetrics, compare_metrics


# Ablation configurations (incremental feature addition)
ABLATION_CONFIGS = [
    {
        'name': '1_baseline',
        'description': 'Perfect sensors, no advanced features',
        'config': 'baseline'
    },
    {
        'name': '2_+sensors',
        'description': '+ Noisy GPS/IMU sensors',
        'config': '+sensors'
    },
    {
        'name': '3_+perception',
        'description': '+ Object detection pipeline',
        'config': '+perception'
    },
    {
        'name': '4_+latency',
        'description': '+ Sensor/actuation latency',
        'config': '+latency'
    },
    {
        'name': '5_full',
        'description': '+ 3D dynamics + moving agents',
        'config': 'full'
    }
]


def run_ablation_study(
    algorithm: str = "SAC",
    seeds: int = 3,
    total_timesteps: int = 500000,
    n_eval_episodes: int = 100,
    save_dir: str = "experiments/results/ablation"
):
    """
    Run complete ablation study.
    
    Args:
        algorithm: Algorithm to use (SAC, PPO, TD3)
        seeds: Number of random seeds per configuration
        total_timesteps: Training timesteps
        n_eval_episodes: Evaluation episodes
        save_dir: Save directory
        
    Returns:
        Dict of results
    """
    print("\n" + "="*70)
    print("ABLATION STUDY: Measuring Impact of Research Features")
    print("="*70)
    print(f"\nAlgorithm: {algorithm}")
    print(f"Seeds: {seeds}")
    print(f"Configurations: {len(ABLATION_CONFIGS)}")
    print("\nFeature Progression:")
    for i, ablation in enumerate(ABLATION_CONFIGS, 1):
        print(f"  {i}. {ablation['name']}: {ablation['description']}")
    print("\n" + "="*70 + "\n")
    
    results = {}
    all_metrics = []
    
    for ablation in ABLATION_CONFIGS:
        config_name = ablation['config']
        ablation_name = ablation['name']
        
        print(f"\n{'='*70}")
        print(f"Running: {ablation_name}")
        print(f"Config: {config_name}")
        print(f"{'='*70}\n")
        
        config_metrics = []
        
        for seed in range(seeds):
            try:
                # Train
                model = train_algorithm(
                    algorithm_name=algorithm,
                    config_name=config_name,
                    seed=seed,
                    total_timesteps=total_timesteps,
                    save_dir=save_dir
                )
                
                # Evaluate
                metrics = evaluate_model(
                    model=model,
                    algorithm_name=algorithm,
                    config_name=ablation_name,  # Use ablation name for clarity
                    seed=seed,
                    n_eval_episodes=n_eval_episodes,
                    save_dir=save_dir
                )
                
                config_metrics.append(metrics)
                all_metrics.append(metrics)
                
            except Exception as e:
                print(f"\n❌ Error in {ablation_name}/seed_{seed}: {e}")
                continue
        
        # Aggregate results for this configuration
        if config_metrics:
            results[ablation_name] = {
                'metrics': config_metrics,
                'success_rate_mean': np.mean([m.success_rate for m in config_metrics]),
                'success_rate_std': np.std([m.success_rate for m in config_metrics]),
                'reward_mean': np.mean([m.avg_episode_reward for m in config_metrics]),
                'reward_std': np.std([m.avg_episode_reward for m in config_metrics]),
                'path_dev_mean': np.mean([m.path_deviation_mean for m in config_metrics]),
                'jerk_mean': np.mean([m.control_jerk_mean for m in config_metrics]),
            }
    
    # Save aggregate results
    save_path = Path(save_dir) / "ablation_summary.json"
    with open(save_path, 'w') as f:
        # Convert to serializable format
        summary = {
            name: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in data.items() if k != 'metrics'}
            for name, data in results.items()
        }
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Ablation summary saved to: {save_path}")
    
    # Generate comparison
    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE - Results Summary")
    print("="*70 + "\n")
    
    compare_metrics(all_metrics, save_path=f"{save_dir}/ablation_comparison.csv")
    
    # Generate analysis report
    generate_ablation_report(results, save_dir)
    
    # Generate plots
    plot_ablation_results(results, save_dir)
    
    return results


def generate_ablation_report(results: Dict, save_dir: str):
    """Generate text report analyzing ablation results."""
    report_path = Path(save_dir) / "ABLATION_REPORT.md"
    
    with open(report_path, 'w') as f:
        f.write("# Ablation Study Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report quantifies the impact of each research-grade feature ")
        f.write("on drift control performance.\n\n")
        
        f.write("## Feature Impact Analysis\n\n")
        f.write("| Configuration | Success Rate | Avg Reward | Path Deviation | Control Jerk |\n")
        f.write("|---------------|--------------|------------|----------------|---------------|\n")
        
        for name, data in results.items():
            f.write(f"| {name} | ")
            f.write(f"{data['success_rate_mean']*100:.1f}% ± {data['success_rate_std']*100:.1f} | ")
            f.write(f"{data['reward_mean']:.2f} ± {data['reward_std']:.2f} | ")
            f.write(f"{data['path_dev_mean']:.3f}m | ")
            f.write(f"{data['jerk_mean']:.3f} |\n")
        
        f.write("\n## Incremental Impact\n\n")
        
        configs = list(results.keys())
        for i in range(1, len(configs)):
            prev_name = configs[i-1]
            curr_name = configs[i]
            
            delta_success = (results[curr_name]['success_rate_mean'] - 
                           results[prev_name]['success_rate_mean']) * 100
            delta_reward = (results[curr_name]['reward_mean'] - 
                          results[prev_name]['reward_mean'])
            
            f.write(f"### {prev_name} → {curr_name}\n\n")
            f.write(f"- **Success rate change**: {delta_success:+.1f}%\n")
            f.write(f"- **Reward change**: {delta_reward:+.2f}\n")
            
            if delta_success > 0:
                f.write(f"- **Impact**: Positive (feature helps)\n\n")
            elif delta_success < -5:
                f.write(f"- **Impact**: Negative (feature hurts - needs investigation)\n\n")
            else:
                f.write(f"- **Impact**: Neutral (minimal effect)\n\n")
        
        f.write("## Recommendations\n\n")
        
        # Find best configuration
        best_config = max(results.items(), key=lambda x: x[1]['success_rate_mean'])
        f.write(f"**Best Configuration**: {best_config[0]} ")
        f.write(f"({best_config[1]['success_rate_mean']*100:.1f}% success rate)\n\n")
        
        # Identify helpful features
        f.write("**Key Findings**:\n\n")
        f.write("1. Compare baseline vs final to quantify total impact of research features\n")
        f.write("2. Features that decrease performance may need better implementation or tuning\n")
        f.write("3. Use these results to prioritize future research directions\n")
    
    print(f"✅ Ablation report saved to: {report_path}")


def plot_ablation_results(results: Dict, save_dir: str):
    """Generate plots comparing ablation configurations."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    configs = list(results.keys())
    
    # Extract metrics
    success_rates = [results[c]['success_rate_mean'] * 100 for c in configs]
    success_stds = [results[c]['success_rate_std'] * 100 for c in configs]
    rewards = [results[c]['reward_mean'] for c in configs]
    reward_stds = [results[c]['reward_std'] for c in configs]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ablation Study Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Success Rate
    ax = axes[0, 0]
    x = range(len(configs))
    ax.bar(x, success_rates, yerr=success_stds, capsize=5, alpha=0.7, color='#2ecc71')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=0, ha='center', fontsize=9)
    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_title('Success Rate Across Configurations')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Plot 2: Average Reward
    ax = axes[0, 1]
    ax.bar(x, rewards, yerr=reward_stds, capsize=5, alpha=0.7, color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=0, ha='center', fontsize=9)
    ax.set_ylabel('Average Reward', fontweight='bold')
    ax.set_title('Episode Reward Across Configurations')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Incremental Impact (Success Rate Delta)
    ax = axes[1, 0]
    deltas = [0]  # Baseline has no delta
    for i in range(1, len(configs)):
        delta = success_rates[i] - success_rates[i-1]
        deltas.append(delta)
    
    colors = ['green' if d >= 0 else 'red' for d in deltas]
    ax.bar(x, deltas, alpha=0.7, color=colors)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=0, ha='center', fontsize=9)
    ax.set_ylabel('Success Rate Change (%)', fontweight='bold')
    ax.set_title('Incremental Impact of Each Feature')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Path Deviation
    ax = axes[1, 1]
    path_devs = [results[c]['path_dev_mean'] for c in configs]
    ax.bar(x, path_devs, alpha=0.7, color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=0, ha='center', fontsize=9)
    ax.set_ylabel('Path Deviation (m)', fontweight='bold')
    ax.set_title('Path Quality Across Configurations')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(save_dir) / "ablation_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Ablation plots saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--algorithm", type=str, default="SAC", choices=["SAC", "PPO", "TD3"],
                        help="Algorithm to use")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--timesteps", type=int, default=500000, help="Training timesteps")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Evaluation episodes")
    parser.add_argument("--save-dir", type=str, default="experiments/results/ablation",
                        help="Save directory")
    
    args = parser.parse_args()
    
    # Run ablation study
    results = run_ablation_study(
        algorithm=args.algorithm,
        seeds=args.seeds,
        total_timesteps=args.timesteps,
        n_eval_episodes=args.eval_episodes,
        save_dir=args.save_dir
    )
    
    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.save_dir}")
    print("\nNext steps:")
    print("1. Review ABLATION_REPORT.md for detailed analysis")
    print("2. Check ablation_plots.png for visualizations")
    print("3. Use findings to guide future research")


if __name__ == "__main__":
    main()

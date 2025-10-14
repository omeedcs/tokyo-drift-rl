"""
Multi-Algorithm Benchmark Script

Trains and evaluates multiple RL algorithms on the drift task.

Algorithms tested:
- SAC (Soft Actor-Critic)
- PPO (Proximal Policy Optimization)
- TD3 (Twin Delayed DDPG)

Each algorithm is trained with 5 random seeds for statistical significance.

Usage:
    python experiments/benchmark_algorithms.py --config baseline --seeds 5

Output:
    - Trained models in experiments/results/models/
    - Training logs in experiments/results/logs/
    - Evaluation metrics in experiments/results/metrics/
    - Comparison plots in experiments/results/plots/
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.evaluation import DriftEvaluator, EvaluationMetrics, compare_metrics
from drift_gym.envs.drift_car_env_advanced import AdvancedDriftCarEnv

# Try to import RL libraries
try:
    from stable_baselines3 import SAC, PPO, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    SB3_AVAILABLE = True
except ImportError:
    print("Warning: stable-baselines3 not installed. Install with: pip install stable-baselines3")
    SB3_AVAILABLE = False


# Environment configurations
ENV_CONFIGS = {
    'baseline': {
        'use_noisy_sensors': False,
        'use_perception_pipeline': False,
        'use_latency': False,
        'use_3d_dynamics': False,
        'use_moving_agents': False,
    },
    '+sensors': {
        'use_noisy_sensors': True,
        'use_perception_pipeline': False,
        'use_latency': False,
        'use_3d_dynamics': False,
        'use_moving_agents': False,
    },
    '+perception': {
        'use_noisy_sensors': True,
        'use_perception_pipeline': True,
        'use_latency': False,
        'use_3d_dynamics': False,
        'use_moving_agents': False,
    },
    '+latency': {
        'use_noisy_sensors': True,
        'use_perception_pipeline': True,
        'use_latency': True,
        'use_3d_dynamics': False,
        'use_moving_agents': False,
    },
    'full': {
        'use_noisy_sensors': True,
        'use_perception_pipeline': True,
        'use_latency': True,
        'use_3d_dynamics': True,
        'use_moving_agents': True,
    }
}


def make_env(config_name: str, scenario: str = "loose", seed: int = 0):
    """Create environment with specified configuration."""
    config = ENV_CONFIGS[config_name]
    
    env = AdvancedDriftCarEnv(
        scenario=scenario,
        **config,
        seed=seed
    )
    
    return env


def train_algorithm(
    algorithm_name: str,
    config_name: str,
    seed: int,
    total_timesteps: int = 500000,
    save_dir: str = "experiments/results"
):
    """
    Train a single algorithm with given configuration and seed.
    
    Args:
        algorithm_name: "SAC", "PPO", or "TD3"
        config_name: Environment configuration name
        seed: Random seed
        total_timesteps: Total training timesteps
        save_dir: Directory to save results
        
    Returns:
        Trained model
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 required for training")
    
    print(f"\n{'='*70}")
    print(f"Training {algorithm_name} with {config_name} config (seed={seed})")
    print(f"{'='*70}\n")
    
    # Create directories
    model_dir = Path(save_dir) / "models" / config_name / algorithm_name.lower() / f"seed_{seed}"
    log_dir = Path(save_dir) / "logs" / config_name / algorithm_name.lower() / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    env = DummyVecEnv([lambda: make_env(config_name, seed=seed)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: make_env(config_name, seed=seed+1000)])
    
    # Algorithm hyperparameters (tuned for continuous control)
    if algorithm_name == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=seed
        )
    elif algorithm_name == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=seed
        )
    elif algorithm_name == "TD3":
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=seed
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(model_dir),
        name_prefix=algorithm_name.lower()
    )
    
    # Train
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    training_time = time.time() - start_time
    
    # Save final model
    model.save(model_dir / "final_model")
    
    print(f"\n✅ Training complete! Time: {training_time/60:.1f} minutes")
    print(f"Model saved to: {model_dir}")
    
    env.close()
    eval_env.close()
    
    return model


def evaluate_model(
    model,
    algorithm_name: str,
    config_name: str,
    seed: int,
    n_eval_episodes: int = 100,
    save_dir: str = "experiments/results"
):
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model
        algorithm_name: Algorithm name
        config_name: Configuration name
        seed: Random seed used for training
        n_eval_episodes: Number of evaluation episodes
        save_dir: Directory to save results
        
    Returns:
        EvaluationMetrics
    """
    print(f"\nEvaluating {algorithm_name} ({config_name}, seed={seed})...")
    
    # Create evaluator
    evaluator = DriftEvaluator(
        env_fn=lambda: make_env(config_name, seed=seed+2000),
        n_episodes=n_eval_episodes,
        seed=seed+2000
    )
    
    # Evaluate
    metrics = evaluator.evaluate(
        model,
        algorithm_name=algorithm_name,
        config_name=config_name,
        verbose=False
    )
    
    # Save metrics
    metrics_dir = Path(save_dir) / "metrics" / config_name / algorithm_name.lower()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"seed_{seed}.json"
    metrics.save_to_json(str(metrics_path))
    
    print(f"✅ Evaluation complete! Saved to: {metrics_path}")
    
    return metrics


def run_benchmark(
    algorithms: list = ["SAC", "PPO", "TD3"],
    configs: list = ["baseline", "+sensors"],
    seeds: list = [0, 1, 2, 3, 4],
    total_timesteps: int = 500000,
    n_eval_episodes: int = 100,
    save_dir: str = "experiments/results"
):
    """
    Run full benchmark suite.
    
    Args:
        algorithms: List of algorithm names to test
        configs: List of environment configurations
        seeds: List of random seeds
        total_timesteps: Training timesteps per run
        n_eval_episodes: Evaluation episodes per run
        save_dir: Directory to save all results
    """
    print("\n" + "="*70)
    print("DRIFT CONTROL BENCHMARK SUITE")
    print("="*70)
    print(f"\nAlgorithms: {algorithms}")
    print(f"Configs: {configs}")
    print(f"Seeds: {len(seeds)}")
    print(f"Total runs: {len(algorithms) * len(configs) * len(seeds)}")
    print(f"Training timesteps: {total_timesteps:,}")
    print(f"Evaluation episodes: {n_eval_episodes}")
    print("\n" + "="*70 + "\n")
    
    all_metrics = []
    
    for config_name in configs:
        for algorithm_name in algorithms:
            for seed in seeds:
                try:
                    # Train
                    model = train_algorithm(
                        algorithm_name=algorithm_name,
                        config_name=config_name,
                        seed=seed,
                        total_timesteps=total_timesteps,
                        save_dir=save_dir
                    )
                    
                    # Evaluate
                    metrics = evaluate_model(
                        model=model,
                        algorithm_name=algorithm_name,
                        config_name=config_name,
                        seed=seed,
                        n_eval_episodes=n_eval_episodes,
                        save_dir=save_dir
                    )
                    
                    all_metrics.append(metrics)
                    
                except Exception as e:
                    print(f"\n❌ Error in {algorithm_name}/{config_name}/seed_{seed}: {e}")
                    continue
    
    # Generate comparison
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE - Generating Comparison")
    print("="*70 + "\n")
    
    compare_metrics(all_metrics, save_path=f"{save_dir}/comparison_table.csv")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Benchmark RL algorithms on drift task")
    parser.add_argument("--algorithms", nargs="+", default=["SAC"], choices=["SAC", "PPO", "TD3"],
                        help="Algorithms to benchmark")
    parser.add_argument("--config", type=str, default="baseline",
                        choices=list(ENV_CONFIGS.keys()),
                        help="Environment configuration")
    parser.add_argument("--seeds", type=int, default=1, help="Number of random seeds")
    parser.add_argument("--timesteps", type=int, default=500000, help="Training timesteps")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Evaluation episodes")
    parser.add_argument("--save-dir", type=str, default="experiments/results", help="Save directory")
    
    args = parser.parse_args()
    
    if not SB3_AVAILABLE:
        print("ERROR: stable-baselines3 not installed!")
        print("Install with: pip install stable-baselines3")
        sys.exit(1)
    
    # Run benchmark
    run_benchmark(
        algorithms=args.algorithms,
        configs=[args.config],
        seeds=list(range(args.seeds)),
        total_timesteps=args.timesteps,
        n_eval_episodes=args.eval_episodes,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()

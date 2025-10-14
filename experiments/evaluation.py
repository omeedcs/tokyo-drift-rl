"""
Standardized Evaluation Protocol for Drift Control

Provides consistent metrics across algorithms and ablation studies.

Metrics computed:
- Success rate (% of successful gate passages)
- Average completion time
- Path deviation (cross-track error)
- Control smoothness (jerk)
- Collision rate
- Trajectory consistency (across seeds)

Usage:
    evaluator = DriftEvaluator(env, n_episodes=100)
    metrics = evaluator.evaluate(agent)
    metrics.save_to_json("results/sac_baseline.json")
"""

import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
import time


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    # Success metrics
    success_rate: float
    avg_completion_time: float
    avg_episode_reward: float
    
    # Path quality metrics
    path_deviation_mean: float
    path_deviation_std: float
    final_distance_to_goal_mean: float
    
    # Control quality metrics
    control_jerk_mean: float  # Smoothness
    control_jerk_std: float
    
    # Safety metrics
    collision_rate: float
    near_miss_rate: float  # Close calls with obstacles
    
    # Efficiency metrics
    avg_path_length: float
    avg_steps: float
    
    # Additional info
    n_episodes: int
    algorithm: str
    config_name: str
    timestamp: str
    
    def save_to_json(self, filepath: str):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @staticmethod
    def load_from_json(filepath: str) -> 'EvaluationMetrics':
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return EvaluationMetrics(**data)
    
    def __str__(self) -> str:
        """Pretty print metrics."""
        lines = [
            "=" * 60,
            f"Evaluation Results: {self.algorithm} ({self.config_name})",
            "=" * 60,
            f"Episodes: {self.n_episodes}",
            "",
            "Success Metrics:",
            f"  Success Rate:     {self.success_rate*100:.1f}%",
            f"  Avg Completion:   {self.avg_completion_time:.2f}s",
            f"  Avg Reward:       {self.avg_episode_reward:.2f}",
            "",
            "Path Quality:",
            f"  Path Deviation:   {self.path_deviation_mean:.3f} ± {self.path_deviation_std:.3f}m",
            f"  Final Distance:   {self.final_distance_to_goal_mean:.3f}m",
            f"  Avg Path Length:  {self.avg_path_length:.2f}m",
            "",
            "Control Quality:",
            f"  Control Jerk:     {self.control_jerk_mean:.3f} ± {self.control_jerk_std:.3f}",
            "",
            "Safety:",
            f"  Collision Rate:   {self.collision_rate*100:.1f}%",
            f"  Near Miss Rate:   {self.near_miss_rate*100:.1f}%",
            "=" * 60
        ]
        return "\n".join(lines)


class DriftEvaluator:
    """
    Standardized evaluator for drift control algorithms.
    
    Ensures consistent evaluation across different algorithms and configurations.
    """
    
    def __init__(
        self,
        env_fn: callable,
        n_episodes: int = 100,
        goal_tolerance: float = 0.5,  # meters
        near_miss_threshold: float = 0.3,  # meters from obstacle
        seed: int = 42
    ):
        """
        Initialize evaluator.
        
        Args:
            env_fn: Function that creates the environment
            n_episodes: Number of episodes to evaluate
            goal_tolerance: Distance threshold for success (meters)
            near_miss_threshold: Distance for near miss detection (meters)
            seed: Random seed for reproducibility
        """
        self.env_fn = env_fn
        self.n_episodes = n_episodes
        self.goal_tolerance = goal_tolerance
        self.near_miss_threshold = near_miss_threshold
        self.seed = seed
        
    def evaluate(
        self,
        agent: Any,
        algorithm_name: str = "unknown",
        config_name: str = "default",
        verbose: bool = True
    ) -> EvaluationMetrics:
        """
        Evaluate agent performance.
        
        Args:
            agent: Agent with forward() or predict() method
            algorithm_name: Name of algorithm (e.g., "SAC", "PPO")
            config_name: Configuration name (e.g., "baseline", "+sensors")
            verbose: Print progress
            
        Returns:
            EvaluationMetrics with all computed metrics
        """
        env = self.env_fn()
        
        # Storage for episode data
        successes = []
        completion_times = []
        episode_rewards = []
        path_deviations = []
        final_distances = []
        control_jerks = []
        collisions = []
        near_misses = []
        path_lengths = []
        episode_steps = []
        
        if verbose:
            print(f"\nEvaluating {algorithm_name} ({config_name})...")
            print(f"Running {self.n_episodes} episodes...")
        
        for episode in range(self.n_episodes):
            # Reset environment
            obs, info = env.reset(seed=self.seed + episode)
            
            # Episode tracking
            done = False
            episode_reward = 0.0
            episode_step = 0
            trajectory = []
            actions = []
            obstacles_min_dist = float('inf')
            
            while not done and episode_step < env.max_steps:
                # Get action from agent
                if hasattr(agent, 'forward'):
                    action = agent.forward(obs, from_cpu=True)
                elif hasattr(agent, 'predict'):
                    action, _ = agent.predict(obs, deterministic=True)
                else:
                    raise ValueError("Agent must have forward() or predict() method")
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_step += 1
                
                # Track trajectory
                state = env.sim_env.vehicle.get_state()
                trajectory.append((state.x, state.y))
                actions.append(action)
                
                # Track minimum distance to obstacles
                for obstacle in env.sim_env.obstacles:
                    dist = np.hypot(obstacle.x - state.x, obstacle.y - state.y) - obstacle.radius
                    obstacles_min_dist = min(obstacles_min_dist, dist)
            
            # Compute episode metrics
            final_state = env.sim_env.vehicle.get_state()
            final_dist = np.hypot(
                env.gate_pos[0] - final_state.x,
                env.gate_pos[1] - final_state.y
            )
            
            success = final_dist < self.goal_tolerance and not info.get('collision', False)
            completion_time = episode_step * 0.05  # dt = 0.05s
            
            # Path deviation (cross-track error from straight line to goal)
            if len(trajectory) > 1:
                path_dev = self._compute_path_deviation(trajectory, env.gate_pos)
                path_length = self._compute_path_length(trajectory)
            else:
                path_dev = 0.0
                path_length = 0.0
            
            # Control jerk (smoothness)
            if len(actions) > 1:
                jerk = self._compute_control_jerk(actions)
            else:
                jerk = 0.0
            
            # Safety metrics
            collision = info.get('termination_reason', '') == 'collision'
            near_miss = obstacles_min_dist < self.near_miss_threshold and not collision
            
            # Store results
            successes.append(success)
            completion_times.append(completion_time if success else env.max_steps * 0.05)
            episode_rewards.append(episode_reward)
            path_deviations.append(path_dev)
            final_distances.append(final_dist)
            control_jerks.append(jerk)
            collisions.append(collision)
            near_misses.append(near_miss)
            path_lengths.append(path_length)
            episode_steps.append(episode_step)
            
            if verbose and (episode + 1) % 20 == 0:
                print(f"  {episode + 1}/{self.n_episodes} episodes complete...")
        
        env.close()
        
        # Compute aggregate metrics
        metrics = EvaluationMetrics(
            success_rate=np.mean(successes),
            avg_completion_time=np.mean([t for t, s in zip(completion_times, successes) if s]) if any(successes) else 0.0,
            avg_episode_reward=np.mean(episode_rewards),
            path_deviation_mean=np.mean(path_deviations),
            path_deviation_std=np.std(path_deviations),
            final_distance_to_goal_mean=np.mean(final_distances),
            control_jerk_mean=np.mean(control_jerks),
            control_jerk_std=np.std(control_jerks),
            collision_rate=np.mean(collisions),
            near_miss_rate=np.mean(near_misses),
            avg_path_length=np.mean(path_lengths),
            avg_steps=np.mean(episode_steps),
            n_episodes=self.n_episodes,
            algorithm=algorithm_name,
            config_name=config_name,
            timestamp=time.strftime("%Y%m%d_%H%M%S")
        )
        
        if verbose:
            print(metrics)
        
        return metrics
    
    @staticmethod
    def _compute_path_deviation(trajectory: List[Tuple[float, float]], goal: np.ndarray) -> float:
        """
        Compute average cross-track error from ideal straight line.
        
        Args:
            trajectory: List of (x, y) positions
            goal: Goal position [x, y]
            
        Returns:
            Mean cross-track error (meters)
        """
        if len(trajectory) < 2:
            return 0.0
        
        start = np.array(trajectory[0])
        end = goal
        
        # Compute perpendicular distance from each point to line
        deviations = []
        for pos in trajectory:
            pos = np.array(pos)
            # Distance from point to line segment
            line_vec = end - start
            line_len = np.linalg.norm(line_vec)
            if line_len < 1e-6:
                dist = np.linalg.norm(pos - start)
            else:
                line_unitvec = line_vec / line_len
                point_vec = pos - start
                # Project onto line
                projection_length = np.dot(point_vec, line_unitvec)
                projection_length = np.clip(projection_length, 0, line_len)
                projection = start + projection_length * line_unitvec
                dist = np.linalg.norm(pos - projection)
            deviations.append(dist)
        
        return np.mean(deviations)
    
    @staticmethod
    def _compute_path_length(trajectory: List[Tuple[float, float]]) -> float:
        """Compute total path length."""
        if len(trajectory) < 2:
            return 0.0
        
        length = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            length += np.hypot(dx, dy)
        
        return length
    
    @staticmethod
    def _compute_control_jerk(actions: List[np.ndarray]) -> float:
        """
        Compute control jerk (rate of change of acceleration).
        
        Higher jerk = less smooth control
        
        Args:
            actions: List of action vectors
            
        Returns:
            Mean absolute jerk
        """
        if len(actions) < 3:
            return 0.0
        
        actions = np.array(actions)
        # First derivative (velocity)
        d_actions = np.diff(actions, axis=0)
        # Second derivative (acceleration)
        dd_actions = np.diff(d_actions, axis=0)
        # Third derivative (jerk)
        ddd_actions = np.diff(dd_actions, axis=0)
        
        # L2 norm of jerk
        jerk = np.mean(np.linalg.norm(ddd_actions, axis=1))
        
        return jerk


def compare_metrics(metrics_list: List[EvaluationMetrics], save_path: Optional[str] = None):
    """
    Compare multiple evaluation results.
    
    Args:
        metrics_list: List of EvaluationMetrics to compare
        save_path: Optional path to save comparison table
    """
    import pandas as pd
    
    # Build comparison dataframe
    data = []
    for m in metrics_list:
        data.append({
            'Algorithm': m.algorithm,
            'Config': m.config_name,
            'Success Rate (%)': f"{m.success_rate*100:.1f}",
            'Avg Reward': f"{m.avg_episode_reward:.2f}",
            'Path Dev (m)': f"{m.path_deviation_mean:.3f}",
            'Jerk': f"{m.control_jerk_mean:.3f}",
            'Collision (%)': f"{m.collision_rate*100:.1f}",
        })
    
    df = pd.DataFrame(data)
    
    print("\nComparison Table:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\nSaved to: {save_path}")


if __name__ == "__main__":
    # Example usage
    print("Evaluation Protocol Test")
    print("=" * 60)
    print("\nThis module provides standardized evaluation for drift control.")
    print("\nUsage example:")
    print("""
    from experiments.evaluation import DriftEvaluator
    from drift_gym.envs.drift_car_env_advanced import AdvancedDriftCarEnv
    
    # Create environment factory
    def make_env():
        return AdvancedDriftCarEnv(
            scenario="loose",
            use_noisy_sensors=True,
            use_perception_pipeline=False
        )
    
    # Create evaluator
    evaluator = DriftEvaluator(make_env, n_episodes=100)
    
    # Evaluate agent
    metrics = evaluator.evaluate(agent, algorithm_name="SAC", config_name="baseline")
    
    # Save results
    metrics.save_to_json("results/sac_baseline.json")
    """)

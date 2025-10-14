#!/usr/bin/env python3
"""
Train SAC agent on drift environment with optional visualization.

Supports:
- SAC training with replay buffer
- Real-time visualization during training
- Curriculum learning (loose -> tight)
- Model checkpointing
- TensorBoard logging
"""

import numpy as np
import torch
import argparse
from pathlib import Path
import time
from collections import deque

from src.rl.gym_drift_env import GymDriftEnv
from src.rl.sac_agent import SACAgent


class RLTrainer:
    """Trainer for SAC on drift environment."""
    
    def __init__(
        self,
        env,
        agent,
        eval_env=None,
        log_dir="logs/sac",
        checkpoint_dir="checkpoints/sac"
    ):
        """
        Initialize trainer.
        
        Args:
            env: Training environment
            agent: SAC agent
            eval_env: Evaluation environment (optional)
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
        """
        self.env = env
        self.agent = agent
        self.eval_env = eval_env
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rate = deque(maxlen=100)
        
        self.total_steps = 0
        self.episode_count = 0
    
    def train(
        self,
        total_timesteps: int = 100000,
        batch_size: int = 256,
        eval_freq: int = 5000,
        checkpoint_freq: int = 10000,
        log_freq: int = 1000,
        start_steps: int = 1000
    ):
        """
        Train agent.
        
        Args:
            total_timesteps: Total training steps
            batch_size: Batch size for updates
            eval_freq: Evaluation frequency
            checkpoint_freq: Checkpoint save frequency
            log_freq: Logging frequency
            start_steps: Random exploration steps before training
        """
        print("\n" + "="*60)
        print("Starting SAC Training")
        print("="*60)
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Batch size: {batch_size}")
        print(f"Start steps (random): {start_steps}")
        print("="*60 + "\n")
        
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        start_time = time.time()
        
        for step in range(total_timesteps):
            # Select action
            if step < start_steps:
                # Random exploration
                action = self.env.action_space.sample()
            else:
                action = self.agent.select_action(obs, evaluate=False)
            
            # Execute action
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.agent.replay_buffer.push(obs, action, reward, next_obs, float(done))
            
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            obs = next_obs
            
            # Update agent
            if step >= start_steps:
                self.agent.update(batch_size)
            
            # Episode end
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Track success
                success = info.get("termination_reason") == "success"
                self.success_rate.append(float(success))
                
                self.episode_count += 1
                
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            
            # Logging
            if step % log_freq == 0 and step > 0:
                self._log_training_stats(step, start_time)
            
            # Evaluation
            if step % eval_freq == 0 and step > 0 and self.eval_env is not None:
                self._evaluate(step)
            
            # Checkpoint
            if step % checkpoint_freq == 0 and step > 0:
                self._save_checkpoint(step)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        
        # Final checkpoint
        self._save_checkpoint(total_timesteps)
    
    def _log_training_stats(self, step, start_time):
        """Log training statistics."""
        elapsed = time.time() - start_time
        fps = step / elapsed
        
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
        success_rate = np.mean(self.success_rate) if self.success_rate else 0
        
        print(f"[{step:6d}] "
              f"Episodes: {self.episode_count:4d} | "
              f"Reward: {avg_reward:7.2f} | "
              f"Success: {success_rate*100:5.1f}% | "
              f"Length: {avg_length:5.1f} | "
              f"FPS: {fps:5.1f}")
    
    def _evaluate(self, step, num_episodes=5):
        """Evaluate current policy."""
        print(f"\n[EVAL at {step}] Running {num_episodes} episodes...")
        
        eval_rewards = []
        eval_successes = []
        
        for ep in range(num_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            
            while True:
                action = self.agent.select_action(obs, evaluate=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                
                # Optional: render evaluation
                if hasattr(self.eval_env, 'render'):
                    self.eval_env.render()
                
                if terminated or truncated:
                    success = info.get("termination_reason") == "success"
                    eval_successes.append(float(success))
                    break
            
            eval_rewards.append(episode_reward)
        
        avg_reward = np.mean(eval_rewards)
        success_rate = np.mean(eval_successes)
        
        print(f"[EVAL] Avg Reward: {avg_reward:.2f} | Success Rate: {success_rate*100:.1f}%\n")
    
    def _save_checkpoint(self, step):
        """Save checkpoint."""
        filepath = self.checkpoint_dir / f"sac_agent_step_{step}.pt"
        self.agent.save(str(filepath))
        print(f"[CHECKPOINT] Saved to {filepath}")


def main():
    """Run RL training."""
    parser = argparse.ArgumentParser(description="Train SAC on Drift Environment")
    parser.add_argument(
        "--scenario",
        type=str,
        default="loose",
        choices=["loose", "tight"],
        help="Training scenario"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization during training (slower)"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to train on"
    )
    
    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("#  SAC Training on Drift Environment")
    print("#" + "#"*60)
    print(f"\nScenario: {args.scenario}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Visualization: {'Enabled' if args.visualize else 'Disabled'}")
    
    # Create environments
    render_mode = "human" if args.visualize else None
    
    train_env = GymDriftEnv(
        scenario=args.scenario,
        render_mode=render_mode,
        dense_rewards=True
    )
    
    # Separate eval environment
    eval_env = GymDriftEnv(
        scenario=args.scenario,
        render_mode=None,  # No rendering for eval (faster)
        dense_rewards=True
    )
    
    # Create agent
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lr=3e-4,
        device=args.device
    )
    
    # Create trainer
    trainer = RLTrainer(
        env=train_env,
        agent=agent,
        eval_env=eval_env,
        log_dir=f"logs/sac_{args.scenario}",
        checkpoint_dir=f"checkpoints/sac_{args.scenario}"
    )
    
    # Train
    try:
        trainer.train(
            total_timesteps=args.timesteps,
            batch_size=args.batch_size,
            eval_freq=5000,
            checkpoint_freq=10000,
            log_freq=1000,
            start_steps=1000
        )
    
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()

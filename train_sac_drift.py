#!/usr/bin/env python3
"""
Train SAC (Soft Actor-Critic) agent for autonomous drifting.

Uses Jake's deep-rl-algos implementation.
"""

import sys
import os
sys.path.insert(0, 'jake-deep-rl-algos')

import gymnasium as gym
import numpy as np
import torch
import json
from pathlib import Path

import deep_control as dc

# Import our drift environment
from src.rl.gym_drift_env import GymDriftEnv


class SACDriftTrainer:
    """Train SAC agent for drift control."""
    
    def __init__(
        self,
        scenario="loose",
        total_timesteps=100000,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        save_dir="trained_agents/sac"
    ):
        self.scenario = scenario
        self.total_timesteps = total_timesteps
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create environment
        print("\n" + "="*60)
        print("Initializing SAC Training")
        print("="*60)
        print(f"  Scenario: {scenario}")
        print(f"  Total timesteps: {total_timesteps:,}")
        print(f"  Buffer size: {buffer_size:,}")
        print(f"  Batch size: {batch_size}")
        
        self.env = GymDriftEnv(
            scenario=scenario,
            max_steps=200,
            render_mode=None
        )
        
        # Get environment specs
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        print(f"\n  Observation dim: {obs_dim}")
        print(f"  Action dim: {action_dim}")
        
        # Create SAC agent
        self.agent = SAC(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_space=self.env.action_space,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            automatic_entropy_tuning=True,
            hidden_dim=256,
            lr=3e-4
        )
        
        # Replay buffer
        self.buffer = ReplayBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_size=buffer_size
        )
        
        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
        self.training_losses = []
        
    def train(self):
        """Train the SAC agent."""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_num = 0
        successes = []
        
        for timestep in range(self.total_timesteps):
            # Select action
            if timestep < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.agent.select_action(obs, evaluate=False)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # Store transition
            self.buffer.add(obs, action, reward, next_obs, done)
            
            obs = next_obs
            
            # Train agent
            if timestep >= self.learning_starts:
                batch = self.buffer.sample(self.batch_size)
                losses = self.agent.update_parameters(
                    batch['obs'],
                    batch['action'],
                    batch['reward'],
                    batch['next_obs'],
                    batch['done']
                )
                
                if timestep % 1000 == 0:
                    self.training_losses.append({
                        'timestep': timestep,
                        'critic_loss': losses.get('critic_loss', 0),
                        'actor_loss': losses.get('actor_loss', 0),
                        'alpha_loss': losses.get('alpha_loss', 0)
                    })
            
            # Episode end
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Track success
                success = info.get('success', False)
                successes.append(1 if success else 0)
                
                # Compute success rate (last 100 episodes)
                if len(successes) >= 100:
                    self.success_rate.append(np.mean(successes[-100:]))
                
                episode_num += 1
                
                # Log progress
                if episode_num % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])
                    recent_success = np.mean(successes[-10:]) if successes else 0
                    
                    print(f"Episode {episode_num:4d} | "
                          f"Timestep {timestep:6d} | "
                          f"Reward: {avg_reward:7.2f} | "
                          f"Length: {avg_length:5.1f} | "
                          f"Success: {recent_success:.2%}")
                
                # Reset
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            
            # Save checkpoint
            if timestep % 10000 == 0 and timestep > 0:
                self.save_checkpoint(timestep)
        
        # Final save
        self.save_final_model()
        self.save_training_stats()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"  Total episodes: {episode_num}")
        print(f"  Final success rate: {self.success_rate[-1]:.2%}" if self.success_rate else "")
        print(f"  Model saved to: {self.save_dir}")
    
    def save_checkpoint(self, timestep):
        """Save training checkpoint."""
        checkpoint_path = self.save_dir / f"checkpoint_{timestep}.pt"
        torch.save({
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'critic_target_state_dict': self.agent.critic_target.state_dict(),
            'timestep': timestep
        }, checkpoint_path)
        print(f"  💾 Checkpoint saved: {checkpoint_path}")
    
    def save_final_model(self):
        """Save final trained model."""
        model_path = self.save_dir / "sac_final.pt"
        torch.save({
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'scenario': self.scenario
        }, model_path)
        print(f"\n  💾 Final model saved: {model_path}")
    
    def save_training_stats(self):
        """Save training statistics."""
        stats = {
            'scenario': self.scenario,
            'total_timesteps': self.total_timesteps,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rate': self.success_rate,
            'training_losses': self.training_losses,
            'final_success_rate': self.success_rate[-1] if self.success_rate else 0,
            'avg_final_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else 0
        }
        
        stats_path = self.save_dir / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  📊 Training stats saved: {stats_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SAC for drift control")
    parser.add_argument("--scenario", type=str, default="loose",
                        choices=["loose", "tight"],
                        help="Drift scenario")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total training timesteps")
    parser.add_argument("--buffer-size", type=int, default=100000,
                        help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--learning-starts", type=int, default=1000,
                        help="Timesteps before training starts")
    parser.add_argument("--save-dir", type=str, default="trained_agents/sac",
                        help="Directory to save models")
    
    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("#  Train SAC Agent for Autonomous Drifting")
    print("#" + "#"*60)
    print(f"\nTraining configuration:")
    print(f"  Algorithm: SAC (Soft Actor-Critic)")
    print(f"  Scenario: {args.scenario}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Save directory: {args.save_dir}")
    
    # Create trainer
    trainer = SACDriftTrainer(
        scenario=args.scenario,
        total_timesteps=args.timesteps,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
    
    # Train
    trainer.train()
    
    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    print("\n1. Evaluate the trained agent:")
    print(f"   python evaluate_sac.py --model {args.save_dir}/sac_final.pt")
    print("\n2. Compare to baseline and IKD:")
    print("   python compare_all_methods.py")


if __name__ == "__main__":
    main()

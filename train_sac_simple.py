#!/usr/bin/env python3
"""
Simple SAC training for drift control using Jake's repo.
"""

import sys
sys.path.insert(0, 'jake-deep-rl-algos')

import argparse
import deep_control as dc
from src.rl.gym_drift_env import GymDriftEnv


class OldGymWrapper:
    """Wrapper to convert new gym API to old gym API for Jake's code."""
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    def reset(self):
        obs, info = self.env.reset()
        return obs
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
    
    def close(self):
        return self.env.close()


def create_drift_env(scenario="loose", seed=0):
    """Create drift environment."""
    env = GymDriftEnv(
        scenario=scenario,
        max_steps=200,
        render_mode=None
    )
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return OldGymWrapper(env)


def train_sac_drift(args):
    """Train SAC on drift task."""
    print("\n" + "#"*60)
    print("#  SAC Training for Autonomous Drifting")
    print("#" + "#"*60)
    print(f"\nScenario: {args.scenario}")
    print(f"Timesteps: {args.total_timesteps:,}")
    print(f"Seed: {args.seed}")
    
    # Create environments
    train_env = create_drift_env(args.scenario, args.seed)
    test_env = create_drift_env(args.scenario, args.seed)
    
    state_space = train_env.observation_space
    action_space = train_env.action_space
    
    print(f"\nObservation space: {state_space.shape}")
    print(f"Action space: {action_space.shape}")
    
    # Create SAC agent
    agent = dc.sac.SACAgent(
        obs_space_size=state_space.shape[0],
        act_space_size=action_space.shape[0],
        log_std_low=args.log_std_low,
        log_std_high=args.log_std_high,
        hidden_size=args.hidden_size
    )
    
    # Create replay buffer
    buffer = dc.replay.ReplayBuffer(
        args.buffer_size,
        state_shape=state_space.shape,
        state_dtype=float,
        action_shape=action_space.shape
    )
    
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    
    # Train
    dc.sac.sac(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        **vars(args)
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModel saved to: {args.save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC for drift control")
    
    # Environment args
    parser.add_argument("--scenario", type=str, default="loose",
                        choices=["loose", "tight"],
                        help="Drift scenario")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    
    # SAC args (from Jake's repo)
    dc.sac.add_args(parser)
    
    # Override defaults
    parser.set_defaults(
        total_timesteps=100000,
        buffer_size=100000,
        batch_size=256,
        hidden_size=256,
        learning_starts=1000,
        log_interval=1000,
        eval_interval=5000,
        save_interval=10000,
        save_dir="trained_agents/sac",
        log_std_low=-20,
        log_std_high=2,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha=0.2,
        automatic_entropy_tuning=True
    )
    
    args = parser.parse_args()
    train_sac_drift(args)

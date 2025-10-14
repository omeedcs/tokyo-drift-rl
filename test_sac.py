#!/usr/bin/env python3
"""
Test SAC policy on drift task.
"""

import sys
sys.path.insert(0, 'jake-deep-rl-algos')

import numpy as np
import torch
import deep_control as dc
from src.rl.gym_drift_env import GymDriftEnv

def load_sac_agent(model_dir="dc_saves/sac_loose_2"):
    """Load trained SAC agent."""
    env = GymDriftEnv(scenario="loose", max_steps=200, render_mode=None)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create agent (match training config)
    agent = dc.sac.SACAgent(
        obs_space_size=obs_dim,
        act_space_size=action_dim,
        log_std_low=-20,
        log_std_high=2,
        hidden_size=256  # Match training config
    )
    
    # Load weights
    agent.actor.load_state_dict(torch.load(f"{model_dir}/actor.pt", map_location='cpu'))
    agent.critic1.load_state_dict(torch.load(f"{model_dir}/critic1.pt", map_location='cpu'))
    agent.critic2.load_state_dict(torch.load(f"{model_dir}/critic2.pt", map_location='cpu'))
    agent.eval()
    
    return agent, env

def test_sac(num_trials=10):
    """Test SAC policy."""
    print("\n" + "="*60)
    print("Testing SAC Policy")
    print("="*60)
    
    agent, env = load_sac_agent()
    
    rewards = []
    steps_list = []
    successes = []
    
    for trial in range(num_trials):
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 200:
            # Get action from SAC
            with torch.no_grad():
                action = agent.forward(torch.FloatTensor(obs))
            
            # Step environment
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            total_reward += reward
            steps += 1
        
        rewards.append(total_reward)
        steps_list.append(steps)
        successes.append(1 if info.get('success', steps < 200) else 0)
        
        print(f"  Trial {trial+1}: Reward={total_reward:.2f}, Steps={steps}, Success={successes[-1]}")
    
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"  Average Reward: {np.mean(rewards):.2f}")
    print(f"  Success Rate: {np.mean(successes)*100:.1f}%")
    print(f"  Average Steps: {np.mean(steps_list):.1f}")
    
    return {
        'avg_reward': np.mean(rewards),
        'success_rate': np.mean(successes),
        'avg_steps': np.mean(steps_list),
        'rewards': rewards,
        'steps': steps_list,
        'successes': successes
    }

if __name__ == "__main__":
    test_sac(num_trials=20)

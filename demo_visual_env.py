#!/usr/bin/env python3
"""
Demo script for visual Gymnasium environment.

Shows the environment in action with different policies:
- Random policy
- Heuristic policy (using trajectory planner)
- Keyboard control (interactive)
"""

import numpy as np
import pygame
import argparse
from src.rl.gym_drift_env import GymDriftEnv


def demo_random_policy(env, episodes=3):
    """Run environment with random actions."""
    print("\n" + "="*60)
    print("DEMO: Random Policy")
    print("="*60)
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        
        while True:
            # Random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Render
            env.render()
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
            
            if terminated or truncated:
                reason = info.get("termination_reason") or info.get("truncation_reason")
                print(f"  Episode ended: {reason}")
                print(f"  Steps: {steps}")
                print(f"  Total reward: {episode_reward:.2f}")
                break
        
        # Pause between episodes
        pygame.time.wait(1000)


def demo_keyboard_control(env):
    """Interactive keyboard control."""
    print("\n" + "="*60)
    print("DEMO: Keyboard Control")
    print("="*60)
    print("\nControls:")
    print("  Arrow Keys: Control velocity and steering")
    print("  UP: Increase velocity")
    print("  DOWN: Decrease velocity")
    print("  LEFT: Turn left")
    print("  RIGHT: Turn right")
    print("  R: Reset episode")
    print("  ESC: Quit")
    print("="*60)
    
    obs, info = env.reset()
    episode_reward = 0
    steps = 0
    
    velocity_cmd = 0.0
    angular_velocity_cmd = 0.0
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    episode_reward = 0
                    steps = 0
                    velocity_cmd = 0.0
                    angular_velocity_cmd = 0.0
                    print("\nEpisode reset")
        
        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        # Update commands based on keys
        if keys[pygame.K_UP]:
            velocity_cmd = min(velocity_cmd + 0.05, 1.0)
        elif keys[pygame.K_DOWN]:
            velocity_cmd = max(velocity_cmd - 0.05, -1.0)
        else:
            velocity_cmd *= 0.95  # Decay
        
        if keys[pygame.K_LEFT]:
            angular_velocity_cmd = min(angular_velocity_cmd + 0.05, 1.0)
        elif keys[pygame.K_RIGHT]:
            angular_velocity_cmd = max(angular_velocity_cmd - 0.05, -1.0)
        else:
            angular_velocity_cmd *= 0.95  # Decay
        
        # Create action
        action = np.array([velocity_cmd, angular_velocity_cmd], dtype=np.float32)
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        
        # Render
        env.render()
        
        # Check termination
        if terminated or truncated:
            reason = info.get("termination_reason") or info.get("truncation_reason")
            print(f"\nEpisode ended: {reason}")
            print(f"  Steps: {steps}")
            print(f"  Total reward: {episode_reward:.2f}")
            
            # Auto-reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            episode_reward = 0
            steps = 0
            velocity_cmd = 0.0
            angular_velocity_cmd = 0.0
        
        clock.tick(20)  # 20 FPS


def demo_smart_policy(env, episodes=3):
    """Run with a simple heuristic policy."""
    print("\n" + "="*60)
    print("DEMO: Smart Heuristic Policy")
    print("="*60)
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        
        while True:
            # Simple heuristic: go toward gate
            # obs = [vel, ang_vel, x, y, cos(theta), sin(theta), 
            #        dist_gate, cos(angle_gate), sin(angle_gate), min_obs_dist]
            
            dist_to_gate = obs[6]
            angle_to_gate = np.arctan2(obs[8], obs[7])  # From cos/sin
            
            # Proportional control
            velocity_cmd = np.clip(0.5 + dist_to_gate * 0.5, -1.0, 1.0)
            angular_velocity_cmd = np.clip(angle_to_gate * 2.0, -1.0, 1.0)
            
            action = np.array([velocity_cmd, angular_velocity_cmd], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Render
            env.render()
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
            
            if terminated or truncated:
                reason = info.get("termination_reason") or info.get("truncation_reason")
                print(f"  Episode ended: {reason}")
                print(f"  Steps: {steps}")
                print(f"  Total reward: {episode_reward:.2f}")
                break
        
        # Pause between episodes
        pygame.time.wait(1000)


def main():
    """Run demos."""
    parser = argparse.ArgumentParser(description="Visual Drift Environment Demo")
    parser.add_argument(
        "--mode",
        type=str,
        default="keyboard",
        choices=["random", "keyboard", "smart"],
        help="Demo mode"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="loose",
        choices=["loose", "tight"],
        help="Drift scenario"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes (for random/smart modes)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("#  Visual Drift Environment Demo")
    print("#" + "#"*60)
    print(f"\nMode: {args.mode}")
    print(f"Scenario: {args.scenario}")
    
    # Create environment with rendering
    env = GymDriftEnv(
        scenario=args.scenario,
        render_mode="human",
        dense_rewards=True
    )
    
    try:
        if args.mode == "random":
            demo_random_policy(env, episodes=args.episodes)
        elif args.mode == "keyboard":
            demo_keyboard_control(env)
        elif args.mode == "smart":
            demo_smart_policy(env, episodes=args.episodes)
    
    finally:
        env.close()
        print("\n" + "="*60)
        print("Demo complete!")
        print("="*60)


if __name__ == "__main__":
    main()

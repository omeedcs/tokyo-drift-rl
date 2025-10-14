#!/usr/bin/env python3
"""
Enhanced 3D-like Demo with Amazing Visual Effects!

Features:
- Tire smoke particles
- Skid marks
- Motion trails
- Professional racing HUD
- Enhanced vehicle graphics
- Shadow effects
- Drift detection with "DRIFT!" indicator

Usage:
    python demo_enhanced.py --mode keyboard
    python demo_enhanced.py --mode smart --episodes 5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np
import pygame

# Import directly to avoid circular imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'autonomous_drift/envs'))
from drift_env_enhanced import EnhancedDriftEnv


def demo_keyboard_control(env):
    """Interactive keyboard control with enhanced graphics."""
    print("\n" + "="*60)
    print("🏎️  ENHANCED DRIFT DEMO - Keyboard Control")
    print("="*60)
    print("\n Controls:")
    print("  ↑ Arrow Up:    Accelerate")
    print("  ↓ Arrow Down:  Brake")
    print("  ← Arrow Left:  Turn left")
    print("  → Arrow Right: Turn right")
    print("  R: Reset episode")
    print("  ESC: Quit")
    print("\n Watch for:")
    print("  💨 Tire smoke when drifting")
    print("  ⚫ Skid marks on track")
    print("  👻 Motion trail (ghost car)")
    print("  🎯 'DRIFT!' indicator")
    print("="*60)
    
    pygame.init()
    
    obs, info = env.reset()
    env.render()  # Initialize display
    
    episode_reward = 0
    steps = 0
    
    velocity_cmd = 0.0
    angular_velocity_cmd = 0.0
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
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
                    print("\n✨ Episode reset")
        
        keys = pygame.key.get_pressed()
        
        # Update commands
        if keys[pygame.K_UP]:
            velocity_cmd = min(velocity_cmd + 0.05, 1.0)
        elif keys[pygame.K_DOWN]:
            velocity_cmd = max(velocity_cmd - 0.05, -1.0)
        else:
            velocity_cmd *= 0.95
        
        if keys[pygame.K_LEFT]:
            angular_velocity_cmd = min(angular_velocity_cmd + 0.05, 1.0)
        elif keys[pygame.K_RIGHT]:
            angular_velocity_cmd = max(angular_velocity_cmd - 0.05, -1.0)
        else:
            angular_velocity_cmd *= 0.95
        
        action = np.array([velocity_cmd, angular_velocity_cmd], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        
        env.render()
        
        if terminated or truncated:
            reason = info.get("termination_reason") or info.get("truncation_reason")
            print(f"\n{'='*60}")
            print(f"Episode ended: {reason}")
            print(f"  Steps: {steps}")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"{'='*60}")
            
            pygame.time.wait(2000)
            obs, info = env.reset()
            episode_reward = 0
            steps = 0
            velocity_cmd = 0.0
            angular_velocity_cmd = 0.0
        
        clock.tick(20)


def demo_smart_policy(env, episodes=3):
    """Demo with smart heuristic showing off effects."""
    print("\n" + "="*60)
    print("🏎️  ENHANCED DRIFT DEMO - Smart Policy")
    print("="*60)
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\n Episode {episode + 1}/{episodes}")
        
        while True:
            # Heuristic control
            dist_to_gate = obs[6]
            angle_to_gate = np.arctan2(obs[8], obs[7])
            
            velocity_cmd = np.clip(0.6 + dist_to_gate * 0.4, -1.0, 1.0)
            angular_velocity_cmd = np.clip(angle_to_gate * 2.5, -1.0, 1.0)
            
            action = np.array([velocity_cmd, angular_velocity_cmd], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            env.render()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
            
            if terminated or truncated:
                reason = info.get("termination_reason") or info.get("truncation_reason")
                print(f"  Result: {reason} | Steps: {steps} | Reward: {episode_reward:.2f}")
                break
        
        pygame.time.wait(1500)


def main():
    parser = argparse.ArgumentParser(description="Enhanced Drift Demo")
    parser.add_argument(
        "--mode",
        type=str,
        default="keyboard",
        choices=["keyboard", "smart"],
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
        help="Number of episodes (for smart mode)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("#  🏎️  ENHANCED DRIFT SIMULATOR")
    print("#" + "#"*60)
    print(f"\n Mode: {args.mode}")
    print(f" Scenario: {args.scenario}")
    print(f"\n✨ New Features:")
    print("   💨 Tire smoke particles")
    print("   ⚫ Skid marks")
    print("   👻 Motion trail")
    print("   🎯 Drift indicator")
    print("   📊 Pro racing HUD")
    
    env = EnhancedDriftEnv(
        scenario=args.scenario,
        render_mode="human",
        dense_rewards=True
    )
    
    try:
        if args.mode == "keyboard":
            demo_keyboard_control(env)
        elif args.mode == "smart":
            demo_smart_policy(env, episodes=args.episodes)
    
    finally:
        env.close()
        print("\n" + "="*60)
        print("✨ Demo complete!")
        print("="*60)


if __name__ == "__main__":
    main()

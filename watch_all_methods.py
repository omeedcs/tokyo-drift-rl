#!/usr/bin/env python3
"""
Watch all methods perform the drift maneuver visually.
Simple script to compare Baseline, IKD, and SAC.
"""

import sys
sys.path.insert(0, 'jake-deep-rl-algos')

import pygame
import numpy as np
import torch
import deep_control as dc
from src.simulator.environment import SimulationEnvironment
from src.simulator.controller import DriftController
from src.models.ikd_model import IKDModel

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
PURPLE = (138, 92, 246)
ORANGE = (255, 140, 0)

class MethodDemo:
    def __init__(self, method_name, color, use_gym_env=False):
        self.method_name = method_name
        self.color = color
        self.use_gym_env = use_gym_env
        
        if use_gym_env:
            from src.rl.gym_drift_env import GymDriftEnv
            self.gym_env = GymDriftEnv(scenario="loose", max_steps=200, render_mode=None)
            self.env = None  # Will use gym_env
        else:
            self.env = SimulationEnvironment(dt=0.05)
            self.env.setup_loose_drift_test()
            self.gym_env = None
            
        self.trajectory = []
        self.complete = False
        self.steps = 0
        self.gym_obs = None
        self.gym_done = False
        
def load_sac_agent():
    """Load SAC agent."""
    from src.rl.gym_drift_env import GymDriftEnv
    env = GymDriftEnv(scenario="loose", max_steps=200, render_mode=None)
    
    agent = dc.sac.SACAgent(
        obs_space_size=10,
        act_space_size=2,
        log_std_low=-20,
        log_std_high=2,
        hidden_size=256
    )
    
    agent.actor.load_state_dict(torch.load("dc_saves/sac_loose_2/actor.pt", map_location='cpu'))
    agent.eval()
    return agent

def main():
    print("\n" + "="*60)
    print("Visual Comparison: Baseline vs IKD vs SAC")
    print("="*60)
    print("\nLoading models...")
    
    # Load IKD model
    ikd_model = IKDModel()
    ikd_model.load_state_dict(torch.load("trained_models/ikd_final.pt"))
    ikd_model.eval()
    
    # Load SAC agent
    sac_agent = load_sac_agent()
    
    print("✓ All models loaded!")
    print("\nStarting simulation...")
    print("Watch all three methods perform simultaneously!")
    print("Press ESC to exit\n")
    
    # Initialize Pygame
    pygame.init()
    width, height = 1400, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Drift Methods Comparison - Baseline vs IKD vs SAC")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    big_font = pygame.font.Font(None, 36)
    
    # Create three demos (SAC uses GymDriftEnv)
    baseline = MethodDemo("Baseline", BLUE, use_gym_env=False)
    ikd_demo = MethodDemo("IKD", PURPLE, use_gym_env=False)
    sac_demo = MethodDemo("SAC", GREEN, use_gym_env=True)
    
    # Setup controllers
    gate_center = (3.0, 1.065)
    gate_width = 2.13
    
    baseline_controller = DriftController(use_optimizer=False)
    obstacles_list = [(obs.x, obs.y, obs.radius) for obs in baseline.env.obstacles]
    baseline_controller.plan_trajectory(
        start_pos=(0.0, 0.0),
        gate_center=gate_center,
        gate_width=gate_width,
        direction="ccw",
        obstacles=obstacles_list
    )
    
    ikd_controller = DriftController(use_optimizer=False)
    ikd_controller.plan_trajectory(
        start_pos=(0.0, 0.0),
        gate_center=gate_center,
        gate_width=gate_width,
        direction="ccw",
        obstacles=obstacles_list
    )
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        
        screen.fill(BLACK)
        
        # Update each method
        if not baseline.complete:
            state = baseline.env.vehicle.get_state()
            vel_cmd, av_cmd = baseline_controller.update(state.x, state.y, state.theta, state.velocity)
            baseline.env.set_control(vel_cmd, av_cmd)
            baseline.env.step()
            baseline.trajectory.append((state.x, state.y))
            baseline.steps += 1
            if baseline.env.check_collision() or baseline_controller.is_complete() or baseline.steps > 200:
                baseline.complete = True
        
        if not ikd_demo.complete:
            state = ikd_demo.env.vehicle.get_state()
            vel_cmd, av_cmd = ikd_controller.update(state.x, state.y, state.theta, state.velocity)
            # Apply IKD correction
            with torch.no_grad():
                model_input = torch.FloatTensor([vel_cmd, av_cmd]).unsqueeze(0)
                correction = ikd_model(model_input).item()
                vel_cmd = vel_cmd + correction
            ikd_demo.env.set_control(vel_cmd, av_cmd)
            ikd_demo.env.step()
            ikd_demo.trajectory.append((state.x, state.y))
            ikd_demo.steps += 1
            if ikd_demo.env.check_collision() or ikd_controller.is_complete() or ikd_demo.steps > 200:
                ikd_demo.complete = True
        
        if not sac_demo.complete:
            # SAC uses GymDriftEnv
            if sac_demo.gym_obs is None:
                # First step
                result = sac_demo.gym_env.reset()
                sac_demo.gym_obs = result[0] if isinstance(result, tuple) else result
            
            # Get action from SAC
            with torch.no_grad():
                action = sac_agent.forward(torch.FloatTensor(sac_demo.gym_obs))
            
            # Step the gym environment
            step_result = sac_demo.gym_env.step(action)
            if len(step_result) == 5:
                sac_demo.gym_obs, reward, terminated, truncated, info = step_result
                sac_demo.gym_done = terminated or truncated
            else:
                sac_demo.gym_obs, reward, sac_demo.gym_done, info = step_result
            
            # Get position for visualization
            state = sac_demo.gym_env.sim_env.vehicle.get_state()
            sac_demo.trajectory.append((state.x, state.y))
            sac_demo.steps += 1
            
            if sac_demo.gym_done or sac_demo.steps > 200:
                sac_demo.complete = True
        
        # Draw all three side by side
        demos = [baseline, ikd_demo, sac_demo]
        panel_width = width // 3
        
        for i, demo in enumerate(demos):
            x_offset = i * panel_width
            
            # Draw track
            scale = 80
            center_x = x_offset + panel_width // 2
            center_y = height // 2
            
            # Draw obstacles
            env_to_use = demo.gym_env.sim_env if demo.use_gym_env else demo.env
            for obs in env_to_use.obstacles:
                screen_x = center_x + int(obs.x * scale)
                screen_y = center_y - int(obs.y * scale)
                pygame.draw.circle(screen, RED, (screen_x, screen_y), int(obs.radius * scale))
            
            # Draw goal
            goal_x = center_x + int(gate_center[0] * scale)
            goal_y = center_y - int(gate_center[1] * scale)
            pygame.draw.rect(screen, GREEN, (goal_x - 15, goal_y - 15, 30, 30), 3)
            
            # Draw trajectory
            if len(demo.trajectory) > 1:
                points = [(center_x + int(x * scale), center_y - int(y * scale)) 
                         for x, y in demo.trajectory]
                pygame.draw.lines(screen, demo.color, False, points, 2)
            
            # Draw current position
            if demo.trajectory:
                x, y = demo.trajectory[-1]
                screen_x = center_x + int(x * scale)
                screen_y = center_y - int(y * scale)
                pygame.draw.circle(screen, demo.color, (screen_x, screen_y), 8)
            
            # Draw info
            title = big_font.render(demo.method_name, True, demo.color)
            screen.blit(title, (x_offset + 20, 20))
            
            status = "✓ Complete" if demo.complete else "Running..."
            status_color = GREEN if demo.complete else WHITE
            status_text = font.render(status, True, status_color)
            screen.blit(status_text, (x_offset + 20, 60))
            
            steps_text = font.render(f"Steps: {demo.steps}", True, WHITE)
            screen.blit(steps_text, (x_offset + 20, 90))
            
            # Draw separator
            if i < 2:
                pygame.draw.line(screen, WHITE, (x_offset + panel_width, 0), 
                               (x_offset + panel_width, height), 1)
        
        # Check if all complete
        if all(d.complete for d in demos):
            # Show final comparison
            result_y = height - 100
            final_text = big_font.render("FINAL RESULTS", True, WHITE)
            screen.blit(final_text, (width // 2 - 100, result_y))
            
            for i, demo in enumerate(demos):
                result = font.render(f"{demo.method_name}: {demo.steps} steps", True, demo.color)
                screen.blit(result, (50 + i * panel_width, result_y + 40))
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Baseline: {baseline.steps} steps")
    print(f"IKD:      {ikd_demo.steps} steps")
    print(f"SAC:      {sac_demo.steps} steps")
    print()

if __name__ == "__main__":
    main()

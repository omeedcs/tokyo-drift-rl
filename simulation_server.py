#!/usr/bin/env python3
"""
WebSocket Simulation Server for Live Streaming

Captures pygame simulation frames and streams them to the web UI in real-time.
"""

import sys
import os

# Set pygame to use dummy video driver (no window) - MUST be before pygame import
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

sys.path.insert(0, 'jake-deep-rl-algos')

import asyncio
import base64
import io
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import pygame
import numpy as np
from PIL import Image
import threading
import time
import torch

from src.utils.model_loader import PretrainedModelLoader
from src.rl.gym_drift_env import GymDriftEnv
from drift_gym.envs import AdvancedDriftCarEnv
import deep_control as dc

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global simulation state
simulation_running = False
current_simulation = None
model_loader = PretrainedModelLoader()

# Environment types
ENV_SIMPLE = 'simple'      # Original demonstration environment
ENV_RESEARCH = 'research'  # Research-grade with sensors/EKF


class SimulationRunner:
    """Runs simulation and streams frames via WebSocket"""
    
    def __init__(self, model_name: str, scenario: str = "loose", max_steps: int = 500, env_type: str = ENV_SIMPLE):
        self.model_name = model_name
        self.scenario = scenario
        self.max_steps = max_steps
        self.env_type = env_type
        self.running = False
        self.env = None
        self.agent = None
        
    def start(self):
        """Start the simulation in a separate thread"""
        self.running = True
        thread = threading.Thread(target=self._run_simulation)
        thread.daemon = True
        thread.start()
        
    def stop(self):
        """Stop the simulation"""
        self.running = False
        if self.env:
            self.env.close()
    
    def _run_simulation(self):
        """Run the simulation loop"""
        try:
            # Load model - detect type from name
            print(f"Loading model: {self.model_name}")
            is_ikd = 'ikd' in self.model_name.lower()
            
            if is_ikd:
                # Load IKD model
                self.agent, _ = model_loader.load_ikd(self.model_name)
                print(f"Loaded IKD model: {self.model_name}")
            else:
                # Load SAC model
                self.agent, _ = model_loader.load_sac(self.model_name)
                print(f"Loaded SAC model: {self.model_name}")
            
            # Create environment based on type
            if self.env_type == ENV_RESEARCH:
                print(f"Creating research-grade environment (sensors + EKF)")
                self.env = AdvancedDriftCarEnv(
                    scenario=self.scenario,
                    use_noisy_sensors=True,
                    use_perception_pipeline=False,
                    use_latency=False,
                    render_mode="rgb_array",
                    seed=42
                )
            else:
                print(f"Creating simple demonstration environment")
                self.env = GymDriftEnv(
                    scenario=self.scenario,
                    max_steps=self.max_steps,
                    render_mode="rgb_array"
                )
            
            obs, _ = self.env.reset()
            step = 0
            
            while self.running and step < self.max_steps:
                # Get action from agent
                if is_ikd:
                    # IKD uses simple tracking (this is simplified - adjust based on your IKD implementation)
                    action = np.array([0.5, 0.2])  # Replace with actual IKD forward pass
                else:
                    # SAC forward pass
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs)
                        action = self.agent.forward(obs_tensor)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Render frame
                frame = self.env.render()
                if frame is not None:
                    # Convert numpy array to base64
                    frame_base64 = self._frame_to_base64(frame)
                    
                    # Emit frame via WebSocket
                    socketio.emit('simulation_frame', {'frame': frame_base64})
                    
                    # Emit metrics (handle different environment types)
                    try:
                        if self.env_type == ENV_RESEARCH:
                            # Research env has different access pattern
                            state = self.env.sim_env.vehicle.get_state()
                        else:
                            # Simple env
                            state = self.env.sim_env.vehicle.get_state()
                        
                        metrics = {
                            'step': step,
                            'reward': float(reward),
                            'position': {'x': float(state.x), 'y': float(state.y)},
                            'velocity': float(state.velocity),
                            'angular_velocity': float(state.angular_velocity),
                            'env_type': self.env_type
                        }
                        socketio.emit('simulation_metrics', metrics)
                    except Exception as e:
                        print(f"Warning: Could not emit metrics: {e}")
                
                step += 1
                
                # Control frame rate
                time.sleep(0.03)  # ~30 FPS
                
                if terminated or truncated:
                    socketio.emit('simulation_complete', {
                        'steps': step,
                        'success': 'success' in info.get('termination_reason', ''),
                        'reason': info.get('termination_reason', 'unknown')
                    })
                    break
            
        except Exception as e:
            print(f"Simulation error: {e}")
            socketio.emit('simulation_error', {'error': str(e)})
        finally:
            if self.env:
                self.env.close()
            self.running = False
    
    @staticmethod
    def _frame_to_base64(frame: np.ndarray) -> str:
        """Convert numpy frame to base64 string"""
        # Convert RGB array to PIL Image
        img = Image.fromarray(frame.astype('uint8'), 'RGB')
        
        # Encode to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_base64


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'status': 'ok'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')
    global current_simulation
    if current_simulation:
        current_simulation.stop()
        current_simulation = None


@socketio.on('start_simulation')
def handle_start_simulation(data):
    """Start a new simulation"""
    global current_simulation, simulation_running
    
    if current_simulation:
        current_simulation.stop()
    
    model_name = data.get('model', 'sac_loose_2')
    scenario = data.get('scenario', 'loose')
    max_steps = data.get('max_steps', 500)
    env_type = data.get('env_type', ENV_SIMPLE)  # Default to simple
    
    print(f"Starting simulation: model={model_name}, scenario={scenario}, max_steps={max_steps}, env={env_type}")
    
    current_simulation = SimulationRunner(model_name, scenario, max_steps, env_type)
    current_simulation.start()
    simulation_running = True
    
    emit('simulation_started', {
        'model': model_name,
        'scenario': scenario,
        'max_steps': max_steps,
        'env_type': env_type
    })


@socketio.on('stop_simulation')
def handle_stop_simulation():
    """Stop current simulation"""
    global current_simulation, simulation_running
    
    if current_simulation:
        print("Stopping simulation")
        current_simulation.stop()
        current_simulation = None
        simulation_running = False
        emit('simulation_stopped', {})


@socketio.on('restart_simulation')
def handle_restart_simulation(data):
    """Restart simulation"""
    handle_stop_simulation()
    time.sleep(0.5)
    handle_start_simulation(data)


@socketio.on('get_available_models')
def handle_get_models():
    """Get list of available models"""
    models = model_loader.list_available_models()
    emit('available_models', {
        'ikd': [m.name for m in models['ikd']],
        'sac': [m.name for m in models['sac']]
    })


@socketio.on('get_environment_types')
def handle_get_env_types():
    """Get list of available environment types"""
    emit('environment_types', {
        'types': [
            {
                'id': ENV_SIMPLE,
                'name': 'Simple (Demo)',
                'description': 'Original environment for demonstration - fast, works with old models'
            },
            {
                'id': ENV_RESEARCH,
                'name': 'Research-Grade',
                'description': 'Advanced environment with GPS, IMU, and EKF - requires new models'
            }
        ],
        'default': ENV_SIMPLE
    })


if __name__ == '__main__':
    print("="*70)
    print("🚀 Simulation WebSocket Server")
    print("="*70)
    print("\nStarting server on http://localhost:5001")
    print("Waiting for connections...\n")
    
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)

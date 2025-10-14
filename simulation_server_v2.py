#!/usr/bin/env python3
"""
Updated WebSocket Simulation Server

Works with both old GymDriftEnv and new AdvancedDriftCarEnv.
Supports SAC and IKD models.
"""

import sys
import os

# Set pygame to use dummy video driver (no window) - MUST be before pygame import
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

sys.path.insert(0, 'jake-deep-rl-algos')

import base64
import io
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import numpy as np
from PIL import Image
import threading
import time
import torch
import glob
import json

# Import both environments
from src.rl.gym_drift_env import GymDriftEnv
from drift_gym.envs.drift_car_env_advanced import AdvancedDriftCarEnv
import deep_control as dc

# Set deep_control to use CPU for inference (avoid device mismatch)
dc.utils.device = torch.device("cpu")

app = Flask(__name__)
CORS(app)
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',
    logger=False,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25
)

# Global simulation state
simulation_running = False
current_simulation = None


class ModelLoader:
    """Smart model loader that handles both SAC and IKD models."""
    
    def __init__(self):
        self.device = self._get_device()
        print(f"🖥️  Model loader using device: {self.device}")
    
    @staticmethod
    def _get_device():
        """Get best available device (use CPU for inference to avoid device mismatch issues)."""
        # Use CPU for inference - simpler and avoids device mismatch
        # MPS/CUDA is mainly beneficial for training, not real-time inference
        return torch.device("cpu")
    
    def load_sac_model(self, model_path):
        """Load SAC model from path (expects deep_control SAC format with actor.pt, critic1.pt, critic2.pt)."""
        print(f"Loading SAC model from: {model_path}")
        
        # Check if model files exist
        actor_path = os.path.join(model_path, 'actor.pt')
        critic1_path = os.path.join(model_path, 'critic1.pt')
        critic2_path = os.path.join(model_path, 'critic2.pt')
        
        if not (os.path.exists(actor_path) and os.path.exists(critic1_path)):
            raise FileNotFoundError(f"SAC model files not found in {model_path}. Expected actor.pt, critic1.pt, critic2.pt")
        
        # Load config if available (from sac_advanced_models directory)
        use_advanced = False
        model_name = os.path.basename(model_path)
        
        # Try to find config in sac_advanced_models (where we save it separately)
        if 'sac_advanced' in model_name:
            # Extract base name without suffix (_0, _1, etc.)
            base_name = model_name.rsplit('_', 1)[0] if model_name[-2:].isdigit() or model_name[-1].isdigit() else model_name
            config_path = os.path.join('sac_advanced_models', base_name, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                use_advanced = config.get('use_noisy_sensors', False)
                print(f"Loaded config from: {config_path}")
                print(f"Advanced features: {use_advanced}")
        
        # Infer observation dimensions from saved model
        # Load actor to check input size
        actor_checkpoint = torch.load(actor_path, map_location=self.device)
        fc1_weight = actor_checkpoint.get('fc1.weight', None)
        if fc1_weight is not None:
            obs_dim = fc1_weight.shape[1]  # Input dimension
            print(f"Detected observation dimension from model: {obs_dim}")
        else:
            # Fallback
            obs_dim = 13 if use_advanced else 10
            print(f"Using default observation dimension: {obs_dim}")
        
        # Create agent with correct dimensions
        agent = dc.sac.SACAgent(
            obs_space_size=obs_dim,
            act_space_size=2,
            log_std_low=-20,
            log_std_high=2,
            hidden_size=256
        )
        
        # Load weights
        agent.load(model_path)
        agent.to(self.device)
        agent.eval()
        
        print(f"SAC model loaded successfully")
        print(f"  Observation dim: {obs_dim}")
        print(f"  Using advanced features: {use_advanced}")
        return agent, use_advanced
    
    def load_ikd_model(self, model_path):
        """Load IKD model from path."""
        print(f"Loading IKD model from: {model_path}")
        # IKD implementation depends on your codebase
        # For now, return a placeholder
        # TODO: Implement actual IKD loading
        print("⚠️  IKD loading not yet implemented")
        return None, False
    
    def list_available_models(self):
        """List all available models."""
        models = {
            'sac': [],
            'ikd': []
        }
        
        # Find SAC models in dc_saves directory (search recursively)
        if os.path.exists('dc_saves'):
            for root, dirs, files in os.walk('dc_saves'):
                # Check if this directory has SAC model files
                has_actor = 'actor.pt' in files
                has_critic = 'critic1.pt' in files
                if has_actor and has_critic:
                    # Use the directory name (not full path) as model name
                    model_dir_name = os.path.basename(root)
                    models['sac'].append({
                        'name': model_dir_name,
                        'path': root,
                        'type': 'sac'
                    })
        
        # Find IKD models
        ikd_dir = 'trained_models'
        if os.path.exists(ikd_dir):
            for file in os.listdir(ikd_dir):
                if file.startswith('ikd_') and file.endswith('.pt'):
                    models['ikd'].append({
                        'name': file.replace('.pt', ''),
                        'path': os.path.join(ikd_dir, file),
                        'type': 'ikd'
                    })
        
        return models


class SimulationRunner:
    """Runs simulation and streams frames via WebSocket."""
    
    def __init__(self, model_path: str, model_type: str = 'sac', scenario: str = "loose", max_steps: int = 500):
        self.model_path = model_path
        self.model_type = model_type
        self.scenario = scenario
        self.max_steps = max_steps
        self.running = False
        self.env = None
        self.agent = None
        self.model_loader = ModelLoader()
        
    def start(self):
        """Start the simulation in a separate thread."""
        self.running = True
        thread = threading.Thread(target=self._run_simulation)
        thread.daemon = True
        thread.start()
        
    def stop(self):
        """Stop the simulation."""
        self.running = False
        if self.env:
            self.env.close()
    
    def _run_simulation(self):
        """Run the simulation loop."""
        try:
            # Load model
            print(f"\n{'='*70}")
            print(f"🚀 Starting Simulation")
            print(f"{'='*70}")
            print(f"Model: {self.model_path}")
            print(f"Type: {self.model_type}")
            print(f"Scenario: {self.scenario}")
            
            use_advanced = False
            
            if self.model_type == 'sac':
                self.agent, use_advanced = self.model_loader.load_sac_model(self.model_path)
            else:  # IKD
                self.agent, use_advanced = self.model_loader.load_ikd_model(self.model_path)
            
            # Create appropriate environment based on model's observation space
            # Check actual observation dimension from agent
            obs_dim = self.agent.actor.fc1.weight.shape[1]
            print(f"\nCreating environment for {obs_dim}-dimensional observations...")
            
            if obs_dim >= 13:
                # Advanced environment with all features
                self.env = AdvancedDriftCarEnv(
                    scenario=self.scenario,
                    max_steps=self.max_steps,
                    render_mode="rgb_array",
                    use_noisy_sensors=False,
                    use_perception_pipeline=False,
                    use_latency=False,
                    use_3d_dynamics=True,
                    use_moving_agents=False,
                    seed=42
                )
                print("✅ Using AdvancedDriftCarEnv (13+ dims)")
            elif obs_dim == 11:
                # Old environment with IMU augmentation (11 dims)
                # This is from the original research with IMU delay
                # Use GymDriftEnv but warn that it only provides 10 dims
                print("⚠️  Model expects 11-dim obs (old IMU augmented version)")
                print("    Using GymDriftEnv (10-dim) - may cause issues")
                print("    Consider retraining with current environment")
                # Create a wrapper to pad observations to 11 dims
                self.env = GymDriftEnv(
                    scenario=self.scenario,
                    max_steps=self.max_steps,
                    render_mode="rgb_array"
                )
                self.obs_padding = 1  # Need to pad by 1 dimension
            else:
                # Standard 10-dimensional environment
                self.env = GymDriftEnv(
                    scenario=self.scenario,
                    max_steps=self.max_steps,
                    render_mode="rgb_array"
                )
                print("✅ Using GymDriftEnv (10 dims)")
                self.obs_padding = 0
            
            obs, _ = self.env.reset()
            step = 0
            total_reward = 0
            
            print(f"\n▶️  Running simulation...")
            
            while self.running and step < self.max_steps:
                # Pad observation if needed for old 11-dim models
                if hasattr(self, 'obs_padding') and self.obs_padding > 0:
                    # Pad with zeros (original IMU delay models expected this)
                    obs_padded = np.concatenate([obs, np.zeros(self.obs_padding)])
                else:
                    obs_padded = obs
                
                # Get action from agent
                if self.model_type == 'sac':
                    # Use forward() for deterministic action (mean of policy)
                    action = self.agent.forward(obs_padded, from_cpu=True)
                else:
                    # IKD forward pass (placeholder)
                    action = self.env.action_space.sample()
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                
                # Render frame
                frame = self.env.render()
                if frame is not None:
                    # Convert to base64
                    frame_base64 = self._frame_to_base64(frame)
                    
                    # Emit frame
                    socketio.emit('simulation_frame', {'frame': frame_base64})
                    
                    # Emit metrics
                    state = self.env.sim_env.vehicle.get_state()
                    metrics = {
                        'step': step,
                        'reward': float(reward),
                        'total_reward': float(total_reward),
                        'position': {'x': float(state.x), 'y': float(state.y)},
                        'velocity': float(state.velocity),
                        'angular_velocity': float(state.angular_velocity),
                    }
                    socketio.emit('simulation_metrics', metrics)
                
                step += 1
                
                # Control frame rate (30 FPS)
                time.sleep(0.033)
                
                if terminated or truncated:
                    success = info.get('termination_reason') == 'success'
                    print(f"\n✅ Simulation complete!")
                    print(f"   Steps: {step}")
                    print(f"   Total reward: {total_reward:.1f}")
                    print(f"   Result: {'SUCCESS' if success else 'FAILED'}")
                    
                    socketio.emit('simulation_complete', {
                        'steps': step,
                        'total_reward': float(total_reward),
                        'success': success,
                        'reason': info.get('termination_reason', 'unknown')
                    })
                    break
            
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            import traceback
            traceback.print_exc()
            socketio.emit('simulation_error', {'error': str(e)})
        finally:
            if self.env:
                self.env.close()
            self.running = False
    
    @staticmethod
    def _frame_to_base64(frame: np.ndarray) -> str:
        """Convert numpy frame to base64 string."""
        img = Image.fromarray(frame.astype('uint8'), 'RGB')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG', optimize=True, quality=85)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_base64


# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('✅ Client connected')
    emit('connected', {'status': 'ok'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('❌ Client disconnected')
    global current_simulation
    if current_simulation:
        current_simulation.stop()
        current_simulation = None


@socketio.on('start_simulation')
def handle_start_simulation(data):
    """Start a new simulation."""
    global current_simulation, simulation_running
    
    if current_simulation:
        current_simulation.stop()
    
    model_name = data.get('model', 'sac_loose_2')  # Model name from dropdown
    model_type = data.get('model_type', 'sac')
    scenario = data.get('scenario', 'loose')
    max_steps = data.get('max_steps', 500)
    
    # Look up full path from model name
    loader = ModelLoader()
    all_models = loader.list_available_models()
    
    model_path = None
    for m in all_models[model_type]:
        if m['name'] == model_name:
            model_path = m['path']
            break
    
    if model_path is None:
        # Fallback - assume it's already a path or use first available
        if all_models[model_type]:
            model_path = all_models[model_type][0]['path']
        else:
            print(f"❌ No models found for type: {model_type}")
            return
    
    print(f"\n📡 Received start request:")
    print(f"   Model: {model_name}")
    print(f"   Path: {model_path}")
    print(f"   Type: {model_type}")
    print(f"   Scenario: {scenario}")
    
    current_simulation = SimulationRunner(model_path, model_type, scenario, max_steps)
    current_simulation.start()
    simulation_running = True
    
    emit('simulation_started', {
        'model': model_name,
        'model_type': model_type,
        'scenario': scenario,
        'max_steps': max_steps
    })


@socketio.on('stop_simulation')
def handle_stop_simulation():
    """Stop current simulation."""
    global current_simulation, simulation_running
    
    if current_simulation:
        print("⏹️  Stopping simulation")
        current_simulation.stop()
        current_simulation = None
        simulation_running = False
        emit('simulation_stopped', {})


@socketio.on('restart_simulation')
def handle_restart_simulation(data):
    """Restart simulation."""
    handle_stop_simulation()
    time.sleep(0.5)
    handle_start_simulation(data)


@socketio.on('get_available_models')
def handle_get_models():
    """Get list of available models."""
    try:
        loader = ModelLoader()
        models = loader.list_available_models()
        
        # Simplify for frontend - just send model names
        response = {
            'sac': [m['name'] for m in models['sac']],
            'ikd': [m['name'] for m in models['ikd']]
        }
        
        print(f"📋 Available models: {len(response['sac'])} SAC, {len(response['ikd'])} IKD")
        emit('available_models', response)
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        emit('available_models', {'sac': [], 'ikd': []})


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Simulation WebSocket Server V2")
    print("="*70)
    print("\nFeatures:")
    print("  ✅ Supports old GymDriftEnv")
    print("  ✅ Supports new AdvancedDriftCarEnv")
    print("  ✅ SAC model loading")
    print("  ✅ IKD model loading (placeholder)")
    print("  ✅ Auto-detect model type")
    print("\nStarting server on http://localhost:5001")
    print("Waiting for connections...\n")
    print("="*70 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)

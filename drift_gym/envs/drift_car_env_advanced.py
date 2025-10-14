"""
Advanced Drift Car Environment with Research-Grade Features

Integrates:
- Sensor noise models (GPS drift, IMU bias)
- Perception pipeline (object detection with false positives)
- Latency modeling (sensor → compute → actuation delay)
- 3D dynamics (pitch/roll, weight transfer)
- Moving obstacles (other vehicles with behaviors)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
import pygame
import math

# Import research-grade modules
from drift_gym.sensors import GPSSensor, IMUSensor, LatencyBuffer, SensorReading
from drift_gym.perception import ObjectDetector, TrackingFilter, Detection, ObjectClass
from drift_gym.dynamics import Vehicle3DDynamics, Vehicle3DState
from drift_gym.agents import MovingAgentSimulator, AgentBehavior
from drift_gym.estimation import ExtendedKalmanFilter, EKFState

from src.simulator.environment import SimulationEnvironment


class AdvancedDriftCarEnv(gym.Env):
    """
    Research-grade drift environment with realistic sensors, perception, and dynamics.
    
    Key Features:
    - Noisy sensors (GPS, IMU) instead of perfect state
    - Object detection with false positives/negatives
    - Realistic latency in perception-action loop
    - 3D vehicle dynamics with weight transfer
    - Moving traffic agents with behaviors
    
    This environment is suitable for:
    - Sim-to-real research
    - Sensor fusion algorithms
    - Robust control under uncertainty
    - Multi-agent scenarios
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }
    
    def __init__(
        self,
        scenario: str = "loose",
        max_steps: int = 400,
        render_mode: Optional[str] = None,
        use_noisy_sensors: bool = True,
        use_perception_pipeline: bool = True,
        use_latency: bool = True,
        use_3d_dynamics: bool = True,
        use_moving_agents: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize advanced environment.
        
        Args:
            scenario: Scenario type
            max_steps: Maximum episode length
            render_mode: Rendering mode
            use_noisy_sensors: Enable GPS/IMU noise
            use_perception_pipeline: Enable object detection pipeline
            use_latency: Enable sensing/actuation delays
            use_3d_dynamics: Enable 3D dynamics (roll/pitch/weight transfer)
            use_moving_agents: Enable moving traffic agents
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.scenario = scenario
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.use_noisy_sensors = use_noisy_sensors
        self.use_perception_pipeline = use_perception_pipeline
        self.use_latency = use_latency
        self.use_3d_dynamics = use_3d_dynamics
        self.use_moving_agents = use_moving_agents
        
        # Action space: [velocity_cmd, angular_velocity_cmd]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # NEW Observation space: Task-relevant with EKF estimates
        # Observations:
        # [rel_goal_x, rel_goal_y, rel_goal_heading,  (3) - Task-relevant goal info
        #  v_estimated, omega_estimated,               (2) - EKF state estimates
        #  v_std, omega_std,                           (2) - EKF uncertainties
        #  n_obstacles, closest_obs_x, closest_obs_y,  (3) - Perception
        #  prev_action_v, prev_action_omega]           (2) - Action history
        # Total: 12 dimensions (more meaningful than old 13)
        
        obs_dim = 12
        self.observation_space = spaces.Box(
            low=-20.0,  # More reasonable bounds
            high=20.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Physical limits
        self.max_velocity = 4.0
        self.max_angular_velocity = 3.0
        
        # Initialize sensors (with calibrated parameters)
        if use_noisy_sensors:
            self.gps_sensor = GPSSensor(
                noise_std=0.3,  # Calibrated for RTK GPS
                drift_rate=0.005,
                update_rate=10.0,
                seed=seed
            )
            self.imu_sensor = IMUSensor(
                gyro_noise_std=0.0087,  # Calibrated for BMI088/MPU9250
                gyro_bias_std=0.0017,
                update_rate=100.0,
                seed=seed
            )
        else:
            self.gps_sensor = None
            self.imu_sensor = None
        
        # Initialize Extended Kalman Filter for sensor fusion
        self.ekf = ExtendedKalmanFilter(dt=0.05)
        self.use_ekf = use_noisy_sensors  # Use EKF when sensors are noisy
        
        # Initialize perception pipeline
        if use_perception_pipeline:
            self.object_detector = ObjectDetector(
                max_range=50.0,
                fov_angle=np.pi,
                false_positive_rate=0.05,
                false_negative_rate=0.10,
                seed=seed
            )
            self.tracking_filter = TrackingFilter()
        else:
            self.object_detector = None
            self.tracking_filter = None
        
        # Initialize latency buffer
        if use_latency:
            self.latency_buffer = LatencyBuffer(
                sensor_delay=0.05,
                compute_delay=0.03,
                actuation_delay=0.02,
                dt=0.05
            )
        else:
            self.latency_buffer = None
        
        # Initialize 3D dynamics
        if use_3d_dynamics:
            self.vehicle_3d = Vehicle3DDynamics(
                mass=1.5,
                wheelbase=0.25,
                cog_height=0.05
            )
            self.vehicle_state_3d = None
        else:
            self.vehicle_3d = None
            self.vehicle_state_3d = None
        
        # Initialize moving agents
        if use_moving_agents:
            self.agent_simulator = MovingAgentSimulator(seed=seed)
        else:
            self.agent_simulator = None
        
        # Simulation environment (2D fallback)
        self.sim_env = SimulationEnvironment(dt=0.05)
        
        # Episode tracking
        self.steps = 0
        self.episode_reward = 0.0
        self.current_time = 0.0
        self.previous_action = np.zeros(2)  # Track previous action for observation
        
        # Scenario configuration
        if scenario == "loose":
            self.gate_width = 2.13
            self.gate_pos = np.array([3.0, 1.065])
        else:  # tight
            self.gate_width = 0.81
            self.gate_pos = np.array([3.0, 0.405])
        
        # Pygame rendering
        self.window = None
        self.clock = None
        self.window_size = 800
        self.scale = 60
        self.camera_offset = np.array([100, self.window_size // 2])
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset simulation
        self.sim_env.reset()
        
        # Setup scenario
        if self.scenario == "loose":
            self.sim_env.setup_loose_drift_test()
        else:
            self.sim_env.setup_tight_drift_test()
        
        # Reset sensors
        if self.gps_sensor:
            self.gps_sensor.reset()
        if self.imu_sensor:
            self.imu_sensor.reset()
        
        # Reset perception
        if self.tracking_filter:
            self.tracking_filter.reset()
        
        # Reset latency buffer
        if self.latency_buffer:
            self.latency_buffer.reset()
        
        # Initialize 3D state
        if self.use_3d_dynamics:
            state_2d = self.sim_env.vehicle.get_state()
            self.vehicle_state_3d = Vehicle3DState(
                x=state_2d.x, y=state_2d.y, z=0.05,
                roll=0.0, pitch=0.0, yaw=state_2d.theta,
                vx=state_2d.velocity, vy=0.0, vz=0.0,
                wx=0.0, wy=0.0, wz=state_2d.angular_velocity
            )
        
        # Setup moving agents
        if self.use_moving_agents:
            self.agent_simulator.reset()
            # Add some traffic agents
            self.agent_simulator.add_agent(
                position=np.array([2.0, 2.0]),
                behavior=AgentBehavior.CIRCULAR,
                size=(1.0, 0.5),
                max_speed=2.0
            )
            self.agent_simulator.add_agent(
                position=np.array([5.0, 0.0]),
                behavior=AgentBehavior.LANE_FOLLOW,
                size=(2.0, 1.0),
                max_speed=3.0
            )
        
        self.steps = 0
        self.episode_reward = 0.0
        self.current_time = 0.0
        self.previous_action = np.zeros(2)
        
        # Reset EKF with initial state
        if self.use_ekf:
            state_2d = self.sim_env.vehicle.get_state()
            initial_state = np.array([
                state_2d.x, state_2d.y, state_2d.theta,
                state_2d.velocity, 0.0, state_2d.angular_velocity
            ])
            self.ekf.reset(initial_state)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step with realistic delays and noise."""
        
        # Add action to latency buffer
        if self.latency_buffer:
            self.latency_buffer.add_action(action)
            delayed_action = self.latency_buffer.get_delayed_action()
        else:
            delayed_action = action
        
        # Scale action to physical limits
        velocity_cmd = float(delayed_action[0]) * self.max_velocity
        angular_velocity_cmd = float(delayed_action[1]) * self.max_angular_velocity
        
        # Execute in simulator
        self.sim_env.set_control(velocity_cmd, angular_velocity_cmd)
        measurements = self.sim_env.step()
        
        # Update 3D dynamics if enabled
        if self.use_3d_dynamics and self.vehicle_state_3d:
            # Convert commands to steering/throttle
            steering = angular_velocity_cmd * 0.1  # Simplified mapping
            throttle = velocity_cmd / self.max_velocity
            
            self.vehicle_state_3d = self.vehicle_3d.step(
                self.vehicle_state_3d,
                steering,
                throttle,
                dt=0.05
            )
        
        # Update moving agents
        if self.use_moving_agents:
            self.agent_simulator.step(dt=0.05)
        
        # Update EKF with sensor measurements
        if self.use_ekf:
            self._update_ekf()
        
        # Get observation (with noise/perception)
        observation = self._get_obs()
        
        # Store current action for next observation
        self.previous_action = action.copy()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check termination
        terminated, truncated, info = self._check_done()
        
        self.steps += 1
        self.current_time += 0.05
        self.episode_reward += reward
        
        info["episode_reward"] = self.episode_reward
        info["steps"] = self.steps
        info["time"] = self.current_time
        
        return observation, reward, terminated, truncated, info
    
    def _update_ekf(self):
        """Update EKF with sensor measurements."""
        state_2d = self.sim_env.vehicle.get_state()
        
        # EKF prediction step
        self.ekf.predict()
        
        # GPS update
        if self.gps_sensor:
            gps_reading = self.gps_sensor.measure(
                np.array([state_2d.x, state_2d.y]),
                self.current_time
            )
            if gps_reading.valid:
                self.ekf.update_gps(gps_reading.data, gps_reading.variance)
        
        # IMU update
        if self.imu_sensor:
            true_gyro = np.array([0.0, 0.0, state_2d.angular_velocity])
            true_accel = np.array([0.0, 0.0, 9.81])
            
            imu_readings = self.imu_sensor.measure(true_gyro, true_accel, self.current_time)
            
            self.ekf.update_imu(
                imu_readings['gyro'].data[2],
                imu_readings['accel'].data[:2],
                np.concatenate([imu_readings['gyro'].variance[[2]], imu_readings['accel'].variance[:2]])
            )
    
    def _get_obs(self) -> np.ndarray:
        """
        Get task-relevant observation with EKF state estimates.
        
        NEW DESIGN: Observations are task-centric, not sensor-centric.
        Agent receives what it needs for control, not raw sensor data.
        
        Observation vector (12-dim):
        [rel_goal_x, rel_goal_y, rel_goal_heading,  # Task-relevant goal info
         v_estimated, omega_estimated,               # EKF state estimates
         v_std, omega_std,                           # EKF uncertainties
         n_obstacles, closest_obs_x, closest_obs_y,  # Perception
         prev_action_v, prev_action_omega]           # Action history
        """
        # Get state estimate from EKF or ground truth
        if self.use_ekf:
            ekf_state = self.ekf.get_state()
            x_est = ekf_state.x
            y_est = ekf_state.y
            theta_est = ekf_state.theta
            v_est = ekf_state.vx
            omega_est = ekf_state.omega
            pos_std, v_std, omega_std = ekf_state.uncertainty_norms
        else:
            # Perfect state (for baseline comparison)
            state_2d = self.sim_env.vehicle.get_state()
            x_est = state_2d.x
            y_est = state_2d.y
            theta_est = state_2d.theta
            v_est = state_2d.velocity
            omega_est = state_2d.angular_velocity
            v_std = 0.0
            omega_std = 0.0
        
        # Compute relative goal position in body frame
        dx_goal = self.gate_pos[0] - x_est
        dy_goal = self.gate_pos[1] - y_est
        
        # Rotate to body frame
        cos_theta = np.cos(theta_est)
        sin_theta = np.sin(theta_est)
        rel_goal_x = dx_goal * cos_theta + dy_goal * sin_theta
        rel_goal_y = -dx_goal * sin_theta + dy_goal * cos_theta
        
        # Compute relative heading to goal
        angle_to_goal = np.arctan2(dy_goal, dx_goal)
        rel_goal_heading = angle_to_goal - theta_est
        # Normalize to [-pi, pi]
        rel_goal_heading = np.arctan2(np.sin(rel_goal_heading), np.cos(rel_goal_heading))
        
        # Obstacle perception
        if self.use_perception_pipeline and self.object_detector:
            # Build list of true objects
            true_objects = []
            for obs in self.sim_env.obstacles:
                true_objects.append({
                    'position': np.array([obs.x, obs.y]),
                    'velocity': np.zeros(2),
                    'size': (obs.radius * 2, obs.radius * 2),
                    'class': ObjectClass.OBSTACLE
                })
            
            # Run object detection
            detections = self.object_detector.detect_objects(
                true_objects,
                np.array([x_est, y_est]),
                theta_est
            )
            
            # Update tracking
            tracks = self.tracking_filter.update(detections, dt=0.05)
            
            # Extract closest detection (in body frame)
            n_obstacles = len(tracks)
            if tracks:
                closest = min(tracks, key=lambda t: np.linalg.norm(t['position']))
                closest_obs_x, closest_obs_y = closest['position']
            else:
                closest_obs_x, closest_obs_y = 0.0, 0.0
        else:
            # Perfect detection in body frame
            n_obstacles = len(self.sim_env.obstacles)
            if self.sim_env.obstacles:
                # Find closest obstacle and transform to body frame
                min_dist = float('inf')
                closest_obs_x, closest_obs_y = 0.0, 0.0
                for obs in self.sim_env.obstacles:
                    dx = obs.x - x_est
                    dy = obs.y - y_est
                    dist = np.hypot(dx, dy)
                    if dist < min_dist:
                        min_dist = dist
                        # Transform to body frame
                        closest_obs_x = dx * cos_theta + dy * sin_theta
                        closest_obs_y = -dx * sin_theta + dy * cos_theta
            else:
                closest_obs_x, closest_obs_y = 0.0, 0.0
        
        # Build observation vector
        obs = np.array([
            rel_goal_x / 5.0,           # Normalize by typical distance
            rel_goal_y / 5.0,
            rel_goal_heading / np.pi,   # Normalize angle to [-1, 1]
            v_est / self.max_velocity,  # Normalize velocity
            omega_est / self.max_angular_velocity,  # Normalize omega
            v_std,                      # Already in reasonable range
            omega_std,
            n_obstacles / 10.0,         # Normalize count
            closest_obs_x / 5.0,        # Normalize distance
            closest_obs_y / 5.0,
            self.previous_action[0],    # Already in [-1, 1]
            self.previous_action[1]
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(self) -> float:
        """Compute reward - similar to base environment."""
        state = self.sim_env.vehicle.get_state()
        
        reward = 0.0
        
        # Progress toward gate
        dx = self.gate_pos[0] - state.x
        dy = self.gate_pos[1] - state.y
        dist_to_gate = math.sqrt(dx**2 + dy**2)
        
        reward += -0.1 * dist_to_gate
        
        # Collision penalty
        if self.sim_env.check_collision():
            reward += -50.0
        
        # Success bonus
        if self._passed_gate():
            reward += 50.0
        
        # Clip reward
        reward = np.clip(reward, -10.0, 10.0)
        
        return reward
    
    def _check_done(self) -> Tuple[bool, bool, Dict[str, Any]]:
        """Check termination conditions."""
        state = self.sim_env.vehicle.get_state()
        info = {}
        
        terminated = False
        
        if self.sim_env.check_collision():
            terminated = True
            info["termination_reason"] = "collision"
        elif self._passed_gate():
            terminated = True
            info["termination_reason"] = "success"
        
        truncated = False
        
        if self.steps >= self.max_steps:
            truncated = True
            info["truncation_reason"] = "max_steps"
        
        return terminated, truncated, info
    
    def _passed_gate(self) -> bool:
        """Check if vehicle passed through gate."""
        state = self.sim_env.vehicle.get_state()
        
        if state.x <= self.gate_pos[0] - 0.5:
            return False
        
        if state.x >= self.gate_pos[0] + 0.5:
            y_min = self.gate_pos[1] - self.gate_width / 2
            y_max = self.gate_pos[1] + self.gate_width / 2
            
            if y_min <= state.y <= y_max:
                return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dict."""
        state = self.sim_env.vehicle.get_state()
        info = {
            "x": state.x,
            "y": state.y,
            "theta": state.theta,
            "velocity": state.velocity,
            "angular_velocity": state.angular_velocity,
            "scenario": self.scenario
        }
        
        # Add 3D state if available
        if self.use_3d_dynamics and self.vehicle_state_3d:
            info["roll"] = self.vehicle_state_3d.roll
            info["pitch"] = self.vehicle_state_3d.pitch
        
        # Add sensor info
        if self.use_noisy_sensors:
            info["sensor_mode"] = "noisy"
        else:
            info["sensor_mode"] = "perfect"
        
        return info
    
    def render(self):
        """Render environment (basic implementation)."""
        if self.render_mode == "human":
            # Could add visualization of detections, etc.
            pass
        elif self.render_mode == "rgb_array":
            # Return RGB array
            pass
    
    def close(self):
        """Clean up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

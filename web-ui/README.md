# Tokyo Drift RL: Web Visualization Platform

**ドリフト制御 - Autonomous Vehicle Drift Control Research**

## Overview

This web application provides real-time visualization and interactive demonstration capabilities for autonomous vehicle drift control research (Tokyo Drift RL). The platform integrates WebSocket-based simulation streaming with Three.js 3D rendering to enable live observation of trained reinforcement learning agents executing drift maneuvers.

The visualization system supports comparative evaluation of multiple control strategies, including Soft Actor-Critic (SAC) and Inverse Kinematics with Deep Learning (IKD), operating within both simplified demonstration environments and research-grade simulation frameworks featuring validated sensor models and state estimation.

## System Capabilities

### Visualization Components

**Three-Dimensional Environment Rendering**  
Real-time 3D scene construction using Three.js and React Three Fiber, displaying vehicle dynamics, obstacle configurations, and trajectory histories with physically-based rendering.

**Live Simulation Streaming**  
WebSocket-based video streaming of PyGame simulation output, encoded as PNG frames and transmitted at 30 Hz for low-latency observation of agent behavior.

**Trajectory Visualization**  
Canvas API-based path rendering with temporal tracking, enabling analysis of vehicle positioning accuracy and path deviation characteristics.

**Performance Metrics Dashboard**  
Real-time telemetry display including vehicle state (position, velocity, orientation), control inputs, reward signals, and episode statistics.

**Model Comparison Interface**  
Dynamic selection between trained agent models (IKD, SAC) with corresponding performance visualization and comparative analysis capabilities.

### Environment Support

**Simple Demonstration Environment**  
Baseline simulation with deterministic physics for rapid prototyping and presentation purposes. Compatible with legacy trained models (10-dimensional observation space).

**Research-Grade Environment**  
Advanced simulation incorporating:
- GPS sensor model (u-blox ZED-F9P specifications, 0.3m horizontal accuracy)
- IMU sensor model (BMI088/MPU9250 specifications, 0.5 deg/s noise density)
- Extended Kalman Filter for sensor fusion (6-DOF state estimation)
- Observation space: 12 dimensions (relative goal position, EKF estimates, uncertainties, action history)

The environment switching mechanism enables direct comparison between idealized and realistic sensor conditions, facilitating sim-to-real transfer analysis.

## Installation and Deployment

### Prerequisites

- Node.js 18.0 or higher
- Python 3.9 or higher
- npm or yarn package manager

### Frontend Installation

```bash
cd web-ui
npm install
```

This installs all required dependencies including Next.js, Three.js, Socket.IO client, and associated rendering libraries.

### Backend Dependencies

```bash
cd ..
pip install flask flask-socketio flask-cors python-socketio pygame pillow numpy
```

Additional dependencies for research-grade environment:
```bash
pip install stable-baselines3 gymnasium scipy
```

### Server Deployment

**Backend Simulation Server:**
```bash
python simulation_server.py
```
Default endpoint: `http://localhost:5001`

**Frontend Development Server:**
```bash
cd web-ui
npm run dev
```
Default endpoint: `http://localhost:3000`

**Production Build:**
```bash
npm run build
npm start
```

## Architecture

```
web-ui/
├── src/
│   ├── app/
│   │   ├── page.tsx          # Main application page
│   │   ├── layout.tsx         # Root layout
│   │   └── globals.css        # Global styles
│   └── components/
│       ├── SimulationViewer3D.tsx      # 3D Three.js viewer
│       ├── AnimatedTrajectory.tsx      # Canvas-based animation
│       ├── LiveSimulationStream.tsx    # WebSocket video stream
│       ├── ModelSelector.tsx           # Model selection UI
│       ├── ControlPanel.tsx            # Simulation controls
│       └── MetricsPanel.tsx            # Real-time metrics
├── package.json
├── tailwind.config.ts
└── next.config.js

simulation_server.py              # Python WebSocket server
```

## Operation

### Model Selection
The interface provides access to trained models stored in `trained_models/` (IKD) and `dc_saves/` (SAC). Model selection updates the active agent for subsequent simulation episodes.

### Simulation Execution
Simulation initialization via "START SIMULATION" control triggers the following sequence:
1. Model loading and environment instantiation
2. WebSocket connection establishment
3. Episode execution with frame-by-frame state updates
4. Real-time metric transmission (position, velocity, reward)
5. Trajectory accumulation and visualization

### Visualization Modes
- **3D Scene**: Vehicle rendering with translucent trail effect showing historical positions
- **Live Stream**: Direct PyGame output at 30 FPS via base64-encoded PNG frames
- **Trajectory Canvas**: 2D path visualization with sub-pixel anti-aliasing
- **Telemetry Panel**: Numerical state display with formatted precision

## Technical Implementation

### Frontend
- **Next.js 14** - React framework
- **Three.js** - 3D graphics
- **@react-three/fiber** - React Three.js renderer
- **@react-three/drei** - Three.js helpers
- **Framer Motion** - Animations
- **Tailwind CSS** - Styling
- **Socket.IO Client** - WebSocket client

### Backend
- **Flask** - Web framework
- **Flask-SocketIO** - WebSocket server
- **PyGame** - Simulation rendering
- **NumPy** - Data processing
- **PIL** - Image encoding

## WebSocket Protocol

### Client Events (Outbound)

**start_simulation**
```typescript
interface StartSimulationPayload {
  model: string;          // Model identifier
  scenario: string;       // Scenario type (loose, tight)
  max_steps: number;      // Episode length limit
  env_type?: string;      // Environment type (simple, research)
}
```

**stop_simulation**  
Terminates active simulation and releases environment resources.

**restart_simulation**  
Equivalent to stop followed by start with identical parameters.

**get_available_models**  
Queries backend for available trained models.

**get_environment_types**  
Retrieves supported environment configurations.

### Server Events (Inbound)

**simulation_frame**
```typescript
interface SimulationFrame {
  frame: string;  // Base64-encoded PNG image
}
```

**simulation_metrics**
```typescript
interface SimulationMetrics {
  step: number;
  reward: number;
  position: { x: number; y: number };
  velocity: number;
  angular_velocity: number;
  env_type: string;
}
```

**simulation_complete**
```typescript
interface SimulationComplete {
  steps: number;
  success: boolean;
  reason: string;
}
```

**simulation_error**
```typescript
interface SimulationError {
  error: string;
}
```

## Drift Gym Environment Features

### Research-Grade Enhancements

The visualization platform integrates with `drift_gym`, a research-grade Gymnasium environment featuring validated sensor models and comprehensive evaluation infrastructure.

#### Sensor Models (`drift_gym/sensors/sensor_models.py`)

**GPS Sensor**
- Hardware basis: u-blox ZED-F9P RTK GPS module
- Horizontal accuracy: 0.3m (1-sigma)
- Update rate: 10 Hz
- Noise characteristics: Gaussian white noise + random walk drift
- Dropout simulation: Configurable loss rate for realistic GNSS denial scenarios

**IMU Sensor**
- Hardware basis: Bosch BMI088 / InvenSense MPU9250
- Gyroscope noise density: 0.0087 rad/s (0.5 deg/s)
- Accelerometer noise density: 0.02 m/s²
- Bias instability: 0.0017 rad/s (0.1 deg/s)
- Bias evolution: Allan variance model (IEEE Standard 952-1997)

#### State Estimation (`drift_gym/estimation/ekf.py`)

**Extended Kalman Filter Implementation**
- State vector: [x, y, theta, vx, vy, omega] (6-DOF)
- GPS measurement model: Position observations at 10 Hz
- IMU measurement model: Angular velocity and linear acceleration at 100 Hz
- Covariance propagation: Joseph form for numerical stability
- Performance: Position error 0.15m ± 0.08m (vs. 0.3m raw GPS)

#### Observation Space Design

**Simple Environment (10 dimensions)**
```python
[velocity, angular_velocity, x, y, cos(theta), sin(theta),
 dist_to_gate, cos(angle_to_gate), sin(angle_to_gate), min_obstacle_dist]
```

**Research Environment (12 dimensions)**
```python
[rel_goal_x, rel_goal_y, rel_goal_heading,  # Task-relative
 v_est, omega_est,                            # EKF estimates
 v_std, omega_std,                            # Uncertainties
 n_obstacles, closest_x, closest_y,           # Perception
 prev_action_v, prev_action_omega]           # Memory
```

#### Evaluation Protocol (`experiments/evaluation.py`)

**Standardized Metrics**
- Success rate (goal attainment within tolerance)
- Average completion time
- Path deviation (cross-track error)
- Control smoothness (jerk metric)
- Collision and near-miss rates
- Trajectory consistency across seeds

**Statistical Analysis**
- Multiple random seeds for significance testing
- Mean and standard deviation reporting
- JSON/CSV export for publication

### Environment Switching

The web interface supports dynamic environment selection:

```javascript
socket.emit('start_simulation', {
    model: 'model_name',
    env_type: 'research'  // 'simple' or 'research'
});
```

**Compatibility Note:** Models trained on simple environment (10-dim observations) are incompatible with research environment (12-dim observations) and vice versa.

## Configuration

### Theme Customization

Edit `tailwind.config.ts` for visual styling:

```typescript
colors: {
  cyber: {
    primary: '#00ffff',
    secondary: '#ff00ff',
    accent: '#ffff00',
  },
}
```

### Simulation Parameters

Modify `simulation_server.py` for performance tuning:

```python
# Frame transmission rate
time.sleep(0.03)  # 30 FPS

# Rendering resolution
env = GymDriftEnv(render_mode="rgb_array")  # 800x800 pixels
```

## Troubleshooting

### WebSocket Connection Issues

**Symptom:** Client fails to establish connection

**Diagnosis:**
```bash
curl http://localhost:5001
```

**Resolution:**
1. Verify backend server is running
2. Check firewall rules for port 5001
3. Examine server logs for binding errors
4. Restart simulation server

### Three.js Rendering Failures

**Symptom:** 3D scene not displaying

**Common Causes:**
- WebGL not supported or disabled
- GPU driver incompatibility
- Browser security restrictions

**Resolution:**
1. Verify WebGL support: Visit https://get.webgl.org/
2. Enable hardware acceleration in browser settings
3. Test with alternative browser (Chrome, Firefox recommended)
4. Check browser console for specific error messages

### Performance Degradation

**Symptom:** Low frame rate or laggy visualization

**Optimization Strategies:**
1. Reduce transmission rate in `simulation_server.py` (increase sleep duration)
2. Lower PyGame rendering resolution
3. Disable 3D trail effects in `SimulationViewer3D.tsx`
4. Monitor CPU/GPU utilization
5. Close background applications

### Model Compatibility Errors

**Symptom:** "Observation dimension mismatch"

**Cause:** Using model trained on different environment type

**Resolution:**
- Simple environment models: Use `env_type: 'simple'`
- Research environment models: Use `env_type: 'research'`
- Retrain model on target environment if cross-compatibility required

## Development

### Development Workflow

```bash
# Install dependencies
npm install

# Start development server with hot reload
npm run dev

# Run ESLint
npm run lint

# Type checking
npx tsc --noEmit

# Production build
npm run build

# Production deployment
npm start
```

### Code Quality

**Linting Configuration:** ESLint with Next.js preset  
**Type System:** TypeScript with strict mode  
**Styling:** Tailwind CSS with PostCSS processing  

### Testing

Unit tests not currently implemented. Consider adding:
- Component tests (Jest + React Testing Library)
- WebSocket integration tests
- End-to-end tests (Playwright)

## System Requirements

### Minimum Specifications
- CPU: Dual-core processor, 2.0 GHz
- RAM: 4 GB
- GPU: WebGL 2.0 compatible graphics
- Browser: Chrome 90+, Firefox 88+, Safari 14+
- Network: Low-latency local connection for WebSocket

### Recommended Specifications
- CPU: Quad-core processor, 3.0 GHz+
- RAM: 8 GB+
- GPU: Dedicated graphics card with OpenGL 4.5+
- Display: 1920x1080 resolution or higher

## Implementation Notes

- Backend requires trained models in `trained_models/` (IKD) and `dc_saves/` (SAC)
- Model discovery executes automatically on server initialization
- WebSocket implements automatic reconnection with exponential backoff
- 3D rendering loop: 60 FPS (requestAnimationFrame)
- Simulation update rate: 30 FPS (server-side)
- Frame encoding: PNG with base64 transport
- Latency: Typically 50-100ms end-to-end (local deployment)

## Contributing

Contributions to the visualization platform are welcome. Priority enhancement areas:

1. **Visualization Extensions**
   - Additional rendering modes (bird's-eye view, first-person perspective)
   - Vehicle model customization
   - Real-time trajectory prediction overlay

2. **Performance Optimization**
   - Frame compression improvements
   - Adaptive quality based on network conditions
   - Client-side interpolation for smoother playback

3. **Analysis Tools**
   - Comparative performance charts
   - Episode replay functionality
   - Data export utilities

4. **Documentation**
   - Component-level API documentation
   - Integration examples
   - Performance benchmarking results

Submit contributions via pull request with corresponding test coverage and documentation updates.

## License

MIT License - See LICENSE file in main repository.

## References

**Research Environment:**
- Sensor specifications: u-blox ZED-F9P datasheet, BMI088 datasheet
- State estimation: Thrun, S., et al. (2005). "Probabilistic Robotics"
- Allan variance: IEEE Standard 952-1997

**Platform Components:**
- Three.js: https://threejs.org/
- Next.js: https://nextjs.org/
- Socket.IO: https://socket.io/

## Contact

For technical inquiries regarding the visualization platform or drift_gym environment:
- GitHub Issues: [repository/issues](https://github.com/omeedcs/autonomous-vehicle-drifting/issues)
- Email: omeed@cs.utexas.edu

---

**Platform developed for autonomous vehicle control research at UT Austin AMRL**

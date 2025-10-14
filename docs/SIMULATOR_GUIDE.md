# IKD Simulator Guide

Since you no longer have access to physical F1/10 vehicles, this simulator allows you to:
- Test IKD models in realistic scenarios
- Generate synthetic training data
- Reproduce paper experiments
- Develop new algorithms without hardware

## 🚗 Vehicle Specifications

The simulator replicates the **UT AUTOmata F1/10** vehicle with high fidelity:

### Physical Parameters
- **Scale**: 1/10 F1 RC car
- **Wheelbase**: 0.324 meters (12.76 inches)
- **Width**: 0.48 meters (19 inches)
- **Weight**: ~4.5 kg (estimated)

### Actuators
- **Motor**: TRAXXAS Titan 550 brushless
- **Controller**: Flipsky VESC 4.12 50A
- **Max Speed**: 4.219 m/s (ERPM-limited)
- **Acceleration Limit**: 6.0 m/s²

### Steering
- **Type**: Ackerman steering
- **Max Turn Rate**: 0.25 radians
- **Servo Range**: 0.05 - 0.95 (normalized)

### Sensors
- **IMU**: Vectornav VN-100
  - Measures angular velocity (z-axis)
  - Typical delay: 0.18-0.20 seconds
  - Noise: σ ≈ 0.01 rad/s
- **Odometry**: Motor encoder
  - Measures linear velocity
  - Minimal noise: σ ≈ 0.001 m/s

### Dynamics
- **Drive**: Four-wheel drive
- **Tire Friction**: μ_static = 0.9, μ_kinetic = 0.7
- **Slip Threshold**: 3.0 rad/s
- **Control Rate**: 20 Hz

## 🎮 Quick Start

### Basic Circle Test
```bash
# Replicate Table I from paper (commanded curvature = 0.7)
python simulate.py --mode circle --velocity 2.0 --curvature 0.7 --duration 10.0

# Try different curvatures
python simulate.py --mode circle --velocity 2.0 --curvature 0.12
python simulate.py --mode circle --velocity 2.0 --curvature 0.63
python simulate.py --mode circle --velocity 2.0 --curvature 0.80
```

### Drift Tests
```bash
# Loose drift (Section IV-D1)
python simulate.py --mode drift-loose --duration 15.0

# Tight drift (Section IV-D2)
python simulate.py --mode drift-tight --duration 15.0
```

### With IKD Correction
```bash
# First train a model
python train.py

# Then test with IKD
python simulate.py \
  --mode circle \
  --velocity 2.0 \
  --curvature 0.7 \
  --use-ikd \
  --model experiments/ikd_baseline/checkpoints/best_model.pt
```

### Compare Baseline vs IKD
```bash
python simulate.py \
  --mode circle \
  --velocity 2.0 \
  --curvature 0.7 \
  --compare \
  --model experiments/ikd_baseline/checkpoints/best_model.pt
```

## 📊 Simulation Modes

### 1. Circle Navigation
Replicates Section IV-C from the paper.

**Purpose**: Test curvature tracking accuracy

**Command**:
```bash
python simulate.py --mode circle --velocity 2.0 --curvature 0.7
```

**Expected Output**:
- Trajectory plot (circle)
- Data plots (velocity, angular velocity, curvature)
- Measured radius and curvature deviation

**Paper Results** (Table I):
| Commanded | Expected Deviation |
|-----------|-------------------|
| 0.12 m    | 2.33%             |
| 0.63 m    | 0.11%             |
| 0.80 m    | 1.78%             |

### 2. Loose Drifting
Replicates Section IV-D1 from the paper.

**Setup**:
- 2 cones, 2 boxes
- Distance: 2.13 meters (84 inches)
- Turbo speed: 5.0 m/s

**Command**:
```bash
python simulate.py --mode drift-loose --duration 15.0
```

**Expected Behavior**:
- Vehicle drifts around cones
- Should NOT collide (loose spacing)
- IKD should tighten trajectory

**Paper Results** (Table II):
- CCW: 100% tightened turn rate
- CW: 50% tightened turn rate

### 3. Tight Drifting
Replicates Section IV-D2 from the paper.

**Setup**:
- 2 cones, 2 boxes
- Distance: 0.81 meters (32 inches)
- Vehicle width: 0.48 meters
- Clearance: Only 0.33 meters!

**Command**:
```bash
python simulate.py --mode drift-tight --duration 15.0
```

**Expected Behavior**:
- Very challenging maneuver
- Baseline may collide
- IKD may over-tighten (from paper findings)

**Paper Finding**: Model struggles with tight trajectories due to limited training data.

## 🔬 Advanced Usage

### Custom Control Logic

```python
from src.simulator import SimulationEnvironment, VirtualJoystick

# Create environment
env = SimulationEnvironment(dt=0.05, enable_slip=True)
joystick = VirtualJoystick()

# Setup circle test
env.setup_circle_test(velocity=2.0, curvature=0.7)

# Custom control loop
env.start_recording()
for i in range(200):
    # Your control logic here
    velocity = 2.0
    angular_velocity = 2.0 * 0.7
    
    env.set_control(velocity, angular_velocity)
    measurements = env.step()
    
    # Access measurements
    print(f"t={measurements['time']:.2f}, "
          f"av={measurements['angular_velocity']:.3f}")

env.stop_recording()
data = env.get_recorded_data()
```

### Generate Synthetic Training Data

```python
from src.simulator import SimulationEnvironment
import numpy as np

env = SimulationEnvironment(dt=0.05)
env.start_recording()

# Random trajectories
for episode in range(100):
    env.reset()
    
    for step in range(200):
        # Random control
        velocity = np.random.uniform(1.0, 4.0)
        curvature = np.random.uniform(-0.8, 0.8)
        angular_velocity = velocity * curvature
        
        env.set_control(velocity, angular_velocity)
        env.step()

# Save synthetic data
env.save_recorded_data("synthetic_training_data.csv")
```

### Test IKD Model

```python
import torch
from src.models.ikd_model import IKDModel
from src.simulator import SimulationEnvironment, VirtualJoystick

# Load model
model = IKDModel(2, 1)
checkpoint = torch.load("path/to/model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Setup
env = SimulationEnvironment()
joystick = VirtualJoystick()
joystick.load_ikd_model(model)
joystick.enable_ikd_correction(True)

# Run with IKD
env.setup_circle_test(2.0, 0.7)
joystick.set_circle_mode(2.0, 0.7)

for i in range(200):
    velocity_cmd, av_cmd = joystick.get_control(
        env.vehicle.get_velocity(),
        env.vehicle.get_angular_velocity()
    )
    env.set_control(velocity_cmd, av_cmd)
    env.step()
```

## 📐 Physics and Dynamics

### Kinematic Bicycle Model

The simulator uses the kinematic bicycle model with Ackerman steering:

```
dx/dt = v * cos(θ)
dy/dt = v * sin(θ)
dθ/dt = ω = (v / L) * tan(δ)
```

Where:
- `v` = linear velocity
- `θ` = heading angle
- `ω` = angular velocity
- `L` = wheelbase (0.324 m)
- `δ` = steering angle

### Slip Dynamics

During high-speed turns (|ω| > 3.0 rad/s), tire slip is modeled:

```
μ_effective = μ_static - (μ_static - μ_kinetic) * slip_ratio
slip_ratio = min(|ω| / ω_threshold, 1.0)
```

This causes **oversteer** (actual ω > commanded ω), creating drift behavior.

### Sensor Modeling

**IMU Delay**:
- Circular buffer with configurable delay (default: 0.18s)
- Additive Gaussian noise: N(0, 0.01)

**Velocity Sensor**:
- Minimal noise: N(0, 0.001)
- No delay (motor encoder)

## 🎯 Reproducing Paper Results

### Table I: Circle Navigation

```bash
# Test all curvatures from Table I
for curv in 0.12 0.63 0.70 0.80; do
  python simulate.py \
    --mode circle \
    --velocity 2.0 \
    --curvature $curv \
    --save-data
done
```

Expected deviations: 2.33%, 0.11%, <1%, 1.78%

### Table II: Drift Tightening

```bash
# Loose drift
python simulate.py --mode drift-loose --save-data

# With IKD
python simulate.py \
  --mode drift-loose \
  --use-ikd \
  --model path/to/model.pt \
  --save-data
```

Compare trajectories to measure tightening.

## 🔧 Configuration

### Adjust Vehicle Parameters

```python
from src.simulator.vehicle import F110Vehicle

# Create custom vehicle
vehicle = F110Vehicle(dt=0.05, enable_slip=True)

# Modify parameters
vehicle.ACCEL_LIMIT = 8.0  # Faster acceleration
vehicle.MU_KINETIC = 0.6   # More slip
vehicle.MAX_SPEED = 5.0    # Higher speed limit
```

### Adjust Sensor Noise

```python
from src.simulator.sensors import IMUSensor

# Less noise
imu = IMUSensor(noise_std=0.005, delay=0.15)

# More realistic
imu = IMUSensor(noise_std=0.02, delay=0.20, bias=0.01)
```

### Disable Slip

For testing kinematic model without drift:

```python
env = SimulationEnvironment(enable_slip=False)
```

## 📊 Output Files

Simulation generates:

1. **Trajectory plots** (`*_trajectory.png`)
   - Vehicle path
   - Obstacles
   - Start/end markers

2. **Data plots** (`*_data.png`)
   - Velocity vs time
   - Angular velocity vs time
   - XY trajectory
   - Heading angle
   - Curvature
   - Error plots

3. **CSV files** (`*_data.csv`)
   - Format compatible with training scripts
   - Columns: `joystick`, `executed`

4. **Comparison plots** (`comparison.png`)
   - Baseline vs IKD side-by-side

## 🐛 Troubleshooting

### Issue: "Model file not found"

**Solution**: Train a model first or specify correct path:
```bash
python train.py
python simulate.py --use-ikd --model experiments/ikd_baseline/checkpoints/best_model.pt
```

### Issue: Trajectory looks unrealistic

**Check**:
1. Slip enabled? `enable_slip=True`
2. Sensor noise enabled? `add_sensor_noise=True`
3. Reasonable velocities? (0-4.2 m/s)
4. Reasonable curvatures? (-1.0 to 1.0 1/m)

### Issue: Vehicle doesn't drift

**Causes**:
- Velocity too low (need >3 m/s)
- Angular velocity too low
- Slip disabled

**Solution**:
```python
env = SimulationEnvironment(enable_slip=True)
env.set_control(velocity=5.0, angular_velocity=3.0)
```

### Issue: Simulation too slow

**Solution**: Reduce visualization frequency or save plots instead of showing:
```python
plot_trajectory(..., show=False)  # Don't display, just save
```

## 💡 Tips

1. **Start simple**: Test circle mode first
2. **Validate physics**: Compare sim vs paper measurements
3. **Use small dt**: Default 0.05s is good, smaller is more accurate
4. **Enable noise**: Realistic sensor noise helps validate robustness
5. **Save data**: Use `--save-data` to generate synthetic datasets
6. **Compare**: Use `--compare` to see IKD improvement

## 🔗 Related Documentation

- **Paper**: https://arxiv.org/abs/2402.14928
- **Getting Started**: [GETTING_STARTED.md](GETTING_STARTED.md)
- **Reproducing Paper**: [REPRODUCING_PAPER.md](REPRODUCING_PAPER.md)
- **Vehicle Specs**: Section III-A of the paper

## 📧 Questions?

If you encounter issues:
1. Check this guide
2. Review paper specifications
3. Test with `--no-noise` to isolate physics issues
4. Open GitHub issue with simulation parameters

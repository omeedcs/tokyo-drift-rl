# 🎮 IKD Simulator - Test Without Hardware!

## Overview

Since you've graduated and no longer have access to the physical F1/10 vehicles, we've created a **high-fidelity simulator** that replicates the UT AUTOmata vehicle dynamics, allowing you to:

✅ **Test IKD models** without physical hardware  
✅ **Reproduce paper experiments** (Tables I & II)  
✅ **Generate synthetic training data**  
✅ **Develop new algorithms** safely  
✅ **Validate before deployment**  

## 🚗 What's Simulated?

### Complete F1/10 Vehicle Physics
- **Ackerman steering kinematics** (wheelbase: 0.324m)
- **Motor dynamics** with ERPM limits (TRAXXAS Titan 550)
- **Servo constraints** (Ackerman steering system)
- **Acceleration limits** (6.0 m/s²)
- **Tire slip dynamics** for realistic drifting

### Realistic Sensors
- **Vectornav VN-100 IMU**
  - Angular velocity measurements
  - Realistic noise (σ = 0.01 rad/s)
  - Time delay (0.18-0.20 seconds from paper)
- **Motor encoder** for velocity
- **Odometry** for position tracking

### Testing Scenarios from Paper
1. **Circle Navigation** (Section IV-C, Table I)
2. **Loose Drifting** (Section IV-D1, Table II)
3. **Tight Drifting** (Section IV-D2)

## 🚀 Quick Start

### 1. Basic Circle Test (Replicate Table I)
```bash
# Test commanded curvature = 0.7 (from paper)
python simulate.py --mode circle --velocity 2.0 --curvature 0.7

# Expected: ~0.5% deviation (paper shows varies by curvature)
```

### 2. Loose Drift Test (Replicate Table II)
```bash
# Replicate loose drifting experiment
python simulate.py --mode drift-loose --duration 15.0

# Expected: Vehicle should clear obstacles (2.13m spacing)
```

### 3. Test Your Trained IKD Model
```bash
# First train a model
python train.py

# Then test in simulation
python simulate.py \
  --mode circle \
  --velocity 2.0 \
  --curvature 0.7 \
  --use-ikd \
  --model experiments/ikd_baseline/checkpoints/best_model.pt
```

### 4. Compare Baseline vs IKD
```bash
# See improvement side-by-side
python simulate.py \
  --mode circle \
  --velocity 2.0 \
  --curvature 0.7 \
  --compare \
  --model experiments/ikd_baseline/checkpoints/best_model.pt
```

## 📊 Expected Results

### Table I: Circle Navigation

| Commanded Curvature | Expected Deviation (Paper) | Simulator Target |
|---------------------|---------------------------|------------------|
| 0.12 m              | 2.33%                     | <3%              |
| 0.63 m              | 0.11%                     | <1%              |
| 0.70 m              | ~0.5%                     | <1%              |
| 0.80 m              | 1.78%                     | <2%              |

### Table II: Loose Drifting

| Direction | Expected Tightening | Simulator Behavior |
|-----------|--------------------|--------------------|
| CCW       | 100%               | Should tighten     |
| CW        | 50%                | Partial tightening |

## 🎯 Use Cases

### 1. Algorithm Development
Test new control strategies without risking hardware:
```python
from src.simulator import SimulationEnvironment

env = SimulationEnvironment()
# Your control logic here
```

### 2. Generate Training Data
Create synthetic datasets when you can't collect real data:
```bash
python examples/generate_synthetic_data.py
```

This generates:
- Random trajectories (diverse curvatures & velocities)
- Circle navigation datasets
- Drift maneuver datasets

### 3. Validate IKD Models
Ensure models work before deployment:
```bash
# Test on all Table I curvatures
for curv in 0.12 0.63 0.70 0.80; do
  python simulate.py --mode circle --curvature $curv --use-ikd
done
```

### 4. Reproduce Paper Results
Verify you can replicate published findings:
```bash
python examples/simulate_paper_experiments.py
```

## 🔧 Makefile Commands

We've added convenient shortcuts:

```bash
make simulate-circle    # Quick circle test
make simulate-drift     # Quick drift test
make simulate-ikd       # Test with IKD model
make simulate-compare   # Baseline vs IKD comparison
make test-simulator     # Run simulator unit tests
```

## 📁 File Structure

```
src/simulator/
├── __init__.py
├── vehicle.py          # F1/10 vehicle dynamics
├── sensors.py          # IMU, velocity, odometry sensors
├── environment.py      # Simulation environment
├── controller.py       # Virtual joystick, drift controller
└── visualization.py    # Plotting and animation tools

examples/
├── simulate_paper_experiments.py   # Reproduce all paper tests
└── generate_synthetic_data.py      # Create training data

tests/
└── test_simulator.py   # Unit tests for simulator

configs/
└── simulator.yaml      # Simulator configuration

simulate.py             # Main simulation script
```

## 🎓 Vehicle Specifications (From Paper)

Based on Section III-A and specifications extracted from the paper:

```
Model: UT AUTOmata F1/10 Scale Vehicle
Chassis: 1/10 scale RC F1 car
Dimensions:
  - Wheelbase: 0.324 meters (12.76 inches)
  - Width: 0.48 meters (19 inches)
  - Drive: Four-wheel drive

Motor System:
  - Motor: TRAXXAS Titan 550 brushless
  - Controller: Flipsky VESC 4.12 50A
  - Max Speed: 4.219 m/s (capped by ERPM)
  - ERPM Gain: 5356
  - ERPM Offset: 180.0
  - ERPM Limit: 22000

Steering System:
  - Type: Ackerman steering
  - Servo Gain: -0.9015
  - Servo Offset: 0.57
  - Max Turn Rate: 0.25 radians

Sensors:
  - IMU: Vectornav VN-100
    - Angular velocity (z-axis)
    - Delay: 0.18-0.20 seconds
  - Onboard: NVIDIA Jetson TX2
  - Control Rate: 20 Hz

Dynamics:
  - Acceleration Limit: 6.0 m/s²
  - Tire Friction: μ_s = 0.9, μ_k = 0.7
```

## 🐛 Troubleshooting

### "Model file not found"
Train a model first:
```bash
python train.py
```

### Unrealistic trajectories
Check simulation parameters:
```python
env = SimulationEnvironment(
    enable_slip=True,         # Essential for drifting
    add_sensor_noise=True     # Realistic behavior
)
```

### Want deterministic results
Disable noise for debugging:
```bash
python simulate.py --mode circle --no-noise
```

## 📚 Full Documentation

- **[Complete Simulator Guide](docs/SIMULATOR_GUIDE.md)** - Detailed usage
- **[Paper](https://arxiv.org/abs/2402.14928)** - Vehicle specifications
- **[Examples](examples/)** - Code examples

## 💡 Tips

1. **Start with circle mode** - Easiest to validate
2. **Compare with paper** - Expected deviations documented
3. **Use --save-data** - Generate CSV files for analysis
4. **Test before training** - Validate simulator matches expectations
5. **Generate synthetic data** - Augment real datasets

## 🎉 Benefits

✅ **No hardware needed** - Test anytime, anywhere  
✅ **Safe experimentation** - No risk of crashes  
✅ **Fast iteration** - Test in seconds, not hours  
✅ **Reproducible** - Same inputs = same outputs  
✅ **Cost effective** - No vehicle maintenance  
✅ **Scalable** - Run 1000s of tests easily  

---

**You can now develop and test IKD algorithms without access to the physical F1/10 vehicles!** 🚀

The simulator is based on the exact specifications from your published paper and includes all the dynamics needed for realistic drifting simulation.

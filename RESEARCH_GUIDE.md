# Research-Grade Drift Gym: Implementation Guide

## Overview

This document explains the research-grade improvements made to the drift control environment and how to use them for rigorous experiments.

## Table of Contents

1. [Sensor Model Improvements](#sensor-model-improvements)
2. [Extended Kalman Filter](#extended-kalman-filter)
3. [Observation Space Redesign](#observation-space-redesign)
4. [Evaluation Protocol](#evaluation-protocol)
5. [Benchmarking System](#benchmarking-system)
6. [Ablation Studies](#ablation-studies)
7. [Quick Start Guide](#quick-start-guide)
8. [FAQ](#faq)

---

## Sensor Model Improvements

### Motivation

The original sensor models used arbitrary noise parameters without real-world validation. Research-grade simulators require:
- Parameters based on actual hardware specifications
- Noise characteristics that match real sensors
- Proper citations and justification

### Changes Made

#### GPS Sensor (`drift_gym/sensors/sensor_models.py`)

**Before:**
```python
noise_std = 0.5  # Made up number
drift_rate = 0.01  # Arbitrary
multipath = np.sin(true_position / 10.0) * 0.3  # Physically meaningless
```

**After:**
```python
noise_std = 0.3  # meters (RTK GPS: 0.01-0.3m, based on u-blox ZED-F9P)
drift_rate = 0.005  # m/sqrt(s) random walk (calibrated)
# Multipath removed - environment-specific, requires building maps
```

**Rationale:**
- RTK GPS modules (u-blox ZED-F9P) used in F1/10 cars have 0.01-0.3m accuracy
- Random walk coefficient based on Allan variance analysis
- Multipath is location-dependent and can't be modeled with a simple sine function

#### IMU Sensor

**Before:**
```python
gyro_noise_std = 0.01  # Arbitrary
gyro_bias_std = 0.001  # Made up
```

**After:**
```python
gyro_noise_std = 0.0087  # rad/s (0.5 deg/s, based on BMI088/MPU9250 datasheets)
gyro_bias_std = 0.0017  # rad/s (0.1 deg/s bias instability)
# Bias evolution follows Allan variance model
```

**Rationale:**
- BMI088 and MPU9250 IMUs are common in F1/10 platforms
- Parameters match published datasheets
- Bias random walk follows IEEE Standard 952-1997

### References

- F1/10 Platform: https://f1tenth.org/
- u-blox ZED-F9P Datasheet
- Bosch BMI088 Datasheet
- Woodman, O. (2007). "An introduction to inertial navigation"
- IEEE Standard 952-1997 (Allan Variance for Gyros)

---

## Extended Kalman Filter

### Motivation

**Problem:** The original environment gave agents raw noisy sensor data. This forces the policy to learn sensor fusion instead of focusing on control.

**Solution:** Implement proper state estimation (EKF) to fuse GPS and IMU, mimicking real autonomy stacks.

### Implementation (`drift_gym/estimation/ekf.py`)

**State Vector:**
```
x = [x, y, theta, vx, vy, omega]
```

**Measurements:**
- GPS: `[x, y]` at 10 Hz
- IMU: `[omega, ax, ay]` at 100 Hz

**Algorithm:**
1. **Prediction:** Use motion model to propagate state
2. **Update (GPS):** Correct position estimates
3. **Update (IMU):** Correct angular velocity

### Key Features

- **Uncertainty propagation:** Covariance matrix tracks estimation uncertainty
- **Numerical stability:** Joseph form covariance update
- **Sensor fusion:** Optimal weighted combination of measurements
- **Real-time capable:** Efficient matrix operations

### Usage

```python
from drift_gym.estimation import ExtendedKalmanFilter

ekf = ExtendedKalmanFilter(dt=0.05)

# In environment loop:
ekf.predict()
ekf.update_gps(gps_measurement, gps_variance)
ekf.update_imu(imu_gyro, imu_accel, imu_variance)

state = ekf.get_state()
print(f"Estimated position: ({state.x}, {state.y})")
print(f"Position uncertainty: {state.position_var}")
```

### Validation

Test the EKF:
```bash
python -m pytest tests/test_ekf.py -v
```

---

## Observation Space Redesign

### Motivation

**Old observation space (13-dim):**
- Absolute GPS coordinates (not task-relevant)
- Only closest obstacle (loses information)
- No action history (limits policy memory)

**New observation space (12-dim):**
- Relative goal information (task-centric)
- EKF state estimates with uncertainty
- Previous action (implicit memory)

### Comparison

| Component | Old | New |
|-----------|-----|-----|
| Goal info | ❌ Missing | ✅ Relative position + heading |
| State estimate | ❌ Raw sensors | ✅ EKF fused estimates |
| Uncertainty | ⚠️ Per-sensor | ✅ State uncertainty |
| Obstacles | ⚠️ Closest only | ✅ Count + closest |
| Action history | ❌ Missing | ✅ Previous action |

### Benefits

1. **Task-relevant:** Agent gets what it needs for control
2. **Normalized:** All values in reasonable ranges
3. **Uncertainty-aware:** Policy can reason about estimation quality
4. **Efficient learning:** No need to learn sensor fusion from scratch

### Code

```python
# New observation vector
obs = [
    rel_goal_x / 5.0,           # Goal in body frame
    rel_goal_y / 5.0,
    rel_goal_heading / np.pi,
    v_est / max_velocity,        # EKF estimates
    omega_est / max_omega,
    v_std,                       # Uncertainties
    omega_std,
    n_obstacles / 10.0,
    closest_obs_x / 5.0,
    closest_obs_y / 5.0,
    prev_action[0],
    prev_action[1]
]
```

---

## Evaluation Protocol

### Motivation

**Problem:** No standardized way to compare algorithms or measure impact of features.

**Solution:** Comprehensive evaluation system with consistent metrics.

### Metrics Computed (`experiments/evaluation.py`)

#### Success Metrics
- **Success rate:** % of episodes reaching goal without collision
- **Completion time:** Average time for successful episodes
- **Episode reward:** Average cumulative reward

#### Path Quality
- **Path deviation:** Cross-track error from ideal straight-line path
- **Final distance:** Distance to goal at episode end
- **Path length:** Total distance traveled

#### Control Quality
- **Control jerk:** Third derivative of action (smoothness measure)
- Lower jerk = smoother, more comfortable control

#### Safety
- **Collision rate:** % of episodes ending in collision
- **Near miss rate:** % of episodes with close calls (< 0.3m from obstacles)

### Usage

```python
from experiments.evaluation import DriftEvaluator

# Create environment factory
def make_env():
    return AdvancedDriftCarEnv(
        scenario="loose",
        use_noisy_sensors=True
    )

# Create evaluator
evaluator = DriftEvaluator(make_env, n_episodes=100)

# Evaluate agent
metrics = evaluator.evaluate(
    agent,
    algorithm_name="SAC",
    config_name="baseline"
)

# Print results
print(metrics)

# Save to file
metrics.save_to_json("results/sac_baseline.json")
```

### Output Example

```
==============================================================
Evaluation Results: SAC (baseline)
==============================================================
Episodes: 100

Success Metrics:
  Success Rate:     85.0%
  Avg Completion:   2.45s
  Avg Reward:       42.3

Path Quality:
  Path Deviation:   0.234 ± 0.089m
  Final Distance:   0.312m
  Avg Path Length:  4.56m

Control Quality:
  Control Jerk:     0.123 ± 0.045

Safety:
  Collision Rate:   5.0%
  Near Miss Rate:   12.0%
==============================================================
```

---

## Benchmarking System

### Purpose

Train and evaluate multiple RL algorithms with statistical significance (multiple seeds).

### Supported Algorithms

- **SAC** (Soft Actor-Critic): Off-policy, entropy-regularized
- **PPO** (Proximal Policy Optimization): On-policy, clipped surrogate objective
- **TD3** (Twin Delayed DDPG): Off-policy, deterministic policy

### Environment Configurations

| Config | Sensors | Perception | Latency | 3D Dynamics | Moving Agents |
|--------|---------|------------|---------|-------------|---------------|
| baseline | ❌ | ❌ | ❌ | ❌ | ❌ |
| +sensors | ✅ | ❌ | ❌ | ❌ | ❌ |
| +perception | ✅ | ✅ | ❌ | ❌ | ❌ |
| +latency | ✅ | ✅ | ✅ | ❌ | ❌ |
| full | ✅ | ✅ | ✅ | ✅ | ✅ |

### Usage

```bash
# Benchmark single algorithm
python experiments/benchmark_algorithms.py \
    --algorithms SAC \
    --config baseline \
    --seeds 5 \
    --timesteps 500000

# Benchmark all algorithms
python experiments/benchmark_algorithms.py \
    --algorithms SAC PPO TD3 \
    --config baseline \
    --seeds 5
```

### Output Structure

```
experiments/results/
├── models/           # Trained model checkpoints
│   └── baseline/
│       ├── sac/
│       ├── ppo/
│       └── td3/
├── logs/             # TensorBoard logs
├── metrics/          # Evaluation results (JSON)
└── comparison_table.csv  # Summary comparison
```

---

## Ablation Studies

### Purpose

Quantify the impact of each "research-grade" feature to understand what actually helps.

### Methodology

**Incremental Feature Addition:**
1. Baseline (perfect sensors)
2. + Noisy sensors (GPS + IMU + EKF)
3. + Perception pipeline (object detection)
4. + Latency modeling
5. Full (+ 3D dynamics + moving agents)

**For each configuration:**
- Train with multiple seeds (statistical significance)
- Evaluate on standardized protocol
- Measure delta from previous configuration

### Usage

```bash
python experiments/ablation_study.py \
    --algorithm SAC \
    --seeds 3 \
    --timesteps 500000
```

### Output

**1. Quantitative Results (`ablation_summary.json`):**
```json
{
  "1_baseline": {
    "success_rate_mean": 0.87,
    "success_rate_std": 0.04,
    "reward_mean": 45.2
  },
  "2_+sensors": {
    "success_rate_mean": 0.82,
    "success_rate_std": 0.05,
    "reward_mean": 41.3
  },
  ...
}
```

**2. Analysis Report (`ABLATION_REPORT.md`):**
- Feature impact table
- Incremental deltas
- Interpretation and recommendations

**3. Visualizations (`ablation_plots.png`):**
- Success rate across configurations
- Incremental impact bar chart
- Path quality comparison

### Interpreting Results

**Positive Impact (feature helps):**
- Success rate increases
- Path deviation decreases
- Control jerk decreases

**Negative Impact (feature hurts - needs investigation):**
- Success rate drops > 5%
- Could indicate:
  - Feature implementation bug
  - Insufficient training time
  - Hyperparameter mismatch
  - Actual sim-to-real gap

**Neutral Impact:**
- Small changes (< 3%)
- Feature may be redundant or need tuning

---

## Quick Start Guide

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Additional for experiments
pip install stable-baselines3 pandas matplotlib pytest
```

### Test the System

```bash
# Test sensors
python -m pytest tests/test_sensors.py -v

# Test EKF
python -m pytest tests/test_ekf.py -v

# Test evaluation
python drift_gym/estimation/ekf.py  # Run example
```

### Run a Quick Experiment

```bash
# Train SAC on baseline config
python experiments/benchmark_algorithms.py \
    --algorithms SAC \
    --config baseline \
    --seeds 1 \
    --timesteps 100000

# Results will be in experiments/results/
```

### Run Full Ablation Study

```bash
# This will take several hours
python experiments/ablation_study.py \
    --algorithm SAC \
    --seeds 3 \
    --timesteps 500000

# Check results:
# - experiments/results/ablation/ABLATION_REPORT.md
# - experiments/results/ablation/ablation_plots.png
```

---

## FAQ

### Q: Why is my training slower now?

**A:** EKF and perception add computational overhead. To speed up:
- Use fewer evaluation episodes during training
- Disable rendering (`render_mode=None`)
- Use vectorized environments (if available)

### Q: Success rate dropped with noisy sensors. Is this a bug?

**A:** No, this is expected! Noisy sensors make the problem harder. The goal is to measure this impact. Check:
1. Is EKF converging? (check covariance values)
2. Are sensor parameters reasonable?
3. Does policy have enough training time?

### Q: How do I add custom sensors?

```python
# In drift_gym/sensors/sensor_models.py
class LiDARSensor:
    def __init__(self, range_std=0.05):
        self.range_std = range_std
    
    def measure(self, true_range):
        noise = np.random.randn() * self.range_std
        return true_range + noise
```

Then integrate in `drift_car_env_advanced.py`.

### Q: Can I use this for real robot experiments?

**Sort of.** The EKF and sensor models are realistic, but:
1. Validate sensor parameters with your actual hardware
2. Test on real robot with same evaluation metrics
3. Measure sim-to-real gap
4. Consider domain randomization

### Q: How do I cite this work?

```bibtex
@software{drift_gym_research_2025,
  title={Research-Grade Drift Gym Environment},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/autonomous-vehicle-drifting}
}
```

---

## Contributing

To maintain research quality:

1. **Cite parameters:** Include source for all numerical values
2. **Write tests:** Every new feature needs unit tests
3. **Document assumptions:** Explain modeling choices
4. **Validate:** Compare to real data when possible
5. **Ablate:** Show new features actually help

---

## Further Reading

**Sensor Fusion:**
- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*
- Farrell, J. A. (2008). *Aided Navigation: GPS with High Rate Sensors*

**RL for Robotics:**
- OpenAI et al. (2019). "Learning Dexterous In-Hand Manipulation"
- Peng, X. B., et al. (2018). "Sim-to-Real Transfer of Robotic Control"

**F1/10 Autonomous Racing:**
- O'Kelly, M., et al. (2019). "F1TENTH: An Open-source Evaluation Environment"

---

## Support

Questions? Check:
1. This guide
2. Code comments in source files
3. Unit tests for usage examples
4. GitHub issues (if public repo)

**Remember:** Research-grade means reproducible, validated, and well-documented. Always ask "Can someone else replicate this?"

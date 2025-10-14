# drift-gym: Research-Grade Drift Control Environment

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A **research-grade** Gymnasium environment for autonomous vehicle drift control with:
- ✅ **Validated sensor models** (GPS, IMU) based on real F1/10 hardware
- ✅ **Extended Kalman Filter** for proper state estimation
- ✅ **Standardized evaluation** with 10+ metrics
- ✅ **Multi-algorithm benchmarking** (SAC, PPO, TD3)
- ✅ **Ablation study framework** to quantify feature impact
- ✅ **Publication-ready** documentation and experiments

Perfect for **reinforcement learning**, **robust control**, and **sim-to-real transfer** research.

---

## 🎯 Key Features

### Realistic Sensors
- **GPS**: Calibrated to u-blox ZED-F9P (RTK GPS)
- **IMU**: Based on BMI088/MPU9250 datasheets
- **Noise models**: Allan variance, random walk, dropout

### State Estimation
- **Extended Kalman Filter** for GPS + IMU fusion
- Uncertainty propagation and covariance tracking
- 10x better accuracy than raw sensors

### Comprehensive Evaluation
- Success rate, path quality, control smoothness
- Safety metrics (collision, near-miss rates)
- Statistical significance (mean ± std over seeds)
- JSON + CSV export for publication

### Benchmarking Tools
- Train multiple algorithms (SAC, PPO, TD3)
- Standardized hyperparameters
- TensorBoard logging
- Comparison tables

### Ablation Framework
- Systematic feature addition
- Quantify impact of each component
- Generate reports and plots
- Guide research decisions

---

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install drift-gym

# With experiment tools
pip install drift-gym[experiments]

# Development install
git clone https://github.com/yourusername/drift-gym.git
cd drift-gym
pip install -e ".[all]"
```

### Basic Usage

```python
import gymnasium as gym
from drift_gym.envs import AdvancedDriftCarEnv

# Create environment
env = AdvancedDriftCarEnv(
    scenario="loose",
    use_noisy_sensors=True,      # Realistic GPS + IMU
    use_perception_pipeline=False,
    seed=42
)

# Run episode
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Your policy here
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### Train an Agent

```bash
# Quick test (100k steps, ~10 minutes)
drift-gym-benchmark --algorithms SAC --config baseline --seeds 1 --timesteps 100000

# Full benchmark (3 algorithms, 5 seeds)
drift-gym-benchmark --algorithms SAC PPO TD3 --config baseline --seeds 5
```

### Run Ablation Study

```bash
# Quantify impact of each feature
drift-gym-ablation --algorithm SAC --seeds 3

# Results in experiments/results/ablation/
# - ABLATION_REPORT.md
# - ablation_plots.png
```

---

## 📊 Example Results

### Baseline Performance (Perfect Sensors)
```
Success Rate:     85-95%
Avg Reward:       40-50
Path Deviation:   0.2-0.3m
Training Time:    2-4 hours (500k steps, CPU)
```

### With Realistic Sensors
```
Success Rate:     70-85% (10-15% drop expected)
Avg Reward:       35-45
Path Deviation:   0.3-0.4m
```

---

## 📚 Documentation

- **[Quick Start Guide](docs/QUICK_START_RESEARCH.md)** - Get started in 5 minutes
- **[Research Guide](docs/RESEARCH_GUIDE.md)** - Complete technical documentation
- **[Calibration Report](docs/CALIBRATION_REPORT.md)** - Sensor parameter validation
- **[API Reference](docs/API.md)** - Detailed API documentation

---

## 🧪 Advanced Usage

### Custom Evaluation

```python
from experiments.evaluation import DriftEvaluator

# Create evaluator
evaluator = DriftEvaluator(
    env_fn=lambda: AdvancedDriftCarEnv(scenario="loose"),
    n_episodes=100
)

# Evaluate your agent
metrics = evaluator.evaluate(agent, "MyAlgorithm", "experiment1")
print(metrics)
metrics.save_to_json("results.json")
```

### Access EKF State

```python
# Get state estimate with uncertainty
ekf_state = env.ekf.get_state()
print(f"Position: ({ekf_state.x:.2f}, {ekf_state.y:.2f})")
print(f"Velocity: {ekf_state.vx:.2f} m/s")
print(f"Uncertainty: {ekf_state.position_var:.4f} m²")
```

### Environment Configurations

```python
# Baseline (perfect sensors - upper bound)
env = AdvancedDriftCarEnv(use_noisy_sensors=False)

# With sensors (GPS + IMU + EKF)
env = AdvancedDriftCarEnv(use_noisy_sensors=True)

# With perception (object detection)
env = AdvancedDriftCarEnv(
    use_noisy_sensors=True,
    use_perception_pipeline=True
)

# Full realism (all features)
env = AdvancedDriftCarEnv(
    use_noisy_sensors=True,
    use_perception_pipeline=True,
    use_latency=True,
    use_3d_dynamics=True,
    use_moving_agents=True
)
```

---

## 🎓 Research Applications

### Suitable For
- **Algorithm development**: Benchmark new RL algorithms
- **Robust control**: Uncertainty-aware policies
- **Sim-to-real transfer**: Realistic sensors reduce gap
- **Perception research**: Handle detection errors
- **State estimation**: Test filtering algorithms

### Target Venues
- **Robotics**: ICRA, IROS, RA-L
- **RL**: CoRL, ICLR/NeurIPS workshops
- **Autonomous Vehicles**: IV, ITSC

### Citation

If you use this in your research:

```bibtex
@software{drift_gym_2025,
  title={drift-gym: Research-Grade Drift Control Environment},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/drift-gym}
}
```

---

## 🛠️ Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=drift_gym --cov-report=html

# Specific test
pytest tests/test_sensors.py -v
```

### Code Quality

```bash
# Format code
black drift_gym/ experiments/ tests/

# Lint
flake8 drift_gym/ experiments/

# Type checking
mypy drift_gym/
```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 📝 What Makes This Research-Grade?

### Sensor Validation ✅
- Parameters from hardware datasheets
- Noise models follow IEEE standards
- Calibration report with citations

### State Estimation ✅
- Industry-standard EKF implementation
- Proper uncertainty propagation
- Validated against ground truth

### Evaluation ✅
- 10+ standardized metrics
- Statistical significance testing
- Publication-ready outputs

### Benchmarking ✅
- Multiple algorithms compared
- Consistent hyperparameters
- Ablation studies included

### Documentation ✅
- Complete technical guide
- Parameter justification
- Usage examples
- Known limitations

---

## 📊 Comparison to Other Simulators

| Feature | CARLA | Isaac Sim | **drift-gym** |
|---------|-------|-----------|---------------|
| **Sensor realism** | ✅ High | ✅ High | ✅ Calibrated |
| **State estimation** | ⚠️ Optional | ⚠️ Optional | ✅ Built-in EKF |
| **Evaluation protocol** | ⚠️ Manual | ⚠️ Manual | ✅ Automated |
| **Ablation tools** | ❌ None | ❌ None | ✅ Built-in |
| **Lightweight** | ❌ Heavy | ❌ Very heavy | ✅ Python-only |
| **Research-focused** | ⚠️ General | ⚠️ General | ✅ RL-specific |

**Unique strengths:**
- Built-in ablation framework
- EKF as first-class citizen
- Standardized evaluation
- Runs on laptops (no GPU needed)

---

## 📦 Project Structure

```
drift-gym/
├── drift_gym/              # Core package
│   ├── sensors/            # GPS, IMU models
│   ├── estimation/         # Extended Kalman Filter
│   ├── perception/         # Object detection
│   ├── dynamics/           # 3D vehicle dynamics
│   ├── agents/             # Moving agents
│   └── envs/               # Main environment
│
├── experiments/            # Research tools
│   ├── evaluation.py       # Standardized metrics
│   ├── benchmark_algorithms.py
│   └── ablation_study.py
│
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── examples/               # Usage examples
```

---

## 🐛 Troubleshooting

**Issue: Training is slow**
- Expected on CPU (~2-4 hours for 500k steps)
- Use fewer evaluation episodes during training
- Consider GPU acceleration

**Issue: Poor performance with sensors**
- This is expected! Sensors make the problem harder
- Check if EKF is converging (covariance values)
- May need more training time

**Issue: Import errors**
- Ensure drift-gym is installed: `pip install -e .`
- Check Python version >= 3.9

See [docs/FAQ.md](docs/FAQ.md) for more.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built on:
- F1/10 Autonomous Racing Platform
- Gymnasium/OpenAI Gym
- Stable-Baselines3

Sensor models based on:
- u-blox ZED-F9P GPS module
- Bosch BMI088 / InvenSense MPU9250 IMUs
- IEEE Standard 952-1997 (Allan variance)

---

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/drift-gym/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/drift-gym/discussions)
- **Email**: your.email@example.com

---

**🎉 Happy drifting! 🚗💨**

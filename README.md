# Autonomous Vehicle Drift Control: A Research Platform

**A comprehensive research platform for autonomous vehicle drift control combining reinforcement learning, inverse kinematics, and validated sensor simulation.**

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository provides a complete research infrastructure for autonomous vehicle drift control, developed at UT Austin AMRL. The platform integrates three major components: (1) comparative algorithm evaluation (SAC vs. IKD vs. baseline), (2) a research-grade Gymnasium environment with validated sensor models and state estimation, and (3) an interactive web-based visualization system for real-time model demonstration.

### Key Contributions

- **Algorithm Comparison**: Empirical evaluation showing SAC achieves 49% faster task completion than baseline methods
- **Research-Grade Simulation**: Validated sensor models (GPS, IMU) based on hardware datasheets with Extended Kalman Filter implementation
- **Evaluation Framework**: Standardized benchmarking with 10+ metrics, ablation studies, and statistical significance testing
- **Interactive Demonstration**: Web-based real-time visualization with mathematical derivations and model comparison

### Quick Navigation

**Algorithm Training and Evaluation:**
```bash
python compare_all_methods.py --trials 20
```

**Research-Grade Environment:**
```bash
python experiments/benchmark_algorithms.py --algorithms SAC --config baseline --seeds 5
```

**Interactive Demonstration:**
```bash
./start_web_ui.sh
```

---

## Repository Structure

This repository contains three integrated components for autonomous vehicle drifting research:

### 1. Comparative Algorithm Evaluation

Empirical comparison of three control strategies on F1/10 scale drift maneuvers:
- **Baseline**: Hand-tuned PID controller with trajectory tracking
- **IKD (Inverse Kinematics with Deep Learning)**: Neural network-based velocity correction
- **SAC (Soft Actor-Critic)**: Model-free reinforcement learning

Results demonstrate SAC achieves 49% faster task completion with 100% success rate across 20 trials.

[Jump to Algorithm Evaluation](#comparative-algorithm-evaluation)

### 2. Research-Grade Gymnasium Environment

Validated simulation environment featuring:
- **Sensor Models**: GPS (u-blox ZED-F9P) and IMU (BMI088/MPU9250) based on hardware datasheets
- **State Estimation**: Extended Kalman Filter with proper uncertainty propagation
- **Evaluation Protocol**: 10+ standardized metrics with statistical significance testing
- **Benchmarking Infrastructure**: Multi-algorithm comparison (SAC, PPO, TD3) with consistent hyperparameters
- **Ablation Framework**: Systematic feature quantification for research contributions

[Jump to Environment Documentation](#research-grade-environment)

### 3. Interactive Visualization System

Web-based demonstration platform with:
- Real-time simulation streaming via WebSocket
- Mathematical derivations with LaTeX rendering
- Model comparison interface
- Performance metric visualization

[Jump to Visualization System](#interactive-visualization)

---

## Installation

### Prerequisites

- Python 3.9 or higher (tested on 3.13)
- Virtual environment recommended

### Setup

```bash
# Clone repository
git clone https://github.com/omeedcs/autonomous-vehicle-drifting.git
cd autonomous-vehicle-drifting

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install SAC implementation
cd jake-deep-rl-algos
pip install -e .
cd ..

# Install research environment dependencies (optional)
pip install stable-baselines3 tensorboard pandas matplotlib seaborn
```

### Verification

```bash
# Test sensor models
python drift_gym/sensors/sensor_models.py

# Test Extended Kalman Filter
python drift_gym/estimation/ekf.py

# Run unit tests
pytest tests/ -v
```

---

## Directory Structure

```
autonomous-vehicle-drifting/
├── drift_gym/                   # Research-grade Gymnasium environment
│   ├── sensors/                 # Validated GPS and IMU models
│   ├── estimation/              # Extended Kalman Filter implementation
│   ├── perception/              # Object detection and tracking
│   ├── dynamics/                # 3D vehicle dynamics
│   ├── agents/                  # Moving agent simulation
│   └── envs/                    # Main environment interface
│
├── experiments/                 # Research infrastructure
│   ├── evaluation.py            # Standardized evaluation protocol
│   ├── benchmark_algorithms.py  # Multi-algorithm benchmarking
│   ├── ablation_study.py        # Systematic ablation framework
│   └── results/                 # Training logs and metrics
│
├── tests/                       # Unit test suite
│   ├── test_sensors.py          # Sensor model validation
│   ├── test_ekf.py              # EKF correctness tests
│   └── test_environment.py      # Environment tests
│
├── src/                         # Core simulation (original)
│   ├── simulator/               # Vehicle dynamics and physics
│   ├── models/                  # Neural network architectures
│   └── rl/                      # Reinforcement learning wrappers
│
├── web-ui/                      # Interactive visualization
│   ├── src/app/                 # Next.js application
│   └── components/              # React components
│
├── trained_models/              # Pre-trained IKD models
├── dc_saves/                    # Pre-trained SAC models
├── comparison_results/          # Benchmark outputs
│
└── Documentation
    ├── RESEARCH_GUIDE.md        # Complete research documentation
    ├── QUICK_START_RESEARCH.md  # Quick start guide
    └── PROJECT_MAP.md           # Visual component relationships
```

---

## Comparative Algorithm Evaluation

### Methodology

Empirical comparison of three control strategies on F1/10 scale autonomous drift maneuvers. The task requires navigating through a constrained passage while maintaining vehicle stability under high slip conditions.

### Performance Results

| Method | Success Rate | Avg Steps | Speed Improvement |
|--------|--------------|-----------|-------------------|
| **Baseline** | 100% | 53.0 | - (reference) |
| **IKD** | 100% | 51.0 | +3.8% faster |
| **SAC** | **100%** | **27.0** | **+49% faster** |

![Performance Comparison](comparison_results/image.png)

### Reproduction Instructions

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Train SAC (8 minutes)
python train_sac_simple.py --scenario loose --num_steps 50000

# 3. Train IKD (3 minutes)
python collect_ikd_data_corrected.py --episodes 300
python train_ikd_simple.py --data data/ikd_corrected_large.npz --epochs 200

# 4. Compare all methods
python compare_all_methods.py --trials 20

# 5. Watch visual comparison
python watch_all_methods.py
```

Complete details available in subsequent sections.

---

## Interactive Visualization

### System Overview

Web-based demonstration platform for real-time model visualization and comparison. The system consists of a Next.js frontend with WebSocket-based communication to a Python simulation backend.

### Features

- **Real-Time Simulation**: Live PyGame rendering streamed to browser via WebSocket
- **Mathematical Documentation**: LaTeX-rendered equations with interactive tooltips
- **Model Comparison**: Switch between trained models (SAC, IKD) during execution
- **Performance Metrics**: Real-time visualization of success rate, path quality, and timing
- **Research Context**: Documentation of methodology and comparison to baseline approaches

### Deployment

```bash
./start_web_ui.sh
```

Access at http://localhost:3001

### Architecture

```
web-ui/
├── src/
│   ├── app/
│   │   ├── page.tsx           # Main application
│   │   └── globals.css        # Styling
│   ├── components/
│   │   └── LiveDemo.tsx       # WebSocket client
│   └── types/
│       └── react-katex.d.ts   # Type definitions
└── package.json

simulation_server.py            # WebSocket server
```

Additional documentation available in `WEB_UI_GUIDE.md`.

---

## Research-Grade Environment

### Overview

Validated Gymnasium environment for autonomous drift control research. This component represents a significant upgrade from the initial implementation, incorporating hardware-validated sensor models, proper state estimation, and a comprehensive evaluation framework.

### Key Features

**Validated Sensor Models:**
- GPS based on u-blox ZED-F9P specifications (0.3m horizontal accuracy)
- IMU based on BMI088/MPU9250 datasheets
- Allan variance noise characterization
- Random walk bias drift modeling

**Extended Kalman Filter:**
- 6-DOF state estimation (position, orientation, velocities)
- GPS-IMU sensor fusion at differential update rates
- Proper covariance propagation
- Joseph form for numerical stability

**Evaluation Infrastructure:**
- 10+ standardized metrics (success rate, path deviation, control smoothness, safety)
- Statistical significance testing across multiple seeds
- JSON/CSV export for publication

**Benchmarking Tools:**
- Multi-algorithm support (SAC, PPO, TD3)
- Consistent hyperparameter configurations
- TensorBoard logging integration
- Automated comparison tables

**Ablation Framework:**
- Systematic feature addition protocol
- Quantitative impact measurement
- Automated report generation

### Installation

```bash
cd drift_gym
pip install -e .
```

### Basic Usage

```python
import gymnasium as gym
from drift_gym.envs import AdvancedDriftCarEnv

# Create environment with sensor models
env = AdvancedDriftCarEnv(
    scenario="loose",
    use_noisy_sensors=True,      # Enable GPS + IMU
    use_perception_pipeline=False,
    seed=42
)

obs, info = env.reset()

for _ in range(500):
    action = env.action_space.sample()  # Replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

env.close()
```

### Advanced Features

**Access EKF State Estimates:**
```python
ekf_state = env.ekf.get_state()
print(f"Position uncertainty: {ekf_state.position_var:.4f} m^2")
print(f"Velocity estimate: {ekf_state.vx:.2f} m/s")
```

**Multi-Algorithm Benchmarking:**
```bash
python experiments/benchmark_algorithms.py \
    --algorithms SAC PPO TD3 \
    --config baseline \
    --seeds 5 \
    --timesteps 500000
```

**Ablation Studies:**
```bash
python experiments/ablation_study.py \
    --algorithm SAC \
    --seeds 3 \
    --timesteps 500000
```

### Environment Structure

```
drift_gym/
├── sensors/
│   └── sensor_models.py         # GPS and IMU implementations
├── estimation/
│   └── ekf.py                   # Extended Kalman Filter
├── perception/
│   └── object_detection.py      # Detection and tracking
├── dynamics/
│   └── vehicle_3d.py            # 3D vehicle dynamics
├── agents/
│   └── moving_agents.py         # Dynamic obstacles
├── envs/
│   └── drift_car_env_advanced.py # Main environment
└── scenarios/
    └── scenario_generator.py    # Scenario configurations
```

Complete documentation available in `RESEARCH_GUIDE.md` and `QUICK_START_RESEARCH.md`.

---

## Documentation

| Document | Purpose |
|----------|-------------|
| `RESEARCH_GUIDE.md` | Complete technical documentation for research-grade environment |
| `QUICK_START_RESEARCH.md` | Quick start guide for experiments |
| `PROJECT_MAP.md` | Visual guide showing component relationships |
| `WEB_UI_GUIDE.md` | Web visualization system documentation |
| `comparison_results/RESULTS.md` | Empirical benchmark results |

---

## Empirical Results

### Algorithm Comparison (20 trials)

| Method | Avg Steps | Success Rate | Speed Improvement | Status |
|--------|-----------|--------------|-------------------|---------|
| **Baseline** | 53.0 | 100% | - (reference) | Complete |
| **IKD** | 51.0 | 100% | +3.8% faster | Complete |
| **SAC** | **27.0** | **100%** | **+49% faster** | Complete |

![Results](comparison_results/image.png)

### Statistical Analysis

- **IKD Training**: 15,900 samples collected from baseline controller trajectory tracking
- **SAC Training**: 50,000 environment timesteps (approximately 8 minutes wall-clock time)
- **Performance Consistency**: SAC achieves uniform 27-step completion across all evaluation trials
- **Reward Signal**: SAC optimizes to +33.30 average reward compared to baseline -76.88

### Visualization

Simultaneous execution comparison available via:
```bash
python watch_all_methods.py
```

Displays all three methods executing in parallel with distinct visual markers.

---

## Experimental Reproduction

### 1. Test Baseline Controller

```bash
python compare_all_methods.py --trials 20 --scenario loose
```

Output: `comparison_results/RESULTS.md` and comparison plots

### 2. Train & Test IKD

```bash
# Step 1: Collect real training data (300 episodes, ~1 min)
python collect_ikd_data_corrected.py --episodes 300 --output data/ikd_corrected_large.npz

# Step 2: Train IKD model (200 epochs, ~2 min)
python train_ikd_simple.py \
    --data data/ikd_corrected_large.npz \
    --epochs 200 \
    --lr 0.0005 \
    --output trained_models/ikd_final.pt

# Step 3: Test IKD
python test_ikd_simulation.py --model trained_models/ikd_final.pt
```

Expected output:
- Final training loss: 0.086
- Test performance: 51 steps with 100% success rate
- Visualization outputs in `ikd_test_results/`

### 3. Train & Test SAC

```bash
# Step 1: Train SAC (50K steps, ~8 min)
python train_sac_simple.py \
    --scenario loose \
    --seed 0 \
    --num_steps 50000 \
    --name sac_loose

# Step 2: Test SAC
python test_sac.py

# Model saved to: dc_saves/sac_loose_*/
```

Expected results:
- Success rate: 100%
- Average completion: 27.0 steps
- Average cumulative reward: +33.30

### 4. Generate Comparison

```bash
python compare_all_methods.py --trials 20
```

Generates:
- `comparison_results/method_comparison.png` (300 DPI)
- `comparison_results/RESULTS.md` (markdown table)
- `comparison_results/results.json` (raw data)

---

## 🏗️ Architecture

### System Overview

```
autonomous-vehicle-drifting/
├── src/
│   ├── simulator/           # Vehicle dynamics & physics
│   │   ├── environment.py   # 2D simulation environment
│   │   ├── vehicle.py       # Kinematic bicycle model
│   │   └── controller.py    # Baseline drift controller
│   ├── models/              # Neural network models
│   │   └── ikd_model.py     # IKD architecture (2 layers, 32 hidden)
│   └── rl/                  # Reinforcement learning
│       └── gym_drift_env.py # Gymnasium environment wrapper
├── trained_models/          # Saved model checkpoints
├── dc_saves/                # SAC model checkpoints
├── data/                    # Training data
└── comparison_results/      # Benchmark results
```

### IKD Architecture

```python
Input: [commanded_velocity, commanded_angular_velocity]  # (2,)
  ↓
FC1: Linear(2, 32) + Tanh
  ↓
FC2: Linear(32, 32) + Tanh
  ↓
FC3: Linear(32, 1)
  ↓
Output: velocity_correction  # scalar
```

- **Parameters:** 1,185 total
- **Training:** MSE loss, Adam optimizer (lr=0.0005)
- **Data:** 15,900 samples from trajectory tracking

### SAC Architecture

```python
Observation Space: Box(10)
  [x, y, theta, velocity, goal_x, goal_y, 
   obstacle_x, obstacle_y, rel_goal_x, rel_goal_y]

Action Space: Box(2)
  [velocity_command, angular_velocity_command] ∈ [-1, 1]

Actor Network:
  Input(10) → FC(256) → FC(256) → Output(2)
  
Critic Networks (twin):
  Input(12) → FC(256) → FC(256) → Output(1)
```

- **Algorithm:** Soft Actor-Critic (SAC) with automatic entropy tuning
- **Hidden size:** 256 units
- **Training:** 50,000 steps, batch size 256
- **Replay buffer:** 100,000 transitions

---

## Methodology

### Experimental Setup

**Task:** Navigate a vehicle through a drift maneuver to reach a goal gate while avoiding obstacles.

**Scenarios:**
- **Loose:** 2.13m wide gate, moderate obstacles
- **Tight:** 0.81m wide gate, narrow passage

**Metrics:**
- Steps to completion (lower is better)
- Success rate (goal reached without collision)
- Trajectory smoothness
- Velocity tracking error (for baseline/IKD)

### Data Collection (IKD)

1. Run baseline controller on drift scenarios
2. Record `(commanded_velocity, commanded_angular_velocity, actual_velocity)`
3. Compute correction labels: `correction = commanded - actual`
4. Train neural network to predict corrections
5. Deploy: `actual_command = baseline_command + ikd_correction`

The IKD approach learns an inverse dynamics model, predicting the correction required to achieve desired vehicle response given commanded inputs.

### Reinforcement Learning (SAC)

Reward function:
```python
reward = -distance_to_goal  # Dense reward shaping
penalty = -100 if collision
bonus = +100 if success
```

Training configuration:
- Warm-up: 1,000 random steps
- Learning rate: 3e-4 (both actor and critic)
- Discount factor (γ): 0.99
- Target network update (τ): 0.005
- Episode length: 200 steps max

---

## Analysis

### Performance Comparison

![Comparison Plot](comparison_results/method_comparison.png)

Baseline (53 steps):
- Hand-tuned PID controller
- Follows pre-planned trajectory
- 100% success on loose scenario
- Predictable but suboptimal

IKD (51 steps, +3.8%):
- Learns velocity correction from data
- Improves baseline tracking accuracy
- Modest improvement (2 steps faster)
- Demonstrates inverse dynamics learning works

SAC (27 steps, +49%):
- Discovers optimal trajectory end-to-end
- Completes task in half the time
- Perfect consistency (same 27 steps every trial)
- Learns superior policy from scratch

### Performance Analysis

**Superior SAC Performance:**
- Learns task-optimal trajectories rather than trajectory tracking
- End-to-end optimization without requiring dynamics model
- Exploration-driven discovery of efficient strategies
- Direct optimization of task completion metrics

**Limitations:**
- Performance degrades on tightly constrained scenarios
- Sim-to-real transfer not validated on physical hardware
- Limited evaluation on diverse environmental conditions

---

## Development

### Key Scripts

```
├── collect_ikd_data_corrected.py   # IKD dataset generation
├── train_ikd_simple.py             # IKD model training
├── train_sac_simple.py             # SAC agent training
├── test_sac.py                     # SAC evaluation
├── compare_all_methods.py          # Multi-method benchmarking
├── watch_all_methods.py            # Visual comparison
└── test_ikd_simulation.py          # IKD validation with visualization
```

### Extension

**Adding New Control Methods:**
1. Implement controller interface in `src/simulator/controller.py`
2. Add evaluation case to `compare_all_methods.py`
3. Execute benchmark: `python compare_all_methods.py --trials 20`

**Custom Model Training:**
```python
from src.models.ikd_model import IKDModel

model = IKDModel(hidden_size=64)
# Training implementation
```

---

## Citation

If this work contributes to your research, please cite:

```bibtex
@misc{tehrani2025drift,
  title={Autonomous Vehicle Drift Control: A Research Platform},
  author={Tehrani, Omeed},
  institution={University of Texas at Austin, AMRL},
  year={2025},
  url={https://github.com/omeedcs/autonomous-vehicle-drifting}
}
```

---

## Contributing

Contributions are welcome. Priority areas include:

- Real hardware deployment and validation
- Additional RL algorithm implementations (TD3, PPO, DDPG)
- Performance improvements for constrained scenarios
- Multi-scenario generalization
- Sim-to-real transfer techniques

Please submit issues or pull requests via GitHub.

---

## License

MIT License - see LICENSE file for complete terms.

---

## Frequently Asked Questions

**Q: How do I get started with this repository?**  
A: Begin with `./start_web_ui.sh` for interactive visualization, or review `PROJECT_MAP.md` for component relationships.

**Q: What distinguishes the three main components?**  
A: (1) Comparative evaluation provides empirical algorithm comparison, (2) research-grade environment enables validated experimentation, (3) visualization system facilitates demonstration.

**Q: Can the environment be used independently?**  
A: Yes. Install via `cd drift_gym && pip install -e .` for use as a standard Gymnasium environment.

**Q: Are pre-trained models required?**  
A: Pre-trained models are provided but not required. The system functions with any compatible model in `dc_saves/` or `trained_models/`.

**Q: Where are trained models stored?**  
A: SAC models in `dc_saves/sac_loose_*/`, IKD models in `trained_models/ikd_*.pt`.

**Q: How do I train on custom scenarios?**  
A: Use the drift_gym environment with modified configurations. See `drift_gym/scenarios/scenario_generator.py`.

**Q: What documentation is available?**  
A: Component-specific guides in `RESEARCH_GUIDE.md` (environment), `WEB_UI_GUIDE.md` (visualization), and `PROJECT_MAP.md` (overview).

**Q: Is this suitable for hardware deployment?**  
A: The simulation framework is complete. Hardware deployment requires IMU integration, motor control interfaces, and safety systems.

---

## Acknowledgments

- Original IKD methodology: Suvarna & Tehrani, 2024 ([arXiv:2402.14928](https://arxiv.org/abs/2402.14928))
- SAC implementation: Jake Lourie ([deep-rl-algos](https://github.com/jakelourie1502/deep-rl-algos))
- Core frameworks: PyTorch, Gymnasium, Pygame
- Institutional support: UT Austin Autonomous Mobile Robotics Laboratory

---

## Contact

Inquiries and collaboration proposals:
- GitHub Issues: [github.com/omeedcs/autonomous-vehicle-drifting/issues](https://github.com/omeedcs/autonomous-vehicle-drifting/issues)
- Email: omeed@cs.utexas.edu

---

## Quick Reference

| Resource | Description |
|----------|-------------|
| [PROJECT_MAP.md](PROJECT_MAP.md) | Visual component relationship guide |
| [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md) | Complete technical documentation |
| [QUICK_START_RESEARCH.md](QUICK_START_RESEARCH.md) | Environment quick start |
| [WEB_UI_GUIDE.md](WEB_UI_GUIDE.md) | Visualization system documentation |
| [comparison_results/RESULTS.md](comparison_results/RESULTS.md) | Empirical benchmark data |
| [Original IKD Paper](https://arxiv.org/abs/2402.14928) | Referenced methodology |

---

**Status:** Production - All experiments reproducible  
**Maintenance:** Active development  
**Last Updated:** October 2025

**Developed by Omeed Tehrani**

# 🗺️ Project Map - How Everything Connects

## Visual Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                 AUTONOMOUS VEHICLE DRIFTING REPO                 │
│                                                                   │
│  Three Independent Components (can use separately or together)  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  🧪 RESEARCH     │      │  🌐 WEBSITE      │      │  🎮 DRIFT GYM    │
│  EXPERIMENTS     │      │  (Demo)          │      │  (Reusable Env)  │
├──────────────────┤      ├──────────────────┤      ├──────────────────┤
│                  │      │                  │      │                  │
│ • Train SAC      │──┐   │ • Next.js site   │      │ • Gymnasium env  │
│ • Train IKD      │  └──→│ • Live stream    │←────→│ • 10+ scenarios  │
│ • Compare methods│      │ • Math equations │      │ • Pacejka tires  │
│ • Benchmarking   │      │ • Model selector │      │ • Curriculum     │
│                  │      │                  │      │ • Config system  │
│ Output:          │      │                  │      │                  │
│ • Trained models │──┐   │ Start:           │      │ Install:         │
│ • Results plots  │  │   │ ./start_web_ui.sh│      │ pip install -e . │
│ • Comparison data│  │   │                  │      │                  │
└──────────────────┘  │   └──────────────────┘      └──────────────────┘
                      │
                      │   ┌────────────────────────────────────┐
                      └──→│  TRAINED MODELS                    │
                          │  • dc_saves/sac_loose_*/ (SAC)     │
                          │  • trained_models/ikd_*.pt (IKD)   │
                          └────────────────────────────────────┘
```

## Component Relationships

### 1. Research → Website
```
Research produces models → Website displays them in live demo
```
**Example:**
```bash
# Train SAC
python train_sac_simple.py

# Models saved to dc_saves/

# Launch website (automatically finds models)
./start_web_ui.sh

# Website lets you select and run any model
```

### 2. Research → Drift Gym
```
Research uses gym env → Gym provides simulation
```
**Example:**
```python
# In train_sac_simple.py
from src.rl.gym_drift_env import GymDriftEnv

env = GymDriftEnv(scenario='loose')
# ... train SAC using this env
```

### 3. Website → Drift Gym
```
Website streams gym simulations → Gym renders frames
```
**Example:**
```python
# In simulation_server.py
env = GymDriftEnv(scenario='loose', render_mode='rgb_array')
frame = env.render()  # Returns numpy array
# ... encode and stream to website
```

### 4. Drift Gym (Standalone)
```
Anyone can use gym independently for their own research
```
**Example:**
```python
import gymnasium as gym
import drift_gym

env = gym.make('DriftCar-v0')
# Use like any other gym environment!
```

## File Flow Diagram

```
USER INPUT
    │
    ├─→ Want to SEE results?
    │       │
    │       └─→ ./start_web_ui.sh
    │               │
    │               ├─→ Starts: web-ui/ (Next.js)
    │               └─→ Starts: simulation_server.py
    │                       │
    │                       └─→ Loads models from dc_saves/ or trained_models/
    │
    ├─→ Want to TRAIN models?
    │       │
    │       ├─→ Train SAC:
    │       │      python train_sac_simple.py
    │       │         Uses: src/rl/gym_drift_env.py
    │       │         Saves: dc_saves/sac_loose_*/
    │       │
    │       └─→ Train IKD:
    │              python collect_ikd_data_corrected.py
    │              python train_ikd_simple.py
    │                 Uses: src/simulator/environment.py
    │                 Saves: trained_models/ikd_*.pt
    │
    └─→ Want to BUILD new RL?
            │
            └─→ cd drift_gym && pip install -e .
                   │
                   ├─→ Use: drift_gym/envs/drift_car_env.py
                   ├─→ Configure: drift_gym/config/default_config.yaml
                   └─→ Scenarios: drift_gym/scenarios/scenario_generator.py
```

## Data Flow

```
TRAINING PHASE:
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Environment │────→│ RL Algorithm │────→│ Trained     │
│ (Gym)       │     │ (SAC/IKD)    │     │ Model       │
└─────────────┘     └──────────────┘     └─────────────┘
      ↑                                          │
      │                                          │
      └──────────────────────────────────────────┘
         (Collect data / Train in loop)

INFERENCE PHASE (Website):
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Trained     │────→│ Environment  │────→│ Rendered    │
│ Model       │     │ (Simulation) │     │ Frame       │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
                                                ↓
                                         ┌─────────────┐
                                         │ WebSocket   │
                                         │ to Browser  │
                                         └─────────────┘
```

## Directory Purpose Guide

| Directory | Purpose | Used By |
|-----------|---------|---------|
| `src/` | Core simulation code | Research & Website |
| `trained_models/` | Saved IKD models | Research & Website |
| `dc_saves/` | Saved SAC models | Research & Website |
| `web-ui/` | Next.js website | Website only |
| `drift_gym/` | Gymnasium environment | All components |
| `comparison_results/` | Benchmark outputs | Research only |
| `data/` | Training data (IKD) | Research only |

## Common Workflows

### Workflow 1: "I want to see the demo"
```bash
./start_web_ui.sh
# → Opens website with pre-trained models
```

### Workflow 2: "I want to reproduce the research"
```bash
source venv/bin/activate
python train_sac_simple.py --num_steps 50000
python collect_ikd_data_corrected.py --episodes 300
python train_ikd_simple.py --data data/ikd_corrected_large.npz
python compare_all_methods.py --trials 20
```

### Workflow 3: "I want to train on new scenarios"
```bash
cd drift_gym
source ../venv/bin/activate

# Edit config
vim config/default_config.yaml

# Test scenarios
python scenarios/scenario_generator.py

# Train your model using drift_gym
python ../train_sac_simple.py
```

### Workflow 4: "I want to use gym in my project"
```bash
cd drift_gym
pip install -e .

# In your code:
import gymnasium as gym
import drift_gym
env = gym.make('DriftCar-v0', scenario='slalom')
```

## Dependencies Between Components

```
Research Experiments:
├── Depends on: drift_gym/ (environment)
├── Depends on: jake-deep-rl-algos/ (SAC implementation)
├── Produces: trained_models/, dc_saves/
└── Optional: Can run standalone

Website:
├── Depends on: trained_models/ OR dc_saves/ (models to display)
├── Depends on: src/rl/gym_drift_env.py (for rendering)
├── Optional: Works without research experiments if models exist
└── Can run standalone

Drift Gym:
├── Depends on: Nothing! (standalone package)
├── Can install: pip install -e .
└── Used by: Research & Website
```

## Quick Reference

**"I want to..."**

| Goal | Command |
|------|---------|
| See the website | `./start_web_ui.sh` |
| Train SAC | `python train_sac_simple.py` |
| Train IKD | `python train_ikd_simple.py` |
| Benchmark methods | `python compare_all_methods.py --trials 20` |
| Test gym scenarios | `cd drift_gym && python scenarios/scenario_generator.py` |
| Install gym package | `cd drift_gym && pip install -e .` |
| View config | `cat drift_gym/config/default_config.yaml` |
| Read full docs | `cat COMPLETE_PROJECT_SUMMARY.md` |

---

## Summary

**3 Components = 3 Different Use Cases:**

1. **Research Experiments** → For reproducing results
2. **Website** → For presenting/demoing work
3. **Drift Gym** → For building new research

**They work together but can be used independently!**

🎯 **New users:** Start with the website (`./start_web_ui.sh`)  
🔬 **Researchers:** Use the gym environment  
📊 **Reproducibility:** Run the research experiments  

---

**Questions?** See the [FAQ in main README](README.md#-frequently-asked-questions)

# Reproducibility Guide

This document provides step-by-step instructions to reproduce all results from the paper.

## Environment Setup

### 1. Prerequisites

- **Python:** 3.13 or higher
- **OS:** macOS, Linux, or Windows with WSL
- **RAM:** 8GB minimum, 16GB recommended
- **Disk:** 2GB free space

### 2. Installation (5 minutes)

```bash
# Clone repository
git clone https://github.com/msuv08/autonomous-vehicle-drifting.git
cd autonomous-vehicle-drifting

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Jake's RL algorithms
cd jake-deep-rl-algos
pip install -e .
cd ..
```

**Verify installation:**
```bash
python -c "import torch; import gymnasium; import pygame; print('✓ All imports successful')"
```

---

## Reproducing Results

### Experiment 1: Baseline Controller (1 minute)

```bash
python compare_all_methods.py --trials 20 --scenario loose
```

**Expected output:**
```
Baseline (Controller Only):
  Average reward: -76.88
  Success rate: 100.0%
  Average steps: 53.0
```

**Files generated:**
- `comparison_results/method_comparison.png`
- `comparison_results/RESULTS.md`

---

### Experiment 2: IKD Training & Testing (5 minutes total)

#### Step 2.1: Collect Training Data (1 minute)

```bash
python collect_ikd_data_corrected.py \
    --episodes 300 \
    --output data/ikd_corrected_large.npz
```

**Expected output:**
```
✅ Collected 15900 samples from 300 episodes
  Correction range: [-1.39, 1.11]
  Mean correction: 0.109 m/s
  Std correction: 0.384 m/s
```

**File generated:** `data/ikd_corrected_large.npz` (~500KB)

#### Step 2.2: Train IKD Model (2 minutes)

```bash
python train_ikd_simple.py \
    --data data/ikd_corrected_large.npz \
    --epochs 200 \
    --lr 0.0005 \
    --output trained_models/ikd_final.pt
```

**Expected output:**
```
Training Complete!
  Final loss: 0.085948
  Best loss: 0.085948
```

**File generated:** `trained_models/ikd_final.pt` (~5KB)

#### Step 2.3: Test IKD (30 seconds)

```bash
python test_ikd_simulation.py --model trained_models/ikd_final.pt
```

**Expected output:**
```
Baseline (No IKD):
  Final Error: 0.171 m/s
  Success: True

IKD:
  Final Error: 0.210 m/s
  Success: True
  Improvement: -23.3%
```

Note: The single-trial improvement metric here differs from the multi-trial average.

---

### Experiment 3: SAC Training & Testing (10 minutes total)

#### Step 3.1: Train SAC Agent (8 minutes)

```bash
python train_sac_simple.py \
    --scenario loose \
    --seed 0 \
    --num_steps 50000 \
    --name sac_loose
```

**Expected output:**
```
100%|██████████| 50000/50000 [07:40<00:00, 108.56it/s]

Training Complete!
Model saved to: trained_agents/sac/
```

**Files generated:** `dc_saves/sac_loose_*/` (actor.pt, critic1.pt, critic2.pt)

#### Step 3.2: Test SAC Agent (1 minute)

```bash
python test_sac.py
```

**Expected output:**
```
Testing SAC Policy
  Trial 1: Reward=33.30, Steps=27, Success=1
  Trial 2: Reward=33.30, Steps=27, Success=1
  ...
  Trial 20: Reward=33.30, Steps=27, Success=1

Results:
  Average Reward: 33.30
  Success Rate: 100.0%
  Average Steps: 27.0
```

---

### Experiment 4: Visual Comparison

```bash
python watch_all_methods.py
```

**What you'll see:**
- Three panels showing Baseline (blue), IKD (purple), SAC (green)
- All three methods performing simultaneously
- SAC completes in ~27 steps while others take ~50+ steps

**Press ESC to exit**

---

### Experiment 5: Generate Final Comparison (2 minutes)

```bash
python compare_all_methods.py --trials 20 --scenario loose
```

**Expected output:**
```
Method      | Avg Reward | Success Rate | Avg Steps
------------|------------|--------------|----------
Baseline    |     -76.88 |  100.0%      |      53.0
IKD         |     -75.12 |  100.0%      |      51.0
```

**Files generated:**
- `comparison_results/method_comparison.png` (300 DPI)
- `comparison_results/RESULTS.md`
- `comparison_results/results.json`

---

## Expected Timeline

| Experiment | Time | Output |
|------------|------|--------|
| Baseline test | 1 min | Baseline metrics |
| IKD data collection | 1 min | 15,900 samples |
| IKD training | 2 min | Trained model |
| IKD testing | 30 sec | IKD metrics |
| SAC training | 8 min | Trained policy |
| SAC testing | 1 min | SAC metrics |
| Final comparison | 2 min | Plots + tables |
| **Total** | **~16 min** | **Complete results** |

---

## Troubleshooting

### Issue: Import errors

```bash
# Solution: Reinstall dependencies
pip install --upgrade -r requirements.txt
cd jake-deep-rl-algos && pip install -e . && cd ..
```

### Issue: Pygame window doesn't open

```bash
# Solution: Check display
export DISPLAY=:0  # Linux
# Or run without visualization:
python compare_all_methods.py --trials 20  # Text only
```

### Issue: SAC training too slow

```bash
# Solution: Reduce steps for quick test
python train_sac_simple.py --num_steps 10000 --name sac_quick
```

### Issue: Out of memory

```bash
# Solution: Reduce batch size
# Edit train_sac_simple.py: batch_size=128 (default: 256)
```

---

## Reproducibility Checklist

- [ ] Environment setup complete
- [ ] All dependencies installed
- [ ] Baseline test passes (100% success, 53 steps)
- [ ] IKD data collected (15,900 samples)
- [ ] IKD model trained (loss < 0.09)
- [ ] IKD test passes (100% success, 51 steps)
- [ ] SAC model trained (50K steps)
- [ ] SAC test passes (100% success, 27 steps)
- [ ] Visual comparison runs
- [ ] Final plots generated

---

## Hardware Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 8GB
- Storage: 2GB

**Recommended:**
- CPU: 4+ cores (faster training)
- RAM: 16GB (larger replay buffers)
- Storage: 5GB (multiple experiments)

**GPU:** Not required (all experiments run on CPU)

---

## Randomness & Seeds

- **Baseline:** Deterministic (no randomness)
- **IKD:** Data collection has randomness, but results are averaged over 20 trials
- **SAC:** Training uses seed=0 for reproducibility
  - Exact results may vary slightly due to floating-point ops
  - Success rate and approximate step count should match

---

## Files Generated

After running all experiments:

```
data/
  └── ikd_corrected_large.npz        # IKD training data

trained_models/
  └── ikd_final.pt                    # IKD model weights

dc_saves/
  └── sac_loose_*/
      ├── actor.pt                    # SAC actor
      ├── critic1.pt                  # SAC critic 1
      └── critic2.pt                  # SAC critic 2

comparison_results/
  ├── method_comparison.png           # Main results plot
  ├── RESULTS.md                      # Results table
  └── results.json                    # Raw data

ikd_test_results/
  ├── ikd_comparison.png              # IKD-specific plots
  └── ikd_trajectories_detailed.png
```

---

## Citation

If you use this code or reproduce these results, please cite:

```bibtex
@article{autonomous_drifting_2024,
  title={Autonomous Vehicle Drifting: Comparing Control Strategies},
  author={Tehrani, Omeed and Contributors},
  journal={arXiv preprint},
  year={2024}
}
```

---

## Support

**Issues:** https://github.com/msuv08/autonomous-vehicle-drifting/issues  
**Email:** your.email@example.com

---

**Last Updated:** October 2024  
**Estimated Reproduction Time:** 15-20 minutes

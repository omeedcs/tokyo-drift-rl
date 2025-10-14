# Reproducing Paper Results

This guide provides step-by-step instructions to reproduce the results from "Learning Inverse Kinodynamics for Autonomous Vehicle Drifting" (Suvarna & Tehrani, 2024).

## Paper Reference

**arXiv**: https://arxiv.org/abs/2402.14928  
**Published**: February 22, 2024

## Overview of Experiments

The paper presents three main experiments:

1. **Circle Navigation Test** (Section IV-C)
2. **Loose Drifting** (Section IV-D1)
3. **Tight Drifting** (Section IV-D2)

## Prerequisites

```bash
# Install all dependencies
pip install -e ".[all]"

# Verify installation
pytest tests/
```

## Experiment 1: Circle Navigation (Table I)

### Goal
Validate IKD model on simple circular trajectories at 2.0 m/s.

### Expected Results

| Commanded Curvature | Executed Curvature | IKD-Corrected | Deviation |
|---------------------|-------------------|---------------|-----------|
| 0.12 m              | 0.135 m           | 0.1172 m      | 2.33%     |
| 0.63 m              | 0.634 m           | 0.6293 m      | 0.11%     |
| 0.70 m              | 0.667 m           | 0.704 m       | ~0.5%     |
| 0.80 m              | 0.810 m           | 0.8142 m      | 1.78%     |

### Steps

#### 1. Prepare Training Data

The circle test requires turning data at various speeds (1.0-5.0 m/s):

```bash
# If you have the turning data files
python src/data_processing/merge_csv.py
```

Expected data characteristics:
- **Velocity range**: 1.0-4.2 m/s (capped by ERPM limit)
- **Duration**: ~5 minutes per speed
- **Curvature distribution**: Bimodal (mostly max curvatures)

#### 2. Train Model

```bash
python train.py \
  --config configs/circle_navigation.yaml \
  --experiment-name circle_navigation_reproduction
```

Training parameters:
- **Epochs**: 50
- **Batch size**: 32
- **Learning rate**: 1e-5
- **Optimizer**: Adam with weight decay 1e-3

Expected training time: ~5-10 minutes on CPU, ~1-2 minutes on GPU

#### 3. Evaluate on Circles

```bash
python evaluate.py \
  --checkpoint experiments/circle_navigation_reproduction/checkpoints/best_model.pt \
  --dataset dataset/circle_test.csv \
  --plot-results
```

#### 4. Measure Physical Trajectories

To replicate the physical measurements:

1. Load model onto vehicle (see `docs/DEPLOYMENT.md`)
2. Execute circles with dry-erase marker attached
3. Measure radius with measuring tape
4. Calculate curvature: `c = 1/r`

Compare with and without IKD correction.

### Validation Criteria

✅ **Pass**: IKD deviation < 2.5% for all test curvatures  
⚠️ **Marginal**: IKD deviation 2.5-5%  
❌ **Fail**: IKD deviation > 5%

---

## Experiment 2: Loose Drifting (Table II)

### Goal
Demonstrate IKD can tighten loose drifting trajectories.

### Expected Results

| Direction | Tightened Turn Rate | IKD Correction Observable |
|-----------|---------------------|---------------------------|
| CCW       | 100%                | Noticeable                |
| CW        | 50%                 | Non-noticeable            |

### Setup

**Physical Setup**:
- 2 AMRL testing cones
- 2 cardboard boxes as boundaries
- Distance: 2.13 meters (84 inches) between cone and box
- Turbo speed: 5.0 m/s
- Surface: AHG gymnasium floor

**Data Collection**:
- Duration: 10 minutes continuous teleoperation
- Primarily counter-clockwise drifts
- Focus: loose clearance around cones

### Steps

#### 1. Collect Data (If Needed)

```bash
# Record bag file
rosbag record /joystick /vectornav/IMU -O loose_drift_data.bag

# Process
python src/data_processing/bag_to_csv.py loose_drift_data.bag
python src/data_processing/align.py
```

Expected IMU delay: 0.18-0.20 seconds

#### 2. Train Model

```bash
python train.py \
  --config configs/loose_drifting.yaml \
  --experiment-name loose_drifting_reproduction
```

#### 3. Evaluate

```bash
python evaluate.py \
  --checkpoint experiments/loose_drifting_reproduction/checkpoints/best_model.pt \
  --dataset dataset/loose_ccw.csv \
  --plot-results
```

Look for:
- Reduced angular velocity deviation
- Tighter predicted trajectories
- Improvement percentage > 0%

#### 4. Deploy and Test

Deploy to vehicle and compare:
1. Baseline (no IKD): Observe loose drift pattern
2. With IKD: Observe tightened trajectory
3. Measure: Should nearly hit cones (was safely clearing)

### Validation Criteria

✅ **Pass**: CCW shows visible tightening, improvement > 50%  
⚠️ **Marginal**: Some tightening but < 50%  
❌ **Fail**: No observable improvement

---

## Experiment 3: Tight Drifting (Challenging Case)

### Goal
Evaluate IKD limits on extremely tight trajectories.

### Expected Results

**Paper Finding**: Model **fails** on tight drifts - vehicle undershoots.

This is an expected negative result that demonstrates:
- Data diversity is critical
- Model struggles with extreme trajectories
- More tight-drift training data needed

### Setup

**Physical Setup**:
- 2 cones, 2 boxes
- Distance: 0.81 meters (32 inches) between cone and box
- Vehicle width: 0.48 meters (19 inches)
- Clearance: Only 0.33 meters (13 inches)
- Turbo speed: 5.0 m/s

### Steps

#### 1. Train Model

```bash
python train.py \
  --config configs/tight_drifting.yaml \
  --experiment-name tight_drifting_reproduction \
  --epochs 75
```

Note: Longer training due to harder task.

#### 2. Evaluate

```bash
python evaluate.py \
  --checkpoint experiments/tight_drifting_reproduction/checkpoints/best_model.pt \
  --dataset dataset/tight_ccw.csv \
  --plot-results
```

Expected observations:
- Lower R² score compared to loose drifts
- Higher curvature error
- May show improvement metrics but fails in practice

#### 3. Deploy and Test

**Expected behavior**: Vehicle will tighten too much and undershoot the turn.

This replicates the paper's finding that the model lacks sufficient training data for extreme trajectories.

### Validation Criteria

✅ **Matches Paper**: Model shows numerical improvement but fails physically  
⚠️ **Different**: Model works (implies better data than paper)  
❌ **Opposite**: Model makes things worse

---

## Key Findings to Replicate

### 1. Data Quality Matters

From Section III-D:

- **Zero curvature samples**: ~40% of raw data (should be pruned)
- **Velocity capping**: Max 4.219 m/s (ERPM limit)
- **IMU delay**: 0.15-0.25 seconds (out of range = corrupted data)
- **Curvature skew**: Bimodal distribution (max turns easier to teleoperate)

### 2. Model Architecture

Simple is effective:
- Input: 2D (velocity, true_angular_velocity)
- Hidden: 2 layers × 32 neurons
- Output: 1D (corrected_angular_velocity)
- Activation: ReLU
- Loss: MSE

### 3. Training Observations

- **Fast convergence**: ~10-20 epochs typically sufficient
- **Batch size**: 32 works well
- **Learning rate**: 1e-5 optimal
- **Overfitting**: Not a major issue with simple architecture

### 4. Performance Patterns

✅ **Works well**:
- Loose trajectories
- Counter-clockwise drifts
- Medium curvatures
- Circle navigation

⚠️ **Struggles with**:
- Tight trajectories
- Extreme curvatures
- Limited training diversity

---

## Troubleshooting

### Issue: Can't Replicate Circle Results

**Check**:
1. Model trained on turning data (1.0-5.0 m/s)?
2. Test velocity exactly 2.0 m/s?
3. Measuring radius accurately?
4. Floor surface similar (smooth, grippy)?

### Issue: Loose Drifting Doesn't Improve

**Check**:
1. Enough training data (>10 minutes)?
2. Data primarily CCW direction?
3. IMU delay in expected range?
4. Zero curvatures removed from training?

### Issue: Better Results Than Paper

**Possible reasons**:
- Better data quality
- Different surface conditions
- More diverse training data
- Improved vehicle calibration

This is good! Consider contributing back.

### Issue: Worse Results Than Paper

**Possible reasons**:
- Insufficient training data
- Corrupted data (check IMU delays)
- Different vehicle/surface
- Implementation bugs (run tests)

---

## Reporting Results

### Benchmark Script

We provide a comprehensive benchmark:

```bash
python scripts/run_benchmarks.py \
  --config configs/default.yaml \
  --output benchmark_results.json
```

### Comparison Format

Please report results in this format:

```markdown
## Reproduction Results

**System**: [CPU/GPU model]
**Training time**: [X minutes]
**Dataset**: [describe any differences]

### Circle Navigation
| Curvature | Paper | Reproduction | Diff |
|-----------|-------|--------------|------|
| 0.12      | 2.33% | X.XX%        | ±X%  |
| 0.63      | 0.11% | X.XX%        | ±X%  |
| 0.80      | 1.78% | X.XX%        | ±X%  |

### Loose Drifting
- CCW Improvement: XX% (Paper: 100%)
- CW Improvement: XX% (Paper: 50%)

### Notes
[Any observations or differences]
```

---

## Citation

If you reproduce these results, please cite:

```bibtex
@article{suvarna2024learning,
  title={Learning Inverse Kinodynamics for Autonomous Vehicle Drifting},
  author={Suvarna, Mihir and Tehrani, Omeed},
  journal={arXiv preprint arXiv:2402.14928},
  year={2024}
}
```

---

## Questions?

- **Issues**: https://github.com/msuv08/autonomous-vehicle-drifting/issues
- **Email**: msuvarna@cs.utexas.edu, omeed@cs.utexas.edu
- **Paper Discussion**: https://arxiv.org/abs/2402.14928

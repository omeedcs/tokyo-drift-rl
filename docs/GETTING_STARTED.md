# Getting Started with IKD Drifting

This guide will help you get started with training and evaluating Inverse Kinodynamic models for autonomous vehicle drifting.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Collection](#data-collection)
- [Training](#training)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)

## Installation

### Option 1: Install from Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/msuv08/autonomous-vehicle-drifting.git
cd autonomous-vehicle-drifting

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with all dependencies
pip install -e .

# Or install with development tools
pip install -e ".[dev]"

# Or install with experiment tracking
pip install -e ".[experiment]"
```

### Option 2: Install from PyPI (Coming Soon)

```bash
pip install ikd-drifting
```

### Verify Installation

```bash
# Run tests
pytest tests/

# Check imports
python -c "from src.models.ikd_model import IKDModel; print('Import successful!')"
```

## Quick Start

### 1. Train a Model (5 minutes)

Train a baseline model with default settings:

```bash
python train.py
```

This will:
- Load data from `./dataset/ikddata2.csv`
- Train for 50 epochs
- Save checkpoints to `./experiments/ikd_baseline/`
- Generate training history plots

### 2. Evaluate the Model (2 minutes)

Evaluate on test datasets:

```bash
python evaluate.py --checkpoint experiments/ikd_baseline/checkpoints/best_model.pt --plot-results
```

This will:
- Test on all configured datasets (loose/tight, CW/CCW)
- Compute comprehensive metrics
- Generate comparison plots
- Save results to `./evaluation_results/`

### 3. Visualize Results

View training history:

```python
import json
import matplotlib.pyplot as plt

# Load training history
with open('experiments/ikd_baseline/training_history.json') as f:
    history = json.load(f)

# Plot
plt.plot(history['train_losses'], label='Train')
plt.plot(history['val_losses'], label='Val')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.show()
```

## Data Collection

### Understanding the Data Format

The model expects CSV files with two columns:
- `joystick`: List of `[velocity, angular_velocity]` from controller
- `executed`: List of `[true_angular_velocity]` from IMU

Example row:
```
joystick,executed
"[2.5, 0.7]","[0.65]"
```

### Collecting Your Own Data

If you have access to a UT AUTOmata F1/10 vehicle:

#### 1. Record ROS Bag Files

```bash
# Record joystick and IMU data
rosbag record /joystick /vectornav/IMU -O my_drift_data.bag
```

#### 2. Convert to CSV

```bash
cd bag-files-and-data/
python ../src/data_processing/bag_to_csv.py my_drift_data.bag
```

#### 3. Align and Process

```bash
# Align IMU and joystick data
python src/data_processing/align.py

# Merge multiple datasets
python src/data_processing/merge_csv.py
```

#### 4. Validate Data

```bash
python train.py --validate-data
```

### Data Quality Tips

From the paper's findings:

1. **Velocity Range**: Ensure data covers 1.0-4.2 m/s
2. **Curvature Diversity**: Collect data at various curvatures (not just max turns)
3. **Direction Balance**: Collect both CW and CCW drifts
4. **Environment**: Use smooth, grippy surfaces (like gym floors)
5. **IMU Delay**: Expected range is 0.15-0.25 seconds

## Training

### Basic Training

```bash
# Train with default config
python train.py

# Train with custom config
python train.py --config configs/loose_drifting.yaml

# Override specific parameters
python train.py --epochs 100 --batch-size 64 --experiment-name my_experiment
```

### Configuration Options

Edit `configs/default.yaml` or create your own:

```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 1.0e-5
  weight_decay: 1.0e-3
  optimizer: "adam"  # or "adamw", "sgd"
```

### Monitor Training

#### Option 1: Console Output

Training progress is printed every epoch:
```
[INFO] epoch 10 | loss   0.001234
```

#### Option 2: TensorBoard

Enable in config:
```yaml
experiment:
  tensorboard: true
```

Then run:
```bash
tensorboard --logdir experiments/
```

#### Option 3: Weights & Biases

Enable in config:
```yaml
experiment:
  wandb:
    enabled: true
    project: "autonomous-vehicle-drifting"
    entity: "your-username"
```

### Resume Training

```bash
# Load checkpoint and continue
python train.py --resume experiments/ikd_baseline/checkpoints/epoch_30.pt
```

## Evaluation

### Full Evaluation

```bash
python evaluate.py \
  --checkpoint experiments/ikd_baseline/checkpoints/best_model.pt \
  --plot-results \
  --save-predictions
```

### Evaluate on Specific Dataset

```bash
python evaluate.py \
  --checkpoint path/to/model.pt \
  --dataset dataset/loose_ccw.csv \
  --output-dir my_results/
```

### Understanding Metrics

Key metrics from the paper:

- **MSE/MAE/RMSE**: Standard regression metrics
- **Curvature Error**: Deviation in executed curvature
- **Improvement %**: How much IKD improves over baseline
- **Baseline MAE**: Error between commanded and true AV (without IKD)
- **Model MAE**: Error between predicted and true AV (with IKD)

### Circle Navigation Test

Replicate Table I from the paper:

```bash
python evaluate.py \
  --config configs/circle_navigation.yaml \
  --checkpoint path/to/model.pt
```

Expected results:
- **0.12m curvature**: ~2.33% deviation
- **0.63m curvature**: ~0.11% deviation
- **0.80m curvature**: ~1.78% deviation

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Make sure you're in the right directory
cd autonomous-vehicle-drifting

# Reinstall in editable mode
pip install -e .
```

#### 2. CUDA Out of Memory

Reduce batch size in config:
```yaml
training:
  batch_size: 16  # or 8
```

#### 3. Poor Model Performance

- **Insufficient data**: Collect at least 10 minutes per scenario
- **Low diversity**: Ensure varied curvatures and speeds
- **Wrong alignment**: Check IMU delay is in 0.15-0.25s range

#### 4. Data Validation Failures

```bash
# Run validation to see specific issues
python train.py --validate-data

# Common fixes:
# - Remove zero-velocity samples
# - Check for NaN/Inf values
# - Verify curvature distribution
```

### Getting Help

- **Issues**: https://github.com/msuv08/autonomous-vehicle-drifting/issues
- **Paper**: https://arxiv.org/abs/2402.14928
- **Email**: msuvarna@cs.utexas.edu, omeed@cs.utexas.edu

## Next Steps

1. ✅ Train baseline model
2. ✅ Evaluate on test sets
3. 📊 Analyze results and metrics
4. 🔧 Tune hyperparameters if needed
5. 🚗 Deploy on vehicle (see `docs/DEPLOYMENT.md`)
6. 📝 Contribute improvements!

## Citation

If you use this code, please cite our paper:

```bibtex
@article{suvarna2024learning,
  title={Learning Inverse Kinodynamics for Autonomous Vehicle Drifting},
  author={Suvarna, Mihir and Tehrani, Omeed},
  journal={arXiv preprint arXiv:2402.14928},
  year={2024}
}
```

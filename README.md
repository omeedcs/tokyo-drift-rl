# Learning Inverse Kinodynamics for Autonomous Vehicle Drifting

A research project implementing modified Inverse Kinodynamic Learning for safe slippage and tight turning in autonomous drifting scenarios.

## Overview

This work demonstrates that a simple neural network architecture can effectively learn inverse kinodynamics for loose drifting trajectories. However, tight trajectories present challenges where the vehicle undershoots during test time. The research emphasizes the critical importance of data quality and evaluation in learning inverse kinodynamic functions.

### Key Findings
- ✅ Effective for loose drifting trajectories
- ⚠️ Performance degrades on tight turning trajectories
- 📊 Data quality and diversity are essential
- 🧠 Simple architecture (2 hidden layers, 32 units each) is sufficient

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Problem Formulation](#problem-formulation)
- [Dataset](#dataset)
- [Results](#results)

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- ROS (for bag file processing)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd autonomous-vehicle-drifting

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
.
├── src/
│   ├── models/              # Neural network models and training
│   │   ├── ikd_model.py     # IKD model definition
│   │   ├── ikd_training.py  # Training script
│   │   ├── make_predictions.py  # Inference script
│   │   └── trace_trained_model.py
│   ├── data_processing/     # Data extraction and alignment
│   │   ├── align.py         # Time alignment
│   │   ├── bag_to_csv.py    # ROS bag conversion
│   │   ├── extractors.py    # IMU/joystick extraction
│   │   ├── interpolation.py # Time interpolation
│   │   └── merge_csv.py     # Data merging
│   ├── visualization/       # Plotting and analysis
│   │   ├── graph.py
│   │   ├── graph_training_data.py
│   │   ├── plot_drifting_data.py
│   │   └── plot_turning_data.py
│   └── utils/              # Utility scripts
├── dataset/                # Training and test data
├── vehicle edits/          # Vehicle configuration files
├── navigation.cc           # C++ navigation code
├── vesc_driver.cpp         # VESC driver code
└── README.md
```

## Usage

### 1. Data Processing

Convert ROS bag files to CSV:
```bash
cd bag-files-and-data/
python ../src/data_processing/bag_to_csv.py <bagfile.bag>
```

Align and merge training data:
```bash
python src/data_processing/align.py
python src/data_processing/merge_csv.py
```

### 2. Training

Train the IKD model:
```bash
python src/models/ikd_training.py
```

The training script will:
- Load data from `dataset/ikddata2.csv`
- Split into 90% train / 10% test
- Train for 50 epochs with batch size 32
- Save model weights to `ikddata2.pt`
- Display training/testing loss curves

### 3. Inference

Make predictions on test datasets:
```bash
python src/models/make_predictions.py
```

This will generate plots for all available test trajectories:
- Loose counter-clockwise (CCW)
- Loose clockwise (CW)
- Tight CCW
- Tight CW

### 4. Visualization

Plot training data characteristics:
```bash
python src/visualization/graph_training_data.py
python src/visualization/plot_turning_data.py
```

## Model Architecture

The Inverse Kinodynamic Model uses a simple feedforward neural network:

```
Input Layer (2):  [velocity, true_angular_velocity]
    ↓
Hidden Layer 1 (32): ReLU activation
    ↓
Hidden Layer 2 (32): ReLU activation
    ↓
Output Layer (1): [predicted_joystick_angular_velocity]
```

![Model Architecture](https://user-images.githubusercontent.com/61725820/234666525-a226c27b-9a0b-4167-bca0-e47f078894b6.png)

**Training Details:**
- Optimizer: Adam (lr=1e-5, weight_decay=1e-3)
- Loss: Mean Squared Error (MSE)
- Batch size: 32
- Epochs: 50

## Problem Formulation

We denote:
- $x$ = linear velocity (joystick command)
- $z$ = angular velocity (joystick command)
- $z'$ = angular velocity (IMU measurement)
- $u_z$ = desired control input

### Goal
Learn a function approximator $f_{\theta}^{+}$ that maps onboard inertial observations to corrected control inputs:

$$f_{\theta}^{+}: (x, z') \rightarrow z$$

At training time:
- **Input**: Joystick velocity $x$ and ground truth angular velocity $z'$ from IMU
- **Output**: Predicted joystick angular velocity $z$

At test time:
- The learned model acts as an inverse kinodynamic model
- Provides corrected angular velocity $u_z$ to match real-world observations

## Dataset

The dataset contains synchronized joystick commands and IMU measurements:
- **Joystick data**: Linear velocity, angular velocity commands
- **IMU data**: Ground truth angular velocity from VectorNav IMU
- **Time alignment**: Optimal delay computed via least squares (~0.18-0.20s)

Test trajectories included:
- `loose_ccw.csv` - Loose counter-clockwise drift
- `loose_cw.csv` - Loose clockwise drift
- `tight_ccw.csv` - Tight counter-clockwise drift
- `tight_cw.csv` - Tight clockwise drift

## Results

The model performs well on loose drifting trajectories but struggles with tight turns, indicating:
1. The need for more diverse training data covering tight maneuvers
2. Potential architecture limitations for high-curvature scenarios
3. The importance of data quality over model complexity

## Future Work

- 🔄 Collect more diverse training data with varied curvatures
- 📡 Incorporate additional sensor modalities (LiDAR, RealSense)
- 🎯 Explore architectures better suited for tight trajectories
- ⚙️ Implement real-time deployment on autonomous vehicles

## References

Inspired by the UT AUTOmata Inverse Kinodynamics work:
- GitHub: [ut-amrl/ikd](https://github.com/ut-amrl/ikd)

## License

Open-sourced for research and educational purposes.

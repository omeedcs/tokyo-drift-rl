# 🧪 Testing Guide

This guide will help you test the simulator and verify everything is working.

## Quick Start (First Time Setup)

### 1. Install Dependencies

Run the setup script to create a virtual environment and install all dependencies:

```bash
./quick_setup.sh
```

Or manually:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Installation

Test that all modules can be imported:

```bash
python3 test_imports.py
```

You should see all ✅ checks pass.

## Testing the Simulator

### Basic Functionality Tests

**1. Run Unit Tests**

```bash
# Activate virtual environment first
source venv/bin/activate

# Run all simulator tests
pytest tests/test_simulator.py -v

# Run all tests
pytest tests/ -v
```

**2. Test Circle Navigation**

```bash
python3 simulate.py --mode circle --velocity 2.0 --curvature 0.7 --duration 10.0
```

This will:
- Create a circular reference trajectory
- Simulate vehicle following the trajectory
- Display plots of position, velocity, and control inputs
- Save data to `sim_data/`

**3. Test Loose Drifting**

```bash
python3 simulate.py --mode drift-loose --duration 15.0
```

This simulates the loose drifting scenario from the paper.

**4. Test Tight Drifting**

```bash
python3 simulate.py --mode drift-tight --duration 15.0
```

This simulates the tight drifting scenario from the paper.

### Advanced Tests

**5. Test with Visualization**

```bash
python3 simulate.py --mode circle --velocity 2.0 --curvature 0.7 --visualize
```

**6. Test Data Generation**

```bash
python3 simulate.py --mode circle --velocity 2.0 --curvature 0.7 --save-data
```

Check the `sim_data/` directory for generated CSV files.

**7. Run All Paper Experiments**

```bash
python3 examples/simulate_paper_experiments.py
```

This reproduces all experiments from the paper.

## What Should Work

### ✅ Implemented Features

1. **Vehicle Dynamics**
   - Ackerman steering geometry
   - Motor ERPM limits (0-20000)
   - Servo constraints (-0.45 to 0.45 rad)
   - Acceleration/deceleration limits
   - Tire slip dynamics

2. **Sensor Simulation**
   - IMU with Gaussian noise (std=0.02 rad/s)
   - 10ms delay
   - Velocity sensor with noise

3. **Test Scenarios**
   - Circle navigation (Table I from paper)
   - Loose drifting (Figure 5 from paper)
   - Tight drifting (Figure 5 from paper)

4. **Controllers**
   - Virtual joystick control
   - Drift controller (sinusoidal trajectory)
   - Trajectory follower with PID

5. **Visualization**
   - Real-time trajectory plots
   - Control input plots
   - State variable plots
   - Comparison plots (baseline vs IKD)

6. **Data Export**
   - CSV format compatible with training pipeline
   - Includes: position, velocity, orientation, angular velocity, servo, motor

## Known Limitations

1. **No IKD Model Testing Yet**
   - Requires training a model first
   - Use `--use-ikd` flag once you have a trained model

2. **No Animation Support**
   - Install optional dependencies: `pip install pillow ffmpeg-python`
   - Then use `--animate` flag

3. **No ROS Integration**
   - Simulator is standalone, doesn't interface with ROS bags yet

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:
```bash
source venv/bin/activate  # Make sure venv is activated
pip install -r requirements.txt
```

### Test Failures

If tests fail:
```bash
# Run with verbose output
pytest tests/test_simulator.py -v -s

# Run a specific test
pytest tests/test_simulator.py::test_vehicle_initialization -v
```

### Simulation Crashes

If simulation crashes or produces errors:
```bash
# Try with shorter duration
python3 simulate.py --mode circle --duration 5.0

# Check the error messages for specific issues
```

## Quick Validation Checklist

Run these commands to verify everything works:

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Test imports
python3 test_imports.py

# 3. Run unit tests
pytest tests/test_simulator.py -v

# 4. Run a quick simulation
python3 simulate.py --mode circle --duration 5.0

# 5. Check output
ls sim_data/
```

If all of these complete successfully, your simulator is working! 🎉

## Next Steps

1. **Generate Training Data**: Use the simulator to create synthetic datasets
2. **Train IKD Model**: Run `python3 train.py` with generated data
3. **Test with IKD**: Use `--use-ikd` flag with your trained model
4. **Compare Performance**: Use `--compare` to see baseline vs IKD

## Getting Help

- Check `docs/SIMULATOR_GUIDE.md` for detailed simulator documentation
- Check `docs/GETTING_STARTED.md` for training and evaluation guides
- Open an issue on GitHub if you encounter bugs

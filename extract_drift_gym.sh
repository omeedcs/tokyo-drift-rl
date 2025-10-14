#!/bin/bash
# Script to extract drift-gym into a standalone repository

set -e  # Exit on error

echo "========================================"
echo "drift-gym Repository Extraction Script"
echo "========================================"
echo ""

# Configuration
SOURCE_DIR="/Users/omeedtehrani/autonomous-vehicle-drifting"
TARGET_DIR="$HOME/drift-gym"
REPO_NAME="drift-gym"

echo "Source directory: $SOURCE_DIR"
echo "Target directory: $TARGET_DIR"
echo ""

# Confirm with user
read -p "This will create a new directory at $TARGET_DIR. Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Create target directory
echo "[1/10] Creating target directory..."
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Initialize git
echo "[2/10] Initializing git repository..."
git init

# Copy core package
echo "[3/10] Copying drift_gym package..."
cp -r "$SOURCE_DIR/drift_gym" .

# Copy experiments
echo "[4/10] Copying experiments..."
cp -r "$SOURCE_DIR/experiments" .

# Copy tests
echo "[5/10] Copying tests..."
cp -r "$SOURCE_DIR/tests" .

# Create docs directory and copy documentation
echo "[6/10] Setting up documentation..."
mkdir -p docs
cp "$SOURCE_DIR/RESEARCH_GUIDE.md" docs/
cp "$SOURCE_DIR/CALIBRATION_REPORT.md" docs/
cp "$SOURCE_DIR/QUICK_START_RESEARCH.md" docs/
cp "$SOURCE_DIR/RESEARCH_IMPROVEMENTS_SUMMARY.md" docs/

# Copy package files
echo "[7/10] Copying package configuration..."
cp "$SOURCE_DIR/NEW_REPO_FILES/setup.py" .
cp "$SOURCE_DIR/NEW_REPO_FILES/pyproject.toml" .
cp "$SOURCE_DIR/NEW_REPO_FILES/README.md" .
cp "$SOURCE_DIR/NEW_REPO_FILES/.gitignore" .
cp "$SOURCE_DIR/NEW_REPO_FILES/LICENSE" .

# Create minimal requirements.txt
echo "[8/10] Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Core dependencies
numpy>=2.0.0,<3.0.0
gymnasium>=0.29.0
pygame>=2.6.0
scipy>=1.14.0

# Optional: For experiments
# stable-baselines3>=2.0.0
# tensorboard>=2.0.0
# pandas>=2.2.0
# matplotlib>=3.9.0
# seaborn>=0.13.0
EOF

# Create examples directory
echo "[9/10] Creating examples..."
mkdir -p examples
cat > examples/train_sac_baseline.py << 'EOF'
"""
Example: Train SAC on baseline configuration
"""
from experiments.benchmark_algorithms import train_algorithm, evaluate_model

if __name__ == "__main__":
    # Train
    model = train_algorithm(
        algorithm_name="SAC",
        config_name="baseline",
        seed=0,
        total_timesteps=100000,
        save_dir="results"
    )
    
    # Evaluate
    metrics = evaluate_model(
        model=model,
        algorithm_name="SAC",
        config_name="baseline",
        seed=0,
        n_eval_episodes=100,
        save_dir="results"
    )
    
    print(metrics)
EOF

# Fix import in drift_car_env_advanced.py
echo "[10/10] Fixing imports..."
# Note: This needs manual adjustment depending on Option A or B
echo "WARNING: You need to manually fix the import in drift_gym/envs/drift_car_env_advanced.py"
echo "         Change: from src.simulator.environment import SimulationEnvironment"
echo "         To something that works in the standalone package"

# Create initial commit
echo ""
echo "Creating initial commit..."
git add .
git commit -m "Initial commit: drift-gym v1.0.0

Research-grade drift control environment with:
- Validated sensor models (GPS, IMU)
- Extended Kalman Filter
- Standardized evaluation
- Benchmarking tools
- Ablation framework
"

echo ""
echo "========================================"
echo "✅ Extraction complete!"
echo "========================================"
echo ""
echo "New repository created at: $TARGET_DIR"
echo ""
echo "Next steps:"
echo "1. cd $TARGET_DIR"
echo "2. Fix imports in drift_gym/envs/drift_car_env_advanced.py"
echo "3. Test installation: pip install -e ."
echo "4. Run tests: pytest tests/ -v"
echo "5. Create GitHub repo and push:"
echo "   git remote add origin https://github.com/yourusername/$REPO_NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "See EXTRACT_DRIFT_GYM_REPO.md for more details."

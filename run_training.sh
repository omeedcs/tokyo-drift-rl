#!/bin/bash
# Quick training script for IKD model
# Usage: ./run_training.sh

echo "================================================"
echo "Autonomous Vehicle Drifting - Training Script"
echo "================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "[INFO] Installing dependencies..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "[INFO] Running IKD training..."
python -m src.models.ikd_training

echo ""
echo "[INFO] Training complete!"
echo "[INFO] Model saved to ikddata2.pt"

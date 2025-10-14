#!/bin/bash
# Quick inference script for IKD model
# Usage: ./run_predictions.sh

echo "================================================"
echo "Autonomous Vehicle Drifting - Inference Script"
echo "================================================"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if model exists
if [ ! -f "ikddata2.pt" ]; then
    echo "[ERROR] Model file 'ikddata2.pt' not found!"
    echo "[INFO] Please run training first: ./run_training.sh"
    exit 1
fi

echo "[INFO] Running predictions on test datasets..."
python -m src.models.make_predictions

echo ""
echo "[INFO] Predictions complete!"

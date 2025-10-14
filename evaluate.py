#!/usr/bin/env python3
"""
Evaluation script for trained IKD models.

Usage:
    python evaluate.py --checkpoint experiments/ikd_baseline/checkpoints/best_model.pt
    python evaluate.py --checkpoint path/to/model.pt --dataset dataset/loose_ccw.csv
"""
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json

from src.models.ikd_model import IKDModel
from src.evaluation.metrics import IKDMetrics, CircleMetrics
from src.utils.config import Config
from src.visualization.plot_results import plot_predictions, plot_angular_velocity_comparison


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate IKD Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to test dataset (if None, evaluates all configured test sets)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to CSV"
    )
    parser.add_argument(
        "--plot-results",
        action="store_true",
        help="Generate and save plots"
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        model = IKDModel(
            dim_input=config_dict.get('model', {}).get('input_dim', 2),
            dim_output=config_dict.get('model', {}).get('output_dim', 1)
        )
    else:
        # Default architecture
        model = IKDModel(dim_input=2, dim_output=1)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Training loss: {checkpoint.get('train_losses', [])[-1] if checkpoint.get('train_losses') else 'N/A'}")
    print(f"Validation loss: {checkpoint.get('val_losses', [])[-1] if checkpoint.get('val_losses') else 'N/A'}")
    
    return model


def evaluate_dataset(
    model: torch.nn.Module,
    dataset_path: str,
    device: torch.device,
    dataset_name: str = "test"
):
    """Evaluate model on a single dataset."""
    print(f"\nEvaluating on {dataset_name}...")
    print(f"Dataset: {dataset_path}")
    
    # Load data
    data = pd.read_csv(dataset_path)
    joystick = np.array([eval(i) for i in data["joystick"]])
    executed = np.array([eval(i) for i in data["executed"]])
    
    # Extract components
    velocities = joystick[:, 0]
    commanded_av = joystick[:, 1]
    true_av = executed[:, 0]
    
    # Prepare inputs for model
    inputs = torch.FloatTensor(np.column_stack([velocities, true_av])).to(device)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(inputs).cpu().numpy().flatten()
    
    # Compute metrics
    metrics = IKDMetrics.compute_all_metrics(
        predictions=predictions,
        targets=commanded_av,
        velocities=velocities,
        commanded_av=commanded_av
    )
    
    # Add dataset-specific metrics
    metrics['dataset'] = dataset_name
    metrics['num_samples'] = len(data)
    
    # Print results
    print(f"\nResults for {dataset_name}:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  R²: {metrics['r2']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  Curvature Error: {metrics['curvature_error']:.6f}")
    
    if 'baseline_mae' in metrics:
        print(f"\n  Baseline MAE (commanded vs true): {metrics['baseline_mae']:.6f}")
        print(f"  Model MAE (predicted vs true): {metrics['model_mae']:.6f}")
        print(f"  Improvement: {metrics['improvement_percentage']:.2f}%")
    
    return {
        'metrics': metrics,
        'data': {
            'velocities': velocities,
            'commanded_av': commanded_av,
            'true_av': true_av,
            'predicted_av': predictions
        }
    }


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Determine datasets to evaluate
    if args.dataset:
        datasets = {Path(args.dataset).stem: args.dataset}
    else:
        # Use default test datasets
        datasets = {
            'loose_ccw': './dataset/loose_ccw.csv',
            'loose_cw': './dataset/loose_cw.csv',
            'tight_ccw': './dataset/tight_ccw.csv',
            'tight_cw': './dataset/tight_cw.csv'
        }
    
    # Evaluate on all datasets
    all_results = {}
    for name, path in datasets.items():
        if not Path(path).exists():
            print(f"Warning: Dataset {path} not found, skipping...")
            continue
        
        results = evaluate_dataset(model, path, device, name)
        all_results[name] = results
        
        # Save predictions
        if args.save_predictions:
            pred_path = output_dir / f"{name}_predictions.csv"
            pred_df = pd.DataFrame(results['data'])
            pred_df.to_csv(pred_path, index=False)
            print(f"Predictions saved to {pred_path}")
        
        # Generate plots
        if args.plot_results:
            plot_path = output_dir / f"{name}_comparison.png"
            plot_angular_velocity_comparison(
                commanded_av=results['data']['commanded_av'],
                true_av=results['data']['true_av'],
                predicted_av=results['data']['predicted_av'],
                title=f"Angular Velocity Comparison - {name}",
                save_path=str(plot_path)
            )
            print(f"Plot saved to {plot_path}")
    
    # Save summary metrics
    summary_metrics = {
        name: results['metrics']
        for name, results in all_results.items()
    }
    
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        summary_json = {k: {k2: convert_types(v2) for k2, v2 in v.items()} 
                       for k, v in summary_metrics.items()}
        json.dump(summary_json, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete! Results saved to {output_dir}")
    print(f"Summary metrics saved to {summary_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

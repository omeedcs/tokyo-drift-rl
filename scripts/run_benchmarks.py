#!/usr/bin/env python3
"""
Comprehensive benchmark script for reproducing paper results.

Usage:
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --config configs/default.yaml --output results.json
"""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

from src.models.ikd_model import IKDModel
from src.models.trainer import IKDTrainer, create_data_loaders
from src.evaluation.metrics import IKDMetrics, CircleMetrics
from src.utils.config import load_config
from src.utils.logger import ExperimentLogger
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Run IKD Benchmarks")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    parser.add_argument("--skip-training", action="store_true", 
                       help="Skip training, only evaluate existing models")
    return parser.parse_args()


def benchmark_training_speed(config, logger):
    """Benchmark training speed."""
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK 1: Training Speed")
    logger.info("="*80)
    
    # Load small subset of data
    data = pd.read_csv(config.data.train_path)
    joystick = np.array([eval(i) for i in data["joystick"]])
    executed = np.array([eval(i) for i in data["executed"]])
    data_array = np.concatenate((joystick, executed), axis=1)
    
    # Use subset for speed test
    subset_size = min(1000, len(data_array))
    data_subset = data_array[:subset_size]
    
    # Split
    train_size = int(subset_size * 0.9)
    train_data = data_subset[:train_size]
    val_data = data_subset[train_size:]
    
    # Create loaders
    train_loader, val_loader = create_data_loaders(
        train_data, val_data, 
        batch_size=32, 
        num_workers=0
    )
    
    # Create model
    model = IKDModel(dim_input=2, dim_output=1)
    
    # Benchmark training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = IKDTrainer(model, config, logger, device)
    
    start_time = time.time()
    trainer.train(train_loader, val_loader, num_epochs=10)
    end_time = time.time()
    
    training_time = end_time - start_time
    time_per_epoch = training_time / 10
    
    results = {
        'total_time_seconds': training_time,
        'time_per_epoch_seconds': time_per_epoch,
        'samples_per_second': (len(train_data) * 10) / training_time,
        'device': str(device),
        'batch_size': 32,
        'num_epochs': 10
    }
    
    logger.info(f"Training time: {training_time:.2f}s")
    logger.info(f"Time per epoch: {time_per_epoch:.2f}s")
    logger.info(f"Throughput: {results['samples_per_second']:.1f} samples/sec")
    
    return results


def benchmark_inference_speed(model, device, logger):
    """Benchmark inference speed."""
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK 2: Inference Speed")
    logger.info("="*80)
    
    model.eval()
    model.to(device)
    
    # Warmup
    dummy_input = torch.randn(32, 2).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    num_samples = 10000
    batch_size = 32
    num_batches = num_samples // batch_size
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_batches):
            inputs = torch.randn(batch_size, 2).to(device)
            _ = model(inputs)
    end_time = time.time()
    
    inference_time = end_time - start_time
    latency = (inference_time / num_samples) * 1000  # ms
    
    results = {
        'total_time_seconds': inference_time,
        'samples_per_second': num_samples / inference_time,
        'latency_ms': latency,
        'batch_size': batch_size,
        'device': str(device)
    }
    
    logger.info(f"Inference throughput: {results['samples_per_second']:.1f} samples/sec")
    logger.info(f"Latency: {latency:.3f}ms per sample")
    
    return results


def benchmark_model_accuracy(config, logger):
    """Benchmark model accuracy on all test sets."""
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK 3: Model Accuracy")
    logger.info("="*80)
    
    # This would load a trained model and evaluate
    # For now, return placeholder
    results = {
        'note': 'Run full training and evaluation separately',
        'expected_metrics': {
            'circle_navigation': {
                '0.12_curvature_deviation_pct': 2.33,
                '0.63_curvature_deviation_pct': 0.11,
                '0.80_curvature_deviation_pct': 1.78
            },
            'loose_drifting': {
                'ccw_tightening_rate': 100,
                'cw_tightening_rate': 50
            }
        }
    }
    
    logger.info("Accuracy benchmark requires trained model.")
    logger.info("Run train.py and evaluate.py separately for full metrics.")
    
    return results


def get_system_info():
    """Collect system information."""
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'ram_gb': psutil.virtual_memory().total / (1024**3),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return info


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logger
    logger = ExperimentLogger(
        experiment_name="benchmarks",
        output_dir="./benchmark_results",
        log_to_file=True,
        log_to_console=True,
        use_tensorboard=False,
        use_wandb=False
    )
    
    logger.info("="*80)
    logger.info("IKD Autonomous Drifting - Benchmark Suite")
    logger.info("Paper: Suvarna & Tehrani, 2024 (arXiv:2402.14928)")
    logger.info("="*80)
    
    # Collect system info
    logger.info("\nSystem Information:")
    system_info = get_system_info()
    for key, value in system_info.items():
        logger.info(f"  {key}: {value}")
    
    # Run benchmarks
    results = {
        'timestamp': datetime.now().isoformat(),
        'config_file': args.config,
        'system_info': system_info,
        'benchmarks': {}
    }
    
    try:
        # Benchmark 1: Training Speed
        if not args.skip_training:
            training_results = benchmark_training_speed(config, logger)
            results['benchmarks']['training_speed'] = training_results
        
        # Benchmark 2: Inference Speed
        model = IKDModel(dim_input=2, dim_output=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inference_results = benchmark_inference_speed(model, device, logger)
        results['benchmarks']['inference_speed'] = inference_results
        
        # Benchmark 3: Accuracy
        accuracy_results = benchmark_model_accuracy(config, logger)
        results['benchmarks']['accuracy'] = accuracy_results
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        results['error'] = str(e)
        raise
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info(f"Benchmark complete! Results saved to: {output_path}")
    logger.info("="*80)
    
    logger.close()


if __name__ == "__main__":
    main()

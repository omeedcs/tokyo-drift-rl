#!/usr/bin/env python3
"""
Main training script for IKD model with full experiment tracking.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --experiment-name my_experiment
"""
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.models.ikd_model import IKDModel
from src.models.trainer import IKDTrainer, create_data_loaders
from src.utils.config import load_config, save_config
from src.utils.logger import ExperimentLogger
from src.data_processing.validators import DataValidator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train IKD Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (overrides config)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Training data path (overrides config)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Run data validation before training"
    )
    
    return parser.parse_args()


def load_and_prepare_data(config, logger):
    """Load and prepare training data."""
    logger.info(f"Loading data from {config.data.train_path}")
    
    # Load CSV
    data = pd.read_csv(config.data.train_path)
    
    # Parse data
    joystick = np.array([eval(i) for i in data["joystick"]])
    executed = np.array([eval(i) for i in data["executed"]])
    
    # Concatenate
    data_array = np.concatenate((joystick, executed), axis=1)
    
    logger.info(f"Loaded {len(data_array)} samples")
    logger.info(f"Data shape: {data_array.shape}")
    
    # Split train/validation
    N = len(data_array)
    N_train = int(N * config.data.train_split)
    
    if config.data.shuffle:
        np.random.seed(config.data.random_seed)
        indices = np.random.permutation(N)
        data_array = data_array[indices]
    
    train_data = data_array[:N_train]
    val_data = data_array[N_train:]
    
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    
    # Statistics
    logger.info("\nData Statistics:")
    logger.info(f"  Velocity range: [{data_array[:, 0].min():.3f}, {data_array[:, 0].max():.3f}]")
    logger.info(f"  Angular velocity range: [{data_array[:, 1].min():.3f}, {data_array[:, 1].max():.3f}]")
    logger.info(f"  True angular velocity range: [{data_array[:, 2].min():.3f}, {data_array[:, 2].max():.3f}]")
    
    return train_data, val_data


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    overrides = {}
    if args.experiment_name:
        overrides['experiment'] = {'name': args.experiment_name}
    if args.data_path:
        overrides['data'] = {'train_path': args.data_path}
    if args.epochs:
        overrides['training'] = {'epochs': args.epochs}
    if args.batch_size:
        overrides['training'] = {'batch_size': args.batch_size}
    
    config = load_config(args.config, overrides)
    
    # Setup logger
    logger = ExperimentLogger(
        experiment_name=config.experiment.name,
        output_dir=config.experiment.output_dir,
        use_tensorboard=config.experiment.tensorboard,
        use_wandb=config.experiment.wandb.enabled,
        wandb_config=config.experiment.wandb.to_dict() if hasattr(config.experiment, 'wandb') else None
    )
    
    logger.info("=" * 80)
    logger.info("Training Inverse Kinodynamic Model for Autonomous Vehicle Drifting")
    logger.info("Based on: Suvarna & Tehrani, 2024")
    logger.info("=" * 80)
    
    # Save configuration
    config_save_path = Path(config.experiment.output_dir) / config.experiment.name / "config.yaml"
    save_config(config, str(config_save_path))
    logger.info(f"Configuration saved to {config_save_path}")
    
    # Validate data if requested
    if args.validate_data:
        logger.info("\nValidating training data...")
        validator = DataValidator()
        validation_result = validator.validate_csv(config.data.train_path)
        
        if not validation_result.is_valid:
            logger.error("Data validation failed!")
            for error in validation_result.errors:
                logger.error(f"  - {error}")
            return
        
        if validation_result.warnings:
            for warning in validation_result.warnings:
                logger.warning(f"  - {warning}")
        
        logger.info("Data validation passed!")
        logger.info("\nData Statistics:")
        for key, value in validation_result.statistics.items():
            logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Load data
    train_data, val_data = load_and_prepare_data(config, logger)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data,
        val_data,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers
    )
    
    # Create model
    logger.info("\nInitializing model...")
    model = IKDModel(
        dim_input=config.model.input_dim,
        dim_output=config.model.output_dim
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"Model architecture:\n{model}")
    
    # Create trainer
    trainer = IKDTrainer(
        model=model,
        config=config,
        logger=logger
    )
    
    # Train
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        trainer.save_checkpoint("interrupted.pt")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        logger.close()
    
    logger.info("\n" + "=" * 80)
    logger.info("Training completed successfully!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.6f}")
    logger.info(f"Results saved to: {Path(config.experiment.output_dir) / config.experiment.name}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

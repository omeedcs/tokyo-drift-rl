"""
Logging utilities for training and evaluation.
Supports console logging, file logging, and optional TensorBoard/Wandb.
"""
import os
import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime


class ExperimentLogger:
    """
    Comprehensive experiment logger with multiple backends.
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str,
        log_to_file: bool = True,
        log_to_console: bool = True,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_config: Optional[Dict] = None
    ):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Output directory for logs
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_config: W&B configuration
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Python logger
        self.logger = self._setup_logger(log_to_file, log_to_console)
        
        # Setup TensorBoard
        self.tensorboard_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.output_dir / "tensorboard"
                self.tensorboard_writer = SummaryWriter(tb_dir)
                self.logger.info(f"TensorBoard logging enabled: {tb_dir}")
            except ImportError:
                self.logger.warning("TensorBoard not available. Install with: pip install tensorboard")
        
        # Setup Weights & Biases
        self.use_wandb = False
        if use_wandb:
            try:
                import wandb
                wandb_config = wandb_config or {}
                wandb.init(
                    project=wandb_config.get('project', 'ikd-drifting'),
                    name=experiment_name,
                    entity=wandb_config.get('entity'),
                    config=wandb_config.get('config', {})
                )
                self.use_wandb = True
                self.logger.info("Weights & Biases logging enabled")
            except ImportError:
                self.logger.warning("Wandb not available. Install with: pip install wandb")
    
    def _setup_logger(self, log_to_file: bool, log_to_console: bool) -> logging.Logger:
        """Setup Python logger with handlers."""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)
        logger.handlers = []  # Clear existing handlers
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        if log_to_file:
            log_file = self.output_dir / f"{self.experiment_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log metrics to all backends.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current step/epoch
            prefix: Prefix for metric names (e.g., "train/", "val/")
        """
        # Log to console
        metric_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} | {prefix}{metric_str}")
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            for name, value in metrics.items():
                self.tensorboard_writer.add_scalar(f"{prefix}{name}", value, step)
        
        # Log to Wandb
        if self.use_wandb:
            import wandb
            wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            wandb_metrics['step'] = step
            wandb.log(wandb_metrics)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        self.logger.info("Hyperparameters:")
        for key, value in hparams.items():
            self.logger.info(f"  {key}: {value}")
        
        if self.use_wandb:
            import wandb
            wandb.config.update(hparams)
    
    def save_checkpoint(self, checkpoint_path: str):
        """Log checkpoint save."""
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def close(self):
        """Close all logging backends."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.use_wandb:
            import wandb
            wandb.finish()

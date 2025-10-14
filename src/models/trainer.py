"""
Enhanced training module with experiment tracking, checkpointing, and evaluation.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import json

from src.models.ikd_model import IKDModel
from src.evaluation.metrics import IKDMetrics
from src.utils.logger import ExperimentLogger
from src.utils.config import Config


class IKDTrainer:
    """
    Comprehensive trainer for IKD models with experiment tracking.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        logger: ExperimentLogger,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: IKD model to train
            config: Configuration object
            logger: Experiment logger
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.logger = logger
        
        # Setup device
        if device is None:
            device_str = config.training.device
            if device_str == "auto":
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                elif torch.backends.mps.is_available():
                    device = torch.device("mps")
                else:
                    device = torch.device("cpu")
            else:
                device = torch.device(device_str)
        
        self.device = device
        self.model = self.model.to(device)
        self.logger.info(f"Training on device: {device}")
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup loss function
        self.criterion = self._setup_loss_function()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Output directory
        self.checkpoint_dir = Path(config.experiment.output_dir) / config.experiment.name / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer from config."""
        optimizer_name = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay
        
        if optimizer_name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function from config."""
        loss_name = self.config.training.loss_function.lower()
        
        if loss_name == "mse":
            return nn.MSELoss()
        elif loss_name == "mae" or loss_name == "l1":
            return nn.L1Loss()
        elif loss_name == "huber":
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        return {'loss': avg_loss}
    
    def evaluate(self, val_loader: DataLoader, compute_detailed_metrics: bool = False) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: Validation data loader
            compute_detailed_metrics: Whether to compute detailed metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        all_inputs = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if compute_detailed_metrics:
                    all_predictions.append(outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    all_inputs.append(inputs.cpu().numpy())
        
        avg_loss = epoch_loss / num_batches
        metrics = {'loss': avg_loss}
        
        # Compute detailed metrics
        if compute_detailed_metrics:
            predictions = np.concatenate(all_predictions, axis=0).flatten()
            targets = np.concatenate(all_targets, axis=0).flatten()
            inputs = np.concatenate(all_inputs, axis=0)
            
            velocities = inputs[:, 0]
            commanded_av = predictions  # Our predictions are the corrected commanded AV
            
            detailed_metrics = IKDMetrics.compute_all_metrics(
                predictions=predictions,
                targets=targets,
                velocities=velocities
            )
            metrics.update(detailed_metrics)
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None
    ):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (uses config if None)
        """
        if num_epochs is None:
            num_epochs = self.config.training.epochs
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.log_hyperparameters({
            'num_epochs': num_epochs,
            'batch_size': self.config.training.batch_size,
            'learning_rate': self.config.training.learning_rate,
            'optimizer': self.config.training.optimizer,
            'model_hidden_dim': self.config.model.hidden_dim,
        })
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['loss'])
            
            # Validate
            val_metrics = self.evaluate(
                val_loader,
                compute_detailed_metrics=(epoch % self.config.experiment.log_frequency == 0)
            )
            self.val_losses.append(val_metrics['loss'])
            
            # Log metrics
            self.logger.log_metrics(train_metrics, epoch, prefix="train/")
            self.logger.log_metrics(val_metrics, epoch, prefix="val/")
            
            # Save checkpoint
            if self.config.experiment.save_model:
                if epoch % self.config.experiment.save_frequency == 0 or epoch == num_epochs - 1:
                    self.save_checkpoint(f"epoch_{epoch}.pt")
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint("best_model.pt")
                    self.logger.info(f"New best model at epoch {epoch} with val_loss: {self.best_val_loss:.6f}")
            
            # Early stopping
            if self.config.training.early_stopping.enabled:
                if self._check_early_stopping():
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        self.save_training_history()
        
        self.logger.info("Training complete!")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.save_checkpoint(str(checkpoint_path))
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def save_training_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir.parent / "training_history.json"
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'num_epochs': len(self.train_losses)
        }
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_path}")
    
    def _check_early_stopping(self) -> bool:
        """Check if early stopping criteria is met."""
        patience = self.config.training.early_stopping.patience
        min_delta = self.config.training.early_stopping.min_delta
        
        if len(self.val_losses) < patience + 1:
            return False
        
        # Check if validation loss hasn't improved in 'patience' epochs
        recent_losses = self.val_losses[-patience:]
        best_recent = min(recent_losses)
        
        if self.best_val_loss - best_recent < min_delta:
            return True
        
        return False


def create_data_loaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    batch_size: int,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch data loaders from numpy arrays.
    
    Args:
        train_data: Training data array [N, 3] (velocity, angular_velocity, true_angular_velocity)
        val_data: Validation data array
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Extract inputs and targets
    train_inputs = torch.FloatTensor(train_data[:, [0, 2]])  # velocity, true_av
    train_targets = torch.FloatTensor(train_data[:, [1]])     # angular_velocity
    
    val_inputs = torch.FloatTensor(val_data[:, [0, 2]])
    val_targets = torch.FloatTensor(val_data[:, [1]])
    
    # Create datasets
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

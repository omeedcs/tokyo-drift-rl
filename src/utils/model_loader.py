"""
Pretrained Model Loader for IKD and SAC models.

Provides utilities for loading, managing, and discovering pretrained models.
"""

import os
import sys
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, 'jake-deep-rl-algos')
import deep_control as dc

from src.models.ikd_model import IKDModel


@dataclass
class ModelInfo:
    """Model metadata."""
    name: str
    model_type: str  # "ikd" or "sac"
    path: str
    timestamp: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, float]] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PretrainedModelLoader:
    """
    Unified loader for pretrained IKD and SAC models.
    
    Features:
    - Automatic model discovery
    - Model metadata tracking
    - Easy loading with error handling
    - Performance tracking
    
    Example:
        loader = PretrainedModelLoader()
        ikd_model = loader.load_ikd("ikd_final.pt")
        sac_agent = loader.load_sac("sac_loose")
        
        # List all available models
        models = loader.list_available_models()
    """
    
    def __init__(
        self,
        ikd_dir: str = "trained_models",
        sac_dir: str = "dc_saves",
        metadata_file: str = "model_registry.json"
    ):
        """
        Initialize model loader.
        
        Args:
            ikd_dir: Directory containing IKD models
            sac_dir: Directory containing SAC models
            metadata_file: JSON file storing model metadata
        """
        self.ikd_dir = Path(ikd_dir)
        self.sac_dir = Path(sac_dir)
        self.metadata_file = Path(metadata_file)
        
        # Load or create metadata registry
        self.registry = self._load_registry()
        
        # Auto-discover models if registry is empty
        if not self.registry:
            self.discover_models()
    
    def _load_registry(self) -> Dict[str, ModelInfo]:
        """Load model registry from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return {
                    k: ModelInfo(**v) for k, v in data.items()
                }
        return {}
    
    def _save_registry(self):
        """Save model registry to disk."""
        data = {k: v.to_dict() for k, v in self.registry.items()}
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def discover_models(self) -> List[ModelInfo]:
        """
        Auto-discover models in standard directories.
        
        Returns:
            List of discovered models
        """
        discovered = []
        
        # Discover IKD models
        if self.ikd_dir.exists():
            for model_path in self.ikd_dir.glob("*.pt"):
                if model_path.name not in [m.name for m in self.registry.values()]:
                    info = ModelInfo(
                        name=model_path.stem,
                        model_type="ikd",
                        path=str(model_path),
                        timestamp=datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
                    )
                    self.registry[info.name] = info
                    discovered.append(info)
        
        # Discover SAC models
        if self.sac_dir.exists():
            for model_dir in self.sac_dir.iterdir():
                if model_dir.is_dir() and model_dir.name not in [m.name for m in self.registry.values()]:
                    # Check for SAC checkpoint files (try both .pt and .pth)
                    actor_path = model_dir / "actor.pt"
                    if not actor_path.exists():
                        actor_path = model_dir / "actor.pth"
                    if actor_path.exists():
                        info = ModelInfo(
                            name=model_dir.name,
                            model_type="sac",
                            path=str(model_dir),
                            timestamp=datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat()
                        )
                        self.registry[info.name] = info
                        discovered.append(info)
        
        self._save_registry()
        return discovered
    
    def load_ikd(
        self,
        model_name: str,
        device: str = "cpu"
    ) -> Tuple[IKDModel, ModelInfo]:
        """
        Load pretrained IKD model.
        
        Args:
            model_name: Name of model (with or without .pt extension)
            device: Device to load model on
            
        Returns:
            Tuple of (model, metadata)
            
        Example:
            model, info = loader.load_ikd("ikd_final")
            print(f"Loaded {info.name} from {info.timestamp}")
        """
        # Remove .pt extension if present
        model_name = model_name.replace(".pt", "")
        
        # Check registry first
        if model_name in self.registry:
            info = self.registry[model_name]
            model_path = Path(info.path)
        else:
            # Try to find in ikd_dir
            model_path = self.ikd_dir / f"{model_name}.pt"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"IKD model '{model_name}' not found. "
                    f"Available models: {self.list_ikd_models()}"
                )
            
            # Create new registry entry
            info = ModelInfo(
                name=model_name,
                model_type="ikd",
                path=str(model_path),
                timestamp=datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
            )
            self.registry[model_name] = info
            self._save_registry()
        
        # Load model
        try:
            model = IKDModel()
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if 'parameters' in checkpoint:
                        info.parameters = checkpoint['parameters']
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            
            print(f"✅ Loaded IKD model: {model_name}")
            if info.timestamp:
                print(f"   Trained: {info.timestamp}")
            
            return model, info
            
        except Exception as e:
            raise RuntimeError(f"Failed to load IKD model '{model_name}': {e}")
    
    def load_sac(
        self,
        model_name: str,
        device: str = "cpu",
        obs_space_size: int = 10,
        act_space_size: int = 2,
        hidden_size: int = 256,
        log_std_low: float = -20,
        log_std_high: float = 2
    ) -> Tuple[dc.sac.SACAgent, ModelInfo]:
        """
        Load pretrained SAC agent.
        
        Args:
            model_name: Name of SAC model directory
            device: Device to load model on
            obs_space_size: Observation space size
            act_space_size: Action space size
            hidden_size: Hidden layer size
            log_std_low: Minimum log std for policy
            log_std_high: Maximum log std for policy
            
        Returns:
            Tuple of (agent, metadata)
            
        Example:
            agent, info = loader.load_sac("sac_loose_42")
        """
        # Check registry first
        if model_name in self.registry:
            info = self.registry[model_name]
            if info.path is None:
                raise ValueError(f"SAC model '{model_name}' has no path in registry")
            model_dir = Path(info.path)
        else:
            # Try to find in sac_dir
            if model_name is None:
                raise ValueError("model_name cannot be None")
            model_dir = self.sac_dir / model_name
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"SAC model '{model_name}' not found. "
                    f"Available models: {self.list_sac_models()}"
                )
            
            # Create new registry entry
            info = ModelInfo(
                name=model_name,
                model_type="sac",
                path=str(model_dir),
                timestamp=datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat()
            )
            self.registry[model_name] = info
            self._save_registry()
        
        # Load agent
        try:
            agent = dc.sac.SACAgent(
                obs_space_size=obs_space_size,
                act_space_size=act_space_size,
                hidden_size=hidden_size,
                log_std_low=log_std_low,
                log_std_high=log_std_high
            )
            
            # Load checkpoint (try both .pt and .pth extensions)
            actor_path = model_dir / "actor.pt"
            if not actor_path.exists():
                actor_path = model_dir / "actor.pth"
            
            critic1_path = model_dir / "critic1.pt"
            if not critic1_path.exists():
                critic1_path = model_dir / "critic1.pth"
            
            critic2_path = model_dir / "critic2.pt"
            if not critic2_path.exists():
                critic2_path = model_dir / "critic2.pth"
            
            if not actor_path.exists():
                raise FileNotFoundError(f"Actor checkpoint not found in {model_dir}")
            
            agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
            
            # Load critic1 and critic2 (SAC uses twin critics)
            if critic1_path.exists():
                agent.critic1.load_state_dict(torch.load(critic1_path, map_location=device))
            if critic2_path.exists():
                agent.critic2.load_state_dict(torch.load(critic2_path, map_location=device))
            
            agent.actor.to(device)
            agent.critic1.to(device)
            agent.critic2.to(device)
            agent.actor.eval()
            
            print(f"✅ Loaded SAC agent: {model_name}")
            if info.timestamp:
                print(f"   Trained: {info.timestamp}")
            
            return agent, info
            
        except Exception as e:
            raise RuntimeError(f"Failed to load SAC model '{model_name}': {e}")
    
    def list_available_models(self) -> Dict[str, List[ModelInfo]]:
        """
        List all available models by type.
        
        Returns:
            Dictionary with keys "ikd" and "sac" containing model lists
        """
        self.discover_models()  # Refresh
        
        ikd_models = [info for info in self.registry.values() if info.model_type == "ikd"]
        sac_models = [info for info in self.registry.values() if info.model_type == "sac"]
        
        return {
            "ikd": sorted(ikd_models, key=lambda x: x.timestamp or "", reverse=True),
            "sac": sorted(sac_models, key=lambda x: x.timestamp or "", reverse=True)
        }
    
    def list_ikd_models(self) -> List[str]:
        """List available IKD model names."""
        models = self.list_available_models()
        return [m.name for m in models["ikd"]]
    
    def list_sac_models(self) -> List[str]:
        """List available SAC model names."""
        models = self.list_available_models()
        return [m.name for m in models["sac"]]
    
    def register_model(
        self,
        name: str,
        model_type: str,
        path: str,
        parameters: Optional[Dict[str, Any]] = None,
        performance: Optional[Dict[str, float]] = None,
        description: Optional[str] = None
    ) -> ModelInfo:
        """
        Register a model with metadata.
        
        Args:
            name: Model name
            model_type: "ikd" or "sac"
            path: Path to model file/directory
            parameters: Training parameters
            performance: Performance metrics
            description: Model description
            
        Returns:
            ModelInfo object
        """
        info = ModelInfo(
            name=name,
            model_type=model_type,
            path=path,
            timestamp=datetime.now().isoformat(),
            parameters=parameters,
            performance=performance,
            description=description
        )
        
        self.registry[name] = info
        self._save_registry()
        
        return info
    
    def get_best_model(self, model_type: str, metric: str = "success_rate") -> Optional[ModelInfo]:
        """
        Get best model by performance metric.
        
        Args:
            model_type: "ikd" or "sac"
            metric: Performance metric to optimize
            
        Returns:
            Best model info or None
        """
        models = [
            info for info in self.registry.values()
            if info.model_type == model_type and info.performance and metric in info.performance
        ]
        
        if not models:
            return None
        
        return max(models, key=lambda x: x.performance[metric])
    
    def print_model_summary(self):
        """Print a summary of all available models."""
        models = self.list_available_models()
        
        print("\n" + "="*60)
        print("📦 PRETRAINED MODEL REGISTRY")
        print("="*60)
        
        print(f"\n🔵 IKD Models ({len(models['ikd'])} available):")
        print("-" * 60)
        for info in models['ikd']:
            print(f"  • {info.name}")
            if info.timestamp:
                print(f"    Trained: {info.timestamp[:10]}")
            if info.performance:
                print(f"    Performance: {info.performance}")
            print()
        
        print(f"\n🟢 SAC Models ({len(models['sac'])} available):")
        print("-" * 60)
        for info in models['sac']:
            print(f"  • {info.name}")
            if info.timestamp:
                print(f"    Trained: {info.timestamp[:10]}")
            if info.performance:
                print(f"    Performance: {info.performance}")
            print()
        
        print("="*60 + "\n")


def quick_load_ikd(model_name: str = "ikd_final", device: str = "cpu") -> IKDModel:
    """
    Quick load IKD model without metadata.
    
    Args:
        model_name: Model name
        device: Device to load on
        
    Returns:
        Loaded IKD model
    """
    loader = PretrainedModelLoader()
    model, _ = loader.load_ikd(model_name, device)
    return model


def quick_load_sac(
    model_name: str = "sac_loose",
    device: str = "cpu",
    obs_space_size: int = 10,
    act_space_size: int = 2,
    hidden_size: int = 256,
    log_std_low: float = -20,
    log_std_high: float = 2
) -> dc.sac.SACAgent:
    """
    Quick load SAC agent without metadata.
    
    Args:
        model_name: Model name
        device: Device to load on
        obs_space_size: Observation space size
        act_space_size: Action space size
        hidden_size: Hidden layer size
        log_std_low: Minimum log std for policy
        log_std_high: Maximum log std for policy
        
    Returns:
        Loaded SAC agent
    """
    loader = PretrainedModelLoader()
    agent, _ = loader.load_sac(
        model_name, device, obs_space_size, act_space_size,
        hidden_size, log_std_low, log_std_high
    )
    return agent


if __name__ == "__main__":
    # Demo usage
    loader = PretrainedModelLoader()
    loader.print_model_summary()

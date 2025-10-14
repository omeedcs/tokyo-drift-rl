"""
Configuration management for IKD training and evaluation.
Supports YAML configuration files with overrides.
"""
import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class Config:
    """Configuration manager with nested dictionary access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict
        self._flatten_config()
    
    def _flatten_config(self):
        """Flatten nested config for easy attribute access."""
        for key, value in self._config.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with default.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self._flatten_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self._config
    
    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: Optional[str] = None, overrides: Optional[Dict] = None) -> Config:
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML configuration file
        overrides: Dictionary of configuration overrides
        
    Returns:
        Config object
    """
    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
    
    # Load YAML config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Apply overrides
    if overrides:
        config_dict = _deep_update(config_dict, overrides)
    
    return Config(config_dict)


def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """
    Recursively update nested dictionary.
    
    Args:
        base_dict: Base dictionary
        update_dict: Updates to apply
        
    Returns:
        Updated dictionary
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    
    return base_dict


def save_config(config: Config, save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object
        save_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

"""
Configuration management for the denoising package.

This module provides configuration settings through environment variables,
configuration files, or default values.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


class Config:
    """Configuration manager for denoising package."""
    
    # Default configuration
    DEFAULT_CONFIG = {
        'data': {
            'derivatives_path': None,
            'output_path': './output',
            'atlas_path': './atlas',
        },
        'processing': {
            'n_jobs': -1,
            'memory': 'nilearn_cache',
            'verbose': 1,
        },
        'denoising': {
            'default_strategy': 1,
            'use_gsr': False,
            'use_cosine': True,
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Parameters
        ----------
        config_path : str, optional
            Path to YAML configuration file
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load from environment variables
        self._load_from_env()
        
        # Load from config file if provided
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
        elif os.path.exists('config.yaml'):
            self._load_from_file('config.yaml')
        elif os.path.exists('config.yml'):
            self._load_from_file('config.yml')
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mapping = {
            'DERIVATIVES_PATH': ('data', 'derivatives_path'),
            'OUTPUT_PATH': ('data', 'output_path'),
            'ATLAS_PATH': ('data', 'atlas_path'),
            'N_JOBS': ('processing', 'n_jobs'),
            'MEMORY': ('processing', 'memory'),
            'VERBOSE': ('processing', 'verbose'),
        }
        
        for env_var, (section, key) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert numeric values
                if key == 'n_jobs':
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                elif key == 'verbose':
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                self.config[section][key] = value
    
    def _load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self._deep_update(self.config, file_config)
        except (yaml.YAMLError, IOError) as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
    
    def _deep_update(self, original: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Recursively update nested dictionaries."""
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        try:
            return self.config[section][key]
        except KeyError:
            return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """Set configuration value."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    @property
    def derivatives_path(self) -> Optional[str]:
        """Get derivatives path from configuration."""
        return self.get('data', 'derivatives_path')
    
    @property
    def output_path(self) -> str:
        """Get output path from configuration."""
        return self.get('data', 'output_path', './output')
    
    @property
    def atlas_path(self) -> str:
        """Get atlas path from configuration."""
        return self.get('data', 'atlas_path', './atlas')
    
    @property
    def n_jobs(self) -> int:
        """Get number of jobs for parallel processing."""
        return self.get('processing', 'n_jobs', -1)
    
    @property
    def memory(self) -> str:
        """Get memory setting for caching."""
        return self.get('processing', 'memory', 'nilearn_cache')
    
    @property
    def verbose(self) -> int:
        """Get verbosity level."""
        return self.get('processing', 'verbose', 1)


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get or create global configuration instance.
    
    Parameters
    ----------
    config_path : str, optional
        Path to configuration file
        
    Returns
    -------
    Config
        Configuration instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reset_config() -> None:
    """Reset global configuration (mainly for testing)."""
    global _config
    _config = None
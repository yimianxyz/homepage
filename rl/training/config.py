"""
Training Configuration - Configuration for RL training

This module provides configuration classes and utilities for setting up
PPO training with various parameters and options.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os


@dataclass
class TrainingConfig:
    """
    Configuration for PPO training
    """
    
    # Environment parameters
    num_boids: int = 20
    canvas_width: float = 800
    canvas_height: float = 600
    max_episode_steps: int = 1000
    
    # Model parameters
    model_checkpoint: str = "best_model.pt"
    model_config: Optional[Dict[str, Any]] = None
    
    # PPO parameters
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training parameters
    n_envs: int = 4  # Number of parallel environments
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", or "cuda"
    
    # Logging and saving
    log_dir: str = "logs"
    save_dir: str = "models"
    save_freq: int = 10000
    log_interval: int = 1
    eval_freq: int = 10000
    eval_episodes: int = 10
    
    # Tensorboard logging
    tensorboard_log: Optional[str] = "tensorboard_logs"
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 50
    min_improvement: float = 0.01
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        if self.tensorboard_log:
            os.makedirs(self.tensorboard_log, exist_ok=True)
        
        # Validate parameters
        assert self.num_boids > 0, "num_boids must be positive"
        assert self.max_episode_steps > 0, "max_episode_steps must be positive"
        assert self.total_timesteps > 0, "total_timesteps must be positive"
        assert 0 < self.learning_rate < 1, "learning_rate must be in (0, 1)"
        assert self.n_steps > 0, "n_steps must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.n_epochs > 0, "n_epochs must be positive"
        assert 0 < self.gamma <= 1, "gamma must be in (0, 1]"
        assert 0 <= self.gae_lambda <= 1, "gae_lambda must be in [0, 1]"
        assert 0 < self.clip_range < 1, "clip_range must be in (0, 1)"
        assert self.n_envs > 0, "n_envs must be positive"
        
        # Set default model config if not provided
        if self.model_config is None:
            self.model_config = {
                'd_model': 64,
                'n_heads': 8,
                'n_layers': 4,
                'ffn_hidden': 256,
                'max_boids': self.num_boids,
                'dropout': 0.1
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'num_boids': self.num_boids,
            'canvas_width': self.canvas_width,
            'canvas_height': self.canvas_height,
            'max_episode_steps': self.max_episode_steps,
            'model_checkpoint': self.model_checkpoint,
            'model_config': self.model_config,
            'total_timesteps': self.total_timesteps,
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'clip_range_vf': self.clip_range_vf,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm,
            'n_envs': self.n_envs,
            'seed': self.seed,
            'device': self.device,
            'log_dir': self.log_dir,
            'save_dir': self.save_dir,
            'save_freq': self.save_freq,
            'log_interval': self.log_interval,
            'eval_freq': self.eval_freq,
            'eval_episodes': self.eval_episodes,
            'tensorboard_log': self.tensorboard_log,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'min_improvement': self.min_improvement
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save config to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """Load config from file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined configurations for different scenarios

def get_quick_test_config() -> TrainingConfig:
    """Configuration for quick testing"""
    return TrainingConfig(
        num_boids=10,
        canvas_width=400,
        canvas_height=300,
        max_episode_steps=200,
        total_timesteps=50000,
        n_steps=512,
        batch_size=32,
        n_envs=2,
        save_freq=5000,
        eval_freq=5000
    )

def get_small_scale_config() -> TrainingConfig:
    """Configuration for small-scale training"""
    return TrainingConfig(
        num_boids=15,
        canvas_width=600,
        canvas_height=400,
        max_episode_steps=500,
        total_timesteps=500000,
        n_steps=1024,
        batch_size=64,
        n_envs=4,
        save_freq=10000,
        eval_freq=10000
    )

def get_full_scale_config() -> TrainingConfig:
    """Configuration for full-scale training"""
    return TrainingConfig(
        num_boids=20,
        canvas_width=800,
        canvas_height=600,
        max_episode_steps=1000,
        total_timesteps=2000000,
        n_steps=2048,
        batch_size=64,
        n_envs=8,
        save_freq=20000,
        eval_freq=20000
    )

def get_large_scale_config() -> TrainingConfig:
    """Configuration for large-scale training"""
    return TrainingConfig(
        num_boids=30,
        canvas_width=1000,
        canvas_height=800,
        max_episode_steps=1500,
        total_timesteps=5000000,
        n_steps=2048,
        batch_size=128,
        n_envs=16,
        learning_rate=1e-4,  # Lower learning rate for stability
        save_freq=50000,
        eval_freq=50000
    )
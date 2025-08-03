"""
PPO Configuration Module

Clean and organized configuration for PPO training hyperparameters.
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class PPOConfig:
    """PPO training configuration with sensible defaults"""
    
    # Environment settings
    num_envs: int = 16                    # Number of parallel environments
    canvas_width: int = 800               # Simulation canvas width
    canvas_height: int = 600              # Simulation canvas height
    num_boids: int = 20                   # Number of boids per episode
    max_steps_per_episode: int = 1000     # Max steps before truncation
    max_boids_for_model: int = 50         # Max boids transformer can handle
    
    # PPO hyperparameters
    learning_rate: float = 3e-4           # Learning rate for Adam
    num_epochs: int = 4                   # PPO epochs per batch
    num_minibatches: int = 4              # Minibatches per PPO epoch
    discount_gamma: float = 0.99          # Discount factor
    gae_lambda: float = 0.95              # GAE lambda
    clip_epsilon: float = 0.2             # PPO clipping parameter
    entropy_coef: float = 0.01            # Entropy regularization
    value_loss_coef: float = 0.5          # Value loss coefficient
    max_grad_norm: float = 0.5            # Gradient clipping
    
    # Training settings
    total_frames: int = 1_000_000         # Total environment frames
    frames_per_batch: int = 2048          # Frames collected per training batch
    
    # Model settings
    checkpoint_path: str = "checkpoints/best_model.pt"  # SL checkpoint
    freeze_transformer: bool = False      # Whether to freeze transformer
    action_std: float = 0.2               # Initial action std dev
    value_hidden_dim: int = 256           # Hidden dim for value head
    
    # Evaluation settings
    eval_frequency: int = 10              # Evaluate every N training iterations
    eval_episodes: int = 20               # Episodes for evaluation
    
    # Logging and saving
    log_frequency: int = 10               # Log every N training iterations
    save_frequency: int = 50              # Save checkpoint every N iterations
    experiment_name: str = "ppo_finetune" # Experiment name for logs
    
    # Device settings
    device: Optional[torch.device] = None # Device (auto-detect if None)
    
    def __post_init__(self):
        """Post-initialization setup"""
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Validate settings
        assert self.frames_per_batch % self.num_envs == 0, \
            f"frames_per_batch ({self.frames_per_batch}) must be divisible by num_envs ({self.num_envs})"
        
        assert self.frames_per_batch % (self.num_minibatches * self.num_envs) == 0, \
            f"frames_per_batch must be divisible by (num_minibatches * num_envs)"
        
        # Calculate derived values
        self.batch_size = self.frames_per_batch // self.num_envs
        self.minibatch_size = self.frames_per_batch // self.num_minibatches
        self.num_iterations = self.total_frames // self.frames_per_batch
        
    def print_config(self):
        """Print configuration summary"""
        print("PPO Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Parallel environments: {self.num_envs}")
        print(f"  Total frames: {self.total_frames:,}")
        print(f"  Frames per batch: {self.frames_per_batch}")
        print(f"  Training iterations: {self.num_iterations}")
        print(f"  PPO epochs: {self.num_epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Freeze transformer: {self.freeze_transformer}")


def get_default_config() -> PPOConfig:
    """Get default PPO configuration"""
    return PPOConfig()


def get_fast_config() -> PPOConfig:
    """Get fast configuration for testing"""
    return PPOConfig(
        num_envs=4,
        total_frames=10_000,
        frames_per_batch=512,
        eval_frequency=5,
        eval_episodes=5,
        save_frequency=10,
    )


def get_large_config() -> PPOConfig:
    """Get configuration for large-scale training"""
    return PPOConfig(
        num_envs=32,
        total_frames=10_000_000,
        frames_per_batch=4096,
        learning_rate=1e-4,
        num_epochs=8,
        num_minibatches=8,
    )
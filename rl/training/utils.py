"""
Training Utilities - Helper functions for RL training

This module provides utilities for experience collection, data management,
logging, and other training-related functionality.
"""

import torch
import numpy as np
import sys
import os
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json
import time
from collections import deque, defaultdict
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class ExperienceBuffer:
    """
    Buffer for collecting and storing RL experiences
    
    This buffer collects experiences during rollouts and provides them
    in the format expected by the PPO algorithm.
    """
    
    def __init__(self, max_steps: int = 1000, debug: bool = True):
        """
        Initialize experience buffer
        
        Args:
            max_steps: Maximum number of steps to store
            debug: Enable debug logging
        """
        self.max_steps = max_steps
        self.debug = debug
        
        # Experience storage
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        # Episode tracking
        self.episode_starts = []
        self.current_episode_start = 0
        
        if self.debug:
            print(f"📦 ExperienceBuffer initialized (max_steps: {max_steps})")
    
    def reset(self):
        """Reset the buffer"""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.episode_starts.clear()
        self.current_episode_start = 0
        
        if self.debug:
            print("🔄 Experience buffer reset")
    
    def store(self, 
              observation: Dict[str, Any],
              action: torch.Tensor,
              reward: float,
              done: bool,
              log_prob: torch.Tensor,
              value: torch.Tensor):
        """Store one step of experience"""
        self.observations.append(observation)
        # Ensure consistent tensor shapes
        action = action.detach().cpu()
        if action.dim() == 0:  # Scalar
            action = action.unsqueeze(0)
        elif action.dim() > 1:  # Multi-dimensional
            action = action.squeeze()
        self.actions.append(action)
        
        log_prob = log_prob.detach().cpu()
        if log_prob.dim() > 0:  # Should be scalar
            log_prob = log_prob.squeeze()
        self.log_probs.append(log_prob)
        
        value = value.detach().cpu()
        if value.dim() > 0:  # Should be scalar
            value = value.squeeze()
        self.values.append(value)
        
        self.rewards.append(reward)
        self.dones.append(done)
        
        if done:
            # Mark episode end
            self.episode_starts.append((self.current_episode_start, len(self.observations)))
            self.current_episode_start = len(self.observations)
    
    def get_experiences(self) -> Dict[str, Any]:
        """
        Get experiences in format for PPO update
        
        Returns:
            Dictionary with batched experiences
        """
        if len(self.observations) == 0:
            return {}
        
        # Convert to episode-based format
        episodes = []
        
        # Handle any incomplete episode
        episode_ranges = list(self.episode_starts)
        if self.current_episode_start < len(self.observations):
            episode_ranges.append((self.current_episode_start, len(self.observations)))
        
        for start, end in episode_ranges:
            if end > start:  # Valid episode
                episode = {
                    'observations': self.observations[start:end],
                    'actions': torch.stack(self.actions[start:end]),
                    'rewards': torch.tensor(self.rewards[start:end], dtype=torch.float32),
                    'dones': torch.tensor(self.dones[start:end], dtype=torch.float32),
                    'log_probs': torch.stack(self.log_probs[start:end]),
                    'values': torch.stack(self.values[start:end])
                }
                episodes.append(episode)
        
        if not episodes:
            return {}
        
        # Pad episodes to same length
        max_len = max(ep['actions'].shape[0] for ep in episodes)
        
        padded_observations = []
        padded_actions = []
        padded_rewards = []
        padded_dones = []
        padded_log_probs = []
        padded_values = []
        
        for episode in episodes:
            ep_len = episode['actions'].shape[0]
            pad_len = max_len - ep_len
            
            # Pad observations (keep last observation)
            ep_obs = episode['observations'] + [episode['observations'][-1]] * pad_len
            padded_observations.append(ep_obs)
            
            # Pad tensors with proper dimensions
            if episode['actions'].dim() == 1:  # 1D action vector
                padded_actions.append(F.pad(episode['actions'], (0, pad_len)))
            else:  # 2D action vector [seq_len, action_dim]
                padded_actions.append(F.pad(episode['actions'], (0, 0, 0, pad_len)))
            
            padded_rewards.append(F.pad(episode['rewards'], (0, pad_len)))
            padded_dones.append(F.pad(episode['dones'], (0, pad_len), value=1.0))  # Pad with done=True
            padded_log_probs.append(F.pad(episode['log_probs'], (0, pad_len)))
            padded_values.append(F.pad(episode['values'], (0, pad_len)))
        
        return {
            'observations': padded_observations,
            'actions': torch.stack(padded_actions),
            'rewards': torch.stack(padded_rewards),
            'dones': torch.stack(padded_dones),
            'old_log_probs': torch.stack(padded_log_probs),
            'old_values': torch.stack(padded_values)
        }
    
    def __len__(self):
        return len(self.observations)
    


class TrainingLogger:
    """
    Logger for training metrics and progress
    
    This logger tracks training progress, saves metrics, and provides
    visualization capabilities.
    """
    
    def __init__(self, log_dir: str, save_interval: int = 100, debug: bool = True):
        """
        Initialize training logger
        
        Args:
            log_dir: Directory to save logs
            save_interval: Interval for saving logs to disk
            debug: Enable debug logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.debug = debug
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        self.step_count = 0
        self.episode_count = 0
        
        # Recent metrics for moving averages
        self.recent_rewards = deque(maxlen=100)
        self.recent_episode_lengths = deque(maxlen=100)
        
        if self.debug:
            print(f"📊 TrainingLogger initialized (log_dir: {log_dir})")
    
    def log_step(self, metrics: Dict[str, float]):
        """Log metrics for a training step"""
        for key, value in metrics.items():
            self.metrics[key].append(value)
        
        self.step_count += 1
        
        # Save periodically
        if self.step_count % self.save_interval == 0:
            self.save_metrics()
    
    def log_episode(self, episode_reward: float, episode_length: int, **kwargs):
        """Log metrics for completed episode"""
        self.episode_metrics['reward'].append(episode_reward)
        self.episode_metrics['length'].append(episode_length)
        
        # Log additional episode metrics
        for key, value in kwargs.items():
            self.episode_metrics[key].append(value)
        
        self.recent_rewards.append(episode_reward)
        self.recent_episode_lengths.append(episode_length)
        self.episode_count += 1
        
        if self.debug and self.episode_count % 10 == 0:
            avg_reward = np.mean(self.recent_rewards)
            avg_length = np.mean(self.recent_episode_lengths)
            print(f"📈 Episode {self.episode_count}: avg_reward={avg_reward:.2f}, avg_length={avg_length:.1f}")
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current training statistics"""
        stats = {
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }
        
        if self.recent_rewards:
            stats.update({
                'avg_reward_100': np.mean(self.recent_rewards),
                'avg_episode_length_100': np.mean(self.recent_episode_lengths),
                'min_reward_100': np.min(self.recent_rewards),
                'max_reward_100': np.max(self.recent_rewards)
            })
        
        # Add latest training metrics
        for key, values in self.metrics.items():
            if values:
                stats[f'latest_{key}'] = values[-1]
        
        return stats
    
    def save_metrics(self):
        """Save metrics to disk"""
        try:
            # Save step metrics
            step_metrics_path = self.log_dir / 'step_metrics.json'
            with open(step_metrics_path, 'w') as f:
                json.dump(dict(self.metrics), f, indent=2)
            
            # Save episode metrics
            episode_metrics_path = self.log_dir / 'episode_metrics.json'
            with open(episode_metrics_path, 'w') as f:
                json.dump(dict(self.episode_metrics), f, indent=2)
            
            # Save current stats
            stats_path = self.log_dir / 'current_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(self.get_current_stats(), f, indent=2)
            
            if self.debug and self.step_count % (self.save_interval * 10) == 0:
                print(f"💾 Saved metrics to {self.log_dir}")
        
        except Exception as e:
            print(f"❌ Error saving metrics: {e}")
    
    def plot_training_progress(self, save_path: Optional[str] = None, show: bool = True):
        """Plot training progress"""
        if not self.episode_metrics['reward']:
            print("No episode data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_metrics['reward'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Episode lengths
        axes[0, 1].plot(self.episode_metrics['length'])
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        # Moving average reward
        if len(self.episode_metrics['reward']) > 10:
            window = min(50, len(self.episode_metrics['reward']) // 10)
            moving_avg = np.convolve(self.episode_metrics['reward'], 
                                   np.ones(window)/window, mode='valid')
            axes[1, 0].plot(moving_avg)
            axes[1, 0].set_title(f'Moving Average Reward (window={window})')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Avg Reward')
        
        # Training loss (if available)
        if 'total_loss' in self.metrics:
            axes[1, 1].plot(self.metrics['total_loss'])
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()

class TrainingUtils:
    """
    Collection of training utility functions
    """
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
        """Count model parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get device information"""
        device_info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            device_info.update({
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(),
                'memory_allocated': torch.cuda.memory_allocated(),
                'memory_cached': torch.cuda.memory_reserved()
            })
        
        return device_info
    
    @staticmethod
    def set_seed(seed: int):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    @staticmethod
    def format_number(num: float) -> str:
        """Format large numbers in human-readable format"""
        if num < 1000:
            return f"{num:.0f}"
        elif num < 1000000:
            return f"{num/1000:.1f}K"
        elif num < 1000000000:
            return f"{num/1000000:.1f}M"
        else:
            return f"{num/1000000000:.1f}B"


# Import torch.nn.functional for padding
import torch.nn.functional as F

if __name__ == "__main__":
    # Test training utilities
    print("🧪 Testing Training Utilities...")
    
    # Test experience buffer
    print("\n🔍 Testing ExperienceBuffer...")
    buffer = ExperienceBuffer(max_steps=100, debug=True)
    
    # Add some dummy experiences
    for i in range(10):
        obs = {'dummy': i}
        action = torch.randn(2)
        reward = np.random.random()
        done = (i == 9)  # Last step is done
        log_prob = torch.randn(1)
        value = torch.randn(1)
        
        buffer.store(obs, action, reward, done, log_prob, value)
    
    experiences = buffer.get_experiences()
    print(f"   Experiences shape: {experiences['actions'].shape}")
    
    # Test logger
    print("\n🔍 Testing TrainingLogger...")
    logger = TrainingLogger('test_logs', debug=True)
    
    # Log some dummy metrics
    for i in range(5):
        logger.log_step({'loss': np.random.random(), 'reward': np.random.random()})
        logger.log_episode(np.random.random() * 100, np.random.randint(10, 100))
    
    stats = logger.get_current_stats()
    print(f"   Current stats: {stats}")
    
    # Test utilities
    print("\n🔍 Testing TrainingUtils...")
    print(f"   Device info: {TrainingUtils.get_device_info()}")
    print(f"   Time format: {TrainingUtils.format_time(3661)}")
    print(f"   Number format: {TrainingUtils.format_number(1234567)}")
    
    print("\n✅ Training Utilities tests passed!") 
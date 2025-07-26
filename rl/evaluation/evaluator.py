"""
Model Evaluator - Evaluation utilities for trained RL models

This module provides comprehensive evaluation capabilities for trained models
including performance metrics, behavior analysis, and visualization.
"""

import torch
import numpy as np
import sys
import os
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from ..models.ppo_model import create_ppo_model
from ..training.environment import BoidsEnvironment

class ModelEvaluator:
    """
    Comprehensive model evaluation
    
    This evaluator provides detailed analysis of trained models including
    performance metrics, behavior patterns, and comparison capabilities.
    """
    
    def __init__(self, 
                 model_path: str,
                 device: torch.device = None,
                 debug: bool = True):
        """
        Initialize model evaluator
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device for computation
            debug: Enable debug logging
        """
        self.model_path = model_path
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug = debug
        
        if self.debug:
            print(f"🎯 Initializing ModelEvaluator:")
            print(f"   Model path: {model_path}")
            print(f"   Device: {self.device}")
        
        # Load model
        self.model = create_ppo_model(model_path, self.device, debug=debug)
        if self.model is None:
            raise ValueError(f"Failed to load model from: {model_path}")
        
        self.model.eval()
        
        if self.debug:
            print(f"   ✅ Model loaded successfully")
    
    def evaluate_performance(self,
                           num_episodes: int = 100,
                           env_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate model performance over multiple episodes
        
        Args:
            num_episodes: Number of episodes to evaluate
            env_config: Environment configuration
            
        Returns:
            Performance statistics
        """
        if self.debug:
            print(f"📊 Evaluating performance over {num_episodes} episodes...")
        
        # Default environment config
        if env_config is None:
            env_config = {
                'num_boids_range': (10, 50),
                'canvas_size_range': ((800, 600), (1920, 1080)),
                'max_episode_steps': 500,
                'debug': False
            }
        
        env = BoidsEnvironment(**env_config)
        
        # Evaluation metrics
        episode_rewards = []
        episode_lengths = []
        boids_caught = []
        approaching_rewards = []
        catch_rewards = []
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            observation = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                with torch.no_grad():
                    # Use deterministic action (mean) for evaluation
                    outputs = self.model.forward([observation])
                    action = outputs['action_mean']
                
                observation, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    
                    stats = info['episode_stats']
                    boids_caught.append(stats['boids_caught'])
                    approaching_rewards.append(stats['approaching_reward'])
                    catch_rewards.append(stats['catch_reward'])
                    break
            
            if self.debug and (episode + 1) % 20 == 0:
                print(f"   Completed {episode + 1}/{num_episodes} episodes")
        
        env.close()
        evaluation_time = time.time() - start_time
        
        # Calculate statistics
        stats = {
            'num_episodes': num_episodes,
            'evaluation_time': evaluation_time,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'mean_boids_caught': np.mean(boids_caught),
            'std_boids_caught': np.std(boids_caught),
            'max_boids_caught': np.max(boids_caught),
            'mean_approaching_reward': np.mean(approaching_rewards),
            'mean_catch_reward': np.mean(catch_rewards),
            'success_rate': np.mean([x > 0 for x in boids_caught]),  # Episodes with at least 1 catch
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'boids_caught': boids_caught
        }
        
        if self.debug:
            print(f"   ✅ Evaluation complete:")
            print(f"      Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
            print(f"      Mean length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
            print(f"      Mean boids caught: {stats['mean_boids_caught']:.1f} ± {stats['std_boids_caught']:.1f}")
            print(f"      Success rate: {stats['success_rate']:.1%}")
            print(f"      Evaluation time: {evaluation_time:.1f}s")
        
        return stats
    
    def compare_with_baseline(self,
                            baseline_policy: str = 'closest_pursuit',
                            num_episodes: int = 50,
                            env_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compare model performance with baseline policy
        
        Args:
            baseline_policy: Baseline policy type
            num_episodes: Number of episodes for comparison
            env_config: Environment configuration
            
        Returns:
            Comparison statistics
        """
        if self.debug:
            print(f"⚖️  Comparing with {baseline_policy} baseline...")
        
        # Evaluate model
        model_stats = self.evaluate_performance(num_episodes, env_config)
        
        # Evaluate baseline
        if baseline_policy == 'closest_pursuit':
            baseline_stats = self._evaluate_closest_pursuit(num_episodes, env_config)
        else:
            raise ValueError(f"Unknown baseline policy: {baseline_policy}")
        
        # Calculate comparison metrics
        comparison = {
            'model_mean_reward': model_stats['mean_reward'],
            'baseline_mean_reward': baseline_stats['mean_reward'],
            'reward_improvement': model_stats['mean_reward'] - baseline_stats['mean_reward'],
            'reward_improvement_percent': ((model_stats['mean_reward'] - baseline_stats['mean_reward']) / 
                                         abs(baseline_stats['mean_reward']) * 100 if baseline_stats['mean_reward'] != 0 else 0),
            'model_mean_boids_caught': model_stats['mean_boids_caught'],
            'baseline_mean_boids_caught': baseline_stats['mean_boids_caught'],
            'catch_improvement': model_stats['mean_boids_caught'] - baseline_stats['mean_boids_caught'],
            'model_success_rate': model_stats['success_rate'],
            'baseline_success_rate': baseline_stats['success_rate'],
            'model_stats': model_stats,
            'baseline_stats': baseline_stats
        }
        
        if self.debug:
            print(f"   📈 Comparison results:")
            print(f"      Model reward: {comparison['model_mean_reward']:.2f}")
            print(f"      Baseline reward: {comparison['baseline_mean_reward']:.2f}")
            print(f"      Improvement: {comparison['reward_improvement']:+.2f} ({comparison['reward_improvement_percent']:+.1f}%)")
            print(f"      Model boids caught: {comparison['model_mean_boids_caught']:.1f}")
            print(f"      Baseline boids caught: {comparison['baseline_mean_boids_caught']:.1f}")
            print(f"      Catch improvement: {comparison['catch_improvement']:+.1f}")
        
        return comparison
    
    def _evaluate_closest_pursuit(self, num_episodes: int, env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate closest pursuit baseline policy"""
        from policy.human_prior.closest_pursuit_policy import create_closest_pursuit_policy
        
        env = BoidsEnvironment(**env_config)
        policy = create_closest_pursuit_policy()
        
        episode_rewards = []
        episode_lengths = []
        boids_caught = []
        
        for episode in range(num_episodes):
            observation = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = policy.get_action(observation)
                observation, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    boids_caught.append(info['episode_stats']['boids_caught'])
                    break
        
        env.close()
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_boids_caught': np.mean(boids_caught),
            'success_rate': np.mean([x > 0 for x in boids_caught])
        }
    
    def plot_evaluation_results(self, 
                               stats: Dict[str, Any],
                               save_path: Optional[str] = None,
                               show: bool = True):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        axes[0, 0].hist(stats['episode_rewards'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(stats['mean_reward'], color='red', linestyle='--', 
                          label=f'Mean: {stats["mean_reward"]:.2f}')
        axes[0, 0].set_title('Episode Rewards Distribution')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Episode lengths
        axes[0, 1].hist(stats['episode_lengths'], bins=20, alpha=0.7, edgecolor='black', color='green')
        axes[0, 1].axvline(stats['mean_length'], color='red', linestyle='--',
                          label=f'Mean: {stats["mean_length"]:.1f}')
        axes[0, 1].set_title('Episode Lengths Distribution')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Boids caught
        unique_catches = np.unique(stats['boids_caught'])
        catches_counts = [np.sum(np.array(stats['boids_caught']) == x) for x in unique_catches]
        axes[1, 0].bar(unique_catches, catches_counts, alpha=0.7, edgecolor='black', color='orange')
        axes[1, 0].set_title('Boids Caught Distribution')
        axes[1, 0].set_xlabel('Boids Caught')
        axes[1, 0].set_ylabel('Frequency')
        
        # Performance over time
        window_size = max(1, len(stats['episode_rewards']) // 20)
        moving_avg = np.convolve(stats['episode_rewards'], np.ones(window_size)/window_size, mode='valid')
        axes[1, 1].plot(stats['episode_rewards'], alpha=0.3, color='blue', label='Episode Reward')
        axes[1, 1].plot(range(window_size-1, len(stats['episode_rewards'])), moving_avg, 
                       color='red', label=f'Moving Avg (window={window_size})')
        axes[1, 1].set_title('Performance Over Time')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved evaluation plot: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        from ..training.utils import TrainingUtils
        
        param_count = TrainingUtils.count_parameters(self.model)
        
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'architecture': {
                'd_model': self.model.d_model,
                'n_heads': self.model.n_heads,
                'n_layers': self.model.n_layers,
                'ffn_hidden': self.model.ffn_hidden,
                'max_boids': self.model.max_boids
            },
            'parameters': param_count
        }


if __name__ == "__main__":
    # Test evaluator
    print("🧪 Testing ModelEvaluator...")
    
    # Check if checkpoint exists
    checkpoint_path = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please run training first or provide a valid checkpoint path")
        exit(1)
    
    # Create evaluator
    evaluator = ModelEvaluator(checkpoint_path, debug=True)
    
    # Test evaluation
    print("\n🔍 Testing performance evaluation...")
    stats = evaluator.evaluate_performance(
        num_episodes=10,  # Small for testing
        env_config={
            'num_boids_range': (5, 10),
            'canvas_size_range': ((400, 300), (600, 400)),
            'max_episode_steps': 100,
            'debug': False
        }
    )
    print(f"✅ Evaluation successful!")
    
    # Test comparison
    print("\n🔍 Testing baseline comparison...")
    comparison = evaluator.compare_with_baseline(
        num_episodes=5,  # Small for testing
        env_config={
            'num_boids_range': (5, 10),
            'canvas_size_range': ((400, 300), (600, 400)),
            'max_episode_steps': 100
        }
    )
    print(f"✅ Comparison successful!")
    
    # Test model info
    print("\n🔍 Testing model info...")
    info = evaluator.get_model_info()
    print(f"Model info: {info}")
    
    print("\n✅ ModelEvaluator tests passed!") 
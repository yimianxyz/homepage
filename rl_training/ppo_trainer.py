"""
PPO Trainer - Main training orchestrator for PPO reinforcement learning

This module implements the complete PPO training pipeline:
1. Rollout collection using existing simulation infrastructure
2. PPO loss computation (policy + value + entropy)
3. Model updates with clipped objectives
4. Integration with existing evaluation system

Design principles:
- Reuse all existing infrastructure (StateManager, RewardProcessor, PolicyEvaluator)
- Clean, readable PPO implementation following best practices
- Comprehensive logging and checkpointing
- Easy integration with SL baseline models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .ppo_transformer_model import create_ppo_policy_from_sl, PPOTransformerPolicy
from .ppo_experience_buffer import PPORolloutCollector, PPOExperienceBuffer
from simulation.state_manager import StateManager
from simulation.random_state_generator import generate_random_state
from rewards.reward_processor import RewardProcessor
from evaluation.policy_evaluator import PolicyEvaluator


class PPOTrainer:
    """
    PPO Trainer for transformer policy fine-tuning
    
    Implements PPO algorithm with:
    - Clipped policy objective
    - Value function learning
    - Entropy bonus for exploration
    - Integration with existing simulation infrastructure
    """
    
    def __init__(self,
                 sl_checkpoint_path: str = "checkpoints/best_model.pt",
                 learning_rate: float = 3e-4,
                 clip_epsilon: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 4,
                 mini_batch_size: int = 64,
                 rollout_steps: int = 2048,
                 max_episode_steps: int = 1000,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 device: str = 'auto'):
        """
        Initialize PPO trainer
        
        Args:
            sl_checkpoint_path: Path to supervised learning checkpoint
            learning_rate: Learning rate for optimizer
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of optimization epochs per rollout
            mini_batch_size: Mini-batch size for PPO updates
            rollout_steps: Steps per rollout collection
            max_episode_steps: Maximum steps per episode
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.rollout_steps = rollout_steps
        self.max_episode_steps = max_episode_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        print(f"🚀 Initializing PPO Trainer")
        print(f"  Device: {self.device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Clip epsilon: {clip_epsilon}")
        print(f"  Rollout steps: {rollout_steps}")
        print(f"  PPO epochs: {ppo_epochs}")
        
        # Load policy from SL checkpoint
        self.policy = create_ppo_policy_from_sl(sl_checkpoint_path).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.policy.model.parameters(), lr=learning_rate)
        
        # Initialize simulation components
        self.state_manager = StateManager()
        self.reward_processor = RewardProcessor()
        self.rollout_collector = PPORolloutCollector(
            self.state_manager,
            self.reward_processor,
            self.policy,
            max_episode_steps
        )
        
        # Initialize evaluator for periodic evaluation
        self.evaluator = PolicyEvaluator()
        
        # Training tracking
        self.iteration = 0
        self.total_timesteps = 0
        self.training_stats = []
        
        print(f"✅ PPO Trainer initialized successfully")
        print(f"  Policy loaded from: {sl_checkpoint_path}")
        print(f"  Ready for training")
    
    def compute_ppo_loss(self, batch_data: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO loss (policy + value + entropy)
        
        Args:
            batch_data: Batch of experiences
            
        Returns:
            total_loss: Combined PPO loss
            loss_info: Dictionary with loss components
        """
        # Extract batch data
        structured_inputs = batch_data['structured_inputs']
        actions = batch_data['actions'].to(self.device)
        old_log_probs = batch_data['log_probs'].to(self.device)
        advantages = batch_data['advantages'].to(self.device)
        returns = batch_data['returns'].to(self.device)
        old_values = batch_data['values'].to(self.device)
        
        # Evaluate actions with current policy
        new_log_probs, new_values, entropy = self.policy.evaluate_actions(structured_inputs, actions)
        
        # Policy loss (PPO clipped objective)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (clipped)
        value_pred_clipped = old_values + torch.clamp(
            new_values - old_values, 
            -self.clip_epsilon, 
            self.clip_epsilon
        )
        value_losses = (new_values - returns).pow(2)
        value_losses_clipped = (value_pred_clipped - returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
        
        # Entropy loss (for exploration)
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (policy_loss + 
                     self.value_loss_coef * value_loss + 
                     self.entropy_coef * entropy_loss)
        
        # Loss info for logging
        loss_info = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'entropy': entropy.mean().item(),
            'policy_ratio_mean': ratio.mean().item(),
            'policy_ratio_std': ratio.std().item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item(),
            'return_mean': returns.mean().item(),
            'return_std': returns.std().item(),
        }
        
        return total_loss, loss_info
    
    def update_policy(self, buffer: PPOExperienceBuffer) -> Dict[str, float]:
        """
        Update policy using PPO algorithm
        
        Args:
            buffer: Experience buffer with rollout data
            
        Returns:
            Dictionary with training statistics
        """
        # Get batch data
        batch_data = buffer.get_batch_data()
        batch_size = len(batch_data['actions'])
        
        # Training statistics
        all_losses = []
        
        print(f"  Updating policy with {batch_size} experiences...")
        
        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(batch_size)
            
            epoch_losses = []
            
            for start_idx in range(0, batch_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, batch_size)
                mb_indices = indices[start_idx:end_idx]
                
                # Create mini-batch
                mb_data = {
                    'structured_inputs': [batch_data['structured_inputs'][i] for i in mb_indices],
                    'actions': batch_data['actions'][mb_indices],
                    'log_probs': batch_data['log_probs'][mb_indices],
                    'advantages': batch_data['advantages'][mb_indices],
                    'returns': batch_data['returns'][mb_indices],
                    'values': batch_data['values'][mb_indices],
                }
                
                # Compute loss
                loss, loss_info = self.compute_ppo_loss(mb_data)
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                epoch_losses.append(loss_info)
            
            # Average losses for this epoch
            avg_epoch_loss = {
                key: np.mean([loss[key] for loss in epoch_losses])
                for key in epoch_losses[0].keys()
            }
            all_losses.append(avg_epoch_loss)
            
            print(f"    Epoch {epoch+1}/{self.ppo_epochs}: loss={avg_epoch_loss['total_loss']:.4f}")
        
        # Average across all epochs
        final_stats = {
            key: np.mean([loss[key] for loss in all_losses])
            for key in all_losses[0].keys()
        }
        
        return final_stats
    
    def train_iteration(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run one PPO training iteration (rollout + update)
        
        Args:
            initial_state: Initial simulation state
            
        Returns:
            Training statistics for this iteration
        """
        self.iteration += 1
        iteration_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"PPO Training Iteration {self.iteration}")
        print(f"{'='*60}")
        
        # Collect rollout
        print("📊 Collecting rollout...")
        buffer = self.rollout_collector.collect_rollout(initial_state, self.rollout_steps)
        rollout_stats = buffer.get_statistics()
        
        # Update timesteps
        self.total_timesteps += len(buffer)
        
        # Update policy
        print("🔄 Updating policy...")
        loss_stats = self.update_policy(buffer)
        
        # Compile iteration statistics
        iteration_time = time.time() - iteration_start_time
        
        iteration_stats = {
            'iteration': self.iteration,
            'total_timesteps': self.total_timesteps,
            'iteration_time': iteration_time,
            'rollout': rollout_stats,
            'training': loss_stats,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'clip_epsilon': self.clip_epsilon,
                'rollout_steps': self.rollout_steps,
            }
        }
        
        # Store statistics
        self.training_stats.append(iteration_stats)
        
        # Print summary
        print(f"\n📈 Iteration {self.iteration} Summary:")
        print(f"  Time: {iteration_time:.1f}s")
        print(f"  Rollout reward: {rollout_stats.get('mean_reward', 0):.3f}")
        print(f"  Policy loss: {loss_stats['policy_loss']:.4f}")
        print(f"  Value loss: {loss_stats['value_loss']:.4f}")
        print(f"  Total timesteps: {self.total_timesteps}")
        
        return iteration_stats
    
    def evaluate_policy(self, num_episodes: int = 5) -> Dict[str, float]:
        """
        Evaluate current policy using existing evaluation system
        
        Args:
            num_episodes: Number of episodes for evaluation
            
        Returns:
            Evaluation statistics
        """
        print(f"🎯 Evaluating policy...")
        
        # Set policy to evaluation mode
        self.policy.eval()
        
        # Use existing policy evaluator
        result = self.evaluator.evaluate_policy(self.policy, f"PPO_Iteration_{self.iteration}")
        
        # Extract key metrics
        eval_stats = {
            'overall_catch_rate': result.overall_catch_rate,
            'overall_std_catch_rate': result.overall_std_catch_rate,
            'successful_episodes': result.successful_episodes,
            'total_episodes': result.total_episodes,
            'evaluation_time': result.evaluation_time_seconds
        }
        
        print(f"  📊 Catch rate: {eval_stats['overall_catch_rate']:.3f} ± {eval_stats['overall_std_catch_rate']:.3f}")
        print(f"  ⏱️  Evaluation time: {eval_stats['evaluation_time']:.1f}s")
        
        # Set back to training mode
        self.policy.train()
        
        return eval_stats
    
    def save_checkpoint(self, save_path: str, is_best: bool = False):
        """
        Save training checkpoint
        
        Args:
            save_path: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'iteration': self.iteration,
            'total_timesteps': self.total_timesteps,
            'model_state_dict': self.policy.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'clip_epsilon': self.clip_epsilon,
                'value_loss_coef': self.value_loss_coef,
                'entropy_coef': self.entropy_coef,
                'rollout_steps': self.rollout_steps,
                'ppo_epochs': self.ppo_epochs,
                'mini_batch_size': self.mini_batch_size,
            },
            'architecture': {
                'd_model': self.policy.model.d_model,
                'n_heads': self.policy.model.n_heads,
                'n_layers': self.policy.model.n_layers,
                'ffn_hidden': self.policy.model.ffn_hidden,
                'max_boids': self.policy.model.max_boids,
            },
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, save_path)
        print(f"✅ Checkpoint saved: {save_path}")
        
        if is_best:
            best_path = os.path.join(os.path.dirname(save_path), "best_ppo_model.pt")
            torch.save(checkpoint, best_path)
            print(f"✅ Best model saved: {best_path}")
    
    def train(self, 
             num_iterations: int,
             eval_interval: int = 10,
             save_interval: int = 5,
             initial_boids: int = 20,
             canvas_width: float = 800,
             canvas_height: float = 600) -> List[Dict[str, Any]]:
        """
        Main training loop
        
        Args:
            num_iterations: Number of training iterations
            eval_interval: Evaluate every N iterations
            save_interval: Save checkpoint every N iterations
            initial_boids: Number of boids in simulation
            canvas_width: Canvas width for simulation
            canvas_height: Canvas height for simulation
            
        Returns:
            List of training statistics for each iteration
        """
        print(f"\n{'='*80}")
        print(f"🚀 STARTING PPO TRAINING")
        print(f"{'='*80}")
        print(f"  Iterations: {num_iterations}")
        print(f"  Rollout steps per iteration: {self.rollout_steps}")
        print(f"  Total timesteps: {num_iterations * self.rollout_steps:,}")
        print(f"  Evaluation interval: {eval_interval}")
        print(f"  Save interval: {save_interval}")
        print(f"{'='*80}")
        
        best_reward = float('-inf')
        training_start_time = time.time()
        
        for iteration in range(num_iterations):
            # Generate fresh initial state for variety
            initial_state = generate_random_state(
                num_boids=initial_boids,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                seed=None  # Random seed for variety
            )
            
            # Run training iteration
            iteration_stats = self.train_iteration(initial_state)
            
            # Periodic evaluation
            if (iteration + 1) % eval_interval == 0:
                eval_stats = self.evaluate_policy()
                iteration_stats['evaluation'] = eval_stats
                
                # Check if best model
                current_reward = eval_stats['overall_catch_rate']
                is_best = current_reward > best_reward
                if is_best:
                    best_reward = current_reward
                    print(f"🎯 New best catch rate: {current_reward:.3f}")
            else:
                is_best = False
            
            # Frequent checkpointing for safety
            checkpoint_path = f"checkpoints/ppo_iteration_{iteration + 1}.pt"
            
            # Always save every iteration (for timeout recovery)
            self.save_checkpoint(checkpoint_path, is_best)
            
            # Also save at regular intervals with special naming
            if (iteration + 1) % save_interval == 0:
                milestone_path = f"checkpoints/ppo_milestone_{iteration + 1}.pt"
                self.save_checkpoint(milestone_path, is_best)
                print(f"📁 Milestone checkpoint: {milestone_path}")
        
        total_training_time = time.time() - training_start_time
        
        print(f"\n{'='*80}")
        print(f"🎉 PPO TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"  Total iterations: {num_iterations}")
        print(f"  Total timesteps: {self.total_timesteps:,}")
        print(f"  Total time: {total_training_time:.1f}s")
        print(f"  Best catch rate: {best_reward:.3f}")
        print(f"{'='*80}")
        
        return self.training_stats


if __name__ == "__main__":
    # Test PPO trainer setup
    try:
        trainer = PPOTrainer(
            sl_checkpoint_path="checkpoints/best_model.pt",
            rollout_steps=100,  # Small for testing
            ppo_epochs=2,
            mini_batch_size=32
        )
        
        print("✅ PPO Trainer created successfully!")
        print("Ready for training with:")
        print(f"  - Supervised learning baseline loaded")
        print(f"  - Simulation infrastructure integrated")  
        print(f"  - Evaluation system ready")
        print(f"  - Device: {trainer.device}")
        
        # Test single iteration (optional)
        # initial_state = generate_random_state(5, 400, 300, seed=42)
        # stats = trainer.train_iteration(initial_state)
        # print(f"Test iteration completed: {stats['rollout']['rollout_length']} steps")
        
    except Exception as e:
        print(f"❌ PPO Trainer setup failed: {e}")
        print("Make sure you have a trained SL model at checkpoints/best_model.pt")
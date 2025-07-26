"""
PPO Algorithm - Proximal Policy Optimization for continuous control

This implements the PPO algorithm for training the actor-critic model on the
boids predator-prey environment. It includes all the standard PPO components:

- Generalized Advantage Estimation (GAE)
- Clipped policy objective
- Value function loss
- Entropy regularization
- Multiple epochs per update
- Gradient clipping

The implementation is optimized for the specific characteristics of the
boids environment while following PPO best practices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class PPOAlgorithm:
    """
    PPO algorithm implementation for continuous control
    
    This class handles the PPO training algorithm including experience collection,
    advantage estimation, and policy updates.
    """
    
    def __init__(self,
                 model: nn.Module,
                 lr: float = 3e-4,
                 clip_ratio: float = 0.2,
                 value_loss_coeff: float = 0.5,
                 entropy_coeff: float = 0.01,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 4,
                 mini_batch_size: int = 64,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 normalize_advantages: bool = True,
                 device: torch.device = None,
                 debug: bool = True):
        """
        Initialize PPO algorithm
        
        Args:
            model: The actor-critic model
            lr: Learning rate
            clip_ratio: PPO clipping ratio
            value_loss_coeff: Coefficient for value function loss
            entropy_coeff: Coefficient for entropy regularization
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of optimization epochs per update
            mini_batch_size: Mini-batch size for optimization
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            normalize_advantages: Whether to normalize advantages
            device: Device for computation
            debug: Enable debug logging
        """
        self.model = model
        self.lr = lr
        self.clip_ratio = clip_ratio
        self.value_loss_coeff = value_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug = debug
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training statistics
        self.update_count = 0
        self.total_samples = 0
        
        if self.debug:
            print(f"🎯 Initializing PPO Algorithm:")
            print(f"   Learning rate: {lr}")
            print(f"   Clip ratio: {clip_ratio}")
            print(f"   Value loss coeff: {value_loss_coeff}")
            print(f"   Entropy coeff: {entropy_coeff}")
            print(f"   PPO epochs: {ppo_epochs}")
            print(f"   Mini-batch size: {mini_batch_size}")
            print(f"   Gamma: {gamma}")
            print(f"   GAE lambda: {gae_lambda}")
            print(f"   Device: {device}")
    
    def compute_advantages(self, 
                          rewards: torch.Tensor,
                          values: torch.Tensor,
                          dones: torch.Tensor,
                          next_value: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Rewards tensor [batch_size, seq_len]
            values: Value estimates [batch_size, seq_len]
            dones: Done flags [batch_size, seq_len]
            next_value: Value of next state [batch_size] (optional)
            
        Returns:
            advantages: Advantage estimates [batch_size, seq_len]
            returns: Value targets [batch_size, seq_len]
        """
        batch_size, seq_len = rewards.shape
        
        # Ensure all tensors are on the correct device
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        
        if next_value is not None:
            next_value = next_value.to(self.device)
        else:
            next_value = torch.zeros(batch_size, device=self.device)
        
        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Work backwards through time
        last_advantage = 0
        
        for t in reversed(range(seq_len)):
            # Determine next value
            if t == seq_len - 1:
                next_value_t = next_value * (1 - dones[:, t])
            else:
                next_value_t = values[:, t + 1] * (1 - dones[:, t])
            
            # Compute temporal difference error
            delta = rewards[:, t] + self.gamma * next_value_t - values[:, t]
            
            # Compute advantage with GAE
            advantages[:, t] = delta + self.gamma * self.gae_lambda * (1 - dones[:, t]) * last_advantage
            last_advantage = advantages[:, t]
            
            # Compute returns
            returns[:, t] = advantages[:, t] + values[:, t]
        
        # Normalize advantages if requested
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        if self.debug:  # Debug every update in quick test mode
            print(f"   🧮 GAE computation:")
            print(f"     Rewards - mean: {rewards.mean():.4f}, std: {rewards.std():.4f}, min: {rewards.min():.4f}, max: {rewards.max():.4f}")
            print(f"     Values - mean: {values.mean():.4f}, std: {values.std():.4f}, min: {values.min():.4f}, max: {values.max():.4f}")
            print(f"     Advantages - mean: {advantages.mean():.4f}, std: {advantages.std():.4f}")
            print(f"     Returns - mean: {returns.mean():.4f}, std: {returns.std():.4f}")
            
            # Check for any non-zero rewards
            nonzero_rewards = rewards[rewards != 0]
            if len(nonzero_rewards) > 0:
                print(f"     🎯 Non-zero rewards found: {len(nonzero_rewards)} out of {len(rewards.flatten())}")
                print(f"       Non-zero reward range: {nonzero_rewards.min():.4f} to {nonzero_rewards.max():.4f}")
        
        return advantages, returns
    
    def update(self, 
               experiences: Dict[str, Any]) -> Dict[str, float]:
        """
        Update the model using collected experiences
        
        Args:
            experiences: Dictionary containing:
                - observations: List of structured inputs
                - actions: Action tensor [batch_size, seq_len, 2]
                - rewards: Reward tensor [batch_size, seq_len]
                - dones: Done tensor [batch_size, seq_len]
                - old_log_probs: Old log probabilities [batch_size, seq_len]
                - old_values: Old value estimates [batch_size, seq_len]
                
        Returns:
            Dictionary with training statistics
        """
        if self.debug:
            print(f"\n🔄 PPO Update {self.update_count + 1}")
        
        start_time = time.time()
        
        # Extract data
        observations = experiences['observations']
        actions = experiences['actions'].to(self.device)
        rewards = experiences['rewards'].to(self.device)
        dones = experiences['dones'].to(self.device)
        old_log_probs = experiences['old_log_probs'].to(self.device)
        old_values = experiences['old_values'].to(self.device)
        
        batch_size, seq_len = actions.shape[:2]
        self.total_samples += batch_size * seq_len
        
        if self.debug:
            print(f"   Batch size: {batch_size}, Sequence length: {seq_len}")
            print(f"   Total samples processed: {self.total_samples}")
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, old_values, dones)
        
        # Flatten everything for mini-batch processing
        observations_flat = []
        for b in range(batch_size):
            for t in range(seq_len):
                observations_flat.append(observations[b][t])
        
        actions_flat = actions.view(-1, 2)
        advantages_flat = advantages.view(-1)
        returns_flat = returns.view(-1)
        old_log_probs_flat = old_log_probs.view(-1)
        old_values_flat = old_values.view(-1)
        
        # Training statistics
        stats = defaultdict(list)
        
        # Multiple epochs of optimization
        for epoch in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(len(observations_flat))
            
            for start in range(0, len(indices), self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]
                
                # Mini-batch data
                mb_observations = [observations_flat[i] for i in mb_indices]
                mb_actions = actions_flat[mb_indices]
                mb_advantages = advantages_flat[mb_indices]
                mb_returns = returns_flat[mb_indices]
                mb_old_log_probs = old_log_probs_flat[mb_indices]
                mb_old_values = old_values_flat[mb_indices]
                
                # Forward pass
                new_log_probs, entropy, values = self.model.get_action_logprob_and_value(
                    mb_observations, mb_actions
                )
                
                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if hasattr(self, 'clip_value_loss') and self.clip_value_loss:
                    # Clipped value loss (optional)
                    values_clipped = mb_old_values + torch.clamp(
                        values.squeeze(-1) - mb_old_values,
                        -self.clip_ratio, self.clip_ratio
                    )
                    value_loss1 = F.mse_loss(values.squeeze(-1), mb_returns)
                    value_loss2 = F.mse_loss(values_clipped, mb_returns)
                    value_loss = torch.max(value_loss1, value_loss2)
                else:
                    # Standard value loss
                    value_loss = F.mse_loss(values.squeeze(-1), mb_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (policy_loss + 
                             self.value_loss_coeff * value_loss + 
                             self.entropy_coeff * entropy_loss)
                
                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Collect statistics
                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(-entropy_loss.item())
                stats['total_loss'].append(total_loss.item())
                stats['ratio_mean'].append(ratio.mean().item())
                stats['ratio_std'].append(ratio.std().item())
                stats['advantages_mean'].append(mb_advantages.mean().item())
                stats['advantages_std'].append(mb_advantages.std().item())
        
        # Calculate final statistics
        final_stats = {
            'policy_loss': np.mean(stats['policy_loss']),
            'value_loss': np.mean(stats['value_loss']),
            'entropy': np.mean(stats['entropy']),
            'total_loss': np.mean(stats['total_loss']),
            'ratio_mean': np.mean(stats['ratio_mean']),
            'ratio_std': np.mean(stats['ratio_std']),
            'advantages_mean': np.mean(stats['advantages_mean']),
            'advantages_std': np.mean(stats['advantages_std']),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'update_time': time.time() - start_time,
            'samples_processed': batch_size * seq_len
        }
        
        self.update_count += 1
        
        if self.debug:
            print(f"   Update statistics:")
            print(f"     Policy loss: {final_stats['policy_loss']:.4f}")
            print(f"     Value loss: {final_stats['value_loss']:.4f}")
            print(f"     Entropy: {final_stats['entropy']:.4f}")
            print(f"     Ratio mean: {final_stats['ratio_mean']:.4f} ± {final_stats['ratio_std']:.4f}")
            print(f"     Update time: {final_stats['update_time']:.2f}s")
        
        return final_stats
    
    def save_checkpoint(self, path: str, **kwargs) -> bool:
        """Save algorithm checkpoint"""
        try:
            checkpoint = {
                'optimizer_state_dict': self.optimizer.state_dict(),
                'update_count': self.update_count,
                'total_samples': self.total_samples,
                'hyperparameters': {
                    'lr': self.lr,
                    'clip_ratio': self.clip_ratio,
                    'value_loss_coeff': self.value_loss_coeff,
                    'entropy_coeff': self.entropy_coeff,
                    'max_grad_norm': self.max_grad_norm,
                    'ppo_epochs': self.ppo_epochs,
                    'mini_batch_size': self.mini_batch_size,
                    'gamma': self.gamma,
                    'gae_lambda': self.gae_lambda,
                    'normalize_advantages': self.normalize_advantages
                },
                **kwargs
            }
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(checkpoint, path)
            
            if self.debug:
                print(f"✅ Saved PPO checkpoint: {path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving PPO checkpoint: {e}")
            return False
    
    def load_checkpoint(self, path: str) -> bool:
        """Load algorithm checkpoint"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.update_count = checkpoint.get('update_count', 0)
            self.total_samples = checkpoint.get('total_samples', 0)
            
            # Load hyperparameters (optional)
            if 'hyperparameters' in checkpoint:
                hyperparams = checkpoint['hyperparameters']
                if self.debug:
                    print(f"   Loaded hyperparameters: {hyperparams}")
            
            if self.debug:
                print(f"✅ Loaded PPO checkpoint from: {path}")
                print(f"   Update count: {self.update_count}")
                print(f"   Total samples: {self.total_samples}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading PPO checkpoint: {e}")
            return False
    
    def adjust_learning_rate(self, factor: float):
        """Adjust learning rate by a factor"""
        old_lr = self.optimizer.param_groups[0]['lr']
        new_lr = old_lr * factor
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        if self.debug:
            print(f"📈 Learning rate adjusted: {old_lr:.6f} → {new_lr:.6f} (factor: {factor})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get algorithm statistics"""
        return {
            'update_count': self.update_count,
            'total_samples': self.total_samples,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'hyperparameters': {
                'clip_ratio': self.clip_ratio,
                'value_loss_coeff': self.value_loss_coeff,
                'entropy_coeff': self.entropy_coeff,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda
            }
        }


if __name__ == "__main__":
    # Test PPO algorithm
    print("🧪 Testing PPO Algorithm...")
    
    # Create dummy model for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)
            
        def get_action_logprob_and_value(self, observations, actions):
            # Dummy implementation
            batch_size = actions.shape[0]
            log_probs = torch.randn(batch_size)
            entropy = torch.ones(batch_size)
            values = torch.randn(batch_size, 1)
            return log_probs, entropy, values
    
    model = DummyModel()
    ppo = PPOAlgorithm(model, debug=True)
    
    # Test advantage computation
    print("\n🔍 Testing advantage computation...")
    rewards = torch.randn(2, 10)  # 2 episodes, 10 steps each
    values = torch.randn(2, 10)
    dones = torch.zeros(2, 10)
    dones[:, -1] = 1  # Last step is done
    
    advantages, returns = ppo.compute_advantages(rewards, values, dones)
    print(f"   Advantages shape: {advantages.shape}")
    print(f"   Returns shape: {returns.shape}")
    
    # Test dummy update
    print("\n🔍 Testing update...")
    dummy_experiences = {
        'observations': [[{}] * 10] * 2,  # Dummy observations
        'actions': torch.randn(2, 10, 2),
        'rewards': rewards,
        'dones': dones,
        'old_log_probs': torch.randn(2, 10),
        'old_values': values
    }
    
    stats = ppo.update(dummy_experiences)
    print(f"   Update successful! Loss: {stats['total_loss']:.4f}")
    
    print("\n✅ PPO Algorithm tests passed!") 
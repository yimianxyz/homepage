#!/usr/bin/env python3
"""
Fixed Production Training - Properly implements value pre-training

The previous version had a dummy value pre-training that caused poor performance.
This version implements actual value pre-training to stabilize learning.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ppo_production_trainer import ProductionPPOTrainer, TrainingConfig, PPOTrainer
from ppo_production_monitor import PPOProductionMonitor
from simulation.random_state_generator import generate_random_state


class PPOTrainerWithRealValuePretraining(PPOTrainer):
    """PPO trainer with actual value pre-training implementation"""
    
    def pretrain_value_function(self, iterations: int, learning_rate: float) -> List[float]:
        """Actually pre-train value function to match SL policy behavior"""
        print(f"\n🎯 REAL VALUE FUNCTION PRE-TRAINING")
        print(f"  Iterations: {iterations}")
        print(f"  Learning rate: {learning_rate}")
        
        # Get model parameters
        value_params = []
        policy_params = []
        
        for name, param in self.policy.model.named_parameters():
            if 'value_head' in name:
                value_params.append(param)
            else:
                policy_params.append(param)
        
        # Freeze policy parameters
        for param in policy_params:
            param.requires_grad = False
        
        # Create value-only optimizer
        value_optimizer = optim.Adam(value_params, lr=learning_rate)
        
        print(f"  Frozen policy params: {len(policy_params)}")
        print(f"  Trainable value params: {len(value_params)}")
        
        # Training loop
        value_losses = []
        
        for iteration in range(1, iterations + 1):
            # Collect experience with frozen policy
            initial_state = generate_random_state(12, 400, 300)
            
            # Simplified rollout collection
            experience_buffer = self.rollout_collector.collect_rollout(
                self.policy, initial_state, 256  # Small rollout for pre-training
            )
            
            # Compute returns
            next_value = 0.0
            advantages, returns = experience_buffer.compute_advantages_and_returns(next_value)
            
            # Get states
            states = experience_buffer.get_stacked_observations()
            
            # Train value function
            total_loss = 0.0
            num_batches = 0
            
            # Multiple epochs over data
            for epoch in range(3):  # 3 epochs per iteration
                indices = torch.randperm(len(states))
                
                # Mini-batch training
                batch_size = 64
                for start_idx in range(0, len(states), batch_size):
                    end_idx = min(start_idx + batch_size, len(states))
                    batch_indices = indices[start_idx:end_idx]
                    
                    batch_states = states[batch_indices].to(self.device)
                    batch_returns = returns[batch_indices].to(self.device)
                    
                    # Forward pass
                    _, values = self.policy.model(batch_states)
                    values = values.squeeze(-1)
                    
                    # MSE loss
                    value_loss = nn.MSELoss()(values, batch_returns)
                    
                    # Backward pass
                    value_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(value_params, 0.5)
                    value_optimizer.step()
                    
                    total_loss += value_loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            value_losses.append(avg_loss)
            
            # Progress update
            if iteration % 5 == 0 or iteration <= 3:
                print(f"    Iteration {iteration}: Loss = {avg_loss:.4f}")
            
            # Early stopping if converged
            if avg_loss < 0.5:
                print(f"  ✅ Converged at iteration {iteration}")
                break
        
        # Unfreeze policy parameters
        for param in policy_params:
            param.requires_grad = True
        
        # Reset main optimizer
        self.optimizer = optim.Adam(self.policy.model.parameters(), lr=self.learning_rate)
        
        print(f"  Final value loss: {value_losses[-1]:.4f}")
        print(f"  Value pre-training complete!")
        
        return value_losses


class FixedProductionPPOTrainer(ProductionPPOTrainer):
    """Production trainer with fixed value pre-training"""
    
    def create_trainer(self):
        """Create PPO trainer with proper value pre-training"""
        if self.resume_checkpoint:
            self.logger.info("\n🔄 LOADING TRAINER FROM CHECKPOINT")
            self.trainer = self._load_trainer_from_checkpoint(self.resume_checkpoint)
        else:
            self.logger.info("\n🆕 CREATING NEW TRAINER WITH REAL VALUE PRE-TRAINING")
            self.trainer = PPOTrainerWithRealValuePretraining(
                sl_checkpoint_path=self.config.sl_checkpoint_path,
                learning_rate=self.config.learning_rate,
                clip_epsilon=self.config.clip_epsilon,
                ppo_epochs=self.config.ppo_epochs,
                rollout_steps=self.config.rollout_steps,
                max_episode_steps=2500,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                device=self.config.device
            )
    
    def _load_trainer_from_checkpoint(self, checkpoint_path: str):
        """Load trainer with proper implementation"""
        trainer = PPOTrainerWithRealValuePretraining(
            sl_checkpoint_path=self.config.sl_checkpoint_path,
            learning_rate=self.config.learning_rate,
            clip_epsilon=self.config.clip_epsilon,
            ppo_epochs=self.config.ppo_epochs,
            rollout_steps=self.config.rollout_steps,
            max_episode_steps=2500,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            device=self.config.device
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        trainer.policy.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return trainer


def main():
    """Run fixed production training"""
    print("🚀 FIXED PPO PRODUCTION TRAINING")
    print("=" * 80)
    print("FIXES:")
    print("• Real value function pre-training (not dummy)")
    print("• Proper JSON serialization")
    print("• Expected to show positive improvement")
    print("=" * 80)
    
    # Clean previous run
    import shutil
    if os.path.exists("production_checkpoints"):
        shutil.rmtree("production_checkpoints")
    
    # Configuration
    config = TrainingConfig(
        max_iterations=50,  # Shorter for testing the fix
        value_pretrain_iterations=15,  # Real pre-training
        checkpoint_interval=5,
        validation_interval=5,
        patience=15,
        learning_rate=0.00005,
        value_learning_rate=0.0005
    )
    
    # Create and run trainer
    trainer = FixedProductionPPOTrainer(config)
    
    try:
        trainer.run_production_training()
        print("\n✅ Training completed successfully!")
        
        # Show final stats
        if trainer.training_state['best_performance'] > 0:
            improvement = ((trainer.training_state['best_performance'] - 
                          trainer.training_state['sl_baseline']['mean']) / 
                          trainer.training_state['sl_baseline']['mean'] * 100)
            print(f"\n📊 FINAL RESULTS:")
            print(f"   SL Baseline: {trainer.training_state['sl_baseline']['mean']:.4f}")
            print(f"   Best PPO: {trainer.training_state['best_performance']:.4f}")
            print(f"   Improvement: {improvement:+.1f}%")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    return trainer


if __name__ == "__main__":
    main()
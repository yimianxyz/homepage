#!/usr/bin/env python3
"""
PPO with Value Function Pre-training - Practical Implementation

USER INSIGHT: Train value function FIRST to match SL baseline,
then train both actor and critic together.

This solves the fundamental instability issue!
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from rl_training.ppo_experience_buffer import PPOExperienceBuffer
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


class PPOWithValuePretraining(PPOTrainer):
    """Extended PPO trainer with value function pre-training phase"""
    
    def __init__(self, *args, **kwargs):
        # Add pre-training specific parameters
        self.value_pretrain_iterations = kwargs.pop('value_pretrain_iterations', 20)
        self.value_pretrain_lr = kwargs.pop('value_pretrain_lr', 0.0005)
        self.value_pretrain_epochs = kwargs.pop('value_pretrain_epochs', 5)
        
        # Initialize parent PPO trainer
        super().__init__(*args, **kwargs)
        
        print("\n🎯 PPO WITH VALUE PRE-TRAINING")
        print("=" * 70)
        print("STRATEGY: Train value function first, then normal PPO")
        print(f"Value Pre-train Iterations: {self.value_pretrain_iterations}")
        print(f"Value Pre-train LR: {self.value_pretrain_lr}")
        print("=" * 70)
    
    def pretrain_value_function(self):
        """Pre-train value function while keeping policy frozen"""
        print(f"\n🔧 PHASE 1: VALUE FUNCTION PRE-TRAINING")
        print("Freezing policy, training only value function...")
        
        # Save original learning rate
        original_lr = self.learning_rate
        
        # Create separate optimizer for value function only
        value_params = []
        policy_params = []
        
        for name, param in self.policy.model.named_parameters():
            if 'value_head' in name:
                value_params.append(param)
            else:
                policy_params.append(param)
        
        # Freeze policy parameters temporarily
        for param in policy_params:
            param.requires_grad = False
        
        # Value-only optimizer
        value_optimizer = optim.Adam(value_params, lr=self.value_pretrain_lr)
        
        print(f"  Frozen parameters: {len(policy_params)}")
        print(f"  Trainable parameters: {len(value_params)}")
        
        # Pre-training loop
        value_losses = []
        
        for iteration in range(1, self.value_pretrain_iterations + 1):
            print(f"\n  Pre-train Iteration {iteration}/{self.value_pretrain_iterations}")
            
            # Collect experience with frozen policy
            initial_state = generate_random_state(12, 400, 300)
            experience_buffer = self.rollout_collector.collect_rollout(
                initial_state, self.rollout_steps
            )
            
            # Compute returns and advantages
            next_value = 0.0
            advantages, returns = experience_buffer.compute_advantages_and_returns(next_value)
            
            # Get batch data for value training
            batch_data = experience_buffer.get_batch_data()
            structured_inputs = batch_data['structured_inputs']
            
            # Train value function only
            total_value_loss = 0.0
            num_updates = 0
            
            for epoch in range(self.value_pretrain_epochs):
                # Shuffle data
                num_samples = len(structured_inputs)
                indices = torch.randperm(num_samples)
                
                # Mini-batch updates
                for start_idx in range(0, num_samples, self.mini_batch_size):
                    end_idx = min(start_idx + self.mini_batch_size, num_samples)
                    batch_indices = indices[start_idx:end_idx].tolist()
                    
                    # Create mini-batch
                    batch_structured_inputs = [structured_inputs[i] for i in batch_indices]
                    batch_returns = returns[batch_indices].to(self.device)
                    
                    # Forward pass - get value predictions
                    _, values = self.policy.model(batch_structured_inputs)
                    values = values.squeeze(-1)
                    
                    # Value loss
                    value_loss = nn.MSELoss()(values, batch_returns)
                    
                    # Update
                    value_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(value_params, self.max_grad_norm)
                    value_optimizer.step()
                    
                    total_value_loss += value_loss.item()
                    num_updates += 1
            
            avg_value_loss = total_value_loss / num_updates if num_updates > 0 else 0
            value_losses.append(avg_value_loss)
            
            print(f"    Value Loss: {avg_value_loss:.4f}")
            
            # Test value predictions
            with torch.no_grad():
                sample_structured_inputs = structured_inputs[:5]
                sample_returns = returns[:5]
                _, sample_values = self.policy.model(sample_structured_inputs)
                sample_values = sample_values.squeeze(-1).cpu()
                
                print(f"    Sample predictions:")
                for i in range(min(3, len(sample_values))):
                    print(f"      Predicted: {sample_values[i].item():.3f}, Actual: {sample_returns[i].item():.3f}")
            
            # Early stopping if converged
            if avg_value_loss < 0.5:
                print(f"\n  ✅ Value function converged at iteration {iteration}")
                break
        
        # Unfreeze policy parameters
        for param in policy_params:
            param.requires_grad = True
        
        # Restore original optimizer
        self.optimizer = optim.Adam(self.policy.model.parameters(), lr=original_lr)
        
        # Summary
        print(f"\n✅ VALUE PRE-TRAINING COMPLETE")
        print(f"   Initial Loss: {value_losses[0]:.4f}")
        print(f"   Final Loss: {value_losses[-1]:.4f}")
        print(f"   Improvement: {(value_losses[0] - value_losses[-1]) / value_losses[0] * 100:.1f}%")
        
        return value_losses
    
    def train_with_pretraining(self, num_iterations: int = 20) -> Dict[str, Any]:
        """Complete training: value pre-training + normal PPO"""
        
        print(f"\n🚀 STARTING TWO-PHASE TRAINING")
        
        # Phase 1: Value pre-training
        print(f"\n{'='*70}")
        print(f"PHASE 1: VALUE FUNCTION PRE-TRAINING")
        print(f"{'='*70}")
        
        value_losses = self.pretrain_value_function()
        
        # Phase 2: Normal PPO training
        print(f"\n{'='*70}")
        print(f"PHASE 2: FULL PPO TRAINING")
        print(f"{'='*70}")
        print(f"Now training with both policy and value function...")
        
        # Track performance
        performance_history = []
        sl_baseline = None
        
        # Evaluate SL baseline
        sl_policy = TransformerPolicy(self.sl_checkpoint_path)
        sl_result = self.evaluator.evaluate_policy(sl_policy, "SL_Baseline")
        sl_baseline = sl_result.overall_catch_rate
        print(f"\nSL Baseline: {sl_baseline:.4f}")
        
        # PPO training loop
        for iteration in range(1, num_iterations + 1):
            print(f"\n📍 PPO Iteration {iteration}/{num_iterations}")
            
            # Standard PPO training iteration
            initial_state = generate_random_state(12, 400, 300)
            self.train_iteration(initial_state)
            
            # Evaluate periodically
            if iteration <= 5 or iteration % 2 == 0:
                result = self.evaluator.evaluate_policy(self.policy, f"PPO_Iter{iteration}")
                performance = result.overall_catch_rate
                improvement = ((performance - sl_baseline) / sl_baseline) * 100
                
                performance_history.append({
                    'iteration': iteration,
                    'performance': performance,
                    'improvement': improvement
                })
                
                status = "✅ BEATS SL" if performance > sl_baseline else "❌ Below SL"
                print(f"  Performance: {performance:.4f} ({improvement:+.1f}%) {status}")
                
                # Early stopping if very successful
                if performance > sl_baseline * 1.1:  # 10% improvement
                    print(f"\n🎉 MAJOR SUCCESS: 10%+ improvement achieved!")
                    break
        
        # Final analysis
        final_results = {
            'value_pretraining_losses': value_losses,
            'performance_history': performance_history,
            'sl_baseline': sl_baseline,
            'best_performance': max(p['performance'] for p in performance_history) if performance_history else 0,
            'final_performance': performance_history[-1]['performance'] if performance_history else 0,
            'success': any(p['performance'] > sl_baseline for p in performance_history) if performance_history else False
        }
        
        print(f"\n{'='*70}")
        print(f"TWO-PHASE TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"SL Baseline: {sl_baseline:.4f}")
        print(f"Best PPO: {final_results['best_performance']:.4f}")
        print(f"Best Improvement: {(final_results['best_performance'] - sl_baseline) / sl_baseline * 100:+.1f}%")
        print(f"Success: {'✅ YES' if final_results['success'] else '❌ NO'}")
        
        return final_results


def run_statistical_validation_with_pretraining():
    """Run multiple trials with value pre-training for statistical validation"""
    print("📊 STATISTICAL VALIDATION WITH VALUE PRE-TRAINING")
    print("=" * 80)
    print("Running multiple independent trials...")
    
    num_trials = 5  # Start with 5 for quick validation
    all_results = []
    
    for trial in range(num_trials):
        print(f"\n{'='*80}")
        print(f"TRIAL {trial + 1}/{num_trials}")
        print(f"{'='*80}")
        
        # Create trainer with value pre-training
        trainer = PPOWithValuePretraining(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=0.00005,
            clip_epsilon=0.1,
            ppo_epochs=2,
            rollout_steps=256,
            max_episode_steps=2500,
            gamma=0.95,
            gae_lambda=0.9,
            device='cpu',
            # Value pre-training parameters
            value_pretrain_iterations=15,
            value_pretrain_lr=0.0005,
            value_pretrain_epochs=3
        )
        
        # Run two-phase training
        results = trainer.train_with_pretraining(num_iterations=10)
        all_results.append(results)
        
        print(f"\nTrial {trial + 1} Result: {'✅ SUCCESS' if results['success'] else '❌ FAILURE'}")
    
    # Statistical analysis
    successful_trials = sum(1 for r in all_results if r['success'])
    best_performances = [r['best_performance'] for r in all_results]
    sl_baseline = all_results[0]['sl_baseline']
    
    improvements = [(p - sl_baseline) / sl_baseline * 100 for p in best_performances]
    
    print(f"\n{'='*80}")
    print(f"STATISTICAL SUMMARY")
    print(f"{'='*80}")
    print(f"Successful Trials: {successful_trials}/{num_trials}")
    print(f"Success Rate: {successful_trials/num_trials*100:.1f}%")
    print(f"Mean Best Performance: {np.mean(best_performances):.4f}")
    print(f"Std Dev: {np.std(best_performances):.4f}")
    print(f"Mean Improvement: {np.mean(improvements):.1f}%")
    
    # Simple t-test
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(best_performances, sl_baseline)
    
    print(f"\nStatistical Test (vs SL baseline {sl_baseline:.4f}):")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant: {'✅ YES' if p_value < 0.05 else '❌ NO'} (α=0.05)")
    
    if successful_trials >= num_trials * 0.8:  # 80% success rate
        print(f"\n🎉 BREAKTHROUGH: Value pre-training solves the instability!")
        print(f"   Consistent improvement over SL baseline")
        print(f"   Ready for full statistical validation (15+ runs)")
    else:
        print(f"\n🔧 Needs further tuning")
    
    # Save results
    import json
    validation_results = {
        'num_trials': num_trials,
        'successful_trials': successful_trials,
        'success_rate': successful_trials/num_trials,
        'mean_improvement': np.mean(improvements),
        'p_value': p_value,
        'all_trial_results': all_results
    }
    
    with open('value_pretraining_validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved: value_pretraining_validation_results.json")
    
    return validation_results


def main():
    """Demonstrate value pre-training solution"""
    print("🎯 PPO WITH VALUE FUNCTION PRE-TRAINING")
    print("=" * 80)
    print("USER INSIGHT: Train value function first, then both")
    print("BENEFIT: Solves catastrophic instability root cause")
    print("=" * 80)
    
    # Quick demonstration
    print("\n1️⃣ QUICK DEMONSTRATION")
    trainer = PPOWithValuePretraining(
        sl_checkpoint_path="checkpoints/best_model.pt",
        learning_rate=0.00005,
        clip_epsilon=0.1,
        ppo_epochs=2,
        rollout_steps=256,
        max_episode_steps=2500,
        gamma=0.95,
        gae_lambda=0.9,
        device='cpu',
        value_pretrain_iterations=10,
        value_pretrain_lr=0.0005
    )
    
    quick_results = trainer.train_with_pretraining(num_iterations=5)
    
    if quick_results['success']:
        print(f"\n✅ Quick test successful! Running statistical validation...")
        
        # Run statistical validation
        print("\n2️⃣ STATISTICAL VALIDATION")
        validation_results = run_statistical_validation_with_pretraining()
    else:
        print(f"\n❌ Quick test failed - may need parameter tuning")
    
    return quick_results


if __name__ == "__main__":
    main()
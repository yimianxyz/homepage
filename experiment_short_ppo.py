#!/usr/bin/env python3
"""
Experiment: Ultra-short PPO Training
Tests if very short PPO training (1-10 iterations) can improve without catastrophic forgetting
"""

import torch
import torch.optim as optim
import numpy as np
import json
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training.ppo_trainer import PPOTrainer
from evaluation import PolicyEvaluator
from policy.transformer.transformer_policy import TransformerPolicy
from simulation.random_state_generator import generate_random_state


def run_short_ppo_experiment():
    """Test ultra-short PPO training"""
    print("=" * 80)
    print("EXPERIMENT: Ultra-short PPO Training")
    print("Hypothesis: Very short training (1-10 iter) improves without forgetting")
    print("=" * 80)
    
    # Create evaluator
    evaluator = PolicyEvaluator(num_episodes=15, base_seed=21000)
    
    # Evaluate baseline
    print("\n1. Evaluating SL baseline...")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    baseline_result = evaluator.evaluate_policy(sl_policy, "SL_Baseline")
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f} ± {baseline_result.std_error:.4f}")
    
    # Test different iteration counts
    iterations_to_test = [1, 2, 3, 5, 7, 10]
    results = []
    
    for n_iter in iterations_to_test:
        print(f"\n2. Testing {n_iter} PPO iterations...")
        
        # Create trainer with optimal settings from experiments
        trainer = PPOTrainer(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=3e-5,
            clip_epsilon=0.1,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            ppo_epochs=2,
            mini_batch_size=64,
            rollout_steps=512,
            max_episode_steps=5000,  # Use optimal episode length
            gamma=0.99,
            gae_lambda=0.95,
            device='cpu'
        )
        
        # Pre-train value function (20 iterations as optimal)
        print("Pre-training value function...")
        pretrain_losses = []
        
        # Freeze policy parameters
        value_params = []
        policy_params = []
        for name, param in trainer.policy.model.named_parameters():
            if 'value_head' in name:
                value_params.append(param)
            else:
                policy_params.append(param)
                param.requires_grad = False
        
        value_optimizer = optim.Adam(value_params, lr=3e-4)
        
        # Value pre-training
        for i in range(20):
            initial_state = generate_random_state(12, 400, 300)
            buffer = trainer.rollout_collector.collect_rollout(initial_state, 256)
            
            batch_data = buffer.get_batch_data()
            if len(batch_data) > 0:
                inputs = batch_data['structured_inputs']
                returns = batch_data['returns'].to(trainer.device)
                
                _, values = trainer.policy.model(inputs)
                values = values.squeeze()
                
                value_loss = torch.nn.functional.mse_loss(values, returns)
                
                value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(value_params, 0.5)
                value_optimizer.step()
                
                pretrain_losses.append(value_loss.item())
        
        # Unfreeze policy parameters
        for param in policy_params:
            param.requires_grad = True
        
        print(f"Value pre-train complete: {pretrain_losses[0]:.4f} → {pretrain_losses[-1]:.4f}")
        
        # Run short PPO training
        print(f"Running {n_iter} PPO iterations...")
        for i in range(n_iter):
            initial_state = generate_random_state(12, 400, 300)
            stats = trainer.train_iteration(initial_state)
            print(f"  Iter {i+1}: Policy loss={stats['training']['policy_loss']:.4f}, "
                  f"Episode reward={stats['rollout']['mean_reward']:.2f}")
        
        # Evaluate
        print(f"3. Evaluating after {n_iter} iterations...")
        result = evaluator.evaluate_policy(trainer.policy, f"ShortPPO_{n_iter}")
        
        improvement = (result.overall_catch_rate - baseline_result.overall_catch_rate) / baseline_result.overall_catch_rate * 100
        print(f"Performance: {result.overall_catch_rate:.4f} ({improvement:+.1f}%)")
        
        # Check if significant
        is_significant = result.confidence_95_lower > baseline_result.confidence_95_upper
        print(f"Statistically significant: {'YES' if is_significant else 'NO'}")
        
        results.append({
            'iterations': n_iter,
            'performance': result.overall_catch_rate,
            'improvement': improvement,
            'is_significant': is_significant,
            'confidence_interval': [result.confidence_95_lower, result.confidence_95_upper],
            'final_policy_loss': stats['training']['policy_loss'],
            'final_value_loss': stats['training']['value_loss']
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY:")
    print("=" * 80)
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f}")
    print("\nShort PPO Training Results:")
    for r in results:
        sig = "✓" if r['is_significant'] else "✗"
        print(f"  {r['iterations']:2d} iterations: {r['performance']:.4f} ({r['improvement']:+.1f}%) {sig}")
    
    # Find optimal
    best = max(results, key=lambda x: x['performance'])
    print(f"\nBest: {best['iterations']} iterations with {best['performance']:.4f} ({best['improvement']:+.1f}%)")
    
    # Save results
    with open('experiment_short_ppo_results.json', 'w') as f:
        json.dump({
            'experiment': 'short_ppo_training',
            'timestamp': datetime.now().isoformat(),
            'baseline': baseline_result.overall_catch_rate,
            'results': results,
            'best': best
        }, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_short_ppo_experiment()
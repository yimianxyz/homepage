#!/usr/bin/env python3
"""
Experiment: Best Approach Based on Findings
Tests the most promising configuration:
- Value pre-training (20 iterations)
- Very short PPO (1-3 iterations)
- Conservative hyperparameters (LR=3e-5, Clip=0.01)
- Match episode length to evaluation (2500)
"""

import torch
import torch.nn as nn
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


def run_best_approach():
    """Test the most promising approach based on all findings"""
    print("=" * 80)
    print("EXPERIMENT: Best Approach")
    print("Configuration based on all findings:")
    print("- Value pre-training: 20 iterations")
    print("- PPO iterations: 1-3 only")
    print("- Learning rate: 3e-5")
    print("- Clip epsilon: 0.01 (very conservative)")
    print("- Episode length: 2500 (match evaluation)")
    print("=" * 80)
    
    # Create evaluator (use 10 episodes for better confidence)
    evaluator = PolicyEvaluator(num_episodes=10, base_seed=26000)
    
    # Evaluate baseline
    print("\n1. Evaluating SL baseline...")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    baseline_result = evaluator.evaluate_policy(sl_policy, "SL_Baseline")
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f} ± {baseline_result.std_error:.4f}")
    print(f"95% CI: [{baseline_result.confidence_95_lower:.4f}, {baseline_result.confidence_95_upper:.4f}]")
    
    results = []
    
    # Test 0, 1, 2, 3 PPO iterations (0 = value pre-training only)
    for n_ppo in [0, 1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"Testing {n_ppo} PPO iterations")
        print(f"{'='*60}")
        
        # Create trainer with optimal conservative settings
        trainer = PPOTrainer(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=3e-5,
            clip_epsilon=0.01,  # Very conservative
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            ppo_epochs=2,
            mini_batch_size=64,
            rollout_steps=512,
            max_episode_steps=2500,  # Match evaluation!
            gamma=0.99,
            gae_lambda=0.95,
            device='cpu'
        )
        
        print("2. Value pre-training (20 iterations)...")
        
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
        value_losses = []
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
                
                value_losses.append(value_loss.item())
        
        print(f"Value pre-training complete: {np.mean(value_losses[:5]):.4f} → {np.mean(value_losses[-5:]):.4f}")
        
        # Unfreeze policy parameters
        for param in policy_params:
            param.requires_grad = True
        
        # Run PPO if n_ppo > 0
        if n_ppo > 0:
            print(f"\n3. Running {n_ppo} PPO iterations...")
            for i in range(n_ppo):
                initial_state = generate_random_state(12, 400, 300)
                stats = trainer.train_iteration(initial_state)
                print(f"  Iter {i+1}: Policy loss={stats['training']['policy_loss']:.4f}, "
                      f"Episode reward={stats['rollout']['mean_reward']:.2f}")
        else:
            print("\n3. Skipping PPO (testing value pre-training only)")
        
        # Evaluate
        print(f"\n4. Evaluating after {n_ppo} PPO iterations...")
        result = evaluator.evaluate_policy(trainer.policy, f"BestApproach_PPO{n_ppo}")
        
        improvement = (result.overall_catch_rate - baseline_result.overall_catch_rate) / baseline_result.overall_catch_rate * 100
        
        # Check statistical significance
        is_significant = result.confidence_95_lower > baseline_result.confidence_95_upper
        
        print(f"Performance: {result.overall_catch_rate:.4f} ± {result.std_error:.4f}")
        print(f"95% CI: [{result.confidence_95_lower:.4f}, {result.confidence_95_upper:.4f}]")
        print(f"Improvement: {improvement:+.1f}%")
        print(f"Statistically significant: {'YES ✓' if is_significant else 'NO ✗'}")
        
        results.append({
            'ppo_iterations': n_ppo,
            'performance': result.overall_catch_rate,
            'std_error': result.std_error,
            'improvement': improvement,
            'is_significant': is_significant,
            'confidence_interval': [result.confidence_95_lower, result.confidence_95_upper]
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY:")
    print("=" * 80)
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f} ± {baseline_result.std_error:.4f}")
    print("\nBest Approach Results:")
    for r in results:
        sig = "✓" if r['is_significant'] else "✗"
        print(f"  {r['ppo_iterations']} PPO iter: {r['performance']:.4f} ± {r['std_error']:.4f} "
              f"({r['improvement']:+.1f}%) {sig}")
    
    # Find best
    best = max(results, key=lambda x: x['performance'])
    print(f"\nBest configuration: {best['ppo_iterations']} PPO iterations")
    print(f"Performance: {best['performance']:.4f} ({best['improvement']:+.1f}%)")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print("1. Use value pre-training (20 iterations) to prepare critic")
    print("2. Run only 1-3 PPO iterations to avoid catastrophic forgetting")
    print("3. Use conservative hyperparameters (LR=3e-5, Clip=0.01)")
    print("4. Match training episode length to evaluation (2500 steps)")
    print("5. Consider early stopping if performance degrades")
    
    # Save results
    with open('experiment_best_approach_results.json', 'w') as f:
        json.dump({
            'experiment': 'best_approach',
            'timestamp': datetime.now().isoformat(),
            'baseline': {
                'performance': baseline_result.overall_catch_rate,
                'std_error': baseline_result.std_error,
                'confidence_interval': [baseline_result.confidence_95_lower, baseline_result.confidence_95_upper]
            },
            'results': results,
            'best': best,
            'configuration': {
                'value_pretrain_iterations': 20,
                'learning_rate': 3e-5,
                'clip_epsilon': 0.01,
                'episode_length': 2500
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: experiment_best_approach_results.json")
    
    return results


if __name__ == "__main__":
    results = run_best_approach()
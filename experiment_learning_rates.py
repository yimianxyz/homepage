#!/usr/bin/env python3
"""
Experiment: Learning Rate Comparison
Tests if lower learning rates prevent catastrophic forgetting
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


def run_learning_rate_experiment():
    """Test different learning rates"""
    print("=" * 80)
    print("EXPERIMENT: Learning Rate Comparison")
    print("Hypothesis: Lower learning rates prevent catastrophic forgetting")
    print("=" * 80)
    
    # Create evaluator
    evaluator = PolicyEvaluator(num_episodes=15, base_seed=22000)
    
    # Evaluate baseline
    print("\n1. Evaluating SL baseline...")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    baseline_result = evaluator.evaluate_policy(sl_policy, "SL_Baseline")
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f} ± {baseline_result.std_error:.4f}")
    
    # Test different learning rates
    learning_rates = [3e-5, 1e-5, 5e-6, 1e-6]
    n_iterations = 20  # Fixed number of iterations
    results = []
    
    for lr in learning_rates:
        print(f"\n2. Testing learning rate: {lr}")
        
        # Create trainer
        trainer = PPOTrainer(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=lr,
            clip_epsilon=0.1,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            ppo_epochs=2,
            mini_batch_size=64,
            rollout_steps=512,
            max_episode_steps=2500,  # Match evaluation
            gamma=0.99,
            gae_lambda=0.95,
            device='cpu'
        )
        
        # Pre-train value function
        print("Pre-training value function...")
        value_params = []
        policy_params = []
        for name, param in trainer.policy.model.named_parameters():
            if 'value_head' in name:
                value_params.append(param)
            else:
                policy_params.append(param)
                param.requires_grad = False
        
        value_optimizer = optim.Adam(value_params, lr=3e-4)
        
        # Value pre-training (20 iterations)
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
        
        # Unfreeze policy parameters
        for param in policy_params:
            param.requires_grad = True
        
        # Run PPO training
        print(f"Running {n_iterations} PPO iterations with lr={lr}...")
        iteration_performances = []
        
        for i in range(n_iterations):
            initial_state = generate_random_state(12, 400, 300)
            stats = trainer.train_iteration(initial_state)
            
            # Evaluate every 5 iterations
            if (i + 1) % 5 == 0:
                result = trainer.evaluate_policy(num_episodes=5)
                iteration_performances.append({
                    'iteration': i + 1,
                    'performance': result['overall_catch_rate']
                })
                print(f"  Iter {i+1}: Performance={result['overall_catch_rate']:.3f}, "
                      f"Policy loss={stats['training']['policy_loss']:.4f}")
        
        # Final evaluation
        print(f"3. Final evaluation for lr={lr}...")
        final_result = evaluator.evaluate_policy(trainer.policy, f"LR_{lr}")
        
        improvement = (final_result.overall_catch_rate - baseline_result.overall_catch_rate) / baseline_result.overall_catch_rate * 100
        print(f"Final performance: {final_result.overall_catch_rate:.4f} ({improvement:+.1f}%)")
        
        # Check stability (compare early vs late performance)
        early_perf = iteration_performances[0]['performance'] if iteration_performances else 0
        late_perf = iteration_performances[-1]['performance'] if iteration_performances else 0
        stability = (late_perf - early_perf) / early_perf * 100 if early_perf > 0 else 0
        
        results.append({
            'learning_rate': lr,
            'final_performance': final_result.overall_catch_rate,
            'improvement': improvement,
            'stability': stability,
            'iteration_performances': iteration_performances,
            'confidence_interval': [final_result.confidence_95_lower, final_result.confidence_95_upper]
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY:")
    print("=" * 80)
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f}")
    print("\nLearning Rate Results:")
    for r in results:
        stability_indicator = "📈" if r['stability'] > -5 else "📉"
        print(f"  LR {r['learning_rate']}: {r['final_performance']:.4f} ({r['improvement']:+.1f}%) "
              f"Stability: {r['stability']:+.1f}% {stability_indicator}")
    
    # Find most stable
    most_stable = max(results, key=lambda x: x['stability'])
    best_performance = max(results, key=lambda x: x['final_performance'])
    
    print(f"\nMost stable: LR={most_stable['learning_rate']} (stability={most_stable['stability']:+.1f}%)")
    print(f"Best performance: LR={best_performance['learning_rate']} ({best_performance['final_performance']:.4f})")
    
    # Save results
    with open('experiment_learning_rates_results.json', 'w') as f:
        json.dump({
            'experiment': 'learning_rate_comparison',
            'timestamp': datetime.now().isoformat(),
            'baseline': baseline_result.overall_catch_rate,
            'results': results,
            'most_stable': most_stable,
            'best_performance': best_performance
        }, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_learning_rate_experiment()
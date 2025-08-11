#!/usr/bin/env python3
"""
Experiment: Conservative PPO
Tests if smaller clip epsilon prevents catastrophic forgetting
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


def run_conservative_ppo_experiment():
    """Test conservative PPO settings"""
    print("=" * 80)
    print("EXPERIMENT: Conservative PPO")
    print("Hypothesis: Smaller clip epsilon prevents drastic policy changes")
    print("=" * 80)
    
    # Create evaluator
    evaluator = PolicyEvaluator(num_episodes=15, base_seed=23000)
    
    # Evaluate baseline
    print("\n1. Evaluating SL baseline...")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    baseline_result = evaluator.evaluate_policy(sl_policy, "SL_Baseline")
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f} ± {baseline_result.std_error:.4f}")
    
    # Test different clip epsilon values
    clip_epsilons = [0.2, 0.1, 0.05, 0.02, 0.01]
    n_iterations = 20
    results = []
    
    for clip_eps in clip_epsilons:
        print(f"\n2. Testing clip epsilon: {clip_eps}")
        
        # Create trainer
        trainer = PPOTrainer(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=1e-5,  # Use conservative learning rate
            clip_epsilon=clip_eps,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            ppo_epochs=2,
            mini_batch_size=64,
            rollout_steps=512,
            max_episode_steps=2500,
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
        print(f"Running {n_iterations} PPO iterations with clip_epsilon={clip_eps}...")
        clip_fractions = []
        policy_changes = []
        
        for i in range(n_iterations):
            initial_state = generate_random_state(12, 400, 300)
            stats = trainer.train_iteration(initial_state)
            
            # Track metrics
            clip_fractions.append(stats['training'].get('policy_ratio_mean', 1.0))
            policy_changes.append(abs(stats['training'].get('policy_ratio_mean', 1.0) - 1.0))
            
            if (i + 1) % 5 == 0:
                avg_clip = np.mean(clip_fractions[-5:])
                avg_change = np.mean(policy_changes[-5:])
                print(f"  Iter {i+1}: Avg clip ratio={avg_clip:.3f}, "
                      f"Avg policy change={avg_change:.3f}")
        
        # Final evaluation
        print(f"3. Final evaluation for clip_epsilon={clip_eps}...")
        final_result = evaluator.evaluate_policy(trainer.policy, f"ConservativePPO_{clip_eps}")
        
        improvement = (final_result.overall_catch_rate - baseline_result.overall_catch_rate) / baseline_result.overall_catch_rate * 100
        avg_policy_change = np.mean(policy_changes)
        
        print(f"Final performance: {final_result.overall_catch_rate:.4f} ({improvement:+.1f}%)")
        print(f"Average policy change: {avg_policy_change:.4f}")
        
        results.append({
            'clip_epsilon': clip_eps,
            'final_performance': final_result.overall_catch_rate,
            'improvement': improvement,
            'avg_policy_change': avg_policy_change,
            'confidence_interval': [final_result.confidence_95_lower, final_result.confidence_95_upper],
            'is_significant': final_result.confidence_95_lower > baseline_result.confidence_95_upper
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY:")
    print("=" * 80)
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f}")
    print("\nClip Epsilon Results:")
    for r in results:
        sig = "✓" if r['is_significant'] else "✗"
        print(f"  ε={r['clip_epsilon']:.2f}: {r['final_performance']:.4f} ({r['improvement']:+.1f}%) "
              f"Avg change={r['avg_policy_change']:.3f} {sig}")
    
    # Find best trade-off (good performance with low policy change)
    # Score = performance - policy_change penalty
    scored_results = [(r, r['final_performance'] - 0.5 * r['avg_policy_change']) for r in results]
    best_tradeoff = max(scored_results, key=lambda x: x[1])[0]
    
    print(f"\nBest trade-off: ε={best_tradeoff['clip_epsilon']} "
          f"(performance={best_tradeoff['final_performance']:.4f}, "
          f"change={best_tradeoff['avg_policy_change']:.3f})")
    
    # Save results
    with open('experiment_conservative_ppo_results.json', 'w') as f:
        json.dump({
            'experiment': 'conservative_ppo',
            'timestamp': datetime.now().isoformat(),
            'baseline': baseline_result.overall_catch_rate,
            'results': results,
            'best_tradeoff': best_tradeoff
        }, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_conservative_ppo_experiment()
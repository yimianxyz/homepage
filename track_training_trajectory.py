#!/usr/bin/env python3
"""
Track Training Trajectory - Monitor validation performance at each iteration

This experiment will:
1. Train both standard and value pre-trained PPO
2. Evaluate performance after EVERY iteration
3. Plot the learning curves
4. Find the optimal stopping point
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from ppo_with_value_pretraining import PPOWithValuePretraining
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


def track_training_trajectory(max_iterations=30, num_trials=2):
    """Track performance at each iteration"""
    print("📈 TRACKING TRAINING TRAJECTORY")
    print("=" * 60)
    print(f"Max iterations: {max_iterations}")
    print(f"Trials per method: {num_trials}")
    print("=" * 60)
    
    evaluator = PolicyEvaluator()
    
    # Get baseline
    print("\n📊 ESTABLISHING BASELINE")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    sl_result = evaluator.evaluate_policy(sl_policy, "SL_Baseline")
    sl_baseline = sl_result.overall_catch_rate
    print(f"SL Baseline: {sl_baseline:.4f}")
    
    # Storage for results
    standard_trajectories = []
    pretrained_trajectories = []
    
    # Run trials
    for trial in range(num_trials):
        print(f"\n{'='*60}")
        print(f"TRIAL {trial + 1}/{num_trials}")
        print(f"{'='*60}")
        
        # Standard PPO
        print(f"\n🔵 STANDARD PPO - Trial {trial + 1}")
        standard_trainer = PPOTrainer(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=0.00005,
            clip_epsilon=0.1,
            ppo_epochs=2,
            rollout_steps=256,
            max_episode_steps=2500,
            gamma=0.95,
            gae_lambda=0.9,
            device='cpu'
        )
        
        standard_performance = []
        
        # Evaluate at iteration 0 (just SL model)
        result = evaluator.evaluate_policy(standard_trainer.policy, f"Std_T{trial+1}_I0")
        perf = result.overall_catch_rate
        standard_performance.append(perf)
        print(f"Iter 0: {perf:.4f} (initial)")
        
        # Train and evaluate at each iteration
        for i in range(1, max_iterations + 1):
            # Train
            initial_state = generate_random_state(12, 400, 300)
            metrics = standard_trainer.train_iteration(initial_state)
            
            # Evaluate every iteration for first 10, then every 5
            if i <= 10 or i % 5 == 0:
                result = evaluator.evaluate_policy(standard_trainer.policy, f"Std_T{trial+1}_I{i}")
                perf = result.overall_catch_rate
                standard_performance.append(perf)
                improvement = (perf - sl_baseline) / sl_baseline * 100
                
                # Print with visual indicator
                if perf > sl_baseline:
                    status = "✅"
                elif perf > sl_baseline * 0.95:
                    status = "🟡"
                else:
                    status = "❌"
                
                print(f"Iter {i}: {perf:.4f} ({improvement:+.1f}%) {status} | Value loss: {metrics.get('value_loss', 0):.2f}")
        
        standard_trajectories.append(standard_performance)
        
        # Value Pre-trained PPO
        print(f"\n🟢 VALUE PRE-TRAINED PPO - Trial {trial + 1}")
        pretrained_trainer = PPOWithValuePretraining(
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
            value_pretrain_lr=0.0005,
            value_pretrain_epochs=3
        )
        
        # Pre-train value function
        print("Pre-training value function...")
        value_losses = pretrained_trainer.pretrain_value_function()
        print(f"Value pre-training: {value_losses[0]:.2f} → {value_losses[-1]:.2f}")
        
        pretrained_performance = []
        
        # Evaluate at iteration 0 (after pre-training)
        result = evaluator.evaluate_policy(pretrained_trainer.policy, f"Pre_T{trial+1}_I0")
        perf = result.overall_catch_rate
        pretrained_performance.append(perf)
        print(f"Iter 0: {perf:.4f} (after value pre-training)")
        
        # Train and evaluate
        for i in range(1, max_iterations + 1):
            # Train
            initial_state = generate_random_state(12, 400, 300)
            metrics = pretrained_trainer.train_iteration(initial_state)
            
            # Evaluate
            if i <= 10 or i % 5 == 0:
                result = evaluator.evaluate_policy(pretrained_trainer.policy, f"Pre_T{trial+1}_I{i}")
                perf = result.overall_catch_rate
                pretrained_performance.append(perf)
                improvement = (perf - sl_baseline) / sl_baseline * 100
                
                # Status indicator
                if perf > sl_baseline:
                    status = "✅"
                elif perf > sl_baseline * 0.95:
                    status = "🟡"
                else:
                    status = "❌"
                
                print(f"Iter {i}: {perf:.4f} ({improvement:+.1f}%) {status} | Value loss: {metrics.get('value_loss', 0):.2f}")
        
        pretrained_trajectories.append(pretrained_performance)
    
    # Analysis
    print("\n" + "="*60)
    print("📊 TRAJECTORY ANALYSIS")
    print("="*60)
    
    # Convert to numpy arrays
    standard_trajectories = np.array(standard_trajectories)
    pretrained_trajectories = np.array(pretrained_trajectories)
    
    # Calculate means and stds
    standard_mean = np.mean(standard_trajectories, axis=0)
    standard_std = np.std(standard_trajectories, axis=0)
    pretrained_mean = np.mean(pretrained_trajectories, axis=0)
    pretrained_std = np.std(pretrained_trajectories, axis=0)
    
    # Find best iteration for each method
    standard_best_iter = np.argmax(standard_mean)
    pretrained_best_iter = np.argmax(pretrained_mean)
    
    print(f"\n🏆 BEST ITERATIONS:")
    print(f"Standard PPO: Iteration {standard_best_iter} → {standard_mean[standard_best_iter]:.4f}")
    print(f"Value Pre-trained: Iteration {pretrained_best_iter} → {pretrained_mean[pretrained_best_iter]:.4f}")
    
    # Success rates (beating baseline)
    print(f"\n✅ SUCCESS RATES (beating {sl_baseline:.4f}):")
    iterations = list(range(len(standard_mean)))
    
    for i in [0, 1, 5, 10, 15, 20, 25, 30]:
        if i < len(iterations):
            std_success = np.sum(standard_trajectories[:, i] > sl_baseline) / num_trials * 100
            pre_success = np.sum(pretrained_trajectories[:, i] > sl_baseline) / num_trials * 100
            print(f"Iteration {iterations[i]:2d}: Standard {std_success:3.0f}% | Pre-trained {pre_success:3.0f}%")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Create iteration labels (0-10, then 15, 20, 25, 30)
    x_labels = list(range(11)) + list(range(15, max_iterations + 1, 5))
    
    # Plot trajectories
    plt.subplot(2, 1, 1)
    plt.plot(x_labels, standard_mean, 'b-', label='Standard PPO', linewidth=2)
    plt.fill_between(x_labels, standard_mean - standard_std, standard_mean + standard_std, alpha=0.2, color='blue')
    plt.plot(x_labels, pretrained_mean, 'g-', label='Value Pre-trained PPO', linewidth=2)
    plt.fill_between(x_labels, pretrained_mean - pretrained_std, pretrained_mean + pretrained_std, alpha=0.2, color='green')
    plt.axhline(y=sl_baseline, color='red', linestyle='--', label='SL Baseline', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Performance (Catch Rate)')
    plt.title('Training Trajectories: Standard vs Value Pre-trained PPO')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot improvement over baseline
    plt.subplot(2, 1, 2)
    standard_improvement = (standard_mean - sl_baseline) / sl_baseline * 100
    pretrained_improvement = (pretrained_mean - sl_baseline) / sl_baseline * 100
    
    plt.plot(x_labels, standard_improvement, 'b-', label='Standard PPO', linewidth=2)
    plt.plot(x_labels, pretrained_improvement, 'g-', label='Value Pre-trained PPO', linewidth=2)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Improvement over Baseline (%)')
    plt.title('Relative Improvement During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_trajectories.png', dpi=150)
    print("\n📈 Plot saved: training_trajectories.png")
    
    # Save detailed results
    results = {
        'sl_baseline': sl_baseline,
        'standard_trajectories': standard_trajectories.tolist(),
        'pretrained_trajectories': pretrained_trajectories.tolist(),
        'analysis': {
            'standard_best_iteration': int(standard_best_iter),
            'standard_best_performance': float(standard_mean[standard_best_iter]),
            'pretrained_best_iteration': int(pretrained_best_iter),
            'pretrained_best_performance': float(pretrained_mean[pretrained_best_iter]),
            'standard_final_performance': float(standard_mean[-1]),
            'pretrained_final_performance': float(pretrained_mean[-1])
        }
    }
    
    with open('training_trajectories.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Results saved: training_trajectories.json")
    
    # Final verdict
    print("\n" + "="*60)
    print("🎯 FINAL VERDICT")
    print("="*60)
    
    if pretrained_mean[pretrained_best_iter] > sl_baseline:
        print(f"✅ VALUE PRE-TRAINING BEATS BASELINE!")
        print(f"   Best: {pretrained_mean[pretrained_best_iter]:.4f} at iteration {pretrained_best_iter}")
        print(f"   Improvement: {(pretrained_mean[pretrained_best_iter] - sl_baseline) / sl_baseline * 100:+.1f}%")
    
    if pretrained_mean[-1] > standard_mean[-1]:
        diff = (pretrained_mean[-1] - standard_mean[-1]) / standard_mean[-1] * 100
        print(f"\n✅ Value pre-training outperforms standard PPO by {diff:.1f}%")
    
    return results


if __name__ == "__main__":
    start = time.time()
    results = track_training_trajectory(max_iterations=30, num_trials=2)
    duration = time.time() - start
    print(f"\n⏱️  Total time: {duration/60:.1f} minutes")
#!/usr/bin/env python3
"""
Simple Value Pre-training Test - Direct comparison

Just run both approaches and see which works better.
Focus on the key metric: does value pre-training help or hurt?
"""

import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from ppo_with_value_pretraining import PPOWithValuePretraining
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


def simple_comparison():
    """Simple direct comparison"""
    print("🎯 SIMPLE VALUE PRE-TRAINING TEST")
    print("=" * 50)
    
    evaluator = PolicyEvaluator()
    
    # Baseline
    print("\n📊 SL BASELINE")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    sl_result = evaluator.evaluate_policy(sl_policy, "SL_Baseline")
    sl_perf = sl_result.overall_catch_rate
    print(f"Performance: {sl_perf:.4f}")
    
    # Run 3 trials of each approach
    print("\n🔵 TESTING STANDARD PPO (3 trials)")
    standard_performances = []
    
    for trial in range(3):
        print(f"\nTrial {trial + 1}:")
        trainer = PPOTrainer(
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
        
        # Train 10 iterations
        for i in range(10):
            initial_state = generate_random_state(12, 400, 300)
            trainer.train_iteration(initial_state)
            print(".", end="", flush=True)
        
        # Evaluate
        result = evaluator.evaluate_policy(trainer.policy, f"Standard_T{trial+1}")
        perf = result.overall_catch_rate
        standard_performances.append(perf)
        print(f"\nPerformance: {perf:.4f} ({(perf-sl_perf)/sl_perf*100:+.1f}%)")
    
    print("\n🟢 TESTING VALUE PRE-TRAINED PPO (3 trials)")
    pretrained_performances = []
    
    for trial in range(3):
        print(f"\nTrial {trial + 1}:")
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
            value_pretrain_lr=0.0005,
            value_pretrain_epochs=3
        )
        
        # Pre-train value
        print("Pre-training value function...")
        value_losses = trainer.pretrain_value_function()
        print(f"Value loss: {value_losses[0]:.2f} → {value_losses[-1]:.2f}")
        
        # Train PPO
        for i in range(10):
            initial_state = generate_random_state(12, 400, 300)
            trainer.train_iteration(initial_state)
            print(".", end="", flush=True)
        
        # Evaluate
        result = evaluator.evaluate_policy(trainer.policy, f"Pretrained_T{trial+1}")
        perf = result.overall_catch_rate
        pretrained_performances.append(perf)
        print(f"\nPerformance: {perf:.4f} ({(perf-sl_perf)/sl_perf*100:+.1f}%)")
    
    # Analysis
    print("\n" + "="*50)
    print("📊 RESULTS SUMMARY")
    print("="*50)
    
    standard_mean = np.mean(standard_performances)
    pretrained_mean = np.mean(pretrained_performances)
    
    print(f"\nSL Baseline: {sl_perf:.4f}")
    print(f"\nStandard PPO:")
    print(f"  Trials: {[f'{p:.4f}' for p in standard_performances]}")
    print(f"  Mean: {standard_mean:.4f} ({(standard_mean-sl_perf)/sl_perf*100:+.1f}%)")
    
    print(f"\nValue Pre-trained PPO:")
    print(f"  Trials: {[f'{p:.4f}' for p in pretrained_performances]}")
    print(f"  Mean: {pretrained_mean:.4f} ({(pretrained_mean-sl_perf)/sl_perf*100:+.1f}%)")
    
    # Winner
    print(f"\n🏆 WINNER: ", end="")
    if pretrained_mean > standard_mean:
        print(f"Value Pre-training ({(pretrained_mean-standard_mean)/standard_mean*100:+.1f}% better)")
        print("\n✅ Value pre-training DOES help!")
        print("Next: Scale up to more iterations and trials")
    else:
        print(f"Standard PPO ({(standard_mean-pretrained_mean)/pretrained_mean*100:+.1f}% better)")
        print("\n❌ Value pre-training is HURTING performance!")
        print("Issues to investigate:")
        print("- Pre-training might be overfitting")
        print("- Learning rate might be too high")
        print("- May need to reduce pre-training iterations")


if __name__ == "__main__":
    start = time.time()
    simple_comparison()
    print(f"\n⏱️  Total time: {(time.time()-start)/60:.1f} minutes")
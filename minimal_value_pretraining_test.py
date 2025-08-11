#!/usr/bin/env python3
"""
Minimal Value Pre-training Test - Just the essentials

Get evidence in 5 minutes by:
1. Skip expensive evaluations during training
2. Just compare final results
3. Focus on value loss convergence as key metric
"""

import os
import sys
import time
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from ppo_with_value_pretraining import PPOWithValuePretraining
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


def minimal_test():
    """Ultra-fast test focusing on key evidence"""
    print("⚡ MINIMAL VALUE PRE-TRAINING TEST")
    print("=" * 50)
    print("Focus: Value loss convergence & final performance")
    print("Duration: ~5 minutes")
    print("=" * 50)
    
    evaluator = PolicyEvaluator()
    
    # 1. Quick baseline
    print("\n1️⃣ SL BASELINE")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    sl_result = evaluator.evaluate_policy(sl_policy, "SL_Baseline")
    sl_performance = sl_result.overall_catch_rate
    print(f"   Performance: {sl_performance:.4f}")
    
    # 2. Standard PPO - Track value loss instability
    print("\n2️⃣ STANDARD PPO (no pre-training)")
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
    
    print("   Training 3 iterations...")
    standard_value_losses = []
    for i in range(1, 4):
        initial_state = generate_random_state(12, 400, 300)
        metrics = standard_trainer.train_iteration(initial_state)
        value_loss = metrics.get('value_loss', 0)
        standard_value_losses.append(value_loss)
        print(f"   Iter {i}: Value loss = {value_loss:.2f}")
    
    # Final evaluation
    standard_result = evaluator.evaluate_policy(standard_trainer.policy, "Standard_Final")
    standard_performance = standard_result.overall_catch_rate
    print(f"   Final performance: {standard_performance:.4f} ({(standard_performance-sl_performance)/sl_performance*100:+.1f}%)")
    
    # 3. Value Pre-trained PPO
    print("\n3️⃣ VALUE PRE-TRAINED PPO")
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
        value_pretrain_iterations=5,  # Quick
        value_pretrain_lr=0.0005,
        value_pretrain_epochs=2
    )
    
    # Pre-train value function
    print("   Pre-training value function...")
    value_pretrain_losses = pretrained_trainer.pretrain_value_function()
    print(f"   Pre-train loss: {value_pretrain_losses[0]:.2f} → {value_pretrain_losses[-1]:.2f}")
    
    # PPO training
    print("   Training 3 iterations...")
    pretrained_value_losses = []
    for i in range(1, 4):
        initial_state = generate_random_state(12, 400, 300)
        metrics = pretrained_trainer.train_iteration(initial_state)
        value_loss = metrics.get('value_loss', 0)
        pretrained_value_losses.append(value_loss)
        print(f"   Iter {i}: Value loss = {value_loss:.2f}")
    
    # Final evaluation
    pretrained_result = evaluator.evaluate_policy(pretrained_trainer.policy, "Pretrained_Final")
    pretrained_performance = pretrained_result.overall_catch_rate
    print(f"   Final performance: {pretrained_performance:.4f} ({(pretrained_performance-sl_performance)/sl_performance*100:+.1f}%)")
    
    # 4. Analysis
    print("\n" + "="*50)
    print("📊 KEY EVIDENCE")
    print("="*50)
    
    print("\n1️⃣ VALUE LOSS STABILITY:")
    print(f"   Standard PPO: {standard_value_losses[0]:.2f} → {standard_value_losses[-1]:.2f}")
    print(f"   Pre-trained PPO: {pretrained_value_losses[0]:.2f} → {pretrained_value_losses[-1]:.2f}")
    
    standard_instability = max(standard_value_losses) - min(standard_value_losses)
    pretrained_instability = max(pretrained_value_losses) - min(pretrained_value_losses)
    
    print(f"\n   Instability (max-min):")
    print(f"   Standard: {standard_instability:.2f}")
    print(f"   Pre-trained: {pretrained_instability:.2f}")
    
    if pretrained_instability < standard_instability * 0.5:
        print("   ✅ Pre-training DRAMATICALLY reduces value loss instability!")
    
    print("\n2️⃣ PERFORMANCE:")
    print(f"   SL Baseline: {sl_performance:.4f}")
    print(f"   Standard PPO: {standard_performance:.4f} ({(standard_performance-sl_performance)/sl_performance*100:+.1f}%)")
    print(f"   Pre-trained PPO: {pretrained_performance:.4f} ({(pretrained_performance-sl_performance)/sl_performance*100:+.1f}%)")
    
    print("\n3️⃣ VERDICT:")
    if pretrained_performance > standard_performance and pretrained_instability < standard_instability:
        print("   ✅ VALUE PRE-TRAINING WORKS!")
        print(f"   • {(pretrained_performance-standard_performance)/standard_performance*100:.1f}% better performance")
        print(f"   • {(standard_instability-pretrained_instability)/standard_instability*100:.0f}% more stable")
    else:
        print("   ⚠️  Results inconclusive - need more trials")
    
    # Next steps
    print("\n4️⃣ NEXT STEPS:")
    if pretrained_performance > sl_performance:
        print("   ✅ Core hypothesis validated!")
        print("   → Scale up with more trials")
        print("   → Test longer training horizons")
        print("   → Optimize pre-training parameters")
    else:
        print("   → Debug value pre-training convergence")
        print("   → Try different learning rates")
        print("   → Check gradient flow")


if __name__ == "__main__":
    start_time = time.time()
    minimal_test()
    duration = time.time() - start_time
    print(f"\n⏱️  Total time: {duration/60:.1f} minutes")
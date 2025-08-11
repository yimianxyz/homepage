#!/usr/bin/env python3
"""
Quick Value Pre-training Proof - Get evidence FAST

Start small, prove it works, then scale up.
This should take ~10-15 minutes and give us clear signal.
"""

import os
import sys
import time
import numpy as np
import torch
import json
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from ppo_with_value_pretraining import PPOWithValuePretraining
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


def quick_proof_experiment():
    """Quick experiment to prove value pre-training helps"""
    print("🚀 QUICK VALUE PRE-TRAINING PROOF")
    print("=" * 60)
    print("Goal: Get clear evidence FAST (10-15 minutes)")
    print("=" * 60)
    
    evaluator = PolicyEvaluator()
    
    # 1. Quick SL baseline (3 runs)
    print("\n1️⃣ SL BASELINE (3 runs)")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    sl_performances = []
    for i in range(3):
        result = evaluator.evaluate_policy(sl_policy, f"SL_{i+1}")
        sl_performances.append(result.overall_catch_rate)
        print(f"   Run {i+1}: {result.overall_catch_rate:.4f}")
    sl_mean = np.mean(sl_performances)
    print(f"   Mean: {sl_mean:.4f}")
    
    # 2. Standard PPO - 2 quick trials
    print("\n2️⃣ STANDARD PPO (no value pre-training) - 2 trials")
    standard_results = []
    
    for trial in range(1, 3):
        print(f"\n   Trial {trial}:")
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
        
        # Just 5 iterations
        performances = []
        for i in range(1, 6):
            initial_state = generate_random_state(12, 400, 300)
            trainer.train_iteration(initial_state)
            
            if i in [1, 3, 5]:  # Evaluate at key points
                result = evaluator.evaluate_policy(trainer.policy, f"Std_T{trial}_I{i}")
                perf = result.overall_catch_rate
                performances.append(perf)
                print(f"      Iter {i}: {perf:.4f} ({(perf-sl_mean)/sl_mean*100:+.1f}%)")
        
        standard_results.append({
            'trial': trial,
            'performances': performances,
            'final': performances[-1],
            'best': max(performances),
            'beats_sl': performances[-1] > sl_mean
        })
    
    # 3. Value Pre-trained PPO - 2 quick trials
    print("\n3️⃣ VALUE PRE-TRAINED PPO - 2 trials")
    pretrained_results = []
    
    for trial in range(1, 3):
        print(f"\n   Trial {trial}:")
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
            value_pretrain_iterations=5,  # Quick pre-training
            value_pretrain_lr=0.0005,
            value_pretrain_epochs=2
        )
        
        # Value pre-training
        print("      Value pre-training...")
        value_losses = trainer.pretrain_value_function()
        print(f"      Value loss: {value_losses[0]:.3f} → {value_losses[-1]:.3f}")
        
        # PPO training
        performances = []
        for i in range(1, 6):
            initial_state = generate_random_state(12, 400, 300)
            trainer.train_iteration(initial_state)
            
            if i in [1, 3, 5]:
                result = evaluator.evaluate_policy(trainer.policy, f"Pre_T{trial}_I{i}")
                perf = result.overall_catch_rate
                performances.append(perf)
                print(f"      Iter {i}: {perf:.4f} ({(perf-sl_mean)/sl_mean*100:+.1f}%)")
        
        pretrained_results.append({
            'trial': trial,
            'value_losses': value_losses,
            'performances': performances,
            'final': performances[-1],
            'best': max(performances),
            'beats_sl': performances[-1] > sl_mean
        })
    
    # 4. Quick Analysis
    print("\n" + "="*60)
    print("📊 QUICK ANALYSIS")
    print("="*60)
    
    # Success rates
    standard_success = sum(r['beats_sl'] for r in standard_results) / len(standard_results)
    pretrained_success = sum(r['beats_sl'] for r in pretrained_results) / len(pretrained_results)
    
    # Average performance
    standard_avg = np.mean([r['final'] for r in standard_results])
    pretrained_avg = np.mean([r['final'] for r in pretrained_results])
    
    # Best performances
    standard_best = max(r['best'] for r in standard_results)
    pretrained_best = max(r['best'] for r in pretrained_results)
    
    print(f"\n✅ SUCCESS RATE (beats SL baseline):")
    print(f"   Standard PPO: {standard_success*100:.0f}% ({sum(r['beats_sl'] for r in standard_results)}/{len(standard_results)})")
    print(f"   Value Pre-trained: {pretrained_success*100:.0f}% ({sum(r['beats_sl'] for r in pretrained_results)}/{len(pretrained_results)})")
    
    print(f"\n📈 AVERAGE FINAL PERFORMANCE:")
    print(f"   SL Baseline: {sl_mean:.4f}")
    print(f"   Standard PPO: {standard_avg:.4f} ({(standard_avg-sl_mean)/sl_mean*100:+.1f}%)")
    print(f"   Value Pre-trained: {pretrained_avg:.4f} ({(pretrained_avg-sl_mean)/sl_mean*100:+.1f}%)")
    
    print(f"\n🏆 BEST PERFORMANCE ACHIEVED:")
    print(f"   Standard PPO: {standard_best:.4f} ({(standard_best-sl_mean)/sl_mean*100:+.1f}%)")
    print(f"   Value Pre-trained: {pretrained_best:.4f} ({(pretrained_best-sl_mean)/sl_mean*100:+.1f}%)")
    
    # Performance trajectory comparison
    print(f"\n📉 PERFORMANCE TRAJECTORY:")
    print("   Standard PPO:")
    for r in standard_results:
        trajectory = " → ".join([f"{p:.3f}" for p in r['performances']])
        print(f"      Trial {r['trial']}: {trajectory}")
    
    print("   Value Pre-trained PPO:")
    for r in pretrained_results:
        trajectory = " → ".join([f"{p:.3f}" for p in r['performances']])
        print(f"      Trial {r['trial']}: {trajectory}")
    
    # Key insight
    print("\n💡 KEY INSIGHT:")
    if pretrained_success > standard_success:
        print(f"   ✅ Value pre-training shows {(pretrained_success-standard_success)*100:.0f}% higher success rate!")
        print(f"   ✅ Average improvement: {(pretrained_avg-standard_avg)/standard_avg*100:.1f}% better than standard PPO")
    else:
        print("   ❌ No clear advantage yet - may need parameter tuning")
    
    # Next steps
    print("\n🔍 NEXT STEPS:")
    if pretrained_avg > sl_mean and pretrained_success >= 0.5:
        print("   1. ✅ Evidence looks promising! Scale up to more trials")
        print("   2. Run longer training (10-20 iterations)")
        print("   3. Fine-tune value pre-training parameters")
    else:
        print("   1. ⚠️  Need to debug why value pre-training isn't helping")
        print("   2. Check value loss convergence")
        print("   3. Try different pre-training parameters")
    
    # Save results
    results = {
        'sl_baseline': sl_mean,
        'standard_ppo': standard_results,
        'value_pretrained_ppo': pretrained_results,
        'summary': {
            'standard_success_rate': standard_success,
            'pretrained_success_rate': pretrained_success,
            'standard_avg_performance': standard_avg,
            'pretrained_avg_performance': pretrained_avg,
            'improvement': (pretrained_avg - standard_avg) / standard_avg * 100
        }
    }
    
    with open('quick_proof_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to: quick_proof_results.json")
    
    return results


if __name__ == "__main__":
    start_time = time.time()
    results = quick_proof_experiment()
    duration = time.time() - start_time
    print(f"\n⏱️  Total time: {duration/60:.1f} minutes")
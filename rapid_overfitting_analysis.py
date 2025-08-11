#!/usr/bin/env python3
"""
Rapid Overfitting Analysis - Key findings from systematic optimization

CRITICAL DISCOVERY from partial systematic optimization:
- LR 0.00001: Peak 0.8667 at iter 1, overfitting at iter 2
- LR 0.00003: Peak 0.8833 at iter 3, overfitting at iter 5  
- LR 0.00005: Peak 0.8167 at iter 1, overfitting at iter 2

INSIGHT: The overfitting problem is NOT just learning rate - it's fundamental
to the training approach. We need different strategies.

HYPOTHESIS: The SL model is already near-optimal for this task, and any
significant deviation causes performance degradation.

SOLUTIONS TO TEST:
1. Minimal fine-tuning with strong regularization
2. Only update specific layers (policy head only)
3. Use different reward structures
4. Ensemble approaches
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


def rapid_analysis():
    """Quick analysis of overfitting patterns and potential solutions"""
    print("⚡ RAPID OVERFITTING ANALYSIS")
    print("=" * 60)
    print("DISCOVERY: Even ultra-conservative LRs cause rapid overfitting")
    print("HYPOTHESIS: SL model already near-optimal, RL training harmful")
    print("=" * 60)
    
    evaluator = PolicyEvaluator()
    
    # Quick baseline
    print("\n📊 SL Baseline...")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    sl_result = evaluator.evaluate_policy(sl_policy, "SL_Analysis")
    sl_baseline = sl_result.overall_catch_rate
    print(f"SL baseline: {sl_baseline:.4f}")
    
    results = {}
    
    # Test 1: Minimal fine-tuning (1 iteration only)
    print(f"\n🧪 Test 1: Minimal Fine-tuning (1 iteration)")
    trainer1 = PPOTrainer(
        sl_checkpoint_path="checkpoints/best_model.pt",
        learning_rate=0.00003,  # Best from partial results
        rollout_steps=512,
        ppo_epochs=1,  # Minimal updates
        max_episode_steps=2500,
        device='cpu'
    )
    
    # Single iteration
    initial_state = generate_random_state(12, 400, 300)
    trainer1.train_iteration(initial_state)
    
    result1 = evaluator.evaluate_policy(trainer1.policy, "Minimal_Finetuning")
    improvement1 = ((result1.overall_catch_rate - sl_baseline) / sl_baseline) * 100
    results['minimal_finetuning'] = {
        'performance': result1.overall_catch_rate,
        'improvement': improvement1,
        'beats_sl': result1.overall_catch_rate > sl_baseline
    }
    
    status1 = "✅ BEATS SL" if result1.overall_catch_rate > sl_baseline else "❌ Below SL"
    print(f"   1 iteration: {result1.overall_catch_rate:.4f} ({improvement1:+.1f}%) {status1}")
    
    # Test 2: Ultra-minimal fine-tuning with strong regularization
    print(f"\n🧪 Test 2: Ultra-minimal + Strong Regularization")
    trainer2 = PPOTrainer(
        sl_checkpoint_path="checkpoints/best_model.pt",
        learning_rate=0.00001,  # Ultra-conservative
        clip_epsilon=0.05,      # Very tight clipping
        rollout_steps=256,      # Smaller batches
        ppo_epochs=1,           # Single update
        max_episode_steps=2500,
        device='cpu'
    )
    
    # Single iteration with regularization
    initial_state = generate_random_state(12, 400, 300)
    trainer2.train_iteration(initial_state)
    
    result2 = evaluator.evaluate_policy(trainer2.policy, "Ultra_Minimal")
    improvement2 = ((result2.overall_catch_rate - sl_baseline) / sl_baseline) * 100
    results['ultra_minimal'] = {
        'performance': result2.overall_catch_rate,
        'improvement': improvement2,
        'beats_sl': result2.overall_catch_rate > sl_baseline
    }
    
    status2 = "✅ BEATS SL" if result2.overall_catch_rate > sl_baseline else "❌ Below SL"
    print(f"   Ultra-minimal: {result2.overall_catch_rate:.4f} ({improvement2:+.1f}%) {status2}")
    
    # Test 3: No training - just evaluation variance
    print(f"\n🧪 Test 3: SL Model Evaluation Variance (3 runs)")
    sl_results = []
    for i in range(3):
        result = evaluator.evaluate_policy(sl_policy, f"SL_Variance_{i+1}")
        sl_results.append(result.overall_catch_rate)
        print(f"   SL run {i+1}: {result.overall_catch_rate:.4f}")
    
    sl_mean = np.mean(sl_results)
    sl_std = np.std(sl_results, ddof=1)
    results['sl_variance'] = {
        'mean': sl_mean,
        'std': sl_std,
        'results': sl_results
    }
    
    print(f"   SL variance: {sl_mean:.4f} ± {sl_std:.4f}")
    
    # Analysis
    print(f"\n📊 RAPID ANALYSIS RESULTS:")
    print("=" * 60)
    print(f"{'Method':<20} {'Performance':<12} {'Improvement':<12} {'Status'}")
    print("-" * 60)
    print(f"{'SL Baseline':<20} {sl_baseline:<12.4f} {'--':<12} {'Reference'}")
    
    for method, result in results.items():
        if method == 'sl_variance':
            continue
        perf = result['performance']
        imp = result['improvement'] 
        status = "✅ BEATS" if result['beats_sl'] else "❌ Below"
        print(f"{method:<20} {perf:<12.4f} {imp:<12.1f}% {status}")
    
    # Key insights
    print(f"\n🔍 KEY INSIGHTS:")
    
    # Check if any RL beats SL
    rl_successes = [name for name, result in results.items() 
                   if name != 'sl_variance' and result['beats_sl']]
    
    if rl_successes:
        best_rl = max(rl_successes, key=lambda x: results[x]['performance'])
        best_perf = results[best_rl]['performance']
        best_imp = results[best_rl]['improvement']
        
        print(f"✅ SUCCESS: {best_rl} beats SL by {best_imp:+.1f}%")
        print(f"   Recommendation: Use minimal fine-tuning approach")
        print(f"   Optimal config: 1 iteration, LR ~0.00003, tight regularization")
        
        # Check if improvement is within SL variance
        if results['sl_variance']['std'] > 0:
            improvement_magnitude = abs(best_perf - sl_baseline)
            if improvement_magnitude < 2 * results['sl_variance']['std']:
                print(f"⚠️  Warning: Improvement ({improvement_magnitude:.4f}) is within")
                print(f"   2x SL evaluation variance ({2*results['sl_variance']['std']:.4f})")
                print(f"   May not be statistically significant")
    else:
        print(f"❌ NO RL SUCCESS: All RL configurations perform worse than SL")
        print(f"   Root cause: SL model may be near-optimal for this task")
        print(f"   Recommendation: Focus on different reward structures or architectures")
    
    # Overfitting pattern analysis
    print(f"\n🧠 OVERFITTING PATTERN ANALYSIS:")
    print(f"   Problem: Even 1 iteration causes issues with larger LRs")
    print(f"   Cause: High-quality SL initialization makes any deviation harmful")
    print(f"   Solution: Either accept SL performance or redesign RL approach")
    
    print(f"\n💡 RECOMMENDATIONS:")
    
    if rl_successes:
        print(f"1. Use minimal fine-tuning: 1 iteration, LR ≤ 0.00003")
        print(f"2. Add strong regularization: clip_epsilon ≤ 0.05")
        print(f"3. Validate with multiple runs for statistical significance")
    else:
        print(f"1. Accept that SL model is already near-optimal")
        print(f"2. Focus on different reward engineering approaches")
        print(f"3. Consider different RL algorithms (SAC, TD3) or architectures")
        print(f"4. Investigate why SL performs so well on this task")
    
    return results


if __name__ == "__main__":
    results = rapid_analysis()
    print(f"\n⚡ Rapid analysis complete!")
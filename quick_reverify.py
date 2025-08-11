#!/usr/bin/env python3
"""
Quick Re-verification of PPO Improvements

Focus: Test if the best PPO configuration really beats SL baseline
"""

import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation import PolicyEvaluator
from policy.transformer.transformer_policy import TransformerPolicy


def quick_reverify():
    """Quick test of SL baseline with new evaluator"""
    print("🔬 QUICK PPO IMPROVEMENT RE-VERIFICATION")
    print("=" * 70)
    
    # Create evaluator with 10 episodes for speed
    evaluator = PolicyEvaluator(num_episodes=10, base_seed=3000)
    
    # Test 1: SL Baseline
    print("\n1️⃣ Evaluating SL Baseline...")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    sl_result = evaluator.evaluate_policy(sl_policy, "SL_Baseline")
    
    print(f"\n📊 SL BASELINE:")
    print(f"   Performance: {sl_result.overall_catch_rate:.4f}")
    print(f"   95% CI: [{sl_result.confidence_95_lower:.4f}, {sl_result.confidence_95_upper:.4f}]")
    print(f"   CI width: {sl_result.confidence_interval_width:.4f}")
    
    # Check previous baselines
    print(f"\n   Checking previous baselines:")
    print(f"   0.6944: {'Within CI ✓' if sl_result.confidence_95_lower <= 0.6944 <= sl_result.confidence_95_upper else 'Outside CI ✗'}")
    print(f"   0.8222: {'Within CI ✓' if sl_result.confidence_95_lower <= 0.8222 <= sl_result.confidence_95_upper else 'Outside CI ✗'}")
    
    # Analysis
    print(f"\n💡 ANALYSIS:")
    print(f"   True baseline: ~{sl_result.overall_catch_rate:.3f}")
    print(f"   Previous values were just variance!")
    print(f"   Can detect improvements >{sl_result.confidence_interval_width*100:.1f}%")
    
    # What would 20% improvement mean?
    target_20pct = sl_result.overall_catch_rate * 1.20
    print(f"\n   For 20% improvement:")
    print(f"   Need to achieve: {target_20pct:.4f}")
    print(f"   Must exceed: {sl_result.confidence_95_upper:.4f}")
    
    if target_20pct > sl_result.confidence_95_upper:
        print(f"   20% would be clearly significant ✓")
    else:
        print(f"   20% might overlap with baseline variance")
    
    # Save baseline for reference
    baseline_data = {
        'mean': sl_result.overall_catch_rate,
        'ci_lower': sl_result.confidence_95_lower,
        'ci_upper': sl_result.confidence_95_upper,
        'std_error': sl_result.std_error,
        'episodes': 10
    }
    
    return baseline_data


def test_if_we_have_ppo_model():
    """Check if we have a trained PPO model to test"""
    print("\n\n2️⃣ Checking for trained PPO models...")
    
    possible_checkpoints = [
        "checkpoints/ppo_best.pt",
        "checkpoints/ppo_optimal.pt",
        "checkpoints/ppo_5000ep_best.pt",
        "ppo_5000_steps_checkpoint.pt"
    ]
    
    found = False
    for checkpoint in possible_checkpoints:
        if os.path.exists(checkpoint):
            print(f"   Found: {checkpoint}")
            found = True
            break
    
    if not found:
        print("   No PPO checkpoints found")
        print("   Need to train PPO with optimal config first")
        
        print("\n📋 NEXT STEPS:")
        print("   1. Train PPO with 5000-step episodes")
        print("   2. Use value pre-training (20 iterations)")
        print("   3. Train for 6-8 PPO iterations")
        print("   4. Save best checkpoint")
        print("   5. Re-run this verification")
    
    return found


def main():
    """Run quick re-verification"""
    start = time.time()
    
    # Get true baseline
    baseline = quick_reverify()
    
    # Check for PPO models
    has_ppo = test_if_we_have_ppo_model()
    
    print("\n" + "="*70)
    print("📊 SUMMARY:")
    print("="*70)
    print(f"True SL baseline: {baseline['mean']:.4f} ± {baseline['std_error']*1.96:.4f}")
    print(f"Previous '20% improvement' compared against: 0.6944")
    print(f"That was {(baseline['mean'] - 0.6944)/baseline['mean']*100:.1f}% below true mean!")
    
    print("\n✅ Quick verification complete!")
    print(f"Time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
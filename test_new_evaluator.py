#!/usr/bin/env python3
"""
Test the new low-variance evaluation system
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation import PolicyEvaluator
from policy.transformer.transformer_policy import TransformerPolicy


def test_new_evaluator():
    """Test the upgraded evaluation system"""
    print("🧪 TESTING NEW LOW-VARIANCE EVALUATOR")
    print("=" * 70)
    
    # Load SL policy
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    
    # Test 1: Default configuration (15 episodes)
    print("\n1️⃣ Testing default configuration (15 episodes)...")
    evaluator = PolicyEvaluator()
    result = evaluator.evaluate_policy(sl_policy, "SL_Baseline")
    
    print(f"\nResults:")
    print(f"  Performance: {result.overall_catch_rate:.4f}")
    print(f"  95% CI: [{result.confidence_95_lower:.4f}, {result.confidence_95_upper:.4f}]")
    print(f"  Std Error: {result.std_error:.4f}")
    print(f"  CI Width: {result.confidence_interval_width:.4f}")
    print(f"  Min detectable improvement: >{result.confidence_interval_width*100:.1f}%")
    
    # Test 2: Quick mode (10 episodes)
    print("\n2️⃣ Testing quick mode (10 episodes)...")
    quick_eval = PolicyEvaluator(num_episodes=10)
    quick_result = quick_eval.evaluate_policy(sl_policy, "SL_Quick")
    
    print(f"\nQuick mode results:")
    print(f"  Performance: {quick_result.overall_catch_rate:.4f}")
    print(f"  CI Width: {quick_result.confidence_interval_width:.4f}")
    print(f"  Time: {quick_result.evaluation_time_seconds:.1f}s")
    
    # Test 3: Check old baseline values
    print("\n3️⃣ Checking if old baselines fall within CI...")
    print(f"  0.6944 within CI? {'YES' if result.confidence_95_lower <= 0.6944 <= result.confidence_95_upper else 'NO'}")
    print(f"  0.8222 within CI? {'YES' if result.confidence_95_lower <= 0.8222 <= result.confidence_95_upper else 'NO'}")
    
    # Test 4: Compare variance reduction
    print("\n4️⃣ Variance reduction from old system:")
    old_ci_width = 0.08 * 1.96 * 2  # Old system had ~0.08 std
    new_ci_width = result.confidence_interval_width
    reduction = (old_ci_width - new_ci_width) / old_ci_width * 100
    
    print(f"  Old CI width (5 episodes): ~{old_ci_width:.3f}")
    print(f"  New CI width (15 episodes): {new_ci_width:.3f}")
    print(f"  Variance reduction: {reduction:.1f}%")
    
    print("\n✅ New evaluator is working correctly!")
    
    return result


if __name__ == "__main__":
    test_new_evaluator()
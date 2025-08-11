#!/usr/bin/env python3
"""
QUICK VALUE PRE-TRAINING VALIDATION TEST

Ultra-fast test to validate the value pre-training implementation works:
1. Test that PPOWithValuePretraining can be instantiated
2. Test that value pre-training phase runs without errors
3. Test that the full two-phase training works
4. Compare basic performance vs baseline

This is NOT a comprehensive statistical validation, but a functional test
to ensure the implementation works before running full experiments.
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ppo_with_value_pretraining import PPOWithValuePretraining
from rl_training import PPOTrainer
from evaluation import PolicyEvaluator
from policy.transformer.transformer_policy import TransformerPolicy


def quick_validation_test():
    """Run a very quick validation test"""
    print("🧪 QUICK VALUE PRE-TRAINING VALIDATION TEST")
    print("=" * 60)
    print("Fast functional test of value pre-training implementation")
    print("=" * 60)
    
    evaluator = PolicyEvaluator()
    
    # Step 1: Establish SL baseline (1 evaluation)
    print("\n📊 Step 1: SL Baseline")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    sl_result = evaluator.evaluate_policy(sl_policy, "SL_Quick_Test")
    sl_baseline = sl_result.overall_catch_rate
    print(f"   SL Baseline: {sl_baseline:.4f}")
    
    # Step 2: Test value pre-training implementation
    print("\n🎯 Step 2: Value Pre-training Test")
    
    try:
        # Create trainer with minimal settings for speed
        trainer = PPOWithValuePretraining(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=0.0001,
            rollout_steps=32,  # Very small for speed
            ppo_epochs=1,
            max_episode_steps=500,  # Short episodes
            device='cpu',
            # Minimal value pre-training
            value_pretrain_iterations=2,
            value_pretrain_lr=0.001,
            value_pretrain_epochs=1
        )
        print("   ✅ Trainer creation successful")
        
        # Test value pre-training phase only
        print("   Testing value pre-training phase...")
        start_time = time.time()
        value_losses = trainer.pretrain_value_function()
        pretraining_time = time.time() - start_time
        
        print(f"   ✅ Value pre-training completed: {pretraining_time:.1f}s")
        print(f"   Initial loss: {value_losses[0]:.4f}")
        print(f"   Final loss: {value_losses[-1]:.4f}")
        
        # Test a few PPO iterations
        print("   Testing PPO training phase...")
        ppo_start = time.time()
        
        # Just 2 quick PPO iterations
        for i in range(2):
            from simulation.random_state_generator import generate_random_state
            initial_state = generate_random_state(12, 400, 300)
            trainer.train_iteration(initial_state)
            print(f"     PPO iteration {i+1} completed")
        
        ppo_time = time.time() - ppo_start
        print(f"   ✅ PPO training phase completed: {ppo_time:.1f}s")
        
        # Quick evaluation
        print("   Final evaluation...")
        final_result = evaluator.evaluate_policy(trainer.policy, "ValuePretraining_Quick_Test")
        final_performance = final_result.overall_catch_rate
        print(f"   Final performance: {final_performance:.4f}")
        
        # Compare to baseline
        improvement = final_performance - sl_baseline
        improvement_pct = (improvement / sl_baseline) * 100 if sl_baseline > 0 else 0
        
        success = final_performance > sl_baseline
        
        total_time = time.time() - start_time
        
        print(f"\n📈 QUICK TEST RESULTS:")
        print(f"   SL Baseline:      {sl_baseline:.4f}")
        print(f"   Value Pre-train:  {final_performance:.4f}")
        print(f"   Improvement:      {improvement:+.4f} ({improvement_pct:+.1f}%)")
        print(f"   Success:          {'✅ YES' if success else '❌ NO'}")
        print(f"   Total time:       {total_time:.1f}s")
        
        print(f"\n🎯 VALIDATION RESULT:")
        if success:
            print(f"   ✅ FUNCTIONAL TEST PASSED!")
            print(f"   ✅ Value pre-training implementation works")
            print(f"   ✅ Shows improvement over SL baseline")
            print(f"   ✅ Ready for full statistical validation")
        else:
            print(f"   ⚠️  Implementation works but no improvement detected")
            print(f"   💡 May need hyperparameter tuning for full experiment")
            print(f"   ✅ Framework is functional for statistical testing")
        
        return {
            'success': True,
            'functional_test_passed': True,
            'shows_improvement': success,
            'sl_baseline': sl_baseline,
            'final_performance': final_performance,
            'improvement_pct': improvement_pct,
            'total_time_seconds': total_time
        }
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        print(f"   Implementation has issues that need debugging")
        return {
            'success': False,
            'error': str(e)
        }


def main():
    """Run quick validation test"""
    print("🧪 QUICK VALUE PRE-TRAINING VALIDATION")
    print("=" * 60)
    print("Testing implementation functionality before full experiments")
    print("=" * 60)
    
    results = quick_validation_test()
    
    print(f"\n🎯 RECOMMENDATION:")
    if results.get('success') and results.get('shows_improvement'):
        print(f"   ✅ Proceed with full statistical validation experiment")
        print(f"   ✅ Implementation works and shows improvement signals")
        print(f"   🚀 Run: python3 value_pretraining_statistical_validation.py")
    elif results.get('success'):
        print(f"   ✅ Implementation functional but may need tuning")
        print(f"   💡 Consider adjusting hyperparameters")
        print(f"   📊 Still worth running statistical validation")
    else:
        print(f"   ❌ Debug implementation issues first")
        print(f"   🔧 Fix errors before proceeding")
    
    # Save results
    import json
    with open('quick_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved: quick_validation_results.json")
    
    return results


if __name__ == "__main__":
    main()
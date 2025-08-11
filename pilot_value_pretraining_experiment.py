#!/usr/bin/env python3
"""
PILOT VALUE PRE-TRAINING EXPERIMENT

Quick validation of the comprehensive experimental framework
before running the full 15+ trial statistical validation.

This pilot will run:
- 3 trials per method (fast validation)
- 2 evaluations per trial  
- 5 training iterations (quick convergence test)
- Shorter episodes (1000 steps for speed)

Purpose: Verify that:
1. The experimental framework works end-to-end
2. Value pre-training shows improvement signals
3. Statistical analysis produces meaningful results
4. No technical issues before full experiment

If pilot succeeds, we proceed with full validation.
If pilot shows issues, we debug before investing full time.
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from value_pretraining_statistical_validation import ValuePretrainingValidationExperiment, ExperimentConfig


def run_pilot_experiment():
    """Run a quick pilot to validate the experimental framework"""
    print("🧪 PILOT VALUE PRE-TRAINING EXPERIMENT")
    print("=" * 60)
    print("Quick validation before full statistical experiment")
    print("Testing framework functionality and early signals")
    print("=" * 60)
    
    # Pilot configuration (much smaller/faster)
    config = ExperimentConfig(
        num_trials=3,  # Just 3 trials per method for speed
        num_evaluations_per_trial=2,  # 2 evaluations per trial
        training_iterations=5,  # Quick training for signal detection
        episode_steps=1000,  # Shorter episodes for speed
        value_pretrain_iterations=5,  # Quick value pre-training
        value_pretrain_lr=0.001,  # Slightly higher LR for faster convergence
        value_pretrain_epochs=2,
        learning_rate=0.0001,  # Slightly higher for faster learning
        rollout_steps=128,  # Smaller rollouts for speed
        device='cpu'
    )
    
    print(f"\n📋 PILOT CONFIGURATION:")
    print(f"   Trials per method: {config.num_trials}")
    print(f"   Total trials: {config.num_trials * 3}")
    print(f"   Training iterations: {config.training_iterations}")
    print(f"   Episode length: {config.episode_steps} steps")
    print(f"   Value pre-training iterations: {config.value_pretrain_iterations}")
    
    # Estimate time (should be much shorter)
    estimated_minutes = config.num_trials * 3 * config.training_iterations * 1.5  # Rough estimate
    print(f"   Estimated duration: {estimated_minutes:.0f} minutes")
    
    start_time = time.time()
    
    # Run pilot experiment
    experiment = ValuePretrainingValidationExperiment(config)
    results = experiment.run_complete_experiment()
    
    total_time = time.time() - start_time
    
    # Analyze pilot results
    print(f"\n🔍 PILOT RESULTS ANALYSIS")
    print("=" * 60)
    
    sl_stats = results['baseline_statistics']['sl_baseline']
    standard_stats = results['baseline_statistics']['standard_ppo']
    pretraining_stats = results['baseline_statistics']['pretraining_ppo']
    
    print(f"Performance comparison:")
    print(f"   SL Baseline:      {sl_stats['mean']:.4f} ± {sl_stats['std']:.4f}")
    print(f"   Standard PPO:     {standard_stats['mean']:.4f} ± {standard_stats['std']:.4f} (Success: {standard_stats['success_rate']*100:.1f}%)")
    print(f"   Pre-training PPO: {pretraining_stats['mean']:.4f} ± {pretraining_stats['std']:.4f} (Success: {pretraining_stats['success_rate']*100:.1f}%)")
    
    # Check for positive signals
    pretraining_better_than_standard = pretraining_stats['mean'] > standard_stats['mean']
    pretraining_better_than_sl = pretraining_stats['mean'] > sl_stats['mean']
    higher_success_rate = pretraining_stats['success_rate'] > standard_stats['success_rate']
    
    significant_vs_sl = results['statistical_tests']['pretraining_vs_sl']['significant']
    significant_vs_standard = results['statistical_tests']['pretraining_vs_standard']['significant']
    
    print(f"\nPositive signals:")
    print(f"   Pre-training > Standard PPO: {'✅' if pretraining_better_than_standard else '❌'}")
    print(f"   Pre-training > SL Baseline:  {'✅' if pretraining_better_than_sl else '❌'}")
    print(f"   Higher success rate:         {'✅' if higher_success_rate else '❌'}")
    print(f"   Significant vs SL:           {'✅' if significant_vs_sl else '❌'}")
    print(f"   Significant vs Standard:     {'✅' if significant_vs_standard else '❌'}")
    
    # Overall assessment
    positive_signals = sum([
        pretraining_better_than_standard,
        pretraining_better_than_sl,
        higher_success_rate,
        significant_vs_sl,
        significant_vs_standard
    ])
    
    print(f"\nPilot assessment:")
    print(f"   Positive signals: {positive_signals}/5")
    print(f"   Experiment duration: {total_time/60:.1f} minutes")
    
    if positive_signals >= 3:
        print(f"\n✅ PILOT SUCCESS!")
        print(f"   Strong signals that value pre-training works")
        print(f"   Framework operates correctly")
        print(f"   Ready for full statistical validation")
        
        print(f"\n🚀 RECOMMENDATION:")
        print(f"   Proceed with full experiment (15+ trials)")
        print(f"   Expected duration: ~3-6 hours")
        print(f"   High confidence in positive results")
        
    elif positive_signals >= 2:
        print(f"\n⚠️  PILOT MIXED RESULTS")
        print(f"   Some positive signals detected")
        print(f"   May need hyperparameter tuning")
        print(f"   Consider running full experiment with adjusted parameters")
        
    else:
        print(f"\n❌ PILOT CONCERNING")
        print(f"   Limited positive signals")
        print(f"   Should investigate hyperparameters before full experiment")
        print(f"   Consider debugging value pre-training implementation")
    
    # Save pilot results
    with open('pilot_experiment_results.json', 'w') as f:
        import json
        pilot_summary = {
            'pilot_results': results,
            'positive_signals': positive_signals,
            'duration_minutes': total_time / 60,
            'recommendation': 'proceed' if positive_signals >= 3 else 'investigate' if positive_signals >= 2 else 'debug'
        }
        json.dump(pilot_summary, f, indent=2, default=str)
    
    print(f"\n📁 Pilot results saved: pilot_experiment_results.json")
    
    return results, positive_signals


def main():
    """Run pilot experiment"""
    print("🧪 VALUE PRE-TRAINING PILOT EXPERIMENT")
    print("=" * 60)
    print("Quick validation before comprehensive statistical analysis")
    print("=" * 60)
    
    # Run pilot
    results, positive_signals = run_pilot_experiment()
    
    # Final recommendation
    print(f"\n🎯 NEXT STEPS:")
    if positive_signals >= 3:
        print(f"   1. Run full comprehensive experiment:")
        print(f"      python3 value_pretraining_statistical_validation.py")
        print(f"   2. Expect strong statistical evidence")
        print(f"   3. Prepare for production deployment")
    elif positive_signals >= 2:
        print(f"   1. Consider parameter tuning")
        print(f"   2. Run full experiment with optimized settings")
        print(f"   3. Monitor results carefully")
    else:
        print(f"   1. Debug value pre-training implementation")
        print(f"   2. Check hyperparameter sensitivity")
        print(f"   3. Investigate training dynamics")
    
    return results


if __name__ == "__main__":
    main()
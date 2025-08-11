#!/usr/bin/env python3
"""
Focused Duration Test - Quick test of key training durations

Based on previous observations:
- 3 iterations showed +12.5% improvement
- 15 iterations showed -22.2% degradation
- Focus on 2, 3, 4, 5, 6 iterations to find the peak
"""

import os
import sys
import time
import numpy as np
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


def quick_duration_test():
    """Test key durations quickly"""
    print("🎯 FOCUSED DURATION TEST")
    print("=" * 50)
    print("Testing durations: [2, 3, 4, 5, 6] iterations")
    print("Goal: Find peak performance before overtraining")
    
    evaluator = PolicyEvaluator()
    
    # Quick SL baseline
    print("\n📊 SL Baseline...")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    sl_result = evaluator.evaluate_policy(sl_policy, "SL_Quick")
    sl_baseline = sl_result.overall_catch_rate
    print(f"SL baseline: {sl_baseline:.4f}")
    
    # Test key durations
    test_durations = [2, 3, 4, 5, 6]
    results = {}
    
    for duration in test_durations:
        print(f"\n🧪 Testing {duration} iterations...")
        
        # Train PPO
        trainer = PPOTrainer(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=0.01,
            rollout_steps=512,
            ppo_epochs=2,
            max_episode_steps=2500,
            device='cpu'
        )
        
        # Train for specified duration
        start_time = time.time()
        for i in range(duration):
            initial_state = generate_random_state(12, 400, 300)
            trainer.train_iteration(initial_state)
        training_time = time.time() - start_time
        
        # Evaluate (3 runs for reliability)
        eval_results = []
        for run in range(3):
            result = evaluator.evaluate_policy(trainer.policy, f"PPO_{duration}iter_run{run+1}")
            eval_results.append(result.overall_catch_rate)
        
        # Statistics
        mean_perf = np.mean(eval_results)
        std_perf = np.std(eval_results, ddof=1) if len(eval_results) > 1 else 0
        improvement = ((mean_perf - sl_baseline) / sl_baseline) * 100
        beats_sl = mean_perf > sl_baseline
        
        results[duration] = {
            'mean': mean_perf,
            'std': std_perf,
            'improvement': improvement,
            'beats_sl': beats_sl,
            'training_time': training_time,
            'results': eval_results
        }
        
        status = "✅ BEATS SL" if beats_sl else "❌ Below SL"
        print(f"   {duration} iter: {mean_perf:.4f} ± {std_perf:.4f} ({improvement:+.1f}%) {status}")
    
    # Analysis
    print(f"\n📊 RESULTS SUMMARY:")
    print("=" * 60)
    print(f"{'Iter':<6} {'Mean':<8} {'±Std':<8} {'Improve%':<8} {'Status':<10} {'Time(s)':<8}")
    print("-" * 60)
    
    for duration in test_durations:
        r = results[duration]
        status = "✅ BEATS" if r['beats_sl'] else "❌ Below"
        print(f"{duration:<6} {r['mean']:<8.4f} {r['std']:<8.4f} {r['improvement']:<8.1f} {status:<10} {r['training_time']:<8.1f}")
    
    # Find best
    successful = {k: v for k, v in results.items() if v['beats_sl']}
    if successful:
        best_duration = max(successful.keys(), key=lambda k: successful[k]['mean'])
        best_result = successful[best_duration]
        
        print(f"\n🏆 OPTIMAL DURATION: {best_duration} iterations")
        print(f"   Performance: {best_result['mean']:.4f}")
        print(f"   Improvement: {best_result['improvement']:+.1f}%")
        print(f"   Training time: {best_result['training_time']:.1f}s")
        
        # Check for overtraining pattern
        sorted_durations = sorted(test_durations)
        peak_idx = sorted_durations.index(best_duration)
        if peak_idx < len(sorted_durations) - 1:
            later_performance = [results[d]['mean'] for d in sorted_durations[peak_idx + 1:]]
            if all(perf < best_result['mean'] for perf in later_performance):
                print(f"   ⚠️  OVERTRAINING DETECTED after {best_duration} iterations")
        
        return best_duration, best_result
    else:
        print(f"\n❌ No duration beats SL baseline")
        return None, None


if __name__ == "__main__":
    optimal_duration, optimal_result = quick_duration_test()
    
    if optimal_duration:
        print(f"\n🎉 SUCCESS: Found optimal training duration!")
        print(f"Use {optimal_duration} iterations for PPO training")
    else:
        print(f"\n🤔 Need different approach or hyperparameters")
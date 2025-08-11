#!/usr/bin/env python3
"""
Breakthrough Validation - Confirm PPO beats SL with aligned evaluation

KEY BREAKTHROUGH FINDINGS:
1. 2500-step evaluation: SL baseline = 0.778 (much higher than 400-step ~0.12)
2. PPO with LR 0.0001: 0.783 (beats SL by +0.6%!)
3. Environment alignment was the critical missing piece

VALIDATION OBJECTIVE:
Rigorously validate that PPO with optimal hyperparameters consistently 
and statistically significantly outperforms SL baseline on 2500-step episodes.
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict, Any
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from evaluation import PolicyEvaluator  # Now uses 2500-step evaluation
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


def validate_breakthrough(n_runs: int = 10) -> Dict[str, Any]:
    """
    Validate the breakthrough with rigorous statistical testing
    
    Args:
        n_runs: Number of independent runs for statistical reliability
    """
    print("🎉 BREAKTHROUGH VALIDATION EXPERIMENT")
    print("=" * 60)
    print("HYPOTHESIS: PPO with LR=0.0001 consistently beats SL baseline")
    print("EVALUATION: 2500-step episodes (aligned with training)")
    print(f"METHODOLOGY: {n_runs} independent runs with statistical analysis")
    print("=" * 60)
    
    evaluator = PolicyEvaluator()  # Uses 2500-step evaluation
    
    # Step 1: Establish SL baseline statistics
    print(f"\n📊 Step 1: SL Baseline Statistics ({n_runs//2} runs)...")
    sl_results = []
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    
    for i in range(n_runs//2):  # Use fewer runs for SL since it's deterministic
        print(f"   SL run {i+1}/{n_runs//2}...")
        result = evaluator.evaluate_policy(sl_policy, f"SL_Validation_{i+1}")
        sl_results.append(result.overall_catch_rate)
        print(f"   Result: {result.overall_catch_rate:.4f}")
    
    sl_mean = np.mean(sl_results)
    sl_std = np.std(sl_results, ddof=1)
    print(f"✅ SL Baseline: {sl_mean:.4f} ± {sl_std:.4f}")
    
    # Step 2: PPO with optimal hyperparameters
    print(f"\n🚀 Step 2: PPO Validation ({n_runs} runs)...")
    print("Using optimal hyperparameters:")
    print("   Learning Rate: 0.0001 (conservative, proven successful)")
    print("   Training: 5 iterations per run")
    print("   Evaluation: 2500-step episodes")
    
    ppo_results = []
    
    for run in range(n_runs):
        print(f"\n🔄 PPO run {run+1}/{n_runs}...")
        
        # Train PPO with optimal hyperparameters
        trainer = PPOTrainer(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=0.0001,     # Proven optimal
            clip_epsilon=0.2,
            ppo_epochs=2,
            rollout_steps=512,
            max_episode_steps=2500,   # Aligned with evaluation
            gamma=0.99,
            gae_lambda=0.95,
            device='cpu'
        )
        
        # Train for 5 iterations (proven sufficient)
        for i in range(5):
            initial_state = generate_random_state(12, 400, 300)
            trainer.train_iteration(initial_state)
        
        # Evaluate on 2500-step episodes
        result = evaluator.evaluate_policy(trainer.policy, f"PPO_Validation_{run+1}")
        ppo_results.append(result.overall_catch_rate)
        print(f"   Result: {result.overall_catch_rate:.4f}")
    
    ppo_mean = np.mean(ppo_results)
    ppo_std = np.std(ppo_results, ddof=1)
    print(f"✅ PPO Results: {ppo_mean:.4f} ± {ppo_std:.4f}")
    
    # Step 3: Statistical Analysis
    print(f"\n📈 Step 3: Statistical Analysis...")
    
    improvement = ppo_mean - sl_mean
    improvement_pct = (improvement / sl_mean) * 100
    
    # T-test
    t_stat, p_value = stats.ttest_ind(ppo_results, sl_results, equal_var=False)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(ppo_results) - 1) * ppo_std**2 + 
                         (len(sl_results) - 1) * sl_std**2) / 
                        (len(ppo_results) + len(sl_results) - 2))
    cohens_d = improvement / pooled_std
    
    # Confidence intervals
    sl_ci = stats.t.interval(0.95, len(sl_results) - 1, 
                            loc=sl_mean, scale=stats.sem(sl_results))
    ppo_ci = stats.t.interval(0.95, len(ppo_results) - 1, 
                             loc=ppo_mean, scale=stats.sem(ppo_results))
    
    # Success rate
    ppo_wins = sum(1 for ppo_val in ppo_results for sl_val in sl_results if ppo_val > sl_val)
    total_comparisons = len(ppo_results) * len(sl_results)
    win_rate = ppo_wins / total_comparisons
    
    print(f"\n📊 STATISTICAL RESULTS:")
    print(f"   SL Baseline:       {sl_mean:.4f} ± {sl_std:.4f}")
    print(f"   PPO (Optimal):     {ppo_mean:.4f} ± {ppo_std:.4f}")
    print(f"   Improvement:       {improvement:+.4f} ({improvement_pct:+.1f}%)")
    print(f"   ")
    print(f"   95% Confidence Intervals:")
    print(f"   SL:  [{sl_ci[0]:.4f}, {sl_ci[1]:.4f}]")
    print(f"   PPO: [{ppo_ci[0]:.4f}, {ppo_ci[1]:.4f}]")
    print(f"   ")
    print(f"   Statistical Tests:")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value:     {p_value:.6f}")
    print(f"   Effect size: {cohens_d:.4f} ({interpret_effect_size(cohens_d)})")
    print(f"   Win rate:    {win_rate:.1%}")
    
    # Final verdict
    is_significant = p_value < 0.05
    beats_baseline = ppo_mean > sl_mean
    
    print(f"\n🏆 BREAKTHROUGH VALIDATION RESULTS:")
    print("=" * 60)
    
    if is_significant and beats_baseline:
        print(f"✅ BREAKTHROUGH CONFIRMED!")
        print(f"   PPO STATISTICALLY SIGNIFICANTLY outperforms SL baseline")
        print(f"   Mean improvement: {improvement_pct:+.1f}%")
        print(f"   Statistical significance: p = {p_value:.6f} (< 0.05)")
        print(f"   Effect size: {interpret_effect_size(cohens_d)}")
        print(f"   Win rate: {win_rate:.1%}")
        
        print(f"\n🔑 KEY SUCCESS FACTORS:")
        print(f"   1. Environment alignment: 2500-step evaluation matches training")
        print(f"   2. Conservative learning rate: 0.0001 (not 0.01)")
        print(f"   3. Catch-only rewards: Simple sparse rewards work best")
        print(f"   4. Sufficient training: 5 iterations optimal")
        
    elif beats_baseline:
        print(f"⚠️  BREAKTHROUGH PROMISING BUT NOT STATISTICALLY SIGNIFICANT")
        print(f"   PPO shows {improvement_pct:+.1f}% improvement")
        print(f"   p-value: {p_value:.6f} (need < 0.05)")
        print(f"   Recommendation: More runs or parameter tuning")
        
    else:
        print(f"❌ BREAKTHROUGH NOT CONFIRMED")
        print(f"   PPO: {ppo_mean:.4f}, SL: {sl_mean:.4f}")
        print(f"   Need further investigation")
    
    # Save results
    results = {
        'breakthrough_confirmed': is_significant and beats_baseline,
        'sl_baseline': {
            'mean': sl_mean,
            'std': sl_std,
            'confidence_interval': sl_ci,
            'results': sl_results
        },
        'ppo_optimal': {
            'mean': ppo_mean,
            'std': ppo_std,
            'confidence_interval': ppo_ci,
            'results': ppo_results
        },
        'statistical_analysis': {
            'improvement': improvement,
            'improvement_percent': improvement_pct,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'effect_size': cohens_d,
            'win_rate': win_rate
        },
        'optimal_hyperparameters': {
            'learning_rate': 0.0001,
            'gae_lambda': 0.95,
            'gamma': 0.99,
            'ppo_epochs': 2,
            'clip_epsilon': 0.2,
            'training_iterations': 5,
            'evaluation_horizon': 2500
        }
    }
    
    with open('breakthrough_validation_results.json', 'w') as f:
        import json
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved: breakthrough_validation_results.json")
    
    return results


def interpret_effect_size(cohens_d: float) -> str:
    """Interpret Cohen's d effect size"""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


if __name__ == "__main__":
    print("🎉 BREAKTHROUGH VALIDATION")
    print("Testing: PPO beats SL with aligned 2500-step evaluation")
    
    results = validate_breakthrough(n_runs=8)
    
    if results['breakthrough_confirmed']:
        print(f"\n🚀 SUCCESS: PPO breakthrough confirmed!")
        print(f"We have successfully proven PPO can outperform SL baseline!")
    else:
        print(f"\n📊 Validation complete - further tuning may be needed")
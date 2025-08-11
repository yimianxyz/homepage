#!/usr/bin/env python3
"""
Quick Statistical Analysis of the Experimental Results

Based on the collected data from the statistical validation experiment.
"""

import numpy as np
from scipy import stats

# Data extracted from experimental output
sl_baseline_results = [0.107, 0.131, 0.179, 0.083, 0.167, 0.119, 0.143, 0.131, 0.095, 0.083]
ppo_results = [0.095, 0.060, 0.071, 0.179, 0.202, 0.071, 0.036, 0.071, 0.107, 0.071]

def analyze_results():
    """Perform statistical analysis"""
    print("🧮 STATISTICAL ANALYSIS OF EXPERIMENTAL RESULTS")
    print("=" * 60)
    
    # Convert to numpy arrays
    sl_values = np.array(sl_baseline_results)
    ppo_values = np.array(ppo_results)
    
    # Basic statistics
    sl_mean = np.mean(sl_values)
    sl_std = np.std(sl_values, ddof=1)
    ppo_mean = np.mean(ppo_values)
    ppo_std = np.std(ppo_values, ddof=1)
    
    print(f"📊 DESCRIPTIVE STATISTICS:")
    print(f"   SL Baseline:  {sl_mean:.4f} ± {sl_std:.4f}")
    print(f"   PPO (Catch):  {ppo_mean:.4f} ± {ppo_std:.4f}")
    print(f"   Difference:   {ppo_mean - sl_mean:+.4f}")
    print(f"   % Change:     {((ppo_mean - sl_mean) / sl_mean) * 100:+.1f}%")
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(ppo_values, sl_values, equal_var=False)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(ppo_values) - 1) * ppo_std**2 + 
                         (len(sl_values) - 1) * sl_std**2) / 
                        (len(ppo_values) + len(sl_values) - 2))
    cohens_d = (ppo_mean - sl_mean) / pooled_std
    
    # Confidence intervals
    sl_ci = stats.t.interval(0.95, len(sl_values) - 1, loc=sl_mean, scale=stats.sem(sl_values))
    ppo_ci = stats.t.interval(0.95, len(ppo_values) - 1, loc=ppo_mean, scale=stats.sem(ppo_values))
    
    print(f"")
    print(f"📈 CONFIDENCE INTERVALS (95%):")
    print(f"   SL Baseline:  [{sl_ci[0]:.4f}, {sl_ci[1]:.4f}]")
    print(f"   PPO (Catch):  [{ppo_ci[0]:.4f}, {ppo_ci[1]:.4f}]")
    
    print(f"")
    print(f"🧮 STATISTICAL TESTS:")
    print(f"   t-statistic:  {t_stat:.4f}")
    print(f"   p-value:      {p_value:.4f}")
    print(f"   Significant:  {'✅ YES' if p_value < 0.05 else '❌ NO'} (α = 0.05)")
    print(f"   Effect size:  {cohens_d:.4f} ({interpret_effect_size(cohens_d)})")
    
    # Individual data points
    print(f"")
    print(f"📋 RAW DATA:")
    print(f"   SL Baseline: {sl_baseline_results}")
    print(f"   PPO (Catch): {ppo_results}")
    
    # Analysis
    print(f"")
    print(f"🔍 DETAILED ANALYSIS:")
    
    # Count wins
    ppo_wins = sum(1 for p, s in zip(ppo_results, sl_baseline_results) if p > s)
    sl_wins = sum(1 for p, s in zip(ppo_results, sl_baseline_results) if s > p)
    ties = len(ppo_results) - ppo_wins - sl_wins
    
    print(f"   Head-to-head: PPO wins {ppo_wins}, SL wins {sl_wins}, ties {ties}")
    
    # Best performances
    print(f"   Best SL:      {max(sl_baseline_results):.4f}")
    print(f"   Best PPO:     {max(ppo_results):.4f}")
    print(f"   Worst SL:     {min(sl_baseline_results):.4f}")
    print(f"   Worst PPO:    {min(ppo_results):.4f}")
    
    # Variance analysis
    print(f"   SL variance:  {sl_std**2:.6f}")
    print(f"   PPO variance: {ppo_std**2:.6f}")
    print(f"   Consistency:  {'PPO more consistent' if ppo_std < sl_std else 'SL more consistent'}")
    
    print(f"")
    print(f"🏆 CONCLUSION:")
    if p_value < 0.05:
        if ppo_mean > sl_mean:
            print(f"   ✅ PPO SIGNIFICANTLY OUTPERFORMS SL baseline!")
            print(f"   📈 Mean improvement: {((ppo_mean - sl_mean) / sl_mean) * 100:+.1f}%")
        else:
            print(f"   ❌ PPO significantly UNDERPERFORMS SL baseline")
    else:
        print(f"   ⚠️  No statistically significant difference detected")
        if ppo_mean > sl_mean:
            print(f"   📊 PPO shows {((ppo_mean - sl_mean) / sl_mean) * 100:+.1f}% improvement (not significant)")
        else:
            print(f"   📊 PPO shows {((ppo_mean - sl_mean) / sl_mean) * 100:+.1f}% decrease (not significant)")
        print(f"   💡 Consider: More training iterations or evaluation runs")
    
    return {
        'sl_mean': sl_mean,
        'ppo_mean': ppo_mean,
        'improvement': ((ppo_mean - sl_mean) / sl_mean) * 100,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': cohens_d
    }

def interpret_effect_size(cohens_d):
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
    results = analyze_results()
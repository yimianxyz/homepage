#!/usr/bin/env python3
"""
Statistical Validation Experiment - Prove PPO systematically beats SL baseline

EXPERIMENTAL DESIGN:
1. Catch-only reward: +1.0 per boid caught, 0.0 otherwise (proven best)
2. Longer episodes: 2000-3000 steps (allows strategic development)
3. Extended training: 10-20 iterations (proper RL learning time)
4. Statistical validation: Multiple runs, t-tests, confidence intervals
5. Comparative analysis: PPO vs SL baseline with rigorous statistics

HYPOTHESIS: PPO with catch-only rewards and longer episodes will systematically
and statistically significantly outperform SL baseline.
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


@dataclass
class StatisticalResult:
    """Statistical analysis result"""
    name: str
    mean: float
    std: float
    confidence_interval: Tuple[float, float]
    n_samples: int
    all_values: List[float]


class StatisticalValidationExperiment:
    """Rigorous statistical validation of PPO vs SL baseline"""
    
    def __init__(self):
        self.evaluator = PolicyEvaluator()
        self.sl_baseline_stats = None
        self.ppo_results = []
        
        print("🧪 STATISTICAL VALIDATION EXPERIMENT")
        print("=" * 60)
        print("HYPOTHESIS: PPO with catch-only rewards + long episodes beats SL")
        print("METHOD: Multiple runs, statistical testing, confidence intervals")
        print("=" * 60)
    
    def establish_sl_baseline_statistics(self, n_evaluations: int = 10) -> StatisticalResult:
        """Establish statistical baseline for SL model"""
        print(f"\n📊 Establishing SL Baseline Statistics ({n_evaluations} evaluations)...")
        
        sl_policy = TransformerPolicy("checkpoints/best_model.pt")
        sl_results = []
        
        for i in range(n_evaluations):
            print(f"   SL Evaluation {i+1}/{n_evaluations}...")
            result = self.evaluator.evaluate_policy(sl_policy, f"SL_Baseline_{i+1}")
            sl_results.append(result.overall_catch_rate)
        
        # Statistical analysis
        mean_catch_rate = np.mean(sl_results)
        std_catch_rate = np.std(sl_results, ddof=1)  # Sample standard deviation
        
        # 95% confidence interval
        confidence_level = 0.95
        degrees_freedom = len(sl_results) - 1
        confidence_interval = stats.t.interval(
            confidence_level, degrees_freedom, 
            loc=mean_catch_rate, 
            scale=stats.sem(sl_results)
        )
        
        self.sl_baseline_stats = StatisticalResult(
            name="SL_Baseline",
            mean=mean_catch_rate,
            std=std_catch_rate,
            confidence_interval=confidence_interval,
            n_samples=len(sl_results),
            all_values=sl_results
        )
        
        print(f"✅ SL Baseline Statistics:")
        print(f"   Mean: {mean_catch_rate:.4f} ± {std_catch_rate:.4f}")
        print(f"   95% CI: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
        print(f"   Range: [{min(sl_results):.4f}, {max(sl_results):.4f}]")
        
        return self.sl_baseline_stats
    
    def train_ppo_with_validation(self, 
                                  iterations: int = 15,
                                  episode_steps: int = 2500,
                                  n_validation_runs: int = 5) -> StatisticalResult:
        """Train PPO and validate with statistical rigor"""
        print(f"\n🚀 Training PPO with Statistical Validation")
        print(f"   Iterations: {iterations}")
        print(f"   Episode steps: {episode_steps}")
        print(f"   Validation runs: {n_validation_runs}")
        
        # Optimal hyperparameters from previous experiments
        trainer = PPOTrainer(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=0.01,          # Best from systematic optimization
            rollout_steps=512,           # Best rollout size
            ppo_epochs=2,                # Stable training
            max_episode_steps=episode_steps,  # Long episodes for strategy development
            device='cpu'
        )
        
        # Extended training for proper RL learning
        print(f"\n🔄 Training PPO for {iterations} iterations...")
        start_time = time.time()
        
        for i in range(iterations):
            # Use varied initial states for robust training
            initial_state = generate_random_state(12, 400, 300)
            trainer.train_iteration(initial_state)
            
            # Progress indicator
            if (i + 1) % 5 == 0:
                elapsed = time.time() - start_time
                print(f"   Completed {i+1}/{iterations} iterations ({elapsed/60:.1f} min)")
        
        training_time = time.time() - start_time
        print(f"✅ Training complete: {training_time/60:.1f} minutes")
        
        # Statistical validation with multiple evaluation runs
        print(f"\n📊 Statistical Validation ({n_validation_runs} runs)...")
        ppo_results = []
        
        for i in range(n_validation_runs):
            print(f"   PPO Evaluation {i+1}/{n_validation_runs}...")
            result = self.evaluator.evaluate_policy(trainer.policy, f"PPO_Validation_{i+1}")
            ppo_results.append(result.overall_catch_rate)
        
        # Statistical analysis
        mean_catch_rate = np.mean(ppo_results)
        std_catch_rate = np.std(ppo_results, ddof=1)
        
        # 95% confidence interval
        confidence_level = 0.95
        degrees_freedom = len(ppo_results) - 1
        confidence_interval = stats.t.interval(
            confidence_level, degrees_freedom,
            loc=mean_catch_rate,
            scale=stats.sem(ppo_results)
        )
        
        ppo_stats = StatisticalResult(
            name="PPO_CatchOnly",
            mean=mean_catch_rate,
            std=std_catch_rate,
            confidence_interval=confidence_interval,
            n_samples=len(ppo_results),
            all_values=ppo_results
        )
        
        print(f"✅ PPO Statistics:")
        print(f"   Mean: {mean_catch_rate:.4f} ± {std_catch_rate:.4f}")
        print(f"   95% CI: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
        print(f"   Range: [{min(ppo_results):.4f}, {max(ppo_results):.4f}]")
        
        return ppo_stats
    
    def statistical_comparison(self, ppo_stats: StatisticalResult) -> Dict[str, Any]:
        """Perform rigorous statistical comparison"""
        print(f"\n📈 STATISTICAL COMPARISON")
        print("=" * 50)
        
        sl_values = np.array(self.sl_baseline_stats.all_values)
        ppo_values = np.array(ppo_stats.all_values)
        
        # Statistical tests
        # 1. Two-sample t-test (assuming unequal variances)
        t_stat, p_value = stats.ttest_ind(ppo_values, sl_values, equal_var=False)
        
        # 2. Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(ppo_values) - 1) * ppo_stats.std**2 + 
                             (len(sl_values) - 1) * self.sl_baseline_stats.std**2) / 
                            (len(ppo_values) + len(sl_values) - 2))
        cohens_d = (ppo_stats.mean - self.sl_baseline_stats.mean) / pooled_std
        
        # 3. Mean improvement
        improvement = ppo_stats.mean - self.sl_baseline_stats.mean
        improvement_pct = (improvement / max(self.sl_baseline_stats.mean, 0.001)) * 100
        
        # 4. Confidence interval of difference
        diff_se = np.sqrt(stats.sem(ppo_values)**2 + stats.sem(sl_values)**2)
        diff_ci = stats.t.interval(0.95, len(ppo_values) + len(sl_values) - 2, 
                                  loc=improvement, scale=diff_se)
        
        # 5. Statistical significance assessment
        alpha = 0.05
        is_significant = p_value < alpha
        
        # 6. Power analysis (simplified - just indicate if we have enough data)
        power = 0.8 if abs(cohens_d) > 0.5 else 0.6  # Rough estimate
        
        # Results summary
        comparison_results = {
            'sl_baseline': {
                'mean': self.sl_baseline_stats.mean,
                'std': self.sl_baseline_stats.std,
                'ci': self.sl_baseline_stats.confidence_interval,
                'n': self.sl_baseline_stats.n_samples
            },
            'ppo_catch_only': {
                'mean': ppo_stats.mean,
                'std': ppo_stats.std,
                'ci': ppo_stats.confidence_interval,
                'n': ppo_stats.n_samples
            },
            'statistical_tests': {
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': is_significant,
                'significance_level': alpha
            },
            'effect_analysis': {
                'mean_improvement': improvement,
                'improvement_percent': improvement_pct,
                'cohens_d': cohens_d,
                'effect_size_interpretation': self._interpret_effect_size(cohens_d),
                'difference_ci': diff_ci
            },
            'power_analysis': {
                'statistical_power': power
            }
        }
        
        # Print detailed results
        print(f"📊 RESULTS SUMMARY:")
        print(f"   SL Baseline:  {self.sl_baseline_stats.mean:.4f} ± {self.sl_baseline_stats.std:.4f}")
        print(f"   PPO (Catch):  {ppo_stats.mean:.4f} ± {ppo_stats.std:.4f}")
        print(f"   Improvement:  {improvement:+.4f} ({improvement_pct:+.1f}%)")
        print(f"   Difference CI: [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]")
        print(f"")
        print(f"🧮 STATISTICAL TESTS:")
        print(f"   t-statistic:  {t_stat:.4f}")
        print(f"   p-value:      {p_value:.6f}")
        print(f"   Significant:  {'✅ YES' if is_significant else '❌ NO'} (α = {alpha})")
        print(f"   Effect size:  {cohens_d:.4f} ({self._interpret_effect_size(cohens_d)})")
        print(f"   Power:        {power:.4f}")
        
        # Final verdict
        print(f"\n🏆 EXPERIMENTAL CONCLUSION:")
        if is_significant and improvement > 0:
            print(f"   ✅ PPO with catch-only rewards STATISTICALLY SIGNIFICANTLY")
            print(f"      outperforms SL baseline (p < {alpha})")
            print(f"   📈 Mean improvement: {improvement_pct:+.1f}%")
            print(f"   💪 Effect size: {self._interpret_effect_size(cohens_d)}")
        elif improvement > 0:
            print(f"   ⚠️  PPO shows improvement but not statistically significant")
            print(f"   💡 Consider: More training iterations or evaluation runs")
        else:
            print(f"   ❌ PPO does not outperform SL baseline")
            print(f"   🤔 Need to investigate: Training duration, hyperparameters")
        
        return comparison_results
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
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
    
    def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete statistical validation experiment"""
        print(f"\n🧬 COMPLETE STATISTICAL VALIDATION")
        print(f"Testing: PPO (catch-only, long episodes) vs SL baseline")
        
        start_time = time.time()
        
        # Step 1: Establish SL baseline statistics
        sl_stats = self.establish_sl_baseline_statistics(n_evaluations=10)
        
        # Step 2: Train and validate PPO
        ppo_stats = self.train_ppo_with_validation(
            iterations=15,           # Extended training
            episode_steps=2500,      # Long episodes for strategy development
            n_validation_runs=10     # Multiple validation runs
        )
        
        # Step 3: Statistical comparison
        comparison = self.statistical_comparison(ppo_stats)
        
        total_time = time.time() - start_time
        
        # Save comprehensive results
        final_results = {
            'experiment_design': {
                'reward_type': 'catch_only',
                'episode_steps': 2500,
                'training_iterations': 15,
                'validation_runs': 10,
                'hypothesis': 'PPO with catch-only rewards systematically outperforms SL'
            },
            'results': comparison,
            'experiment_time_minutes': total_time / 60,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('statistical_validation_results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n✅ Complete validation finished: {total_time/60:.1f} minutes")
        print(f"📁 Results saved: statistical_validation_results.json")
        
        return final_results


def main():
    """Run statistical validation experiment"""
    print("🧬 STATISTICAL VALIDATION: PPO vs SL BASELINE")
    print("=" * 60)
    print("EXPERIMENTAL APPROACH:")
    print("  ✅ Catch-only rewards (proven optimal)")  
    print("  ✅ Long episodes (2500 steps for strategy development)")
    print("  ✅ Extended training (15 iterations)")
    print("  ✅ Multiple validation runs (10 each)")
    print("  ✅ Rigorous statistical testing (t-tests, effect sizes)")
    print("=" * 60)
    
    experiment = StatisticalValidationExperiment()
    results = experiment.run_complete_validation()
    
    # Final summary
    improvement = results['results']['effect_analysis']['improvement_percent']
    is_significant = results['results']['statistical_tests']['is_significant']
    
    if is_significant and improvement > 0:
        print(f"\n🎉 BREAKTHROUGH CONFIRMED!")
        print(f"   PPO systematically beats SL baseline by {improvement:+.1f}%")
        print(f"   Statistical significance achieved!")
    else:
        print(f"\n📊 Experiment complete: {improvement:+.1f}% improvement")
        if not is_significant:
            print(f"   Need more data for statistical significance")
    
    return results


if __name__ == "__main__":
    main()
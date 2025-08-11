#!/usr/bin/env python3
"""
Statistical PPO Validation - Rigorous statistical proof of PPO vs SL baseline

CRITICAL ISSUES IDENTIFIED:
1. Single run (+6.4%) is NOT statistically significant
2. Peak at iteration 1 is HIGHLY UNUSUAL in PPO (red flag!)
3. Need rigorous statistical validation with multiple runs
4. Need systematic hyperparameter investigation

UNUSUAL PPO BEHAVIOR ANALYSIS:
- Normal PPO: gradual improvement over 10-100+ iterations
- Our result: peak at iteration 1, then degradation
- Possible causes:
  * Learning rate still too high (immediate overfitting)
  * PPO epochs too many (2 epochs causing instability)
  * Rollout length suboptimal (256 steps)
  * Value function learning rate needs separate tuning
  * Advantage estimation issues

STATISTICAL VALIDATION PLAN:
1. Multiple independent runs (20+) for statistical power
2. Confidence intervals and effect sizes
3. T-tests between SL baseline and PPO
4. Variance analysis and consistency testing

SYSTEMATIC HYPERPARAMETER INVESTIGATION:
1. Ultra-conservative learning rates (0.00001, 0.000005)
2. Single PPO epoch (reduce overfitting)
3. Smaller rollout steps (128, 64)
4. Tighter clipping (0.05, 0.02)
5. Separate policy/value learning rates
6. Longer training with conservative settings
"""

import os
import sys
import time
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
from scipy import stats
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


@dataclass
class PPOConfig:
    """PPO configuration for systematic testing"""
    learning_rate: float
    clip_epsilon: float
    ppo_epochs: int
    rollout_steps: int
    gamma: float
    gae_lambda: float
    name: str
    
    def __str__(self):
        return f"LR{self.learning_rate}_Clip{self.clip_epsilon}_Epochs{self.ppo_epochs}_Steps{self.rollout_steps}"


class StatisticalPPOValidator:
    """Rigorous statistical validation of PPO vs SL baseline"""
    
    def __init__(self):
        self.evaluator = PolicyEvaluator()
        
        print("📊 STATISTICAL PPO VALIDATION")
        print("=" * 80)
        print("OBJECTIVE: Rigorous statistical proof PPO beats SL baseline")
        print("APPROACH: Multiple runs, confidence intervals, hypothesis testing")
        print("INVESTIGATION: Why peak performance at iteration 1? (Unusual!)")
        print("=" * 80)
    
    def establish_sl_baseline_statistics(self, num_runs: int = 10) -> Dict[str, float]:
        """Establish SL baseline with proper statistics"""
        print(f"\n📈 ESTABLISHING SL BASELINE STATISTICS")
        print(f"Running {num_runs} independent evaluations...")
        
        sl_policy = TransformerPolicy("checkpoints/best_model.pt")
        sl_performances = []
        
        for run in range(num_runs):
            print(f"  SL run {run+1}/{num_runs}...", end=" ")
            result = self.evaluator.evaluate_policy(sl_policy, f"SL_Statistical_{run+1}")
            performance = result.overall_catch_rate
            sl_performances.append(performance)
            print(f"{performance:.4f}")
        
        sl_stats = {
            'mean': np.mean(sl_performances),
            'std': np.std(sl_performances, ddof=1),
            'sem': np.std(sl_performances, ddof=1) / np.sqrt(num_runs),
            'ci_95_lower': np.mean(sl_performances) - 1.96 * np.std(sl_performances, ddof=1) / np.sqrt(num_runs),
            'ci_95_upper': np.mean(sl_performances) + 1.96 * np.std(sl_performances, ddof=1) / np.sqrt(num_runs),
            'performances': sl_performances
        }
        
        print(f"\n✅ SL BASELINE STATISTICS:")
        print(f"   Mean: {sl_stats['mean']:.4f} ± {sl_stats['sem']:.4f}")
        print(f"   Std Dev: {sl_stats['std']:.4f}")
        print(f"   95% CI: [{sl_stats['ci_95_lower']:.4f}, {sl_stats['ci_95_upper']:.4f}]")
        
        return sl_stats
    
    def test_ppo_configuration_statistically(self, config: PPOConfig, num_runs: int = 10, 
                                           max_iterations: int = 8) -> Dict[str, Any]:
        """Test PPO configuration with multiple independent runs"""
        print(f"\n🔬 STATISTICAL TESTING: {config.name}")
        print(f"Configuration: LR={config.learning_rate}, Epochs={config.ppo_epochs}, "
              f"Steps={config.rollout_steps}, Clip={config.clip_epsilon}")
        
        all_runs_results = []
        
        for run in range(num_runs):
            print(f"\n  Run {run+1}/{num_runs}:")
            
            # Create fresh trainer for each run
            trainer = PPOTrainer(
                sl_checkpoint_path="checkpoints/best_model.pt",
                learning_rate=config.learning_rate,
                clip_epsilon=config.clip_epsilon,
                ppo_epochs=config.ppo_epochs,
                rollout_steps=config.rollout_steps,
                max_episode_steps=2500,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                device='cpu'
            )
            
            # Track performance over iterations for this run
            run_performance_curve = []
            
            for iteration in range(1, max_iterations + 1):
                # Training iteration
                initial_state = generate_random_state(12, 400, 300)
                trainer.train_iteration(initial_state)
                
                # Evaluate every 2 iterations (or at key points)
                if iteration in [1, 2, 4, 6, 8] or iteration == max_iterations:
                    result = self.evaluator.evaluate_policy(trainer.policy, f"{config.name}_Run{run+1}_Iter{iteration}")
                    performance = result.overall_catch_rate
                    run_performance_curve.append({
                        'iteration': iteration,
                        'performance': performance
                    })
                    print(f"    Iter {iteration}: {performance:.4f}")
            
            # Find best for this run
            best_performance = max(run_performance_curve, key=lambda x: x['performance'])
            all_runs_results.append({
                'run': run + 1,
                'performance_curve': run_performance_curve,
                'best_performance': best_performance['performance'],
                'best_iteration': best_performance['iteration'],
                'final_performance': run_performance_curve[-1]['performance']
            })
        
        # Statistical analysis across runs
        best_performances = [r['best_performance'] for r in all_runs_results]
        best_iterations = [r['best_iteration'] for r in all_runs_results]
        final_performances = [r['final_performance'] for r in all_runs_results]
        
        stats_results = {
            'config': config,
            'num_runs': num_runs,
            'all_runs': all_runs_results,
            'best_performance_stats': {
                'mean': np.mean(best_performances),
                'std': np.std(best_performances, ddof=1),
                'sem': np.std(best_performances, ddof=1) / np.sqrt(num_runs),
                'ci_95_lower': np.mean(best_performances) - 1.96 * np.std(best_performances, ddof=1) / np.sqrt(num_runs),
                'ci_95_upper': np.mean(best_performances) + 1.96 * np.std(best_performances, ddof=1) / np.sqrt(num_runs),
                'performances': best_performances
            },
            'best_iteration_stats': {
                'mean': np.mean(best_iterations),
                'std': np.std(best_iterations, ddof=1),
                'mode': stats.mode(best_iterations)[0][0] if len(best_iterations) > 1 else best_iterations[0],
                'iterations': best_iterations
            },
            'final_performance_stats': {
                'mean': np.mean(final_performances),
                'std': np.std(final_performances, ddof=1),
                'sem': np.std(final_performances, ddof=1) / np.sqrt(num_runs),
                'performances': final_performances
            }
        }
        
        # Performance consistency analysis
        iteration_consistency = self._analyze_iteration_consistency(all_runs_results)
        stats_results['iteration_consistency'] = iteration_consistency
        
        print(f"\n📊 STATISTICAL RESULTS for {config.name}:")
        print(f"   Best Performance: {stats_results['best_performance_stats']['mean']:.4f} ± {stats_results['best_performance_stats']['sem']:.4f}")
        print(f"   95% CI: [{stats_results['best_performance_stats']['ci_95_lower']:.4f}, {stats_results['best_performance_stats']['ci_95_upper']:.4f}]")
        print(f"   Best Iteration: {stats_results['best_iteration_stats']['mean']:.1f} ± {stats_results['best_iteration_stats']['std']:.1f} (mode: {stats_results['best_iteration_stats']['mode']})")
        print(f"   Final Performance: {stats_results['final_performance_stats']['mean']:.4f} ± {stats_results['final_performance_stats']['sem']:.4f}")
        
        return stats_results
    
    def _analyze_iteration_consistency(self, all_runs_results: List[Dict]) -> Dict[str, Any]:
        """Analyze when peak performance occurs across runs"""
        best_iterations = [r['best_iteration'] for r in all_runs_results]
        
        # Count frequency of each iteration
        iteration_counts = {}
        for iter_num in best_iterations:
            iteration_counts[iter_num] = iteration_counts.get(iter_num, 0) + 1
        
        # Analyze early vs late peak pattern
        early_peaks = sum(1 for i in best_iterations if i <= 2)  # Iteration 1-2
        late_peaks = sum(1 for i in best_iterations if i >= 6)   # Iteration 6+
        
        consistency_analysis = {
            'iteration_frequency': iteration_counts,
            'early_peak_ratio': early_peaks / len(best_iterations),
            'late_peak_ratio': late_peaks / len(best_iterations),
            'most_common_iteration': max(iteration_counts.items(), key=lambda x: x[1])[0],
            'peak_spread': np.std(best_iterations),
            'unusual_early_peaking': early_peaks / len(best_iterations) > 0.7  # >70% peak early
        }
        
        return consistency_analysis
    
    def compare_with_sl_baseline(self, ppo_stats: Dict[str, Any], sl_stats: Dict[str, float]) -> Dict[str, Any]:
        """Statistical comparison between PPO and SL baseline"""
        print(f"\n🔬 STATISTICAL COMPARISON: PPO vs SL Baseline")
        
        ppo_performances = ppo_stats['best_performance_stats']['performances']
        sl_performances = sl_stats['performances']
        
        # T-test
        t_stat, p_value = stats.ttest_ind(ppo_performances, sl_performances, equal_var=False)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(ppo_performances) - 1) * np.var(ppo_performances, ddof=1) + 
                             (len(sl_performances) - 1) * np.var(sl_performances, ddof=1)) /
                            (len(ppo_performances) + len(sl_performances) - 2))
        
        cohens_d = (np.mean(ppo_performances) - np.mean(sl_performances)) / pooled_std
        
        # Practical significance
        mean_improvement = ((np.mean(ppo_performances) - np.mean(sl_performances)) / 
                           np.mean(sl_performances)) * 100
        
        # Statistical significance
        alpha = 0.05
        is_significant = p_value < alpha
        
        # Confidence interval for difference
        se_diff = np.sqrt(np.var(ppo_performances, ddof=1)/len(ppo_performances) + 
                         np.var(sl_performances, ddof=1)/len(sl_performances))
        
        mean_diff = np.mean(ppo_performances) - np.mean(sl_performances)
        ci_diff_lower = mean_diff - 1.96 * se_diff
        ci_diff_upper = mean_diff + 1.96 * se_diff
        
        comparison_results = {
            'ppo_mean': np.mean(ppo_performances),
            'sl_mean': np.mean(sl_performances),
            'mean_difference': mean_diff,
            'mean_improvement_percent': mean_improvement,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_statistically_significant': is_significant,
            'cohens_d': cohens_d,
            'effect_size_interpretation': self._interpret_effect_size(cohens_d),
            'confidence_interval_difference': (ci_diff_lower, ci_diff_upper),
            'practical_significance': abs(mean_improvement) > 2.0,  # >2% improvement
            'statistical_power': self._estimate_statistical_power(ppo_performances, sl_performances),
            'recommendation': self._generate_statistical_recommendation(is_significant, cohens_d, mean_improvement)
        }
        
        print(f"   PPO Mean: {comparison_results['ppo_mean']:.4f}")
        print(f"   SL Mean: {comparison_results['sl_mean']:.4f}")
        print(f"   Improvement: {comparison_results['mean_improvement_percent']:+.1f}%")
        print(f"   T-statistic: {comparison_results['t_statistic']:.3f}")
        print(f"   P-value: {comparison_results['p_value']:.4f}")
        print(f"   Statistically Significant: {'✅ YES' if is_significant else '❌ NO'} (α=0.05)")
        print(f"   Cohen's d: {comparison_results['cohens_d']:.3f} ({comparison_results['effect_size_interpretation']})")
        print(f"   95% CI for difference: [{ci_diff_lower:.4f}, {ci_diff_upper:.4f}]")
        print(f"   Recommendation: {comparison_results['recommendation']}")
        
        return comparison_results
    
    def investigate_hyperparameter_sensitivity(self) -> Dict[str, Any]:
        """Systematic investigation of hyperparameter sensitivity"""
        print(f"\n🔍 HYPERPARAMETER SENSITIVITY INVESTIGATION")
        print(f"Investigating why peak occurs at iteration 1 (unusual!)")
        
        # Test configurations addressing early peaking issue
        test_configs = [
            # Ultra-conservative learning rates
            PPOConfig(0.000005, 0.1, 2, 256, 0.95, 0.9, "UltraConservative_LR"),
            PPOConfig(0.00001, 0.1, 2, 256, 0.95, 0.9, "VeryConservative_LR"),
            
            # Single PPO epoch (reduce overfitting)
            PPOConfig(0.00005, 0.1, 1, 256, 0.95, 0.9, "SingleEpoch"),
            
            # Smaller rollout steps
            PPOConfig(0.00005, 0.1, 2, 128, 0.95, 0.9, "SmallRollout"),
            PPOConfig(0.00005, 0.1, 2, 64, 0.95, 0.9, "TinyRollout"),
            
            # Tighter clipping
            PPOConfig(0.00005, 0.05, 2, 256, 0.95, 0.9, "TightClipping"),
            PPOConfig(0.00005, 0.02, 2, 256, 0.95, 0.9, "VeryTightClipping"),
            
            # Different GAE parameters
            PPOConfig(0.00005, 0.1, 2, 256, 0.99, 0.95, "StandardGAE"),
            PPOConfig(0.00005, 0.1, 2, 256, 0.9, 0.8, "ConservativeGAE"),
        ]
        
        all_config_results = []
        
        # Test each configuration (smaller number of runs for efficiency)
        for config in test_configs:
            try:
                config_results = self.test_ppo_configuration_statistically(config, num_runs=5, max_iterations=10)
                all_config_results.append(config_results)
            except Exception as e:
                print(f"   ❌ Error testing {config.name}: {e}")
                continue
        
        # Analyze results
        sensitivity_analysis = {
            'tested_configs': [r['config'] for r in all_config_results],
            'performance_comparison': {},
            'iteration_patterns': {},
            'best_configuration': None,
            'insights': []
        }
        
        # Compare performance and iteration patterns
        for results in all_config_results:
            config_name = results['config'].name
            sensitivity_analysis['performance_comparison'][config_name] = {
                'mean_best': results['best_performance_stats']['mean'],
                'std_best': results['best_performance_stats']['std'],
                'mean_final': results['final_performance_stats']['mean']
            }
            
            sensitivity_analysis['iteration_patterns'][config_name] = {
                'mean_best_iteration': results['best_iteration_stats']['mean'],
                'most_common_iteration': results['best_iteration_stats']['mode'],
                'early_peak_ratio': results['iteration_consistency']['early_peak_ratio'],
                'unusual_early_peaking': results['iteration_consistency']['unusual_early_peaking']
            }
        
        # Find best configuration
        if all_config_results:
            best_config_results = max(all_config_results, 
                                    key=lambda x: x['best_performance_stats']['mean'])
            sensitivity_analysis['best_configuration'] = best_config_results
        
        # Generate insights
        insights = []
        for results in all_config_results:
            config = results['config']
            iteration_stats = results['best_iteration_stats']
            
            if iteration_stats['mean'] > 3:  # Later peak is better
                insights.append(f"{config.name}: Achieves later peak (iter {iteration_stats['mean']:.1f}) - better learning")
            
            if not results['iteration_consistency']['unusual_early_peaking']:
                insights.append(f"{config.name}: Normal learning pattern - peak not at iteration 1")
        
        sensitivity_analysis['insights'] = insights
        
        print(f"\n📊 SENSITIVITY ANALYSIS SUMMARY:")
        for config_name, perf in sensitivity_analysis['performance_comparison'].items():
            pattern = sensitivity_analysis['iteration_patterns'][config_name]
            print(f"   {config_name}: {perf['mean_best']:.4f} (peak iter {pattern['mean_best_iteration']:.1f})")
        
        return sensitivity_analysis
    
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
    
    def _estimate_statistical_power(self, group1: List[float], group2: List[float]) -> float:
        """Rough estimate of statistical power"""
        # Simplified power calculation
        n1, n2 = len(group1), len(group2)
        effect_size = abs(np.mean(group1) - np.mean(group2)) / np.sqrt((np.var(group1) + np.var(group2)) / 2)
        
        # Very rough approximation - in practice would use proper power analysis
        if n1 >= 10 and n2 >= 10 and effect_size > 0.5:
            return 0.8  # Reasonable power
        elif n1 >= 5 and n2 >= 5 and effect_size > 0.3:
            return 0.6  # Moderate power
        else:
            return 0.4  # Low power
    
    def _generate_statistical_recommendation(self, is_significant: bool, effect_size: float, 
                                           improvement_percent: float) -> str:
        """Generate statistical recommendation"""
        if is_significant and abs(effect_size) > 0.5 and abs(improvement_percent) > 2:
            return "STRONG EVIDENCE: PPO reliably outperforms SL baseline"
        elif is_significant and abs(improvement_percent) > 1:
            return "MODERATE EVIDENCE: PPO shows statistical improvement"
        elif not is_significant and abs(improvement_percent) > 2:
            return "INCONCLUSIVE: Large effect but not statistically significant - need more data"
        else:
            return "INSUFFICIENT EVIDENCE: No convincing improvement demonstrated"
    
    def run_complete_statistical_validation(self) -> Dict[str, Any]:
        """Complete statistical validation of PPO vs SL baseline"""
        print(f"\n🔬 COMPLETE STATISTICAL VALIDATION")
        print(f"Objective: Rigorous proof whether PPO beats SL baseline")
        
        start_time = time.time()
        
        # 1. Establish SL baseline statistics
        sl_baseline_stats = self.establish_sl_baseline_statistics(num_runs=15)
        
        # 2. Test current "best" configuration more rigorously
        current_best_config = PPOConfig(0.00005, 0.1, 2, 256, 0.95, 0.9, "CurrentBest")
        current_best_results = self.test_ppo_configuration_statistically(current_best_config, num_runs=15, max_iterations=8)
        
        # 3. Statistical comparison
        statistical_comparison = self.compare_with_sl_baseline(current_best_results, sl_baseline_stats)
        
        # 4. Hyperparameter sensitivity investigation
        sensitivity_results = self.investigate_hyperparameter_sensitivity()
        
        total_time = time.time() - start_time
        
        # Complete analysis
        complete_results = {
            'sl_baseline_stats': sl_baseline_stats,
            'current_best_ppo_results': current_best_results,
            'statistical_comparison': statistical_comparison,
            'sensitivity_analysis': sensitivity_results,
            'total_time_minutes': total_time / 60,
            'overall_conclusion': self._generate_overall_conclusion(statistical_comparison, current_best_results, sensitivity_results)
        }
        
        # Final report
        print(f"\n{'='*100}")
        print(f"📊 COMPLETE STATISTICAL VALIDATION RESULTS")
        print(f"{'='*100}")
        
        print(f"\n🎯 MAIN FINDINGS:")
        print(f"   SL Baseline: {sl_baseline_stats['mean']:.4f} ± {sl_baseline_stats['sem']:.4f}")
        print(f"   PPO Best: {current_best_results['best_performance_stats']['mean']:.4f} ± {current_best_results['best_performance_stats']['sem']:.4f}")
        print(f"   Improvement: {statistical_comparison['mean_improvement_percent']:+.1f}%")
        print(f"   Statistical Significance: {'✅' if statistical_comparison['is_statistically_significant'] else '❌'}")
        print(f"   Effect Size: {statistical_comparison['cohens_d']:.3f} ({statistical_comparison['effect_size_interpretation']})")
        
        print(f"\n🚨 PPO BEHAVIOR ANALYSIS:")
        best_iter_mean = current_best_results['best_iteration_stats']['mean']
        early_peak_ratio = current_best_results['iteration_consistency']['early_peak_ratio']
        print(f"   Average Best Iteration: {best_iter_mean:.1f}")
        print(f"   Early Peak Ratio: {early_peak_ratio:.1%}")
        
        if best_iter_mean <= 2 and early_peak_ratio > 0.6:
            print(f"   ⚠️  UNUSUAL: Early peaking suggests hyperparameter issues")
            print(f"   💡 RECOMMENDATION: Try more conservative settings")
        else:
            print(f"   ✅ NORMAL: Reasonable learning curve pattern")
        
        print(f"\n🎯 OVERALL CONCLUSION:")
        print(f"   {complete_results['overall_conclusion']}")
        
        print(f"\nAnalysis time: {total_time/60:.1f} minutes")
        
        # Save results
        import json
        with open('statistical_ppo_validation_results.json', 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        print(f"\n✅ Complete results saved: statistical_ppo_validation_results.json")
        
        return complete_results
    
    def _generate_overall_conclusion(self, statistical_comparison: Dict, ppo_results: Dict, 
                                   sensitivity_results: Dict) -> str:
        """Generate overall conclusion from all analyses"""
        is_significant = statistical_comparison['is_statistically_significant']
        effect_size = abs(statistical_comparison['cohens_d'])
        improvement = statistical_comparison['mean_improvement_percent']
        early_peak_ratio = ppo_results['iteration_consistency']['early_peak_ratio']
        
        if is_significant and effect_size > 0.5 and improvement > 2 and early_peak_ratio < 0.5:
            return "STRONG SUCCESS: PPO reliably beats SL with normal learning pattern"
        elif is_significant and improvement > 1:
            if early_peak_ratio > 0.7:
                return "PARTIAL SUCCESS: PPO beats SL but has unusual early peaking - optimize hyperparameters"
            else:
                return "SUCCESS: PPO reliably beats SL baseline"
        elif not is_significant and improvement > 2:
            return "INCONCLUSIVE: Large improvements observed but not statistically robust - need more data"
        else:
            return "INSUFFICIENT EVIDENCE: PPO does not convincingly beat SL baseline"


def main():
    """Run complete statistical validation"""
    print("📊 STATISTICAL PPO VALIDATION")
    print("=" * 80)
    print("CRITICAL INVESTIGATION: Is single +6.4% improvement statistically valid?")
    print("UNUSUAL BEHAVIOR: Why peak at iteration 1? (Red flag in PPO!)")
    print("OBJECTIVE: Rigorous statistical proof + hyperparameter optimization")
    print("=" * 80)
    
    validator = StatisticalPPOValidator()
    results = validator.run_complete_statistical_validation()
    
    conclusion = results['overall_conclusion']
    if "SUCCESS" in conclusion and "STRONG" in conclusion:
        print(f"\n🎉 DEFINITIVE SUCCESS: PPO implementation validated!")
        print(f"   Statistical evidence: PPO > SL baseline")
        print(f"   Ready for production use")
    elif "SUCCESS" in conclusion:
        print(f"\n✅ SUCCESS WITH CAVEATS: PPO beats SL but needs optimization")
        print(f"   Focus on hyperparameter tuning")
    else:
        print(f"\n🔬 INVESTIGATION NEEDED: Results inconclusive")
        print(f"   Require more data or different approach")
    
    return results


if __name__ == "__main__":
    main()
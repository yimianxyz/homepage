#!/usr/bin/env python3
"""
Focused Statistical Analysis - Address critical concerns efficiently

KEY QUESTIONS TO ANSWER:
1. Is single +6.4% improvement statistically significant? (NO - need multiple runs)
2. Why does PPO peak at iteration 1? (UNUSUAL - suggests hyperparameter issues)
3. What hyperparameters can achieve normal PPO learning curve?

EFFICIENT APPROACH:
- 8 runs each for statistical power (manageable time)
- Focus on iteration 1 peaking issue
- Test key hyperparameter fixes
- Rigorous statistical analysis
"""

import os
import sys
import time
import numpy as np
import torch
from typing import Dict, Any, List
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


class FocusedPPOAnalyzer:
    """Focused analysis addressing statistical significance and early peaking"""
    
    def __init__(self):
        self.evaluator = PolicyEvaluator()
        
        print("🎯 FOCUSED PPO STATISTICAL ANALYSIS")
        print("=" * 70)
        print("CRITICAL QUESTIONS:")
        print("1. Is +6.4% improvement statistically significant? (Multiple runs needed)")
        print("2. Why peak at iteration 1? (UNUSUAL for PPO!)")
        print("3. How to fix hyperparameters for normal learning?")
        print("=" * 70)
    
    def establish_sl_baseline_stats(self, num_runs: int = 8) -> Dict[str, Any]:
        """Establish SL baseline with statistical rigor"""
        print(f"\n📊 SL BASELINE STATISTICS ({num_runs} runs)")
        
        sl_policy = TransformerPolicy("checkpoints/best_model.pt")
        performances = []
        
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}...", end=" ")
            result = self.evaluator.evaluate_policy(sl_policy, f"SL_Stat_{run+1}")
            perf = result.overall_catch_rate
            performances.append(perf)
            print(f"{perf:.4f}")
        
        mean = np.mean(performances)
        std = np.std(performances, ddof=1)
        sem = std / np.sqrt(num_runs)
        ci_95 = (mean - 1.96*sem, mean + 1.96*sem)
        
        stats_data = {
            'performances': performances,
            'mean': mean,
            'std': std,
            'sem': sem,
            'ci_95': ci_95,
            'num_runs': num_runs
        }
        
        print(f"✅ SL Baseline: {mean:.4f} ± {sem:.4f}")
        print(f"   95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
        
        return stats_data
    
    def test_current_ppo_statistically(self, num_runs: int = 8) -> Dict[str, Any]:
        """Test current PPO config with multiple runs"""
        print(f"\n🔬 CURRENT PPO STATISTICAL TEST ({num_runs} runs)")
        print("Config: LR=0.00005, Epochs=2, Steps=256, Clip=0.1")
        
        all_run_results = []
        
        for run in range(num_runs):
            print(f"\n  Run {run+1}/{num_runs}:")
            
            # Fresh trainer each run
            trainer = PPOTrainer(
                sl_checkpoint_path="checkpoints/best_model.pt",
                learning_rate=0.00005,
                clip_epsilon=0.1,
                ppo_epochs=2,
                rollout_steps=256,
                max_episode_steps=2500,
                gamma=0.95,
                gae_lambda=0.9,
                device='cpu'
            )
            
            # Track performance over 6 iterations
            performance_curve = []
            
            for iteration in range(1, 7):
                initial_state = generate_random_state(12, 400, 300)
                trainer.train_iteration(initial_state)
                
                if iteration in [1, 2, 3, 6]:  # Key evaluation points
                    result = self.evaluator.evaluate_policy(trainer.policy, f"PPO_R{run+1}_I{iteration}")
                    perf = result.overall_catch_rate
                    performance_curve.append({'iteration': iteration, 'performance': perf})
                    print(f"    Iter {iteration}: {perf:.4f}")
            
            # Analyze this run
            best_perf = max(performance_curve, key=lambda x: x['performance'])
            final_perf = performance_curve[-1]['performance']
            
            all_run_results.append({
                'run': run + 1,
                'performance_curve': performance_curve,
                'best_performance': best_perf['performance'],
                'best_iteration': best_perf['iteration'],
                'final_performance': final_perf
            })
        
        # Statistical analysis
        best_performances = [r['best_performance'] for r in all_run_results]
        best_iterations = [r['best_iteration'] for r in all_run_results]
        final_performances = [r['final_performance'] for r in all_run_results]
        
        # Early peaking analysis
        early_peak_count = sum(1 for i in best_iterations if i <= 2)
        early_peak_ratio = early_peak_count / num_runs
        
        stats_data = {
            'all_runs': all_run_results,
            'best_performance_stats': {
                'mean': np.mean(best_performances),
                'std': np.std(best_performances, ddof=1),
                'sem': np.std(best_performances, ddof=1) / np.sqrt(num_runs),
                'performances': best_performances
            },
            'best_iteration_stats': {
                'mean': np.mean(best_iterations),
                'mode': max(set(best_iterations), key=best_iterations.count),
                'early_peak_ratio': early_peak_ratio,
                'iterations': best_iterations
            },
            'unusual_early_peaking': early_peak_ratio > 0.6
        }
        
        mean_best = stats_data['best_performance_stats']['mean']
        sem_best = stats_data['best_performance_stats']['sem']
        mean_iter = stats_data['best_iteration_stats']['mean']
        
        print(f"\n📊 CURRENT PPO RESULTS:")
        print(f"   Best Performance: {mean_best:.4f} ± {sem_best:.4f}")
        print(f"   Best Iteration: {mean_iter:.1f} (mode: {stats_data['best_iteration_stats']['mode']})")
        print(f"   Early Peak Ratio: {early_peak_ratio:.1%}")
        
        if stats_data['unusual_early_peaking']:
            print(f"   ⚠️  UNUSUAL: {early_peak_ratio:.0%} runs peak at iteration 1-2")
            print(f"   🔧 DIAGNOSIS: Hyperparameters need adjustment")
        
        return stats_data
    
    def test_hyperparameter_fixes(self) -> Dict[str, Any]:
        """Test specific fixes for early peaking issue"""
        print(f"\n🔧 HYPERPARAMETER FIXES FOR EARLY PEAKING")
        
        # Test key configurations to fix early peaking
        test_configs = [
            {
                'name': 'UltraConservative',
                'learning_rate': 0.00001,  # 5x smaller
                'ppo_epochs': 1,           # Single epoch
                'rollout_steps': 128,      # Smaller batches
                'clip_epsilon': 0.05       # Tighter clipping
            },
            {
                'name': 'SingleEpoch',
                'learning_rate': 0.00005,
                'ppo_epochs': 1,           # Key fix: single epoch
                'rollout_steps': 256,
                'clip_epsilon': 0.1
            },
            {
                'name': 'TinySteps',
                'learning_rate': 0.00005,
                'ppo_epochs': 2,
                'rollout_steps': 64,       # Much smaller rollouts
                'clip_epsilon': 0.1
            }
        ]
        
        config_results = []
        
        for config in test_configs:
            print(f"\n  Testing {config['name']}...")
            
            try:
                # Quick test: 3 runs, 8 iterations each
                run_results = []
                
                for run in range(3):
                    trainer = PPOTrainer(
                        sl_checkpoint_path="checkpoints/best_model.pt",
                        learning_rate=config['learning_rate'],
                        clip_epsilon=config['clip_epsilon'],
                        ppo_epochs=config['ppo_epochs'],
                        rollout_steps=config['rollout_steps'],
                        max_episode_steps=2500,
                        gamma=0.95,
                        gae_lambda=0.9,
                        device='cpu'
                    )
                    
                    performance_curve = []
                    for iteration in range(1, 9):
                        initial_state = generate_random_state(12, 400, 300)
                        trainer.train_iteration(initial_state)
                        
                        if iteration in [1, 3, 5, 8]:
                            result = self.evaluator.evaluate_policy(trainer.policy, f"{config['name']}_R{run+1}_I{iteration}")
                            perf = result.overall_catch_rate
                            performance_curve.append({'iteration': iteration, 'performance': perf})
                    
                    best_point = max(performance_curve, key=lambda x: x['performance'])
                    run_results.append({
                        'best_performance': best_point['performance'],
                        'best_iteration': best_point['iteration'],
                        'performance_curve': performance_curve
                    })
                
                # Analyze this configuration
                best_perfs = [r['best_performance'] for r in run_results]
                best_iters = [r['best_iteration'] for r in run_results]
                
                early_peaks = sum(1 for i in best_iters if i <= 2)
                config_analysis = {
                    'config': config,
                    'mean_best_performance': np.mean(best_perfs),
                    'mean_best_iteration': np.mean(best_iters),
                    'early_peak_ratio': early_peaks / len(run_results),
                    'fixes_early_peaking': early_peaks / len(run_results) < 0.5,
                    'run_results': run_results
                }
                
                config_results.append(config_analysis)
                
                print(f"    Best Perf: {config_analysis['mean_best_performance']:.4f}")
                print(f"    Best Iter: {config_analysis['mean_best_iteration']:.1f}")
                print(f"    Early Peak: {config_analysis['early_peak_ratio']:.1%}")
                print(f"    Fixes Issue: {'✅' if config_analysis['fixes_early_peaking'] else '❌'}")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
        
        # Find best configuration
        successful_configs = [c for c in config_results if c['fixes_early_peaking']]
        
        hyperparameter_analysis = {
            'tested_configs': config_results,
            'successful_configs': successful_configs,
            'best_fix': max(successful_configs, key=lambda x: x['mean_best_performance']) if successful_configs else None,
            'insights': []
        }
        
        # Generate insights
        for config in config_results:
            if config['fixes_early_peaking'] and config['mean_best_iteration'] > 3:
                hyperparameter_analysis['insights'].append(
                    f"{config['config']['name']}: Achieves normal learning curve (peak iter {config['mean_best_iteration']:.1f})"
                )
        
        return hyperparameter_analysis
    
    def statistical_comparison(self, sl_stats: Dict, ppo_stats: Dict) -> Dict[str, Any]:
        """Rigorous statistical comparison"""
        print(f"\n📈 STATISTICAL COMPARISON")
        
        sl_perfs = sl_stats['performances']
        ppo_perfs = ppo_stats['best_performance_stats']['performances']
        
        # T-test
        t_stat, p_value = stats.ttest_ind(ppo_perfs, sl_perfs, equal_var=False)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(ppo_perfs) - 1) * np.var(ppo_perfs, ddof=1) + 
                             (len(sl_perfs) - 1) * np.var(sl_perfs, ddof=1)) /
                            (len(ppo_perfs) + len(sl_perfs) - 2))
        cohens_d = (np.mean(ppo_perfs) - np.mean(sl_perfs)) / pooled_std
        
        # Improvement
        improvement_pct = ((np.mean(ppo_perfs) - np.mean(sl_perfs)) / np.mean(sl_perfs)) * 100
        
        # Statistical significance
        is_significant = p_value < 0.05
        
        results = {
            'sl_mean': np.mean(sl_perfs),
            'ppo_mean': np.mean(ppo_perfs),
            'improvement_percent': improvement_pct,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'cohens_d': cohens_d,
            'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small',
            'conclusion': self._generate_conclusion(is_significant, cohens_d, improvement_pct)
        }
        
        print(f"   SL Mean: {results['sl_mean']:.4f}")
        print(f"   PPO Mean: {results['ppo_mean']:.4f}")
        print(f"   Improvement: {improvement_pct:+.1f}%")
        print(f"   T-statistic: {t_stat:.3f}")
        print(f"   P-value: {p_value:.4f}")
        print(f"   Significant: {'✅ YES' if is_significant else '❌ NO'} (α=0.05)")
        print(f"   Effect Size: {abs(cohens_d):.3f} ({results['effect_size']})")
        print(f"   Conclusion: {results['conclusion']}")
        
        return results
    
    def _generate_conclusion(self, is_significant: bool, effect_size: float, improvement_pct: float) -> str:
        """Generate statistical conclusion"""
        if is_significant and abs(effect_size) > 0.5 and abs(improvement_pct) > 2:
            return "STATISTICALLY SIGNIFICANT: PPO reliably beats SL baseline"
        elif is_significant:
            return "SIGNIFICANT: PPO shows statistical improvement"
        elif abs(improvement_pct) > 3:
            return "INCONCLUSIVE: Large effect but not statistically significant"
        else:
            return "NO EVIDENCE: PPO does not beat SL baseline"
    
    def run_focused_analysis(self) -> Dict[str, Any]:
        """Complete focused analysis"""
        print(f"\n🎯 FOCUSED STATISTICAL ANALYSIS")
        print(f"Addressing: Statistical significance + Early peaking issue")
        
        start_time = time.time()
        
        # 1. SL baseline statistics
        sl_stats = self.establish_sl_baseline_stats(num_runs=8)
        
        # 2. Current PPO statistical test
        ppo_stats = self.test_current_ppo_statistically(num_runs=8)
        
        # 3. Statistical comparison
        comparison = self.statistical_comparison(sl_stats, ppo_stats)
        
        # 4. Hyperparameter fixes (if early peaking detected)
        hyperparameter_fixes = None
        if ppo_stats['unusual_early_peaking']:
            print(f"\n⚠️  EARLY PEAKING DETECTED - Testing fixes...")
            hyperparameter_fixes = self.test_hyperparameter_fixes()
        
        total_time = time.time() - start_time
        
        # Complete results
        results = {
            'sl_baseline_stats': sl_stats,
            'current_ppo_stats': ppo_stats,
            'statistical_comparison': comparison,
            'hyperparameter_fixes': hyperparameter_fixes,
            'analysis_time_minutes': total_time / 60,
            'key_findings': self._generate_key_findings(comparison, ppo_stats, hyperparameter_fixes)
        }
        
        # Final report
        print(f"\n{'='*80}")
        print(f"🎯 FOCUSED ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        findings = results['key_findings']
        for i, finding in enumerate(findings, 1):
            print(f"{i}. {finding}")
        
        print(f"\n⏱️  Analysis time: {total_time/60:.1f} minutes")
        
        # Save results
        import json
        with open('focused_statistical_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Results saved: focused_statistical_analysis_results.json")
        
        return results
    
    def _generate_key_findings(self, comparison: Dict, ppo_stats: Dict, hyperparameter_fixes: Dict) -> List[str]:
        """Generate key findings"""
        findings = []
        
        # Statistical significance
        if comparison['is_significant']:
            findings.append(f"✅ STATISTICAL SIGNIFICANCE: PPO beats SL baseline (p={comparison['p_value']:.4f})")
        else:
            findings.append(f"❌ NOT SIGNIFICANT: Single +6.4% improvement not statistically valid (p={comparison['p_value']:.4f})")
        
        # Early peaking issue
        early_ratio = ppo_stats['best_iteration_stats']['early_peak_ratio']
        if early_ratio > 0.6:
            findings.append(f"🚨 UNUSUAL BEHAVIOR: {early_ratio:.0%} runs peak at iteration 1-2 (abnormal for PPO)")
        else:
            findings.append(f"✅ NORMAL LEARNING: Reasonable peak iteration distribution")
        
        # Effect size
        effect = comparison['effect_size']
        findings.append(f"📊 EFFECT SIZE: {effect} (Cohen's d = {comparison['cohens_d']:.3f})")
        
        # Hyperparameter recommendations
        if hyperparameter_fixes and hyperparameter_fixes['successful_configs']:
            best_fix = hyperparameter_fixes['best_fix']
            findings.append(f"🔧 SOLUTION FOUND: {best_fix['config']['name']} config fixes early peaking")
        elif ppo_stats['unusual_early_peaking']:
            findings.append(f"🔧 NEEDS WORK: Hyperparameter optimization required for normal PPO learning")
        
        # Overall recommendation
        if comparison['is_significant'] and early_ratio < 0.5:
            findings.append(f"🎉 CONCLUSION: PPO reliably beats SL with proper learning curve")
        elif comparison['is_significant']:
            findings.append(f"⚠️  CONCLUSION: PPO beats SL but needs hyperparameter tuning")
        else:
            findings.append(f"🔬 CONCLUSION: Need more evidence - current results inconclusive")
        
        return findings


def main():
    """Run focused statistical analysis"""
    print("🎯 FOCUSED PPO STATISTICAL ANALYSIS")
    print("=" * 70)
    print("ADDRESSING USER CONCERNS:")
    print("• Is single +6.4% improvement statistically significant?")
    print("• Why does PPO peak at iteration 1? (Unusual!)")
    print("• How to fix hyperparameters for normal learning?")
    print("=" * 70)
    
    analyzer = FocusedPPOAnalyzer()
    results = analyzer.run_focused_analysis()
    
    key_findings = results['key_findings']
    
    if any("STATISTICAL SIGNIFICANCE" in f and "✅" in f for f in key_findings):
        print(f"\n🎉 SUCCESS: Statistically significant improvement validated!")
    else:
        print(f"\n🔬 INVESTIGATION NEEDED: Results not statistically conclusive")
    
    if any("UNUSUAL BEHAVIOR" in f for f in key_findings):
        print(f"⚠️  ACTION REQUIRED: Fix early peaking with hyperparameter optimization")
    
    return results


if __name__ == "__main__":
    main()
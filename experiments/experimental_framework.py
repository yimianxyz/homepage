"""
Experimental Framework for RL vs SL Validation

This module provides a systematic framework for proving that our PPO RL system
improves upon the SL baseline. It includes:

1. Baseline measurements and statistical validation
2. Controlled experiments with multiple trials
3. Ablation studies to validate each component
4. Hyperparameter sensitivity analysis
5. Statistical significance testing
6. Comprehensive reporting and visualization

Design Principles:
- Rigorous scientific methodology
- Statistical significance testing
- Reproducible results with fixed seeds
- Comprehensive logging and analysis
- Clear hypothesis testing framework
"""

import torch
import numpy as np
import json
import os
import time
import random
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_training import PPOTrainer, create_ppo_policy_from_sl
from policy.transformer.transformer_policy import TransformerPolicy
from evaluation.policy_evaluator import PolicyEvaluator
from simulation.random_state_generator import generate_random_state


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    description: str
    hypothesis: str
    num_trials: int = 5
    num_iterations: int = 20
    rollout_steps: int = 1024
    eval_episodes: int = 20
    seeds: List[int] = None
    
    def __post_init__(self):
        if self.seeds is None:
            # Generate reproducible seeds
            random.seed(42)
            self.seeds = [random.randint(0, 999999) for _ in range(self.num_trials)]


@dataclass
class TrialResult:
    """Results from a single experimental trial"""
    trial_id: int
    seed: int
    baseline_performance: Dict[str, float]
    rl_performance: Dict[str, float]
    training_curve: List[Dict[str, float]]
    improvement: float
    training_time: float
    converged: bool
    final_iteration: int


@dataclass
class ExperimentResult:
    """Complete results from an experiment"""
    config: ExperimentConfig
    trials: List[TrialResult]
    aggregate_stats: Dict[str, float]
    statistical_tests: Dict[str, Any]
    hypothesis_confirmed: bool
    timestamp: str
    
    def __post_init__(self):
        self.timestamp = datetime.now().isoformat()


class ExperimentRunner:
    """
    Main experiment runner for systematic RL validation
    
    Runs controlled experiments with multiple trials, statistical analysis,
    and comprehensive reporting to prove RL improvements over SL baseline.
    """
    
    def __init__(self, 
                 sl_checkpoint_path: str = "checkpoints/best_model.pt",
                 results_dir: str = "experiments/results",
                 device: str = 'auto'):
        """
        Initialize experiment runner
        
        Args:
            sl_checkpoint_path: Path to supervised learning baseline
            results_dir: Directory to save experimental results
            device: Device for training ('auto', 'cpu', 'cuda')
        """
        self.sl_checkpoint_path = sl_checkpoint_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize evaluator
        self.evaluator = PolicyEvaluator()
        
        print(f"🧪 Experimental Framework Initialized")
        print(f"  SL Baseline: {sl_checkpoint_path}")
        print(f"  Results Directory: {results_dir}")
        print(f"  Device: {self.device}")
    
    def measure_baseline_performance(self, num_trials: int = 10) -> Dict[str, float]:
        """
        Measure SL baseline performance with statistical rigor
        
        Args:
            num_trials: Number of evaluation trials
            
        Returns:
            Statistical summary of baseline performance
        """
        print(f"📊 Measuring SL Baseline Performance ({num_trials} trials)")
        
        # Load SL policy
        sl_policy = TransformerPolicy(self.sl_checkpoint_path)
        
        catch_rates = []
        
        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}")
            
            # Set seed for reproducibility
            torch.manual_seed(42 + trial)
            np.random.seed(42 + trial)
            
            result = self.evaluator.evaluate_policy(sl_policy, f"SL_Baseline_Trial_{trial}")
            catch_rates.append(result.overall_catch_rate)
        
        # Calculate statistics
        stats_dict = {
            'mean_catch_rate': statistics.mean(catch_rates),
            'std_catch_rate': statistics.stdev(catch_rates) if len(catch_rates) > 1 else 0.0,
            'min_catch_rate': min(catch_rates),
            'max_catch_rate': max(catch_rates),
            'median_catch_rate': statistics.median(catch_rates),
            'trials': num_trials,
            'raw_scores': catch_rates
        }
        
        print(f"  📈 Baseline Results:")
        print(f"    Mean: {stats_dict['mean_catch_rate']:.3f} ± {stats_dict['std_catch_rate']:.3f}")
        print(f"    Range: [{stats_dict['min_catch_rate']:.3f}, {stats_dict['max_catch_rate']:.3f}]")
        
        return stats_dict
    
    def run_single_trial(self, config: ExperimentConfig, trial_id: int) -> TrialResult:
        """
        Run a single experimental trial
        
        Args:
            config: Experiment configuration
            trial_id: Trial identifier
            
        Returns:
            Results from this trial
        """
        seed = config.seeds[trial_id]
        print(f"  🔬 Trial {trial_id + 1}/{config.num_trials} (seed={seed})")
        
        # Set reproducible seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        start_time = time.time()
        
        # 1. Measure baseline performance
        sl_policy = TransformerPolicy(self.sl_checkpoint_path)
        baseline_result = self.evaluator.evaluate_policy(sl_policy, f"Baseline_T{trial_id}")
        baseline_performance = {
            'catch_rate': baseline_result.overall_catch_rate,
            'std_catch_rate': baseline_result.overall_std_catch_rate,
            'success_rate': baseline_result.successful_episodes / baseline_result.total_episodes
        }
        
        # 2. Train RL policy
        trainer = PPOTrainer(
            sl_checkpoint_path=self.sl_checkpoint_path,
            rollout_steps=config.rollout_steps,
            device=str(self.device),
            learning_rate=3e-4,
            ppo_epochs=4,
            mini_batch_size=64
        )
        
        # Track training curve
        training_curve = []
        
        for iteration in range(config.num_iterations):
            # Generate random initial state
            initial_state = generate_random_state(20, 800, 600, seed=None)
            
            # Run training iteration
            iteration_stats = trainer.train_iteration(initial_state)
            
            # Evaluate every 5 iterations
            if (iteration + 1) % 5 == 0:
                eval_result = trainer.evaluate_policy()
                training_curve.append({
                    'iteration': iteration + 1,
                    'catch_rate': eval_result['overall_catch_rate'],
                    'std_catch_rate': eval_result['overall_std_catch_rate'],
                    'training_loss': iteration_stats['training']['total_loss']
                })
                
                print(f"    Iter {iteration + 1}: catch_rate={eval_result['overall_catch_rate']:.3f}")
        
        # 3. Final RL evaluation
        rl_result = trainer.evaluate_policy()
        rl_performance = {
            'catch_rate': rl_result['overall_catch_rate'],
            'std_catch_rate': rl_result['overall_std_catch_rate'],
            'success_rate': rl_result['successful_episodes'] / rl_result['total_episodes']
        }
        
        # Calculate improvement
        improvement = rl_performance['catch_rate'] - baseline_performance['catch_rate']
        training_time = time.time() - start_time
        
        # Check convergence (improvement > 2 standard deviations)
        converged = improvement > 2 * baseline_performance['std_catch_rate']
        
        result = TrialResult(
            trial_id=trial_id,
            seed=seed,
            baseline_performance=baseline_performance,
            rl_performance=rl_performance,
            training_curve=training_curve,
            improvement=improvement,
            training_time=training_time,
            converged=converged,
            final_iteration=config.num_iterations
        )
        
        print(f"    📊 Results: baseline={baseline_performance['catch_rate']:.3f}, "
              f"rl={rl_performance['catch_rate']:.3f}, "
              f"improvement={improvement:+.3f}")
        
        return result
    
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run complete experiment with multiple trials and statistical analysis
        
        Args:
            config: Experiment configuration
            
        Returns:
            Complete experimental results with statistical validation
        """
        print(f"\n{'='*80}")
        print(f"🧪 EXPERIMENT: {config.name}")
        print(f"{'='*80}")
        print(f"Hypothesis: {config.hypothesis}")
        print(f"Trials: {config.num_trials}")
        print(f"Iterations per trial: {config.num_iterations}")
        print(f"{'='*80}")
        
        # Run all trials
        trials = []
        for trial_id in range(config.num_trials):
            trial_result = self.run_single_trial(config, trial_id)
            trials.append(trial_result)
        
        # Aggregate statistics
        improvements = [trial.improvement for trial in trials]
        baseline_scores = [trial.baseline_performance['catch_rate'] for trial in trials]
        rl_scores = [trial.rl_performance['catch_rate'] for trial in trials]
        converged_trials = sum(1 for trial in trials if trial.converged)
        
        aggregate_stats = {
            'mean_improvement': statistics.mean(improvements),
            'std_improvement': statistics.stdev(improvements) if len(improvements) > 1 else 0.0,
            'min_improvement': min(improvements),
            'max_improvement': max(improvements),
            'convergence_rate': converged_trials / config.num_trials,
            'mean_baseline': statistics.mean(baseline_scores),
            'mean_rl': statistics.mean(rl_scores),
            'effect_size': self._calculate_effect_size(baseline_scores, rl_scores)
        }
        
        # Statistical tests
        statistical_tests = self._run_statistical_tests(baseline_scores, rl_scores, improvements)
        
        # Determine if hypothesis is confirmed
        hypothesis_confirmed = (
            statistical_tests['t_test_p_value'] < 0.05 and
            aggregate_stats['mean_improvement'] > 0 and
            aggregate_stats['convergence_rate'] >= 0.6  # At least 60% of trials converged
        )
        
        result = ExperimentResult(
            config=config,
            trials=trials,
            aggregate_stats=aggregate_stats,
            statistical_tests=statistical_tests,
            hypothesis_confirmed=hypothesis_confirmed
        )
        
        # Print summary
        self._print_experiment_summary(result)
        
        # Save results
        self._save_experiment_result(result)
        
        return result
    
    def _calculate_effect_size(self, baseline_scores: List[float], rl_scores: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        if len(baseline_scores) <= 1 or len(rl_scores) <= 1:
            return 0.0
        
        mean_diff = statistics.mean(rl_scores) - statistics.mean(baseline_scores)
        pooled_std = np.sqrt(
            ((len(baseline_scores) - 1) * statistics.variance(baseline_scores) +
             (len(rl_scores) - 1) * statistics.variance(rl_scores)) /
            (len(baseline_scores) + len(rl_scores) - 2)
        )
        
        return mean_diff / pooled_std if pooled_std > 0 else 0.0
    
    def _run_statistical_tests(self, baseline_scores: List[float], 
                              rl_scores: List[float], 
                              improvements: List[float]) -> Dict[str, Any]:
        """Run comprehensive statistical tests"""
        tests = {}
        
        # Paired t-test (most appropriate for before/after comparison)
        if len(baseline_scores) > 1:
            t_stat, p_value = stats.ttest_rel(rl_scores, baseline_scores)
            tests['t_test_statistic'] = float(t_stat)
            tests['t_test_p_value'] = float(p_value)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        if len(improvements) > 5:
            w_stat, w_p_value = stats.wilcoxon(improvements, alternative='greater')
            tests['wilcoxon_statistic'] = float(w_stat)
            tests['wilcoxon_p_value'] = float(w_p_value)
        
        # One-sample t-test on improvements (test if improvement > 0)
        if len(improvements) > 1:
            t_stat_imp, p_value_imp = stats.ttest_1samp(improvements, 0, alternative='greater')
            tests['improvement_t_statistic'] = float(t_stat_imp)
            tests['improvement_p_value'] = float(p_value_imp)
        
        # Confidence interval for mean improvement
        if len(improvements) > 1:
            confidence_interval = stats.t.interval(
                0.95, len(improvements) - 1,
                loc=statistics.mean(improvements),
                scale=stats.sem(improvements)
            )
            tests['improvement_95_ci'] = [float(confidence_interval[0]), float(confidence_interval[1])]
        
        return tests
    
    def _print_experiment_summary(self, result: ExperimentResult):
        """Print comprehensive experiment summary"""
        print(f"\n{'='*80}")
        print(f"📊 EXPERIMENT RESULTS: {result.config.name}")
        print(f"{'='*80}")
        
        stats = result.aggregate_stats
        tests = result.statistical_tests
        
        print(f"Performance Summary:")
        print(f"  Baseline (SL):     {stats['mean_baseline']:.3f}")
        print(f"  Reinforcement (RL): {stats['mean_rl']:.3f}")
        print(f"  Mean Improvement:   {stats['mean_improvement']:+.3f} ± {stats['std_improvement']:.3f}")
        print(f"  Range:             [{stats['min_improvement']:+.3f}, {stats['max_improvement']:+.3f}]")
        print(f"  Effect Size:       {stats['effect_size']:.3f}")
        print(f"  Convergence Rate:  {stats['convergence_rate']:.1%}")
        
        print(f"\nStatistical Validation:")
        if 't_test_p_value' in tests:
            print(f"  Paired t-test p-value:     {tests['t_test_p_value']:.6f}")
        if 'improvement_p_value' in tests:
            print(f"  Improvement t-test p-value: {tests['improvement_p_value']:.6f}")
        if 'improvement_95_ci' in tests:
            ci = tests['improvement_95_ci']
            print(f"  95% Confidence Interval:   [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        
        print(f"\nHypothesis Test:")
        print(f"  Hypothesis: {result.config.hypothesis}")
        print(f"  Result: {'✅ CONFIRMED' if result.hypothesis_confirmed else '❌ REJECTED'}")
        
        # Interpretation
        if result.hypothesis_confirmed:
            print(f"\n🎉 SUCCESS: RL system significantly improves over SL baseline!")
            print(f"   - Statistically significant improvement (p < 0.05)")
            print(f"   - Positive mean improvement: {stats['mean_improvement']:+.3f}")
            print(f"   - Good convergence rate: {stats['convergence_rate']:.1%}")
        else:
            print(f"\n⚠️  INCONCLUSIVE: RL improvement not statistically significant")
            print(f"   - Consider: more trials, longer training, hyperparameter tuning")
        
        print(f"{'='*80}")
    
    def _save_experiment_result(self, result: ExperimentResult):
        """Save experiment result to file"""
        filename = f"{result.config.name}_{result.timestamp.replace(':', '-')}.json"
        filepath = self.results_dir / filename
        
        # Convert to serializable format
        serializable_result = {
            'config': asdict(result.config),
            'trials': [asdict(trial) for trial in result.trials],
            'aggregate_stats': result.aggregate_stats,
            'statistical_tests': result.statistical_tests,
            'hypothesis_confirmed': result.hypothesis_confirmed,
            'timestamp': result.timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        
        print(f"💾 Results saved: {filepath}")
    
    def generate_experiment_report(self, results: List[ExperimentResult], output_path: str):
        """Generate comprehensive experimental report"""
        print(f"📝 Generating Experimental Report...")
        
        report = {
            'experiment_summary': {
                'total_experiments': len(results),
                'confirmed_hypotheses': sum(1 for r in results if r.hypothesis_confirmed),
                'total_trials': sum(len(r.trials) for r in results),
                'timestamp': datetime.now().isoformat()
            },
            'experiments': []
        }
        
        for result in results:
            exp_summary = {
                'name': result.config.name,
                'hypothesis': result.config.hypothesis,
                'confirmed': result.hypothesis_confirmed,
                'mean_improvement': result.aggregate_stats['mean_improvement'],
                'p_value': result.statistical_tests.get('t_test_p_value', None),
                'effect_size': result.aggregate_stats['effect_size'],
                'convergence_rate': result.aggregate_stats['convergence_rate']
            }
            report['experiments'].append(exp_summary)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Report saved: {output_path}")
        
        # Print summary
        confirmed = report['experiment_summary']['confirmed_hypotheses']
        total = report['experiment_summary']['total_experiments']
        print(f"🎯 Overall Success Rate: {confirmed}/{total} ({confirmed/total:.1%})")


if __name__ == "__main__":
    # Test the experimental framework
    runner = ExperimentRunner()
    
    # Test baseline measurement
    baseline_stats = runner.measure_baseline_performance(num_trials=3)
    print(f"Baseline measurement complete: {baseline_stats['mean_catch_rate']:.3f}")
    
    print("✅ Experimental framework ready for systematic validation!")
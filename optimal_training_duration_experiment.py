#!/usr/bin/env python3
"""
Optimal Training Duration Experiment - Find PPO's Peak Performance

RESEARCH QUESTION: At what training duration does PPO achieve peak performance 
before overtraining degrades it below SL baseline?

HYPOTHESIS: PPO has an optimal training duration (likely 3-7 iterations) where 
it significantly outperforms SL baseline before overtraining causes degradation.

EXPERIMENTAL DESIGN:
1. Test PPO at multiple training durations: 1, 2, 3, 4, 5, 6, 7, 8, 10, 12 iterations
2. Use catch-only reward (proven optimal)
3. Use longer episodes (2500 steps)
4. Multiple validation runs per duration for statistical reliability
5. Find the peak performance point and validate it statistically
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
class DurationResult:
    """Result for a specific training duration"""
    iterations: int
    mean_performance: float
    std_performance: float
    confidence_interval: Tuple[float, float]
    all_results: List[float]
    beats_sl_baseline: bool
    improvement_over_sl: float
    training_time: float


class OptimalDurationExperiment:
    """Find optimal PPO training duration before overtraining"""
    
    def __init__(self):
        self.evaluator = PolicyEvaluator()
        self.sl_baseline_mean = None
        self.duration_results = []
        
        print("🎯 OPTIMAL TRAINING DURATION EXPERIMENT")
        print("=" * 60)
        print("GOAL: Find PPO's peak performance before overtraining")
        print("METHOD: Test multiple training durations with statistical validation")
        print("=" * 60)
        
        # Establish SL baseline
        self._establish_sl_baseline()
    
    def _establish_sl_baseline(self):
        """Quick SL baseline establishment"""
        print("\n📊 Establishing SL baseline...")
        sl_policy = TransformerPolicy("checkpoints/best_model.pt")
        
        # Use 5 evaluations for speed while maintaining reliability
        sl_results = []
        for i in range(5):
            result = self.evaluator.evaluate_policy(sl_policy, f"SL_Quick_{i+1}")
            sl_results.append(result.overall_catch_rate)
        
        self.sl_baseline_mean = np.mean(sl_results)
        print(f"✅ SL baseline: {self.sl_baseline_mean:.4f} ± {np.std(sl_results, ddof=1):.4f}")
    
    def test_training_duration(self, iterations: int, n_validations: int = 5) -> DurationResult:
        """Test PPO performance at specific training duration"""
        print(f"\n🧪 Testing {iterations} iterations ({n_validations} validation runs)...")
        
        start_time = time.time()
        
        # Train PPO with optimal settings
        trainer = PPOTrainer(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=0.01,          # Optimal from previous experiments
            rollout_steps=512,           # Optimal rollout size
            ppo_epochs=2,
            max_episode_steps=2500,      # Long episodes for strategy development
            device='cpu'
        )
        
        # Train for specified iterations
        for i in range(iterations):
            initial_state = generate_random_state(12, 400, 300)
            trainer.train_iteration(initial_state)
        
        training_time = time.time() - start_time
        
        # Multiple validation runs
        validation_results = []
        for i in range(n_validations):
            result = self.evaluator.evaluate_policy(trainer.policy, f"PPO_{iterations}iter_val{i+1}")
            validation_results.append(result.overall_catch_rate)
        
        # Statistical analysis
        mean_perf = np.mean(validation_results)
        std_perf = np.std(validation_results, ddof=1)
        
        # Confidence interval
        ci = stats.t.interval(0.95, len(validation_results) - 1, 
                             loc=mean_perf, scale=stats.sem(validation_results))
        
        # Comparison with SL baseline
        improvement = ((mean_perf - self.sl_baseline_mean) / self.sl_baseline_mean) * 100
        beats_baseline = mean_perf > self.sl_baseline_mean
        
        duration_result = DurationResult(
            iterations=iterations,
            mean_performance=mean_perf,
            std_performance=std_perf,
            confidence_interval=ci,
            all_results=validation_results,
            beats_sl_baseline=beats_baseline,
            improvement_over_sl=improvement,
            training_time=training_time
        )
        
        self.duration_results.append(duration_result)
        
        # Print results
        status = "✅ BEATS SL" if beats_baseline else "❌ Below SL"
        print(f"   {iterations} iterations: {mean_perf:.4f} ± {std_perf:.4f} ({improvement:+.1f}%) {status}")
        print(f"   95% CI: [{ci[0]:.4f}, {ci[1]:.4f}] | Time: {training_time:.1f}s")
        
        return duration_result
    
    def run_duration_sweep(self) -> Dict[str, Any]:
        """Test multiple training durations to find optimal point"""
        print(f"\n🔍 TRAINING DURATION SWEEP")
        print(f"Testing durations: [1, 2, 3, 4, 5, 6, 7, 8, 10, 12] iterations")
        
        # Test durations - focus on early iterations where we expect peak
        test_durations = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
        
        start_time = time.time()
        
        for duration in test_durations:
            self.test_training_duration(duration, n_validations=5)
        
        total_time = time.time() - start_time
        
        # Analysis
        print(f"\n📊 DURATION SWEEP RESULTS:")
        print("=" * 70)
        print(f"{'Iter':<6} {'Mean':<8} {'±Std':<8} {'Improve':<8} {'Beats SL':<8} {'95% CI':<20}")
        print("-" * 70)
        
        for result in self.duration_results:
            status = "✅" if result.beats_sl_baseline else "❌"
            ci_str = f"[{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]"
            print(f"{result.iterations:<6} {result.mean_performance:<8.4f} {result.std_performance:<8.4f} "
                  f"{result.improvement_over_sl:<8.1f} {status:<8} {ci_str:<20}")
        
        # Find optimal duration
        successful_results = [r for r in self.duration_results if r.beats_sl_baseline]
        
        if successful_results:
            # Find the duration with highest performance among those that beat SL
            optimal_result = max(successful_results, key=lambda x: x.mean_performance)
            
            print(f"\n🏆 OPTIMAL TRAINING DURATION FOUND:")
            print(f"   Duration: {optimal_result.iterations} iterations")
            print(f"   Performance: {optimal_result.mean_performance:.4f}")
            print(f"   Improvement: {optimal_result.improvement_over_sl:+.1f}% over SL")
            print(f"   Statistical: 95% CI [{optimal_result.confidence_interval[0]:.4f}, {optimal_result.confidence_interval[1]:.4f}]")
            
            # Check for overtraining pattern
            sorted_results = sorted(self.duration_results, key=lambda x: x.iterations)
            peak_idx = next(i for i, r in enumerate(sorted_results) if r.iterations == optimal_result.iterations)
            
            if peak_idx < len(sorted_results) - 1:
                later_results = sorted_results[peak_idx + 1:]
                declining = [r for r in later_results if r.mean_performance < optimal_result.mean_performance]
                if len(declining) >= 2:
                    print(f"   ⚠️  OVERTRAINING DETECTED: Performance declines after {optimal_result.iterations} iterations")
        else:
            print(f"\n❌ NO OPTIMAL DURATION FOUND:")
            print(f"   None of the tested durations beat SL baseline consistently")
            print(f"   Consider: Different hyperparameters or reward design")
            optimal_result = None
        
        # Save results
        results_summary = {
            'sl_baseline_mean': self.sl_baseline_mean,
            'test_durations': test_durations,
            'duration_results': [
                {
                    'iterations': r.iterations,
                    'mean_performance': r.mean_performance,
                    'std_performance': r.std_performance,
                    'confidence_interval': r.confidence_interval,
                    'improvement_over_sl': r.improvement_over_sl,
                    'beats_sl_baseline': r.beats_sl_baseline,
                    'training_time': r.training_time,
                    'all_results': r.all_results
                }
                for r in self.duration_results
            ],
            'optimal_duration': optimal_result.iterations if optimal_result else None,
            'optimal_performance': optimal_result.mean_performance if optimal_result else None,
            'total_experiment_time_minutes': total_time / 60,
            'experiment_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('optimal_duration_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"\n✅ Duration sweep complete: {total_time/60:.1f} minutes")
        print(f"📁 Results saved: optimal_duration_results.json")
        
        return results_summary
    
    def validate_optimal_duration(self, optimal_iterations: int) -> Dict[str, Any]:
        """Statistically validate the optimal duration with more rigorous testing"""
        print(f"\n🎯 VALIDATING OPTIMAL DURATION: {optimal_iterations} iterations")
        print("=" * 60)
        print("Running rigorous statistical validation with 10 independent runs")
        
        validation_results = []
        
        for run in range(10):
            print(f"\n🔄 Validation run {run + 1}/10...")
            
            # Fresh training for each validation
            trainer = PPOTrainer(
                sl_checkpoint_path="checkpoints/best_model.pt",
                learning_rate=0.01,
                rollout_steps=512,
                ppo_epochs=2,
                max_episode_steps=2500,
                device='cpu'
            )
            
            # Train for optimal duration
            for i in range(optimal_iterations):
                initial_state = generate_random_state(12, 400, 300)
                trainer.train_iteration(initial_state)
            
            # Evaluate
            result = self.evaluator.evaluate_policy(trainer.policy, f"OptimalValidation_{run+1}")
            validation_results.append(result.overall_catch_rate)
            
            print(f"   Run {run + 1}: {result.overall_catch_rate:.4f}")
        
        # Statistical analysis vs SL baseline
        ppo_mean = np.mean(validation_results)
        ppo_std = np.std(validation_results, ddof=1)
        
        # T-test against SL baseline
        # Using one-sample t-test (testing if PPO mean significantly > SL baseline)
        t_stat, p_value = stats.ttest_1samp(validation_results, self.sl_baseline_mean)
        
        # Effect size
        effect_size = (ppo_mean - self.sl_baseline_mean) / ppo_std
        
        # Confidence interval
        ci = stats.t.interval(0.95, len(validation_results) - 1, 
                             loc=ppo_mean, scale=stats.sem(validation_results))
        
        improvement = ((ppo_mean - self.sl_baseline_mean) / self.sl_baseline_mean) * 100
        
        print(f"\n📊 VALIDATION RESULTS:")
        print(f"   PPO ({optimal_iterations} iter): {ppo_mean:.4f} ± {ppo_std:.4f}")
        print(f"   SL Baseline:        {self.sl_baseline_mean:.4f}")
        print(f"   Improvement:        {improvement:+.1f}%")
        print(f"   95% CI:            [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"   t-statistic:       {t_stat:.4f}")
        print(f"   p-value:           {p_value:.6f}")
        print(f"   Effect size:       {effect_size:.4f}")
        
        is_significant = p_value < 0.05 and ppo_mean > self.sl_baseline_mean
        
        print(f"\n🏆 FINAL VALIDATION:")
        if is_significant:
            print(f"   ✅ PPO with {optimal_iterations} iterations SIGNIFICANTLY outperforms SL!")
            print(f"   📈 Statistical significance: p < 0.05")
            print(f"   💪 Mean improvement: {improvement:+.1f}%")
        else:
            print(f"   ❌ PPO with {optimal_iterations} iterations does not significantly outperform SL")
            print(f"   📊 p-value: {p_value:.6f} (need p < 0.05)")
        
        validation_summary = {
            'optimal_iterations': optimal_iterations,
            'ppo_mean': ppo_mean,
            'ppo_std': ppo_std,
            'sl_baseline_mean': self.sl_baseline_mean,
            'improvement_percent': improvement,
            'confidence_interval': ci,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'effect_size': effect_size,
            'validation_results': validation_results
        }
        
        return validation_summary


def main():
    """Run optimal training duration experiment"""
    print("🎯 OPTIMAL TRAINING DURATION EXPERIMENT")
    print("=" * 60)
    print("RESEARCH QUESTION: When does PPO peak before overtraining?")
    print("APPROACH:")
    print("  1. Test durations: 1-12 iterations")
    print("  2. Find peak performance point")  
    print("  3. Validate statistically")
    print("=" * 60)
    
    experiment = OptimalDurationExperiment()
    
    # Step 1: Duration sweep
    sweep_results = experiment.run_duration_sweep()
    
    # Step 2: Validate optimal duration if found
    if sweep_results['optimal_duration']:
        validation_results = experiment.validate_optimal_duration(sweep_results['optimal_duration'])
        
        if validation_results['is_significant']:
            print(f"\n🎉 BREAKTHROUGH CONFIRMED!")
            print(f"   Optimal PPO training: {sweep_results['optimal_duration']} iterations")
            print(f"   Significant improvement: {validation_results['improvement_percent']:+.1f}%")
            print(f"   p-value: {validation_results['p_value']:.6f}")
        else:
            print(f"\n📊 Optimal duration found but needs more validation")
    else:
        print(f"\n🤔 Need to investigate different approaches")
        print(f"   Consider: Lower learning rates, different reward scales")
    
    return sweep_results


if __name__ == "__main__":
    main()
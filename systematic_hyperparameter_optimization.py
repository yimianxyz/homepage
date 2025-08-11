#!/usr/bin/env python3
"""
Systematic Hyperparameter Optimization - Scientific approach to PPO tuning

RESEARCH OBJECTIVE: Find optimal PPO hyperparameters to surpass SL baseline
with aligned evaluation horizons (2500 steps).

KEY INSIGHTS TO TEST:
1. Learning Rate: Fine-tune from current 0.01 (may be too high)
2. GAE Lambda: Improve reward propagation to previous steps (0.95 -> variations)
3. Gamma (Discount): Long-horizon reward discounting for 2500-step episodes
4. PPO Epochs: Training iterations per rollout
5. Clip Epsilon: Policy update constraint

SYSTEMATIC APPROACH:
- Grid search on critical parameters
- Statistical validation (3-5 runs per configuration)
- Aligned evaluation horizon (2500 steps)
- Focus on catch-only rewards (proven optimal)
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from itertools import product

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from evaluation import PolicyEvaluator  # Now uses 2500-step evaluation
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration"""
    learning_rate: float
    gae_lambda: float
    gamma: float
    ppo_epochs: int
    clip_epsilon: float
    name: str


@dataclass
class HyperparameterResult:
    """Result from hyperparameter configuration test"""
    config: HyperparameterConfig
    mean_performance: float
    std_performance: float
    confidence_interval: Tuple[float, float]
    improvement_over_sl: float
    beats_sl_baseline: bool
    all_results: List[float]
    training_time: float
    evaluation_time: float


class SystematicHyperparameterOptimizer:
    """Scientific hyperparameter optimization for PPO"""
    
    def __init__(self):
        self.evaluator = PolicyEvaluator()  # Now uses 2500-step evaluation
        self.sl_baseline_mean = None
        self.results = []
        
        print("🧬 SYSTEMATIC HYPERPARAMETER OPTIMIZATION")
        print("=" * 60)
        print("OBJECTIVE: Find optimal PPO hyperparameters for 2500-step horizon")
        print("FOCUS: Learning rate, GAE lambda, gamma, PPO epochs, clip epsilon")
        print("METHOD: Grid search with statistical validation")
        print("=" * 60)
        
        # Establish SL baseline with new 2500-step evaluation
        self._establish_sl_baseline()
    
    def _establish_sl_baseline(self):
        """Establish SL baseline with 2500-step evaluation"""
        print("\n📊 Establishing SL baseline (2500-step evaluation)...")
        sl_policy = TransformerPolicy("checkpoints/best_model.pt")
        
        # Multiple evaluations for reliability
        sl_results = []
        for i in range(3):  # Fewer runs since 2500-step evaluation is longer
            print(f"   SL evaluation {i+1}/3...")
            result = self.evaluator.evaluate_policy(sl_policy, f"SL_2500step_{i+1}")
            sl_results.append(result.overall_catch_rate)
        
        self.sl_baseline_mean = np.mean(sl_results)
        sl_std = np.std(sl_results, ddof=1)
        
        print(f"✅ SL baseline (2500 steps): {self.sl_baseline_mean:.4f} ± {sl_std:.4f}")
        print(f"   This is the new target for PPO to beat")
    
    def test_hyperparameter_config(self, config: HyperparameterConfig, 
                                   n_runs: int = 3, training_iterations: int = 5) -> HyperparameterResult:
        """Test a specific hyperparameter configuration"""
        print(f"\n🧪 Testing: {config.name}")
        print(f"   LR: {config.learning_rate}, GAE λ: {config.gae_lambda}, γ: {config.gamma}")
        print(f"   PPO epochs: {config.ppo_epochs}, Clip ε: {config.clip_epsilon}")
        
        results = []
        total_training_time = 0
        total_evaluation_time = 0
        
        for run in range(n_runs):
            print(f"   Run {run+1}/{n_runs}...")
            
            # Training
            train_start = time.time()
            trainer = PPOTrainer(
                sl_checkpoint_path="checkpoints/best_model.pt",
                learning_rate=config.learning_rate,
                clip_epsilon=config.clip_epsilon,
                ppo_epochs=config.ppo_epochs,
                rollout_steps=512,  # Keep proven rollout size
                max_episode_steps=2500,  # Aligned with evaluation
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                device='cpu'
            )
            
            # Train for specified iterations
            for i in range(training_iterations):
                initial_state = generate_random_state(12, 400, 300)
                trainer.train_iteration(initial_state)
            
            training_time = time.time() - train_start
            total_training_time += training_time
            
            # Evaluation (now 2500 steps, aligned with training)
            eval_start = time.time()
            result = self.evaluator.evaluate_policy(trainer.policy, f"{config.name}_run{run+1}")
            evaluation_time = time.time() - eval_start
            total_evaluation_time += evaluation_time
            
            results.append(result.overall_catch_rate)
            print(f"     Result: {result.overall_catch_rate:.4f}")
        
        # Statistical analysis
        mean_perf = np.mean(results)
        std_perf = np.std(results, ddof=1) if len(results) > 1 else 0
        
        # Confidence interval
        if len(results) > 1:
            ci = stats.t.interval(0.95, len(results) - 1, 
                                 loc=mean_perf, scale=stats.sem(results))
        else:
            ci = (mean_perf, mean_perf)
        
        # Comparison with SL baseline
        improvement = ((mean_perf - self.sl_baseline_mean) / self.sl_baseline_mean) * 100
        beats_baseline = mean_perf > self.sl_baseline_mean
        
        hyperparameter_result = HyperparameterResult(
            config=config,
            mean_performance=mean_perf,
            std_performance=std_perf,
            confidence_interval=ci,
            improvement_over_sl=improvement,
            beats_sl_baseline=beats_baseline,
            all_results=results,
            training_time=total_training_time,
            evaluation_time=total_evaluation_time
        )
        
        self.results.append(hyperparameter_result)
        
        # Print summary
        status = "✅ BEATS SL" if beats_baseline else "❌ Below SL"
        print(f"   RESULT: {mean_perf:.4f} ± {std_perf:.4f} ({improvement:+.1f}%) {status}")
        
        return hyperparameter_result
    
    def experiment_1_learning_rate_fine_tuning(self) -> Dict[str, Any]:
        """
        Experiment 1: Fine-tune learning rate
        
        Current: 0.01 may be too aggressive for long-horizon training
        Test: Conservative to moderate learning rates
        """
        print(f"\n{'='*60}")
        print(f"🧪 EXPERIMENT 1: LEARNING RATE FINE-TUNING")
        print(f"{'='*60}")
        print(f"Hypothesis: Current 0.01 LR is too high for 2500-step training")
        
        # Test learning rates from conservative to moderate
        learning_rates = [0.0001, 0.0003, 0.001, 0.003, 0.005, 0.01]
        
        results = []
        for lr in learning_rates:
            config = HyperparameterConfig(
                learning_rate=lr,
                gae_lambda=0.95,  # Default
                gamma=0.99,       # Default  
                ppo_epochs=2,     # Default
                clip_epsilon=0.2, # Default
                name=f"LR_{lr:.4f}"
            )
            
            result = self.test_hyperparameter_config(config, n_runs=3, training_iterations=5)
            results.append((lr, result))
        
        # Find best learning rate
        best_lr_result = max(results, key=lambda x: x[1].mean_performance)
        best_lr = best_lr_result[0]
        
        print(f"\n📊 LEARNING RATE RESULTS:")
        for lr, result in results:
            status = "✅" if result.beats_sl_baseline else "❌"
            print(f"   LR {lr:.4f}: {result.mean_performance:.4f} ({result.improvement_over_sl:+.1f}%) {status}")
        
        print(f"\n🏆 BEST LEARNING RATE: {best_lr}")
        
        return {
            'best_learning_rate': best_lr,
            'best_result': best_lr_result[1],
            'all_results': results
        }
    
    def experiment_2_gae_lambda_optimization(self, optimal_lr: float) -> Dict[str, Any]:
        """
        Experiment 2: GAE Lambda for reward propagation
        
        Key insight: Catch rewards should propagate to previous steps
        Test different GAE lambda values for better credit assignment
        """
        print(f"\n{'='*60}")
        print(f"🧪 EXPERIMENT 2: GAE LAMBDA OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Hypothesis: Higher GAE λ improves catch reward propagation")
        print(f"Using optimal LR: {optimal_lr}")
        
        # Test GAE lambda values for better credit assignment
        gae_lambdas = [0.90, 0.95, 0.97, 0.99, 0.995]
        
        results = []
        for gae_lambda in gae_lambdas:
            config = HyperparameterConfig(
                learning_rate=optimal_lr,
                gae_lambda=gae_lambda,
                gamma=0.99,       # Default
                ppo_epochs=2,     # Default
                clip_epsilon=0.2, # Default
                name=f"GAE_{gae_lambda:.3f}"
            )
            
            result = self.test_hyperparameter_config(config, n_runs=3, training_iterations=5)
            results.append((gae_lambda, result))
        
        # Find best GAE lambda
        best_gae_result = max(results, key=lambda x: x[1].mean_performance)
        best_gae = best_gae_result[0]
        
        print(f"\n📊 GAE LAMBDA RESULTS:")
        for gae_lambda, result in results:
            status = "✅" if result.beats_sl_baseline else "❌"
            print(f"   GAE λ {gae_lambda:.3f}: {result.mean_performance:.4f} ({result.improvement_over_sl:+.1f}%) {status}")
        
        print(f"\n🏆 BEST GAE LAMBDA: {best_gae}")
        
        return {
            'best_gae_lambda': best_gae,
            'best_result': best_gae_result[1],
            'all_results': results
        }
    
    def experiment_3_gamma_discount_optimization(self, optimal_lr: float, optimal_gae: float) -> Dict[str, Any]:
        """
        Experiment 3: Gamma (discount factor) for long-horizon episodes
        
        2500-step episodes need appropriate discounting
        """
        print(f"\n{'='*60}")
        print(f"🧪 EXPERIMENT 3: GAMMA DISCOUNT OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Hypothesis: 2500-step episodes need optimized discount factor")
        print(f"Using optimal LR: {optimal_lr}, GAE λ: {optimal_gae}")
        
        # Test gamma values for long-horizon episodes
        gammas = [0.95, 0.97, 0.99, 0.995, 0.999]
        
        results = []
        for gamma in gammas:
            config = HyperparameterConfig(
                learning_rate=optimal_lr,
                gae_lambda=optimal_gae,
                gamma=gamma,
                ppo_epochs=2,     # Default
                clip_epsilon=0.2, # Default
                name=f"Gamma_{gamma:.3f}"
            )
            
            result = self.test_hyperparameter_config(config, n_runs=3, training_iterations=5)
            results.append((gamma, result))
        
        # Find best gamma
        best_gamma_result = max(results, key=lambda x: x[1].mean_performance)
        best_gamma = best_gamma_result[0]
        
        print(f"\n📊 GAMMA RESULTS:")
        for gamma, result in results:
            status = "✅" if result.beats_sl_baseline else "❌"
            print(f"   γ {gamma:.3f}: {result.mean_performance:.4f} ({result.improvement_over_sl:+.1f}%) {status}")
        
        print(f"\n🏆 BEST GAMMA: {best_gamma}")
        
        return {
            'best_gamma': best_gamma,
            'best_result': best_gamma_result[1],
            'all_results': results
        }
    
    def experiment_4_ppo_epochs_optimization(self, optimal_lr: float, optimal_gae: float, 
                                            optimal_gamma: float) -> Dict[str, Any]:
        """
        Experiment 4: PPO epochs optimization
        
        More training epochs per rollout might help with long-horizon learning
        """
        print(f"\n{'='*60}")
        print(f"🧪 EXPERIMENT 4: PPO EPOCHS OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Hypothesis: More PPO epochs improve long-horizon learning")
        print(f"Using optimal LR: {optimal_lr}, GAE λ: {optimal_gae}, γ: {optimal_gamma}")
        
        # Test different PPO epochs
        ppo_epochs_list = [1, 2, 3, 4, 5]
        
        results = []
        for ppo_epochs in ppo_epochs_list:
            config = HyperparameterConfig(
                learning_rate=optimal_lr,
                gae_lambda=optimal_gae,
                gamma=optimal_gamma,
                ppo_epochs=ppo_epochs,
                clip_epsilon=0.2, # Default
                name=f"Epochs_{ppo_epochs}"
            )
            
            result = self.test_hyperparameter_config(config, n_runs=3, training_iterations=5)
            results.append((ppo_epochs, result))
        
        # Find best PPO epochs
        best_epochs_result = max(results, key=lambda x: x[1].mean_performance)
        best_epochs = best_epochs_result[0]
        
        print(f"\n📊 PPO EPOCHS RESULTS:")
        for ppo_epochs, result in results:
            status = "✅" if result.beats_sl_baseline else "❌"
            print(f"   Epochs {ppo_epochs}: {result.mean_performance:.4f} ({result.improvement_over_sl:+.1f}%) {status}")
        
        print(f"\n🏆 BEST PPO EPOCHS: {best_epochs}")
        
        return {
            'best_ppo_epochs': best_epochs,
            'best_result': best_epochs_result[1],
            'all_results': results
        }
    
    def run_systematic_optimization(self) -> Dict[str, Any]:
        """Run complete systematic hyperparameter optimization"""
        print(f"\n🚀 SYSTEMATIC HYPERPARAMETER OPTIMIZATION")
        print(f"Goal: Find optimal PPO hyperparameters for 2500-step episodes")
        
        start_time = time.time()
        
        # Experiment 1: Learning rate
        exp1_results = self.experiment_1_learning_rate_fine_tuning()
        optimal_lr = exp1_results['best_learning_rate']
        
        # Experiment 2: GAE lambda (using optimal LR)
        exp2_results = self.experiment_2_gae_lambda_optimization(optimal_lr)
        optimal_gae = exp2_results['best_gae_lambda']
        
        # Experiment 3: Gamma (using optimal LR + GAE)
        exp3_results = self.experiment_3_gamma_discount_optimization(optimal_lr, optimal_gae)
        optimal_gamma = exp3_results['best_gamma']
        
        # Experiment 4: PPO epochs (using all optimal parameters)
        exp4_results = self.experiment_4_ppo_epochs_optimization(optimal_lr, optimal_gae, optimal_gamma)
        optimal_epochs = exp4_results['best_ppo_epochs']
        
        total_time = time.time() - start_time
        
        # Final optimal configuration
        optimal_config = HyperparameterConfig(
            learning_rate=optimal_lr,
            gae_lambda=optimal_gae, 
            gamma=optimal_gamma,
            ppo_epochs=optimal_epochs,
            clip_epsilon=0.2,  # Keep default for now
            name="OPTIMAL"
        )
        
        # Final validation with optimal config
        print(f"\n🎯 FINAL VALIDATION WITH OPTIMAL CONFIG")
        final_result = self.test_hyperparameter_config(optimal_config, n_runs=5, training_iterations=8)
        
        print(f"\n{'='*80}")
        print(f"🏆 SYSTEMATIC HYPERPARAMETER OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"SL Baseline (2500 steps): {self.sl_baseline_mean:.4f}")
        print(f"Final PPO Performance:    {final_result.mean_performance:.4f}")
        print(f"Improvement:              {final_result.improvement_over_sl:+.1f}%")
        print(f"")
        print(f"🔧 OPTIMAL HYPERPARAMETERS:")
        print(f"   Learning Rate:  {optimal_lr}")
        print(f"   GAE Lambda:     {optimal_gae}")
        print(f"   Gamma:          {optimal_gamma}")
        print(f"   PPO Epochs:     {optimal_epochs}")
        print(f"   Clip Epsilon:   0.2")
        print(f"")
        print(f"Total optimization time: {total_time/60:.1f} minutes")
        
        # Save comprehensive results
        optimization_results = {
            'sl_baseline': self.sl_baseline_mean,
            'final_performance': final_result.mean_performance,
            'improvement_percent': final_result.improvement_over_sl,
            'beats_sl_baseline': final_result.beats_sl_baseline,
            'optimal_hyperparameters': {
                'learning_rate': optimal_lr,
                'gae_lambda': optimal_gae,
                'gamma': optimal_gamma,
                'ppo_epochs': optimal_epochs,
                'clip_epsilon': 0.2
            },
            'experiment_1_learning_rate': exp1_results,
            'experiment_2_gae_lambda': exp2_results,
            'experiment_3_gamma': exp3_results,
            'experiment_4_ppo_epochs': exp4_results,
            'final_validation': {
                'mean': final_result.mean_performance,
                'std': final_result.std_performance,
                'confidence_interval': final_result.confidence_interval,
                'all_results': final_result.all_results
            },
            'total_time_minutes': total_time/60,
            'evaluation_horizon': '2500_steps_aligned_with_training'
        }
        
        with open('systematic_hyperparameter_results.json', 'w') as f:
            json.dump(optimization_results, f, indent=2, default=str)
        
        print(f"✅ Results saved: systematic_hyperparameter_results.json")
        
        return optimization_results


def main():
    """Run systematic hyperparameter optimization"""
    print("🧬 SYSTEMATIC HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print("APPROACH:")
    print("  1. Aligned evaluation horizon (2500 steps)")
    print("  2. Learning rate fine-tuning")
    print("  3. GAE lambda for reward propagation")
    print("  4. Gamma optimization for long episodes")
    print("  5. PPO epochs optimization")
    print("  6. Statistical validation")
    print("=" * 60)
    
    optimizer = SystematicHyperparameterOptimizer()
    results = optimizer.run_systematic_optimization()
    
    if results['beats_sl_baseline']:
        print(f"\n🎉 BREAKTHROUGH ACHIEVED!")
        print(f"   PPO systematically beats SL baseline by {results['improvement_percent']:+.1f}%")
        print(f"   Optimal configuration found and validated")
    else:
        print(f"\n📊 Optimization complete: {results['improvement_percent']:+.1f}% vs SL")
        print(f"   Best hyperparameters identified for future work")
    
    return results


if __name__ == "__main__":
    main()
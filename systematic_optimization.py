#!/usr/bin/env python3
"""
Systematic PPO Optimization - Evolutionary approach to surpass SL baseline

This script conducts systematic experiments to optimize PPO training:
1. Early Stopping Validation - Find optimal training duration
2. Learning Rate Optimization - Based on optimal stopping point
3. Rollout Size Optimization - Based on best hyperparameters
4. Training Diversity - Based on optimal configuration

Each experiment builds on previous insights using evolutionary principles.
"""

import os
import sys
import time
import json
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy
from policy.human_prior.random_policy import RandomPolicy
from policy.human_prior.closest_pursuit_policy import ClosestPursuitPolicy


@dataclass
class ExperimentResult:
    """Result from a single experiment"""
    experiment_name: str
    config: Dict[str, Any]
    overall_catch_rate: float
    strategic_insights: Dict[str, Any]
    training_time: float
    evaluation_time: float
    beats_sl_baseline: bool
    improvement_over_sl: float


class SystematicOptimizer:
    """Systematic PPO optimization using evolutionary experimentation"""
    
    def __init__(self):
        self.evaluator = PolicyEvaluator()
        self.experiment_history = []
        self.sl_baseline_rate = None
        
        # Get SL baseline performance once
        self._establish_sl_baseline()
    
    def _establish_sl_baseline(self):
        """Establish SL baseline performance for comparison"""
        print("📊 Establishing SL baseline performance...")
        
        if os.path.exists("checkpoints/best_model.pt"):
            sl_policy = TransformerPolicy("checkpoints/best_model.pt")
            result = self.evaluator.evaluate_policy(sl_policy, "SL_Baseline")
            self.sl_baseline_rate = result.overall_catch_rate
            
            print(f"✅ SL baseline established: {self.sl_baseline_rate:.3f} catch rate")
            print(f"   Strategic profile: {result.strategy_type} | {result.formation_style}")
        else:
            print("❌ No SL baseline found - cannot run optimization")
            sys.exit(1)
    
    def run_experiment(self, name: str, config: Dict[str, Any]) -> ExperimentResult:
        """Run a single PPO experiment with given configuration"""
        print(f"\n🧪 EXPERIMENT: {name}")
        print(f"   Config: {config}")
        
        start_time = time.time()
        
        # Train PPO with config
        trainer = PPOTrainer(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=config.get('learning_rate', 3e-4),
            rollout_steps=config.get('rollout_steps', 512),
            ppo_epochs=config.get('ppo_epochs', 2),
            device='cpu'
        )
        
        # Training
        for i in range(config.get('iterations', 3)):
            initial_state = generate_random_state(
                config.get('boids', 20),
                config.get('canvas_width', 800), 
                config.get('canvas_height', 600)
            )
            trainer.train_iteration(initial_state)
        
        training_time = time.time() - start_time
        
        # Strategic evaluation
        eval_start = time.time()
        result = self.evaluator.evaluate_policy(trainer.policy, f"{name}_Result")
        evaluation_time = time.time() - eval_start
        
        # Analysis
        improvement = result.overall_catch_rate - self.sl_baseline_rate
        improvement_pct = (improvement / max(self.sl_baseline_rate, 0.001)) * 100
        beats_baseline = result.overall_catch_rate > self.sl_baseline_rate
        
        experiment_result = ExperimentResult(
            experiment_name=name,
            config=config,
            overall_catch_rate=result.overall_catch_rate,
            strategic_insights={
                'strategy_type': result.strategy_type,
                'formation_style': result.formation_style,
                'early_phase_rate': result.early_phase_rate,
                'mid_phase_rate': result.mid_phase_rate,
                'late_phase_rate': result.late_phase_rate,
                'adaptability_score': result.adaptability_score,
                'strategy_consistency': result.strategy_consistency
            },
            training_time=training_time,
            evaluation_time=evaluation_time,
            beats_sl_baseline=beats_baseline,
            improvement_over_sl=improvement_pct
        )
        
        self.experiment_history.append(experiment_result)
        
        # Print results
        status = "✅ BEATS SL" if beats_baseline else "❌ Below SL"
        print(f"   Result: {result.overall_catch_rate:.3f} vs SL {self.sl_baseline_rate:.3f} ({improvement_pct:+.1f}%) {status}")
        print(f"   Profile: {result.strategy_type} | {result.formation_style}")
        print(f"   Times: Train {training_time:.1f}s | Eval {evaluation_time:.1f}s")
        
        return experiment_result
    
    def experiment_1_early_stopping(self) -> Dict[str, Any]:
        """
        Experiment 1: Early Stopping Validation
        
        Hypothesis: Optimal training duration exists between 1-6 iterations
        Test different stopping points to find sweet spot
        """
        print(f"\n{'='*60}")
        print(f"🔬 EXPERIMENT 1: EARLY STOPPING VALIDATION")
        print(f"{'='*60}")
        print(f"Hypothesis: PPO_iter2 beat SL, PPO_iter4 regressed → optimal stopping exists")
        
        results = []
        base_config = {
            'learning_rate': 3e-4,
            'rollout_steps': 512,
            'ppo_epochs': 2,
            'boids': 12,  # Faster training
            'canvas_width': 400,
            'canvas_height': 300
        }
        
        # Test different iteration counts
        for iterations in [1, 2, 3, 4, 5, 6]:
            config = base_config.copy()
            config['iterations'] = iterations
            
            result = self.run_experiment(f"EarlyStop_Iter{iterations}", config)
            results.append((iterations, result))
        
        # Find best iteration count
        best_iterations = max(results, key=lambda x: x[1].overall_catch_rate)
        best_result = best_iterations[1]
        
        print(f"\n📊 EXPERIMENT 1 RESULTS:")
        print(f"   Best iteration count: {best_iterations[0]}")
        print(f"   Best performance: {best_result.overall_catch_rate:.3f}")
        print(f"   Improvement over SL: {best_result.improvement_over_sl:+.1f}%")
        
        return {
            'best_iterations': best_iterations[0],
            'best_config': best_result.config,
            'best_performance': best_result.overall_catch_rate,
            'all_results': results
        }
    
    def experiment_2_learning_rate(self, optimal_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Experiment 2: Learning Rate Optimization
        
        Based on optimal stopping point from Experiment 1,
        test different learning rates
        """
        print(f"\n{'='*60}")
        print(f"🔬 EXPERIMENT 2: LEARNING RATE OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Using optimal iterations: {optimal_config['iterations']}")
        
        results = []
        base_config = optimal_config.copy()
        
        # Test different learning rates
        learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
        
        for lr in learning_rates:
            config = base_config.copy()
            config['learning_rate'] = lr
            
            result = self.run_experiment(f"LearningRate_{lr:.0e}", config)
            results.append((lr, result))
        
        # Find best learning rate
        best_lr = max(results, key=lambda x: x[1].overall_catch_rate)
        best_result = best_lr[1]
        
        print(f"\n📊 EXPERIMENT 2 RESULTS:")
        print(f"   Best learning rate: {best_lr[0]:.0e}")
        print(f"   Best performance: {best_result.overall_catch_rate:.3f}")
        print(f"   Improvement over SL: {best_result.improvement_over_sl:+.1f}%")
        
        return {
            'best_learning_rate': best_lr[0],
            'best_config': best_result.config,
            'best_performance': best_result.overall_catch_rate,
            'all_results': results
        }
    
    def experiment_3_rollout_size(self, optimal_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Experiment 3: Rollout Size Optimization
        
        Based on optimal hyperparameters from previous experiments,
        test different rollout sizes for sample efficiency
        """
        print(f"\n{'='*60}")
        print(f"🔬 EXPERIMENT 3: ROLLOUT SIZE OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Using optimal config from previous experiments")
        
        results = []
        base_config = optimal_config.copy()
        
        # Test different rollout sizes
        rollout_sizes = [128, 256, 512, 1024, 2048]
        
        for rollout_steps in rollout_sizes:
            config = base_config.copy()
            config['rollout_steps'] = rollout_steps
            
            result = self.run_experiment(f"Rollout_{rollout_steps}", config)
            results.append((rollout_steps, result))
        
        # Find best rollout size
        best_rollout = max(results, key=lambda x: x[1].overall_catch_rate)
        best_result = best_rollout[1]
        
        print(f"\n📊 EXPERIMENT 3 RESULTS:")
        print(f"   Best rollout size: {best_rollout[0]}")
        print(f"   Best performance: {best_result.overall_catch_rate:.3f}")
        print(f"   Improvement over SL: {best_result.improvement_over_sl:+.1f}%")
        
        return {
            'best_rollout_size': best_rollout[0],
            'best_config': best_result.config,
            'best_performance': best_result.overall_catch_rate,
            'all_results': results
        }
    
    def run_systematic_optimization(self) -> Dict[str, Any]:
        """
        Run complete systematic optimization pipeline
        
        Returns final optimal configuration
        """
        print(f"\n🚀 SYSTEMATIC PPO OPTIMIZATION")
        print(f"Goal: Surpass SL baseline ({self.sl_baseline_rate:.3f}) using evolutionary experimentation")
        
        start_time = time.time()
        
        # Experiment 1: Find optimal training duration
        exp1_results = self.experiment_1_early_stopping()
        
        # Experiment 2: Optimize learning rate based on optimal duration
        exp2_results = self.experiment_2_learning_rate(exp1_results['best_config'])
        
        # Experiment 3: Optimize rollout size based on best hyperparameters
        exp3_results = self.experiment_3_rollout_size(exp2_results['best_config'])
        
        total_time = time.time() - start_time
        
        # Final analysis
        final_config = exp3_results['best_config']
        final_performance = exp3_results['best_performance']
        final_improvement = (final_performance - self.sl_baseline_rate) / self.sl_baseline_rate * 100
        
        print(f"\n{'='*80}")
        print(f"🏆 SYSTEMATIC OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"SL Baseline:     {self.sl_baseline_rate:.3f}")
        print(f"Final PPO:       {final_performance:.3f}")
        print(f"Improvement:     {final_improvement:+.1f}%")
        print(f"")
        print(f"Optimal Configuration:")
        for key, value in final_config.items():
            print(f"  {key}: {value}")
        print(f"")
        print(f"Total optimization time: {total_time/60:.1f} minutes")
        
        # Save results
        optimization_results = {
            'sl_baseline': self.sl_baseline_rate,
            'final_performance': final_performance,
            'improvement_percent': final_improvement,
            'optimal_config': final_config,
            'experiment_1': exp1_results,
            'experiment_2': exp2_results, 
            'experiment_3': exp3_results,
            'total_time_minutes': total_time/60,
            'all_experiments': [exp.__dict__ for exp in self.experiment_history]
        }
        
        with open('systematic_optimization_results.json', 'w') as f:
            json.dump(optimization_results, f, indent=2, default=str)
        
        print(f"✅ Results saved: systematic_optimization_results.json")
        
        return optimization_results


def main():
    """Run systematic PPO optimization"""
    print("🧬 SYSTEMATIC PPO OPTIMIZATION - EVOLUTIONARY APPROACH")
    print("=" * 70)
    
    optimizer = SystematicOptimizer()
    results = optimizer.run_systematic_optimization()
    
    if results['improvement_percent'] > 0:
        print(f"\n🎉 SUCCESS: PPO surpasses SL baseline by {results['improvement_percent']:.1f}%!")
    else:
        print(f"\n⚠️  More optimization needed - current best: {results['improvement_percent']:.1f}%")
    
    return results


if __name__ == "__main__":
    main()
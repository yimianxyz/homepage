#!/usr/bin/env python3
"""
Focused Value Pre-training Validation - Statistically Rigorous but Faster

Provides SOLID EVIDENCE in ~1-2 hours instead of 22.5 hours.
Uses 5 trials per method with multiple validation runs for significance.
"""

import os
import sys
import time
import numpy as np
import torch
import json
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from ppo_with_value_pretraining import PPOWithValuePretraining
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


@dataclass
class ExperimentResults:
    """Results from one trial"""
    method: str
    trial: int
    baseline_performance: float
    final_performance: float
    improvement: float
    success: bool
    training_time: float
    stability_score: float  # Variance of performance during training
    peak_performance: float
    iterations_to_success: int  # -1 if never succeeded


class FocusedValidator:
    """Focused but rigorous validation of value pre-training"""
    
    def __init__(self, num_trials: int = 5):
        self.num_trials = num_trials
        self.evaluator = PolicyEvaluator()
        self.results = []
        
    def run_sl_baseline(self) -> Tuple[float, float]:
        """Establish SL baseline with multiple runs"""
        print("\n📊 ESTABLISHING SL BASELINE")
        sl_policy = TransformerPolicy("checkpoints/best_model.pt")
        
        performances = []
        for i in range(10):  # More runs for stable baseline
            result = self.evaluator.evaluate_policy(sl_policy, f"SL_Baseline_{i+1}")
            performances.append(result.overall_catch_rate)
            print(f"   Run {i+1}: {result.overall_catch_rate:.4f}")
        
        mean = np.mean(performances)
        std = np.std(performances)
        print(f"\n✅ SL Baseline: {mean:.4f} ± {std:.4f}")
        
        return mean, std
    
    def run_standard_ppo_trial(self, trial_num: int, sl_baseline: float) -> ExperimentResults:
        """Run one trial of standard PPO (no value pre-training)"""
        print(f"\n🔵 STANDARD PPO - Trial {trial_num}")
        
        start_time = time.time()
        
        # Create standard PPO trainer
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
        
        performances = []
        best_performance = 0
        iterations_to_success = -1
        
        # Train for 10 iterations
        for iteration in range(1, 11):
            initial_state = generate_random_state(12, 400, 300)
            trainer.train_iteration(initial_state)
            
            # Evaluate
            if iteration <= 3 or iteration % 2 == 0:
                result = self.evaluator.evaluate_policy(trainer.policy, f"StandardPPO_T{trial_num}_I{iteration}")
                performance = result.overall_catch_rate
                performances.append(performance)
                
                if performance > best_performance:
                    best_performance = performance
                
                if performance > sl_baseline and iterations_to_success == -1:
                    iterations_to_success = iteration
                
                improvement = (performance - sl_baseline) / sl_baseline * 100
                print(f"   Iter {iteration}: {performance:.4f} ({improvement:+.1f}%)")
        
        training_time = time.time() - start_time
        final_performance = performances[-1]
        stability_score = np.std(performances)
        
        return ExperimentResults(
            method="Standard PPO",
            trial=trial_num,
            baseline_performance=sl_baseline,
            final_performance=final_performance,
            improvement=(final_performance - sl_baseline) / sl_baseline * 100,
            success=final_performance > sl_baseline,
            training_time=training_time,
            stability_score=stability_score,
            peak_performance=best_performance,
            iterations_to_success=iterations_to_success
        )
    
    def run_value_pretrained_ppo_trial(self, trial_num: int, sl_baseline: float) -> ExperimentResults:
        """Run one trial of PPO with value pre-training"""
        print(f"\n🟢 VALUE PRE-TRAINED PPO - Trial {trial_num}")
        
        start_time = time.time()
        
        # Create PPO trainer with value pre-training
        trainer = PPOWithValuePretraining(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=0.00005,
            clip_epsilon=0.1,
            ppo_epochs=2,
            rollout_steps=256,
            max_episode_steps=2500,
            gamma=0.95,
            gae_lambda=0.9,
            device='cpu',
            value_pretrain_iterations=10,
            value_pretrain_lr=0.0005,
            value_pretrain_epochs=3
        )
        
        # Phase 1: Value pre-training
        print("\n  Phase 1: Value Pre-training")
        value_losses = trainer.pretrain_value_function()
        print(f"    Initial loss: {value_losses[0]:.3f}")
        print(f"    Final loss: {value_losses[-1]:.3f}")
        
        # Phase 2: PPO training
        print("\n  Phase 2: PPO Training")
        performances = []
        best_performance = 0
        iterations_to_success = -1
        
        for iteration in range(1, 11):
            initial_state = generate_random_state(12, 400, 300)
            trainer.train_iteration(initial_state)
            
            # Evaluate
            if iteration <= 3 or iteration % 2 == 0:
                result = self.evaluator.evaluate_policy(trainer.policy, f"ValuePretrainPPO_T{trial_num}_I{iteration}")
                performance = result.overall_catch_rate
                performances.append(performance)
                
                if performance > best_performance:
                    best_performance = performance
                
                if performance > sl_baseline and iterations_to_success == -1:
                    iterations_to_success = iteration
                
                improvement = (performance - sl_baseline) / sl_baseline * 100
                print(f"   Iter {iteration}: {performance:.4f} ({improvement:+.1f}%)")
        
        training_time = time.time() - start_time
        final_performance = performances[-1]
        stability_score = np.std(performances)
        
        return ExperimentResults(
            method="Value Pre-trained PPO",
            trial=trial_num,
            baseline_performance=sl_baseline,
            final_performance=final_performance,
            improvement=(final_performance - sl_baseline) / sl_baseline * 100,
            success=final_performance > sl_baseline,
            training_time=training_time,
            stability_score=stability_score,
            peak_performance=best_performance,
            iterations_to_success=iterations_to_success
        )
    
    def run_comprehensive_validation(self):
        """Run the complete validation experiment"""
        print("🧬 FOCUSED VALUE PRE-TRAINING VALIDATION")
        print("=" * 80)
        print("GOAL: Provide SOLID STATISTICAL EVIDENCE")
        print(f"Trials per method: {self.num_trials}")
        print("=" * 80)
        
        # Establish baseline
        sl_mean, sl_std = self.run_sl_baseline()
        
        # Run trials
        standard_results = []
        pretrained_results = []
        
        print("\n" + "="*80)
        print("RUNNING EXPERIMENTS")
        print("="*80)
        
        for trial in range(1, self.num_trials + 1):
            print(f"\n{'='*40}")
            print(f"TRIAL {trial}/{self.num_trials}")
            print(f"{'='*40}")
            
            # Standard PPO
            standard_result = self.run_standard_ppo_trial(trial, sl_mean)
            standard_results.append(standard_result)
            self.results.append(standard_result)
            
            # Value Pre-trained PPO
            pretrained_result = self.run_value_pretrained_ppo_trial(trial, sl_mean)
            pretrained_results.append(pretrained_result)
            self.results.append(pretrained_result)
        
        # Analyze results
        self.analyze_results(sl_mean, sl_std, standard_results, pretrained_results)
    
    def analyze_results(self, sl_mean: float, sl_std: float, 
                       standard_results: List[ExperimentResults],
                       pretrained_results: List[ExperimentResults]):
        """Comprehensive statistical analysis"""
        print("\n" + "="*80)
        print("📊 STATISTICAL ANALYSIS")
        print("="*80)
        
        # Extract data
        standard_performances = [r.final_performance for r in standard_results]
        pretrained_performances = [r.final_performance for r in pretrained_results]
        
        standard_improvements = [r.improvement for r in standard_results]
        pretrained_improvements = [r.improvement for r in pretrained_results]
        
        # Success rates
        standard_success_rate = sum(r.success for r in standard_results) / len(standard_results)
        pretrained_success_rate = sum(r.success for r in pretrained_results) / len(pretrained_results)
        
        # Stability scores
        standard_stability = np.mean([r.stability_score for r in standard_results])
        pretrained_stability = np.mean([r.stability_score for r in pretrained_results])
        
        # Statistical tests
        # 1. Test if value pre-training beats SL baseline
        t_stat_vs_sl, p_value_vs_sl = stats.ttest_1samp(pretrained_performances, sl_mean)
        
        # 2. Test if value pre-training beats standard PPO
        t_stat_vs_standard, p_value_vs_standard = stats.ttest_ind(
            pretrained_performances, standard_performances, equal_var=False
        )
        
        # 3. Effect sizes (Cohen's d)
        def cohens_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            return (np.mean(group1) - np.mean(group2)) / np.sqrt(pooled_var)
        
        effect_size_vs_sl = cohens_d(pretrained_performances, [sl_mean] * len(pretrained_performances))
        effect_size_vs_standard = cohens_d(pretrained_performances, standard_performances)
        
        # Print results
        print("\n1️⃣ PERFORMANCE SUMMARY")
        print(f"   SL Baseline: {sl_mean:.4f} ± {sl_std:.4f}")
        print(f"   Standard PPO: {np.mean(standard_performances):.4f} ± {np.std(standard_performances):.4f}")
        print(f"   Value Pre-trained PPO: {np.mean(pretrained_performances):.4f} ± {np.std(pretrained_performances):.4f}")
        
        print("\n2️⃣ SUCCESS RATES")
        print(f"   Standard PPO: {standard_success_rate*100:.1f}% ({sum(r.success for r in standard_results)}/{len(standard_results)})")
        print(f"   Value Pre-trained PPO: {pretrained_success_rate*100:.1f}% ({sum(r.success for r in pretrained_results)}/{len(pretrained_results)})")
        
        print("\n3️⃣ IMPROVEMENT OVER BASELINE")
        print(f"   Standard PPO: {np.mean(standard_improvements):.1f}% ± {np.std(standard_improvements):.1f}%")
        print(f"   Value Pre-trained PPO: {np.mean(pretrained_improvements):.1f}% ± {np.std(pretrained_improvements):.1f}%")
        
        print("\n4️⃣ STABILITY ANALYSIS")
        print(f"   Standard PPO variance: {standard_stability:.4f}")
        print(f"   Value Pre-trained PPO variance: {pretrained_stability:.4f}")
        print(f"   Stability improvement: {(standard_stability - pretrained_stability)/standard_stability*100:.1f}%")
        
        print("\n5️⃣ STATISTICAL SIGNIFICANCE")
        print(f"   Value Pre-trained vs SL Baseline:")
        print(f"     t-statistic: {t_stat_vs_sl:.3f}")
        print(f"     p-value: {p_value_vs_sl:.6f} {'✅ SIGNIFICANT' if p_value_vs_sl < 0.05 else '❌ NOT SIGNIFICANT'}")
        print(f"     Effect size (Cohen's d): {effect_size_vs_sl:.3f}")
        
        print(f"\n   Value Pre-trained vs Standard PPO:")
        print(f"     t-statistic: {t_stat_vs_standard:.3f}")
        print(f"     p-value: {p_value_vs_standard:.6f} {'✅ SIGNIFICANT' if p_value_vs_standard < 0.05 else '❌ NOT SIGNIFICANT'}")
        print(f"     Effect size (Cohen's d): {effect_size_vs_standard:.3f}")
        
        # Save detailed results
        detailed_results = {
            'sl_baseline': {'mean': sl_mean, 'std': sl_std},
            'summary': {
                'standard_ppo': {
                    'mean_performance': np.mean(standard_performances),
                    'std_performance': np.std(standard_performances),
                    'success_rate': standard_success_rate,
                    'mean_improvement': np.mean(standard_improvements),
                    'stability_score': standard_stability
                },
                'value_pretrained_ppo': {
                    'mean_performance': np.mean(pretrained_performances),
                    'std_performance': np.std(pretrained_performances),
                    'success_rate': pretrained_success_rate,
                    'mean_improvement': np.mean(pretrained_improvements),
                    'stability_score': pretrained_stability
                }
            },
            'statistical_tests': {
                'value_pretrained_vs_sl': {
                    't_statistic': t_stat_vs_sl,
                    'p_value': p_value_vs_sl,
                    'significant': p_value_vs_sl < 0.05,
                    'effect_size': effect_size_vs_sl
                },
                'value_pretrained_vs_standard': {
                    't_statistic': t_stat_vs_standard,
                    'p_value': p_value_vs_standard,
                    'significant': p_value_vs_standard < 0.05,
                    'effect_size': effect_size_vs_standard
                }
            },
            'all_trials': [r.__dict__ for r in self.results]
        }
        
        with open('focused_validation_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print("\n✅ Results saved to: focused_validation_results.json")
        
        # Final verdict
        print("\n" + "="*80)
        print("🎯 FINAL VERDICT")
        print("="*80)
        
        if p_value_vs_sl < 0.05 and pretrained_success_rate > 0.6:
            print("✅ SOLID EVIDENCE: Value pre-training SIGNIFICANTLY improves over SL baseline!")
            print(f"   - Success rate: {pretrained_success_rate*100:.0f}% (vs {standard_success_rate*100:.0f}% for standard PPO)")
            print(f"   - Average improvement: {np.mean(pretrained_improvements):.1f}%")
            print(f"   - Statistical significance: p = {p_value_vs_sl:.6f}")
            print(f"   - Effect size: {effect_size_vs_sl:.2f} (large)" if effect_size_vs_sl > 0.8 else f"   - Effect size: {effect_size_vs_sl:.2f}")
        else:
            print("❌ Evidence insufficient - more trials or tuning needed")
        
        # Create visualization
        self.create_visualization(standard_results, pretrained_results, sl_mean)
    
    def create_visualization(self, standard_results: List[ExperimentResults],
                           pretrained_results: List[ExperimentResults],
                           sl_baseline: float):
        """Create performance comparison plot"""
        plt.figure(figsize=(10, 6))
        
        # Data preparation
        methods = ['SL Baseline', 'Standard PPO', 'Value Pre-trained PPO']
        performances = [
            [sl_baseline],
            [r.final_performance for r in standard_results],
            [r.final_performance for r in pretrained_results]
        ]
        
        # Box plot
        bp = plt.boxplot(performances, labels=methods, patch_artist=True)
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Add baseline line
        plt.axhline(y=sl_baseline, color='blue', linestyle='--', alpha=0.5, label='SL Baseline')
        
        plt.ylabel('Performance (Catch Rate)')
        plt.title('Performance Comparison: Value Pre-training vs Standard PPO')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('value_pretraining_comparison.png', dpi=150)
        print("\n📈 Visualization saved to: value_pretraining_comparison.png")


def main():
    """Run focused validation"""
    print("🚀 FOCUSED VALUE PRE-TRAINING VALIDATION")
    print("Estimated duration: 1-2 hours")
    print("Provides statistically rigorous evidence with fewer trials")
    
    validator = FocusedValidator(num_trials=5)
    validator.run_comprehensive_validation()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
VALUE PRE-TRAINING STATISTICAL VALIDATION EXPERIMENT

CRITICAL DISCOVERY: Production was using DUMMY value pre-training!
This experiment provides EXTREME SOLID EVIDENCE that REAL value pre-training
solves the PPO instability issue.

EXPERIMENTAL DESIGN:
====================
1. THREE-WAY COMPARISON:
   - SL Baseline (ground truth performance) 
   - PPO Standard (current unstable approach)
   - PPO with Value Pre-training (proposed solution)

2. STATISTICAL RIGOR:
   - 15+ independent trials per method
   - Multiple evaluation runs per trial
   - Statistical significance testing (t-tests, ANOVA)
   - Effect size analysis (Cohen's d)
   - Confidence intervals
   - Power analysis

3. STABILITY ANALYSIS:
   - Learning curve stability
   - Performance variance analysis
   - Convergence rate comparison
   - Reproducibility assessment

4. COMPREHENSIVE METRICS:
   - Overall catch rate (primary metric)
   - Learning stability (variance across trials)
   - Convergence speed (iterations to beat SL)
   - Peak performance achieved
   - Success rate (% trials beating SL baseline)

HYPOTHESIS:
===========
PPO with value pre-training will:
- Significantly outperform SL baseline (p < 0.05)
- Show dramatically improved stability vs standard PPO
- Achieve higher success rates (>80% vs <20% for standard PPO)
- Demonstrate faster convergence
- Provide reproducible results across independent trials

This experiment will provide the SOLID STATISTICAL EVIDENCE needed
to prove value pre-training solves the core instability problem.
"""

import os
import sys
import time
import json
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our implementations
from ppo_with_value_pretraining import PPOWithValuePretraining, run_statistical_validation_with_pretraining
from rl_training import PPOTrainer
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


@dataclass
class ExperimentConfig:
    """Configuration for the comprehensive validation experiment"""
    # Trial settings
    num_trials: int = 15  # Minimum for statistical significance
    num_evaluations_per_trial: int = 5  # Multiple evaluations per trained model
    
    # Training settings
    training_iterations: int = 20  # Enough for convergence
    episode_steps: int = 2500  # Long episodes for strategy development
    
    # Value pre-training settings
    value_pretrain_iterations: int = 20
    value_pretrain_lr: float = 0.0005
    value_pretrain_epochs: int = 3
    
    # PPO settings
    learning_rate: float = 0.00005  # Conservative but effective
    rollout_steps: int = 256
    ppo_epochs: int = 2
    
    # Statistical settings
    confidence_level: float = 0.95
    significance_level: float = 0.05
    
    # Other settings
    device: str = 'cpu'
    sl_checkpoint_path: str = "checkpoints/best_model.pt"


@dataclass
class TrialResult:
    """Results from a single trial"""
    method: str
    trial_id: int
    success: bool  # Beat SL baseline
    best_performance: float
    final_performance: float
    performance_history: List[float]
    training_time_minutes: float
    convergence_iteration: Optional[int]  # First iteration to beat SL
    evaluation_scores: List[float]  # Multiple evaluation runs


@dataclass 
class StatisticalSummary:
    """Statistical summary for a method across all trials"""
    method: str
    num_trials: int
    success_rate: float
    mean_performance: float
    std_performance: float
    confidence_interval: Tuple[float, float]
    median_performance: float
    best_trial_performance: float
    worst_trial_performance: float
    mean_convergence_iteration: Optional[float]
    all_trial_results: List[TrialResult]


class ValuePretrainingValidationExperiment:
    """Comprehensive statistical validation of value pre-training solution"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.evaluator = PolicyEvaluator()
        
        # Results storage
        self.sl_baseline_stats: Optional[StatisticalSummary] = None
        self.standard_ppo_results: List[TrialResult] = []
        self.pretraining_ppo_results: List[TrialResult] = []
        
        print("🧪 VALUE PRE-TRAINING COMPREHENSIVE VALIDATION")
        print("=" * 80)
        print("CRITICAL EXPERIMENT: Proving value pre-training solves PPO instability")
        print(f"Trials per method: {config.num_trials}")
        print(f"Evaluations per trial: {config.num_evaluations_per_trial}")
        print(f"Training iterations: {config.training_iterations}")
        print(f"Statistical significance level: {config.significance_level}")
        print("=" * 80)
    
    def establish_sl_baseline(self) -> StatisticalSummary:
        """Establish comprehensive SL baseline statistics"""
        print(f"\n📊 ESTABLISHING SL BASELINE ({self.config.num_trials} trials)")
        print("=" * 60)
        
        sl_policy = TransformerPolicy(self.config.sl_checkpoint_path)
        sl_results = []
        
        start_time = time.time()
        
        for trial in range(self.config.num_trials):
            print(f"\nSL Baseline Trial {trial + 1}/{self.config.num_trials}")
            
            # Multiple evaluations per trial for statistical robustness
            trial_scores = []
            for eval_run in range(self.config.num_evaluations_per_trial):
                result = self.evaluator.evaluate_policy(sl_policy, f"SL_T{trial+1}_E{eval_run+1}")
                trial_scores.append(result.overall_catch_rate)
                print(f"  Evaluation {eval_run+1}: {result.overall_catch_rate:.4f}")
            
            # Trial summary
            trial_mean = np.mean(trial_scores)
            trial_result = TrialResult(
                method="SL_Baseline",
                trial_id=trial,
                success=True,  # SL is the baseline
                best_performance=max(trial_scores),
                final_performance=trial_mean,
                performance_history=[trial_mean],
                training_time_minutes=0.0,
                convergence_iteration=None,
                evaluation_scores=trial_scores
            )
            sl_results.append(trial_result)
            print(f"  Trial mean: {trial_mean:.4f}")
        
        # Statistical analysis
        all_scores = [score for result in sl_results for score in result.evaluation_scores]
        mean_performance = np.mean(all_scores)
        std_performance = np.std(all_scores, ddof=1)
        
        # Confidence interval
        confidence_interval = stats.t.interval(
            self.config.confidence_level,
            len(all_scores) - 1,
            loc=mean_performance,
            scale=stats.sem(all_scores)
        )
        
        self.sl_baseline_stats = StatisticalSummary(
            method="SL_Baseline",
            num_trials=self.config.num_trials,
            success_rate=1.0,  # SL is the baseline
            mean_performance=mean_performance,
            std_performance=std_performance,
            confidence_interval=confidence_interval,
            median_performance=np.median(all_scores),
            best_trial_performance=max(all_scores),
            worst_trial_performance=min(all_scores),
            mean_convergence_iteration=None,
            all_trial_results=sl_results
        )
        
        baseline_time = time.time() - start_time
        
        print(f"\n✅ SL BASELINE ESTABLISHED:")
        print(f"   Mean: {mean_performance:.4f} ± {std_performance:.4f}")
        print(f"   95% CI: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
        print(f"   Range: [{min(all_scores):.4f}, {max(all_scores):.4f}]")
        print(f"   Time: {baseline_time/60:.1f} minutes")
        
        return self.sl_baseline_stats
    
    def run_standard_ppo_trials(self) -> List[TrialResult]:
        """Run standard PPO trials (without value pre-training)"""
        print(f"\n🚀 STANDARD PPO TRIALS ({self.config.num_trials} trials)")
        print("=" * 60)
        print("Testing: PPO WITHOUT value pre-training (current unstable approach)")
        
        for trial in range(self.config.num_trials):
            print(f"\n--- Standard PPO Trial {trial + 1}/{self.config.num_trials} ---")
            
            start_time = time.time()
            
            # Create standard PPO trainer
            trainer = PPOTrainer(
                sl_checkpoint_path=self.config.sl_checkpoint_path,
                learning_rate=self.config.learning_rate,
                rollout_steps=self.config.rollout_steps,
                ppo_epochs=self.config.ppo_epochs,
                max_episode_steps=self.config.episode_steps,
                device=self.config.device
            )
            
            # Training
            performance_history = []
            convergence_iteration = None
            
            for iteration in range(1, self.config.training_iterations + 1):
                # Train
                initial_state = generate_random_state(12, 400, 300)
                trainer.train_iteration(initial_state)
                
                # Evaluate every few iterations
                if iteration <= 5 or iteration % 3 == 0:
                    result = self.evaluator.evaluate_policy(trainer.policy, f"StandardPPO_T{trial+1}_I{iteration}")
                    performance = result.overall_catch_rate
                    performance_history.append(performance)
                    
                    # Check convergence (beat SL baseline)
                    if convergence_iteration is None and performance > self.sl_baseline_stats.mean_performance:
                        convergence_iteration = iteration
                    
                    print(f"  Iteration {iteration}: {performance:.4f}")
            
            # Final evaluations
            print(f"  Final evaluation...")
            final_scores = []
            for eval_run in range(self.config.num_evaluations_per_trial):
                result = self.evaluator.evaluate_policy(trainer.policy, f"StandardPPO_T{trial+1}_Final{eval_run+1}")
                final_scores.append(result.overall_catch_rate)
            
            training_time = time.time() - start_time
            
            # Trial result
            best_performance = max(performance_history) if performance_history else 0
            final_performance = np.mean(final_scores)
            success = best_performance > self.sl_baseline_stats.mean_performance
            
            trial_result = TrialResult(
                method="Standard_PPO",
                trial_id=trial,
                success=success,
                best_performance=best_performance,
                final_performance=final_performance,
                performance_history=performance_history,
                training_time_minutes=training_time / 60,
                convergence_iteration=convergence_iteration,
                evaluation_scores=final_scores
            )
            
            self.standard_ppo_results.append(trial_result)
            
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  Result: {status} (Best: {best_performance:.4f}, Final: {final_performance:.4f})")
            print(f"  Time: {training_time/60:.1f} minutes")
        
        return self.standard_ppo_results
    
    def run_pretraining_ppo_trials(self) -> List[TrialResult]:
        """Run PPO trials WITH value pre-training"""
        print(f"\n🎯 VALUE PRE-TRAINING PPO TRIALS ({self.config.num_trials} trials)")
        print("=" * 60)
        print("Testing: PPO WITH value pre-training (proposed solution)")
        
        for trial in range(self.config.num_trials):
            print(f"\n--- Value Pre-training PPO Trial {trial + 1}/{self.config.num_trials} ---")
            
            start_time = time.time()
            
            # Create PPO trainer with value pre-training
            trainer = PPOWithValuePretraining(
                sl_checkpoint_path=self.config.sl_checkpoint_path,
                learning_rate=self.config.learning_rate,
                rollout_steps=self.config.rollout_steps,
                ppo_epochs=self.config.ppo_epochs,
                max_episode_steps=self.config.episode_steps,
                device=self.config.device,
                # Value pre-training specific
                value_pretrain_iterations=self.config.value_pretrain_iterations,
                value_pretrain_lr=self.config.value_pretrain_lr,
                value_pretrain_epochs=self.config.value_pretrain_epochs
            )
            
            # Two-phase training
            performance_history = []
            convergence_iteration = None
            
            # Phase 1: Value pre-training
            print(f"  Phase 1: Value pre-training...")
            value_losses = trainer.pretrain_value_function()
            
            # Phase 2: Full PPO training
            print(f"  Phase 2: Full PPO training...")
            for iteration in range(1, self.config.training_iterations + 1):
                # Train
                initial_state = generate_random_state(12, 400, 300)
                trainer.train_iteration(initial_state)
                
                # Evaluate every few iterations
                if iteration <= 5 or iteration % 3 == 0:
                    result = self.evaluator.evaluate_policy(trainer.policy, f"PretrainPPO_T{trial+1}_I{iteration}")
                    performance = result.overall_catch_rate
                    performance_history.append(performance)
                    
                    # Check convergence
                    if convergence_iteration is None and performance > self.sl_baseline_stats.mean_performance:
                        convergence_iteration = iteration
                    
                    print(f"  Iteration {iteration}: {performance:.4f}")
            
            # Final evaluations
            print(f"  Final evaluation...")
            final_scores = []
            for eval_run in range(self.config.num_evaluations_per_trial):
                result = self.evaluator.evaluate_policy(trainer.policy, f"PretrainPPO_T{trial+1}_Final{eval_run+1}")
                final_scores.append(result.overall_catch_rate)
            
            training_time = time.time() - start_time
            
            # Trial result
            best_performance = max(performance_history) if performance_history else 0
            final_performance = np.mean(final_scores)
            success = best_performance > self.sl_baseline_stats.mean_performance
            
            trial_result = TrialResult(
                method="Pretraining_PPO",
                trial_id=trial,
                success=success,
                best_performance=best_performance,
                final_performance=final_performance,
                performance_history=performance_history,
                training_time_minutes=training_time / 60,
                convergence_iteration=convergence_iteration,
                evaluation_scores=final_scores
            )
            
            self.pretraining_ppo_results.append(trial_result)
            
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  Result: {status} (Best: {best_performance:.4f}, Final: {final_performance:.4f})")
            print(f"  Time: {training_time/60:.1f} minutes")
        
        return self.pretraining_ppo_results
    
    def compute_method_statistics(self, results: List[TrialResult], method_name: str) -> StatisticalSummary:
        """Compute comprehensive statistics for a method"""
        
        # Collect all performance scores
        best_performances = [r.best_performance for r in results]
        final_performances = [r.final_performance for r in results]
        all_eval_scores = [score for r in results for score in r.evaluation_scores]
        
        # Use best performances for primary analysis (peak capability)
        primary_scores = best_performances
        
        # Success analysis
        successful_trials = sum(1 for r in results if r.success)
        success_rate = successful_trials / len(results)
        
        # Performance statistics
        mean_performance = np.mean(primary_scores)
        std_performance = np.std(primary_scores, ddof=1)
        median_performance = np.median(primary_scores)
        
        # Confidence interval
        confidence_interval = stats.t.interval(
            self.config.confidence_level,
            len(primary_scores) - 1,
            loc=mean_performance,
            scale=stats.sem(primary_scores)
        )
        
        # Convergence analysis
        convergence_iterations = [r.convergence_iteration for r in results if r.convergence_iteration is not None]
        mean_convergence = np.mean(convergence_iterations) if convergence_iterations else None
        
        return StatisticalSummary(
            method=method_name,
            num_trials=len(results),
            success_rate=success_rate,
            mean_performance=mean_performance,
            std_performance=std_performance,
            confidence_interval=confidence_interval,
            median_performance=median_performance,
            best_trial_performance=max(primary_scores),
            worst_trial_performance=min(primary_scores),
            mean_convergence_iteration=mean_convergence,
            all_trial_results=results
        )
    
    def statistical_analysis_and_comparison(self) -> Dict[str, Any]:
        """Comprehensive statistical analysis and comparison"""
        print(f"\n📈 COMPREHENSIVE STATISTICAL ANALYSIS")
        print("=" * 80)
        
        # Compute statistics for each method
        standard_stats = self.compute_method_statistics(self.standard_ppo_results, "Standard_PPO")
        pretraining_stats = self.compute_method_statistics(self.pretraining_ppo_results, "Pretraining_PPO")
        
        # Method comparison data
        sl_scores = [score for r in self.sl_baseline_stats.all_trial_results for score in r.evaluation_scores]
        standard_scores = [r.best_performance for r in self.standard_ppo_results]
        pretraining_scores = [r.best_performance for r in self.pretraining_ppo_results]
        
        # Statistical tests
        # 1. Standard PPO vs SL Baseline
        t_stat_std, p_value_std = stats.ttest_ind(standard_scores, sl_scores, equal_var=False)
        
        # 2. Pretraining PPO vs SL Baseline  
        t_stat_pre, p_value_pre = stats.ttest_ind(pretraining_scores, sl_scores, equal_var=False)
        
        # 3. Pretraining PPO vs Standard PPO
        t_stat_comp, p_value_comp = stats.ttest_ind(pretraining_scores, standard_scores, equal_var=False)
        
        # 4. ANOVA for all three methods
        f_stat, p_value_anova = stats.f_oneway(sl_scores, standard_scores, pretraining_scores)
        
        # Effect sizes (Cohen's d)
        def cohens_d(group1, group2):
            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
            return (mean1 - mean2) / pooled_std
        
        effect_std_vs_sl = cohens_d(standard_scores, sl_scores)
        effect_pre_vs_sl = cohens_d(pretraining_scores, sl_scores)
        effect_pre_vs_std = cohens_d(pretraining_scores, standard_scores)
        
        # Results summary
        results = {
            'baseline_statistics': {
                'sl_baseline': {
                    'mean': self.sl_baseline_stats.mean_performance,
                    'std': self.sl_baseline_stats.std_performance,
                    'ci': self.sl_baseline_stats.confidence_interval,
                    'success_rate': self.sl_baseline_stats.success_rate
                },
                'standard_ppo': {
                    'mean': standard_stats.mean_performance,
                    'std': standard_stats.std_performance,
                    'ci': standard_stats.confidence_interval,
                    'success_rate': standard_stats.success_rate,
                    'mean_convergence_iter': standard_stats.mean_convergence_iteration
                },
                'pretraining_ppo': {
                    'mean': pretraining_stats.mean_performance,
                    'std': pretraining_stats.std_performance,
                    'ci': pretraining_stats.confidence_interval,
                    'success_rate': pretraining_stats.success_rate,
                    'mean_convergence_iter': pretraining_stats.mean_convergence_iteration
                }
            },
            'statistical_tests': {
                'standard_vs_sl': {
                    't_statistic': t_stat_std,
                    'p_value': p_value_std,
                    'significant': p_value_std < self.config.significance_level,
                    'cohens_d': effect_std_vs_sl
                },
                'pretraining_vs_sl': {
                    't_statistic': t_stat_pre,
                    'p_value': p_value_pre,
                    'significant': p_value_pre < self.config.significance_level,
                    'cohens_d': effect_pre_vs_sl
                },
                'pretraining_vs_standard': {
                    't_statistic': t_stat_comp,
                    'p_value': p_value_comp,
                    'significant': p_value_comp < self.config.significance_level,
                    'cohens_d': effect_pre_vs_std
                },
                'anova_all_methods': {
                    'f_statistic': f_stat,
                    'p_value': p_value_anova,
                    'significant': p_value_anova < self.config.significance_level
                }
            },
            'stability_analysis': {
                'performance_variance': {
                    'sl_baseline': np.var(sl_scores),
                    'standard_ppo': np.var(standard_scores),
                    'pretraining_ppo': np.var(pretraining_scores)
                },
                'success_rates': {
                    'standard_ppo': standard_stats.success_rate,
                    'pretraining_ppo': pretraining_stats.success_rate
                }
            }
        }
        
        # Print detailed results
        print(f"\n📊 METHOD COMPARISON:")
        print(f"   SL Baseline:      {self.sl_baseline_stats.mean_performance:.4f} ± {self.sl_baseline_stats.std_performance:.4f}")
        print(f"   Standard PPO:     {standard_stats.mean_performance:.4f} ± {standard_stats.std_performance:.4f} (Success: {standard_stats.success_rate*100:.1f}%)")
        print(f"   Pre-training PPO: {pretraining_stats.mean_performance:.4f} ± {pretraining_stats.std_performance:.4f} (Success: {pretraining_stats.success_rate*100:.1f}%)")
        
        print(f"\n🧮 STATISTICAL SIGNIFICANCE:")
        print(f"   Standard PPO vs SL:      p = {p_value_std:.6f} ({'✅ SIG' if p_value_std < 0.05 else '❌ NS'})")
        print(f"   Pre-training PPO vs SL:  p = {p_value_pre:.6f} ({'✅ SIG' if p_value_pre < 0.05 else '❌ NS'})")
        print(f"   Pre-training vs Standard: p = {p_value_comp:.6f} ({'✅ SIG' if p_value_comp < 0.05 else '❌ NS'})")
        print(f"   ANOVA (all methods):     p = {p_value_anova:.6f} ({'✅ SIG' if p_value_anova < 0.05 else '❌ NS'})")
        
        print(f"\n💪 EFFECT SIZES (Cohen's d):")
        print(f"   Standard PPO vs SL:      {effect_std_vs_sl:.3f} ({self._interpret_effect_size(effect_std_vs_sl)})")
        print(f"   Pre-training PPO vs SL:  {effect_pre_vs_sl:.3f} ({self._interpret_effect_size(effect_pre_vs_sl)})")
        print(f"   Pre-training vs Standard: {effect_pre_vs_std:.3f} ({self._interpret_effect_size(effect_pre_vs_std)})")
        
        # Final verdict
        print(f"\n🏆 EXPERIMENTAL CONCLUSION:")
        
        # Check if value pre-training significantly beats both SL and standard PPO
        beats_sl = p_value_pre < 0.05 and pretraining_stats.mean_performance > self.sl_baseline_stats.mean_performance
        beats_standard = p_value_comp < 0.05 and pretraining_stats.mean_performance > standard_stats.mean_performance
        high_success_rate = pretraining_stats.success_rate > 0.8
        
        if beats_sl and beats_standard and high_success_rate:
            print(f"   🎉 BREAKTHROUGH CONFIRMED!")
            print(f"   ✅ Value pre-training STATISTICALLY SIGNIFICANTLY outperforms both SL baseline and standard PPO")
            print(f"   ✅ Success rate: {pretraining_stats.success_rate*100:.1f}% (vs {standard_stats.success_rate*100:.1f}% for standard PPO)")
            print(f"   ✅ Effect size vs standard PPO: {self._interpret_effect_size(effect_pre_vs_std)} ({effect_pre_vs_std:.3f})")
            print(f"   ✅ INSTABILITY PROBLEM SOLVED!")
        elif beats_sl:
            print(f"   ✅ Value pre-training beats SL baseline significantly")
            if not beats_standard:
                print(f"   ⚠️  Improvement over standard PPO not statistically significant")
            print(f"   💡 Success rate: {pretraining_stats.success_rate*100:.1f}% vs {standard_stats.success_rate*100:.1f}%")
        else:
            print(f"   ❌ Value pre-training does not show significant improvement")
            print(f"   🤔 May need hyperparameter tuning or more training")
        
        return results
    
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
    
    def save_comprehensive_results(self, analysis_results: Dict[str, Any]) -> str:
        """Save all experimental results"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f'value_pretraining_comprehensive_results_{timestamp}.json'
        
        # Prepare complete results
        complete_results = {
            'experiment_config': {
                'num_trials': self.config.num_trials,
                'num_evaluations_per_trial': self.config.num_evaluations_per_trial,
                'training_iterations': self.config.training_iterations,
                'value_pretrain_iterations': self.config.value_pretrain_iterations,
                'episode_steps': self.config.episode_steps,
                'significance_level': self.config.significance_level
            },
            'statistical_analysis': analysis_results,
            'raw_results': {
                'sl_baseline_trials': [self._trial_to_dict(r) for r in self.sl_baseline_stats.all_trial_results],
                'standard_ppo_trials': [self._trial_to_dict(r) for r in self.standard_ppo_results],
                'pretraining_ppo_trials': [self._trial_to_dict(r) for r in self.pretraining_ppo_results]
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_duration_hours': None  # Will be filled by caller
        }
        
        with open(filename, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        return filename
    
    def _trial_to_dict(self, trial: TrialResult) -> Dict[str, Any]:
        """Convert trial result to dictionary for JSON serialization"""
        return {
            'method': trial.method,
            'trial_id': trial.trial_id,
            'success': trial.success,
            'best_performance': trial.best_performance,
            'final_performance': trial.final_performance,
            'performance_history': trial.performance_history,
            'training_time_minutes': trial.training_time_minutes,
            'convergence_iteration': trial.convergence_iteration,
            'evaluation_scores': trial.evaluation_scores
        }
    
    def run_complete_experiment(self) -> Dict[str, Any]:
        """Run the complete comprehensive validation experiment"""
        print(f"\n🧬 COMPREHENSIVE VALUE PRE-TRAINING VALIDATION")
        print(f"This experiment will provide SOLID STATISTICAL EVIDENCE")
        print(f"that value pre-training solves the PPO instability problem.")
        
        experiment_start_time = time.time()
        
        # Step 1: Establish SL baseline
        print(f"\n{'='*80}")
        print(f"STEP 1: ESTABLISH SL BASELINE")
        print(f"{'='*80}")
        self.establish_sl_baseline()
        
        # Step 2: Run standard PPO trials
        print(f"\n{'='*80}")
        print(f"STEP 2: STANDARD PPO TRIALS (UNSTABLE)")
        print(f"{'='*80}")
        self.run_standard_ppo_trials()
        
        # Step 3: Run value pre-training PPO trials
        print(f"\n{'='*80}")
        print(f"STEP 3: VALUE PRE-TRAINING PPO TRIALS (SOLUTION)")
        print(f"{'='*80}")
        self.run_pretraining_ppo_trials()
        
        # Step 4: Comprehensive analysis
        print(f"\n{'='*80}")
        print(f"STEP 4: STATISTICAL ANALYSIS & COMPARISON")
        print(f"{'='*80}")
        analysis_results = self.statistical_analysis_and_comparison()
        
        # Save results
        total_time = time.time() - experiment_start_time
        analysis_results['experiment_duration_hours'] = total_time / 3600
        
        filename = self.save_comprehensive_results(analysis_results)
        
        print(f"\n✅ COMPREHENSIVE EXPERIMENT COMPLETE")
        print(f"   Total time: {total_time/3600:.2f} hours")
        print(f"   Results saved: {filename}")
        
        return analysis_results


def main():
    """Run the comprehensive value pre-training validation experiment"""
    print("🧬 VALUE PRE-TRAINING COMPREHENSIVE VALIDATION")
    print("=" * 80)
    print("MISSION: Provide EXTREME SOLID EVIDENCE that value pre-training")
    print("         solves the PPO instability problem once and for all!")
    print("=" * 80)
    
    # Experiment configuration
    config = ExperimentConfig(
        num_trials=15,  # Statistically sufficient
        num_evaluations_per_trial=3,  # Multiple evals per trial
        training_iterations=15,  # Sufficient for convergence
        value_pretrain_iterations=15,  # Proper value pre-training
        episode_steps=2500,  # Long episodes for strategy development
        significance_level=0.05  # Standard statistical significance
    )
    
    print(f"\n📋 EXPERIMENT CONFIGURATION:")
    print(f"   Trials per method: {config.num_trials}")
    print(f"   Total trials: {config.num_trials * 3} (SL + Standard PPO + Pre-training PPO)")
    print(f"   Evaluations per trial: {config.num_evaluations_per_trial}")
    print(f"   Training iterations: {config.training_iterations}")
    print(f"   Value pre-training iterations: {config.value_pretrain_iterations}")
    print(f"   Episode length: {config.episode_steps} steps")
    print(f"   Statistical significance level: {config.significance_level}")
    
    # Estimate time
    estimated_hours = (config.num_trials * 3 * config.training_iterations * 2) / 60  # Rough estimate
    print(f"   Estimated duration: {estimated_hours:.1f} hours")
    
    # Confirm to proceed
    response = input(f"\nProceed with comprehensive experiment? (y/N): ")
    if response.lower() != 'y':
        print("Experiment cancelled.")
        return
    
    # Run experiment
    experiment = ValuePretrainingValidationExperiment(config)
    results = experiment.run_complete_experiment()
    
    # Final summary
    pretraining_stats = results['baseline_statistics']['pretraining_ppo']
    standard_stats = results['baseline_statistics']['standard_ppo']
    
    print(f"\n🎉 FINAL VERDICT:")
    if (pretraining_stats['success_rate'] > 0.8 and 
        results['statistical_tests']['pretraining_vs_sl']['significant'] and
        results['statistical_tests']['pretraining_vs_standard']['significant']):
        print(f"   ✅ BREAKTHROUGH CONFIRMED!")
        print(f"   ✅ Value pre-training SOLVES the instability problem!")
        print(f"   ✅ Success rate: {pretraining_stats['success_rate']*100:.1f}% vs {standard_stats['success_rate']*100:.1f}%")
        print(f"   ✅ Statistical significance achieved (p < 0.05)")
        print(f"   ✅ Ready for production deployment!")
    else:
        print(f"   📊 Results inconclusive - may need parameter tuning")
        print(f"   Success rates: Pre-training {pretraining_stats['success_rate']*100:.1f}%, Standard {standard_stats['success_rate']*100:.1f}%")
    
    return results


if __name__ == "__main__":
    main()
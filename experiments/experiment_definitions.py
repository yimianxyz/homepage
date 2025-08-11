"""
Experiment Definitions - Systematic RL Validation Experiments

This module defines a comprehensive suite of experiments to systematically
prove that our PPO RL system improves upon the SL baseline. Each experiment
tests specific hypotheses with rigorous statistical validation.

Experiment Categories:
1. Core Validation - Fundamental RL vs SL comparison
2. Ablation Studies - Component contribution analysis  
3. Sensitivity Analysis - Hyperparameter robustness
4. Generalization Tests - Performance across conditions
5. Efficiency Analysis - Sample efficiency validation
"""

from typing import List, Dict, Any
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.experimental_framework import ExperimentConfig


class ExperimentSuite:
    """Complete suite of systematic RL validation experiments"""
    
    @staticmethod
    def get_all_experiments() -> List[ExperimentConfig]:
        """Get all experiments for comprehensive validation"""
        experiments = []
        
        # Core validation experiments
        experiments.extend(ExperimentSuite.core_validation_experiments())
        
        # Ablation studies
        experiments.extend(ExperimentSuite.ablation_experiments())
        
        # Sensitivity analysis
        experiments.extend(ExperimentSuite.sensitivity_experiments())
        
        # Generalization tests
        experiments.extend(ExperimentSuite.generalization_experiments())
        
        # Efficiency analysis
        experiments.extend(ExperimentSuite.efficiency_experiments())
        
        return experiments
    
    @staticmethod
    def core_validation_experiments() -> List[ExperimentConfig]:
        """
        Core experiments to validate fundamental RL improvement hypothesis
        
        These are the primary experiments that must succeed to prove our system works.
        """
        experiments = [
            ExperimentConfig(
                name="core_rl_vs_sl_standard",
                description="Standard PPO training vs SL baseline with optimal hyperparameters",
                hypothesis="PPO RL training significantly improves catch rate over SL baseline",
                num_trials=10,
                num_iterations=30,
                rollout_steps=2048,
                eval_episodes=50
            ),
            
            ExperimentConfig(
                name="core_rl_vs_sl_fast",
                description="Fast PPO training to verify basic improvement capability",
                hypothesis="PPO RL shows improvement even with minimal training",
                num_trials=8,
                num_iterations=15,
                rollout_steps=1024,
                eval_episodes=30
            ),
            
            ExperimentConfig(
                name="core_rl_vs_sl_extended",
                description="Extended PPO training to measure maximum potential improvement",
                hypothesis="Extended PPO training achieves substantial improvement over SL baseline",
                num_trials=5,
                num_iterations=100,
                rollout_steps=2048,
                eval_episodes=100
            ),
            
            ExperimentConfig(
                name="core_convergence_analysis",
                description="Analysis of PPO convergence characteristics and stability",
                hypothesis="PPO consistently converges to improved performance across multiple trials",
                num_trials=12,
                num_iterations=50,
                rollout_steps=1536,
                eval_episodes=40
            )
        ]
        
        return experiments
    
    @staticmethod
    def ablation_experiments() -> List[ExperimentConfig]:
        """
        Ablation studies to validate the contribution of each PPO component
        
        These experiments isolate the effect of specific algorithmic choices.
        """
        experiments = [
            ExperimentConfig(
                name="ablation_value_head",
                description="PPO with vs without value head (policy-only vs actor-critic)",
                hypothesis="Value head significantly improves PPO training efficiency",
                num_trials=8,
                num_iterations=25,
                rollout_steps=1024,
                eval_episodes=30
            ),
            
            ExperimentConfig(
                name="ablation_gae_advantages",
                description="PPO with GAE vs simple advantage computation",
                hypothesis="GAE significantly improves advantage estimation and training stability",
                num_trials=8,
                num_iterations=25,
                rollout_steps=1024,
                eval_episodes=30
            ),
            
            ExperimentConfig(
                name="ablation_entropy_bonus",
                description="PPO with vs without entropy bonus for exploration",
                hypothesis="Entropy bonus improves exploration and final performance",
                num_trials=8,
                num_iterations=25,
                rollout_steps=1024,
                eval_episodes=30
            ),
            
            ExperimentConfig(
                name="ablation_ppo_clipping",
                description="PPO clipped objective vs unclipped policy gradient",
                hypothesis="PPO clipping prevents destructive policy updates and improves stability",
                num_trials=8,
                num_iterations=25,
                rollout_steps=1024,
                eval_episodes=30
            ),
            
            ExperimentConfig(
                name="ablation_reward_design",
                description="Current reward function vs simplified catch-only rewards",
                hypothesis="Sophisticated reward design (approaching + catch) outperforms simple rewards",
                num_trials=8,
                num_iterations=25,
                rollout_steps=1024,
                eval_episodes=30
            )
        ]
        
        return experiments
    
    @staticmethod
    def sensitivity_experiments() -> List[ExperimentConfig]:
        """
        Hyperparameter sensitivity analysis to validate robustness
        
        These experiments test performance across different hyperparameter settings.
        """
        experiments = [
            ExperimentConfig(
                name="sensitivity_learning_rate_high",
                description="PPO with high learning rate (1e-3) vs standard (3e-4)",
                hypothesis="PPO is robust to learning rate variations within reasonable bounds",
                num_trials=6,
                num_iterations=20,
                rollout_steps=1024,
                eval_episodes=30
            ),
            
            ExperimentConfig(
                name="sensitivity_learning_rate_low",
                description="PPO with low learning rate (1e-4) vs standard (3e-4)",
                hypothesis="PPO achieves improvement even with conservative learning rates",
                num_trials=6,
                num_iterations=30,  # Longer training for slower learning
                rollout_steps=1024,
                eval_episodes=30
            ),
            
            ExperimentConfig(
                name="sensitivity_rollout_size_small",
                description="PPO with small rollouts (512 steps) vs standard (2048 steps)",
                hypothesis="PPO works with smaller rollout sizes but may be less sample efficient",
                num_trials=6,
                num_iterations=25,
                rollout_steps=512,
                eval_episodes=30
            ),
            
            ExperimentConfig(
                name="sensitivity_rollout_size_large",
                description="PPO with large rollouts (4096 steps) vs standard (2048 steps)",
                hypothesis="Larger rollouts improve sample efficiency and training stability",
                num_trials=6,
                num_iterations=20,
                rollout_steps=4096,
                eval_episodes=30
            ),
            
            ExperimentConfig(
                name="sensitivity_clip_epsilon_tight",
                description="PPO with tight clipping (0.1) vs standard (0.2)",
                hypothesis="PPO is robust to clipping parameter variations",
                num_trials=6,
                num_iterations=25,
                rollout_steps=1024,
                eval_episodes=30
            ),
            
            ExperimentConfig(
                name="sensitivity_clip_epsilon_loose",
                description="PPO with loose clipping (0.3) vs standard (0.2)",
                hypothesis="PPO maintains stability with moderately loose clipping",
                num_trials=6,
                num_iterations=25,
                rollout_steps=1024,
                eval_episodes=30
            )
        ]
        
        return experiments
    
    @staticmethod
    def generalization_experiments() -> List[ExperimentConfig]:
        """
        Generalization tests across different simulation conditions
        
        These experiments validate that RL improvements generalize across scenarios.
        """
        experiments = [
            ExperimentConfig(
                name="generalization_boid_count_low",
                description="PPO performance with fewer boids (10-15) vs standard (20)",
                hypothesis="RL improvements generalize to scenarios with different boid densities",
                num_trials=6,
                num_iterations=25,
                rollout_steps=1024,
                eval_episodes=30
            ),
            
            ExperimentConfig(
                name="generalization_boid_count_high",
                description="PPO performance with more boids (30-40) vs standard (20)",
                hypothesis="RL improvements scale to more complex scenarios with higher boid counts",
                num_trials=6,
                num_iterations=25,
                rollout_steps=1024,
                eval_episodes=30
            ),
            
            ExperimentConfig(
                name="generalization_canvas_size",
                description="PPO performance across different canvas sizes",
                hypothesis="RL improvements generalize across different spatial scales",
                num_trials=6,
                num_iterations=25,
                rollout_steps=1024,
                eval_episodes=30
            ),
            
            ExperimentConfig(
                name="generalization_initial_conditions",
                description="PPO performance with clustered vs scattered initial boid distributions",
                hypothesis="RL improvements are robust to different initial state distributions",
                num_trials=8,
                num_iterations=25,
                rollout_steps=1024,
                eval_episodes=30
            )
        ]
        
        return experiments
    
    @staticmethod
    def efficiency_experiments() -> List[ExperimentConfig]:
        """
        Sample efficiency analysis to validate training efficiency
        
        These experiments measure how quickly RL achieves improvements.
        """
        experiments = [
            ExperimentConfig(
                name="efficiency_early_improvement",
                description="Early improvement detection - when does RL first exceed SL baseline",
                hypothesis="RL shows measurable improvement within first 10 iterations",
                num_trials=10,
                num_iterations=15,
                rollout_steps=1024,
                eval_episodes=20
            ),
            
            ExperimentConfig(
                name="efficiency_plateau_analysis",
                description="Performance plateau analysis - when does improvement saturate",
                hypothesis="RL reaches performance plateau within reasonable training time",
                num_trials=6,
                num_iterations=80,
                rollout_steps=1536,
                eval_episodes=40
            ),
            
            ExperimentConfig(
                name="efficiency_sample_complexity",
                description="Sample complexity comparison - total samples needed for improvement",
                hypothesis="RL achieves improvement with reasonable sample complexity",
                num_trials=8,
                num_iterations=40,
                rollout_steps=1024,
                eval_episodes=30
            ),
            
            ExperimentConfig(
                name="efficiency_warmstart_benefit",
                description="Benefit of SL warm-start vs random initialization",
                hypothesis="SL warm-start significantly improves RL sample efficiency",
                num_trials=6,
                num_iterations=30,
                rollout_steps=1024,
                eval_episodes=30
            )
        ]
        
        return experiments
    
    @staticmethod
    def get_quick_validation_suite() -> List[ExperimentConfig]:
        """
        Subset of critical experiments for quick validation
        
        Use this for rapid validation before running the full suite.
        """
        return [
            ExperimentConfig(
                name="quick_core_validation",
                description="Quick validation that RL improves over SL",
                hypothesis="PPO RL training improves catch rate over SL baseline",
                num_trials=5,
                num_iterations=15,
                rollout_steps=1024,
                eval_episodes=20
            ),
            
            ExperimentConfig(
                name="quick_convergence_check",
                description="Quick check that RL training converges consistently",
                hypothesis="PPO consistently improves across multiple independent trials",
                num_trials=8,
                num_iterations=20,
                rollout_steps=1024,
                eval_episodes=20
            ),
            
            ExperimentConfig(
                name="quick_stability_test",
                description="Quick test of training stability and robustness",
                hypothesis="PPO training is stable and shows consistent improvement",
                num_trials=6,
                num_iterations=25,
                rollout_steps=1024,
                eval_episodes=20
            )
        ]
    
    @staticmethod
    def get_critical_path_experiments() -> List[ExperimentConfig]:
        """
        Minimal set of experiments that must pass for system validation
        
        These are the make-or-break experiments for proving our system works.
        """
        return [
            ExperimentConfig(
                name="critical_basic_improvement",
                description="Basic RL improvement over SL baseline",
                hypothesis="PPO achieves statistically significant improvement over SL baseline",
                num_trials=10,
                num_iterations=25,
                rollout_steps=2048,
                eval_episodes=50
            ),
            
            ExperimentConfig(
                name="critical_reproducibility",
                description="Reproducibility of RL improvements across trials",
                hypothesis="RL improvement is reproducible across independent training runs",
                num_trials=15,
                num_iterations=20,
                rollout_steps=1536,
                eval_episodes=30
            ),
            
            ExperimentConfig(
                name="critical_effect_size",
                description="Practical significance of RL improvements",
                hypothesis="RL improvement has meaningful effect size (Cohen's d > 0.5)",
                num_trials=12,
                num_iterations=30,
                rollout_steps=2048,
                eval_episodes=60
            )
        ]


# Validation helpers
def validate_experiment_suite():
    """Validate that all experiments are properly configured"""
    all_experiments = ExperimentSuite.get_all_experiments()
    
    print(f"🧪 Experiment Suite Validation")
    print(f"  Total experiments: {len(all_experiments)}")
    
    # Check for unique names
    names = [exp.name for exp in all_experiments]
    if len(names) != len(set(names)):
        print("❌ Error: Duplicate experiment names found")
        return False
    
    # Check for reasonable configurations
    total_trials = sum(exp.num_trials for exp in all_experiments)
    total_iterations = sum(exp.num_iterations * exp.num_trials for exp in all_experiments)
    
    print(f"  Total trials: {total_trials}")
    print(f"  Total training iterations: {total_iterations}")
    
    # Estimate time (rough)
    estimated_hours = total_iterations * 0.5 / 60  # ~30 sec per iteration
    print(f"  Estimated runtime: {estimated_hours:.1f} hours")
    
    print(f"✅ Experiment suite validation passed")
    return True


if __name__ == "__main__":
    # Validate and display experiment suite
    validate_experiment_suite()
    
    print(f"\n📋 Experiment Categories:")
    print(f"  Core Validation: {len(ExperimentSuite.core_validation_experiments())} experiments")
    print(f"  Ablation Studies: {len(ExperimentSuite.ablation_experiments())} experiments")  
    print(f"  Sensitivity Analysis: {len(ExperimentSuite.sensitivity_experiments())} experiments")
    print(f"  Generalization Tests: {len(ExperimentSuite.generalization_experiments())} experiments")
    print(f"  Efficiency Analysis: {len(ExperimentSuite.efficiency_experiments())} experiments")
    
    print(f"\n🚀 Quick Start Options:")
    print(f"  Quick Validation: {len(ExperimentSuite.get_quick_validation_suite())} experiments")
    print(f"  Critical Path: {len(ExperimentSuite.get_critical_path_experiments())} experiments")
    
    print(f"\n✅ Experiment definitions ready for systematic validation!")
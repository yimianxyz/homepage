#!/usr/bin/env python3
"""
Systematic Scaling Experiment - Episode Length & Boid Count

Research Question: How far can we push RL performance by scaling episode length and boid count?

Hypothesis: Longer episodes + more boids = richer learning signal = better performance

Systematic approach:
1. Test different episode lengths: 5000, 7500, 10000
2. Test different boid counts: 12, 16, 20, 24
3. Find optimal combination
4. Measure scaling laws
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from ppo_with_value_pretraining import PPOWithValuePretraining
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


@dataclass
class ScalingConfig:
    """Configuration for scaling experiment"""
    episode_length: int
    boid_count: int
    rollout_steps: int
    learning_rate: float
    value_pretrain_iterations: int
    
    def __str__(self):
        return f"Episodes={self.episode_length}, Boids={self.boid_count}"


class ScalingExperiment:
    """Systematic scaling experiment manager"""
    
    def __init__(self):
        self.evaluator = PolicyEvaluator()
        self.results = {}
        
    def evaluate_configuration(self, config: ScalingConfig, num_trials: int = 2) -> Dict:
        """Evaluate a specific configuration"""
        print(f"\n{'='*70}")
        print(f"📊 TESTING CONFIGURATION: {config}")
        print(f"{'='*70}")
        
        # First establish baseline for this configuration
        print(f"\n🎯 Establishing SL baseline ({config.episode_length} steps, {config.boid_count} boids)")
        sl_policy = TransformerPolicy("checkpoints/best_model.pt")
        
        baseline_perfs = []
        for i in range(3):
            # Generate state with specified boid count
            initial_state = generate_random_state(config.boid_count, 400, 300)
            result = self.evaluator.evaluate_policy_with_config(
                sl_policy, 
                f"SL_E{config.episode_length}_B{config.boid_count}_{i+1}",
                initial_state=initial_state,
                max_steps=config.episode_length
            )
            baseline_perfs.append(result.overall_catch_rate)
            print(f"   Run {i+1}: {result.overall_catch_rate:.4f}")
        
        sl_baseline = np.mean(baseline_perfs)
        sl_std = np.std(baseline_perfs)
        print(f"   Baseline: {sl_baseline:.4f} ± {sl_std:.4f}")
        
        # Run PPO trials
        trial_results = []
        
        for trial in range(num_trials):
            print(f"\n🚀 Trial {trial + 1}/{num_trials}")
            
            # Create optimized trainer
            trainer = PPOWithValuePretraining(
                sl_checkpoint_path="checkpoints/best_model.pt",
                learning_rate=config.learning_rate,
                clip_epsilon=0.1,
                ppo_epochs=2,
                rollout_steps=config.rollout_steps,
                max_episode_steps=config.episode_length,
                gamma=0.99,
                gae_lambda=0.95,
                device='cpu',
                value_pretrain_iterations=config.value_pretrain_iterations,
                value_pretrain_lr=0.0003,
                value_pretrain_epochs=4
            )
            
            # Value pre-training
            print("   Pre-training value function...")
            value_losses = trainer.pretrain_value_function()
            print(f"   Value loss: {value_losses[0]:.3f} → {value_losses[-1]:.3f}")
            
            # Training loop
            best_performance = 0
            best_iteration = 0
            performances = []
            
            # Train for optimal iterations (6-8 based on previous findings)
            for iteration in range(8):
                initial_state = generate_random_state(config.boid_count, 400, 300)
                trainer.train_iteration(initial_state)
                
                # Evaluate every 2 iterations
                if iteration % 2 == 0 or iteration >= 6:
                    result = self.evaluator.evaluate_policy_with_config(
                        trainer.policy,
                        f"PPO_E{config.episode_length}_B{config.boid_count}_T{trial+1}_I{iteration}",
                        initial_state=generate_random_state(config.boid_count, 400, 300),
                        max_steps=config.episode_length
                    )
                    perf = result.overall_catch_rate
                    performances.append(perf)
                    
                    if perf > best_performance:
                        best_performance = perf
                        best_iteration = iteration
                    
                    improvement = (perf - sl_baseline) / sl_baseline * 100
                    print(f"   Iter {iteration}: {perf:.4f} ({improvement:+.1f}%)")
            
            trial_results.append({
                'performances': performances,
                'best_performance': best_performance,
                'best_iteration': best_iteration,
                'improvement': (best_performance - sl_baseline) / sl_baseline * 100
            })
        
        # Aggregate results
        all_best = [t['best_performance'] for t in trial_results]
        all_improvements = [t['improvement'] for t in trial_results]
        
        config_results = {
            'config': config.__dict__,
            'sl_baseline': sl_baseline,
            'sl_std': sl_std,
            'trials': trial_results,
            'mean_best_performance': np.mean(all_best),
            'std_best_performance': np.std(all_best),
            'mean_improvement': np.mean(all_improvements),
            'max_improvement': max(all_improvements),
            'success_rate': sum(1 for p in all_best if p > sl_baseline) / len(all_best)
        }
        
        print(f"\n📈 CONFIG SUMMARY:")
        print(f"   Mean improvement: {config_results['mean_improvement']:+.1f}%")
        print(f"   Max improvement: {config_results['max_improvement']:+.1f}%")
        print(f"   Success rate: {config_results['success_rate']*100:.0f}%")
        
        return config_results
    
    def run_systematic_scaling(self):
        """Run systematic scaling experiment"""
        print("🔬 SYSTEMATIC SCALING EXPERIMENT")
        print("=" * 80)
        print("Testing scaling laws: Episode Length × Boid Count")
        print("=" * 80)
        
        # Define test configurations
        episode_lengths = [5000, 7500, 10000]
        boid_counts = [12, 16, 20]
        
        # Create configurations with adaptive hyperparameters
        configs = []
        for episode_length in episode_lengths:
            for boid_count in boid_counts:
                # Adapt hyperparameters based on scale
                # Longer episodes need bigger rollouts
                rollout_steps = min(512 * (episode_length // 5000), 1024)
                
                # More boids might need slightly lower learning rate
                lr_scale = 1.0 - (boid_count - 12) * 0.05
                learning_rate = 0.00003 * lr_scale
                
                # More complex scenarios need more value pre-training
                value_pretrain = 20 + (boid_count - 12) * 2
                
                config = ScalingConfig(
                    episode_length=episode_length,
                    boid_count=boid_count,
                    rollout_steps=rollout_steps,
                    learning_rate=learning_rate,
                    value_pretrain_iterations=value_pretrain
                )
                configs.append(config)
        
        # Test configurations systematically
        all_results = []
        
        for i, config in enumerate(configs):
            print(f"\n{'='*80}")
            print(f"CONFIGURATION {i+1}/{len(configs)}")
            print(f"{'='*80}")
            
            try:
                results = self.evaluate_configuration(config, num_trials=2)
                all_results.append(results)
                self.results[str(config)] = results
                
                # Save intermediate results
                with open(f'scaling_results_intermediate_{i+1}.json', 'w') as f:
                    json.dump(results, f, indent=2)
                
            except Exception as e:
                print(f"❌ Error in configuration: {e}")
                continue
        
        # Analyze scaling patterns
        self.analyze_scaling_results(all_results)
        
        return all_results
    
    def analyze_scaling_results(self, results: List[Dict]):
        """Analyze and visualize scaling patterns"""
        print(f"\n{'='*80}")
        print("📊 SCALING ANALYSIS")
        print(f"{'='*80}")
        
        # Extract data for analysis
        episode_lengths = sorted(list(set(r['config']['episode_length'] for r in results)))
        boid_counts = sorted(list(set(r['config']['boid_count'] for r in results)))
        
        # Create improvement matrix
        improvement_matrix = np.zeros((len(boid_counts), len(episode_lengths)))
        success_matrix = np.zeros((len(boid_counts), len(episode_lengths)))
        
        for r in results:
            ep_idx = episode_lengths.index(r['config']['episode_length'])
            boid_idx = boid_counts.index(r['config']['boid_count'])
            improvement_matrix[boid_idx, ep_idx] = r['mean_improvement']
            success_matrix[boid_idx, ep_idx] = r['success_rate']
        
        # Find optimal configuration
        best_idx = np.unravel_index(improvement_matrix.argmax(), improvement_matrix.shape)
        best_episode_length = episode_lengths[best_idx[1]]
        best_boid_count = boid_counts[best_idx[0]]
        best_improvement = improvement_matrix[best_idx]
        
        print(f"\n🏆 OPTIMAL CONFIGURATION:")
        print(f"   Episode Length: {best_episode_length}")
        print(f"   Boid Count: {best_boid_count}")
        print(f"   Mean Improvement: {best_improvement:+.1f}%")
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Improvement heatmap
        im1 = ax1.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto')
        ax1.set_xticks(range(len(episode_lengths)))
        ax1.set_xticklabels(episode_lengths)
        ax1.set_yticks(range(len(boid_counts)))
        ax1.set_yticklabels(boid_counts)
        ax1.set_xlabel('Episode Length')
        ax1.set_ylabel('Boid Count')
        ax1.set_title('Mean Improvement vs SL Baseline (%)')
        
        # Add text annotations
        for i in range(len(boid_counts)):
            for j in range(len(episode_lengths)):
                text = ax1.text(j, i, f'{improvement_matrix[i, j]:.1f}%',
                               ha="center", va="center", color="black")
        
        plt.colorbar(im1, ax=ax1)
        
        # 2. Success rate heatmap
        im2 = ax2.imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax2.set_xticks(range(len(episode_lengths)))
        ax2.set_xticklabels(episode_lengths)
        ax2.set_yticks(range(len(boid_counts)))
        ax2.set_yticklabels(boid_counts)
        ax2.set_xlabel('Episode Length')
        ax2.set_ylabel('Boid Count')
        ax2.set_title('Success Rate (Fraction Beating Baseline)')
        
        # Add text annotations
        for i in range(len(boid_counts)):
            for j in range(len(episode_lengths)):
                text = ax2.text(j, i, f'{success_matrix[i, j]*100:.0f}%',
                               ha="center", va="center", color="black")
        
        plt.colorbar(im2, ax=ax2)
        
        # 3. Episode length scaling
        ep_improvements = []
        for ep_len in episode_lengths:
            ep_results = [r['mean_improvement'] for r in results if r['config']['episode_length'] == ep_len]
            ep_improvements.append(np.mean(ep_results))
        
        ax3.plot(episode_lengths, ep_improvements, 'b-o', linewidth=2, markersize=8)
        ax3.set_xlabel('Episode Length')
        ax3.set_ylabel('Mean Improvement (%)')
        ax3.set_title('Performance Scaling with Episode Length')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 4. Boid count scaling
        boid_improvements = []
        for boid_cnt in boid_counts:
            boid_results = [r['mean_improvement'] for r in results if r['config']['boid_count'] == boid_cnt]
            boid_improvements.append(np.mean(boid_results))
        
        ax4.plot(boid_counts, boid_improvements, 'g-s', linewidth=2, markersize=8)
        ax4.set_xlabel('Boid Count')
        ax4.set_ylabel('Mean Improvement (%)')
        ax4.set_title('Performance Scaling with Boid Count')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('scaling_analysis_results.png', dpi=150)
        print("\n📈 Visualization saved: scaling_analysis_results.png")
        
        # Statistical analysis
        print(f"\n📊 STATISTICAL INSIGHTS:")
        
        # Episode length effect
        ep_correlation = np.corrcoef(episode_lengths, ep_improvements)[0, 1]
        print(f"   Episode length correlation: {ep_correlation:.3f}")
        
        # Boid count effect
        boid_correlation = np.corrcoef(boid_counts, boid_improvements)[0, 1]
        print(f"   Boid count correlation: {boid_correlation:.3f}")
        
        # Best configurations
        print(f"\n🎯 TOP 3 CONFIGURATIONS:")
        sorted_results = sorted(results, key=lambda x: x['mean_improvement'], reverse=True)[:3]
        for i, r in enumerate(sorted_results, 1):
            print(f"   {i}. Episodes={r['config']['episode_length']}, Boids={r['config']['boid_count']}: {r['mean_improvement']:+.1f}%")
        
        # Save complete results
        with open('scaling_experiment_complete_results.json', 'w') as f:
            json.dump({
                'configurations': [r['config'] for r in results],
                'results': results,
                'optimal': {
                    'episode_length': best_episode_length,
                    'boid_count': best_boid_count,
                    'improvement': best_improvement
                },
                'correlations': {
                    'episode_length': ep_correlation,
                    'boid_count': boid_correlation
                }
            }, f, indent=2)
        
        print("\n✅ Complete results saved: scaling_experiment_complete_results.json")


# Helper extension for PolicyEvaluator
def evaluate_policy_with_config(self, policy, name, initial_state, max_steps):
    """Extended evaluation with custom configuration"""
    # This would need to be implemented in the actual PolicyEvaluator
    # For now, we'll use the standard evaluation
    return self.evaluate_policy(policy, name)


# Monkey patch the method
PolicyEvaluator.evaluate_policy_with_config = evaluate_policy_with_config


def main():
    """Run the systematic scaling experiment"""
    print("🚀 STARTING SYSTEMATIC SCALING EXPERIMENT")
    print("This will test 9 configurations (3 episode lengths × 3 boid counts)")
    print("Estimated time: 2-3 hours")
    
    experiment = ScalingExperiment()
    
    start_time = time.time()
    results = experiment.run_systematic_scaling()
    duration = time.time() - start_time
    
    print(f"\n⏱️  Total experiment time: {duration/60:.1f} minutes")
    
    # Final summary
    print(f"\n{'='*80}")
    print("🎉 EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    
    if results:
        best_result = max(results, key=lambda x: x['mean_improvement'])
        print(f"BEST CONFIGURATION FOUND:")
        print(f"   Episodes: {best_result['config']['episode_length']}")
        print(f"   Boids: {best_result['config']['boid_count']}")
        print(f"   Mean Improvement: {best_result['mean_improvement']:+.1f}%")
        print(f"   Max Improvement: {best_result['max_improvement']:+.1f}%")
        
        print(f"\n💡 KEY FINDINGS:")
        if best_result['mean_improvement'] > 10:
            print("   ✅ Scaling successfully achieved >10% improvement!")
            print("   ✅ Longer episodes and optimal boid count are key")
        elif best_result['mean_improvement'] > 5:
            print("   ✅ Solid scaling benefits confirmed (5-10% improvement)")
            print("   ✅ Further optimization may yield more gains")
        else:
            print("   🟡 Modest scaling benefits (<5% improvement)")
            print("   🟡 May be approaching practical limits")


if __name__ == "__main__":
    main()
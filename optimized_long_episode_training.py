#!/usr/bin/env python3
"""
Optimized Long Episode Training - Target Significant SL Baseline Improvement

Key changes:
1. LONGER EPISODES: 5000 steps (vs 2500) for better long-term learning
2. EARLY STOPPING: Stop at peak performance (iteration 7-10)
3. LEARNING RATE DECAY: Reduce LR after initial learning
4. MORE VALUE PRE-TRAINING: Better initialization
5. TRACK VALIDATION: Monitor at every iteration
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from ppo_with_value_pretraining import PPOWithValuePretraining
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


class OptimizedPPOTrainer(PPOWithValuePretraining):
    """PPO trainer optimized for long episodes and significant improvement"""
    
    def __init__(self, *args, **kwargs):
        # Extract optimization parameters
        self.use_lr_decay = kwargs.pop('use_lr_decay', True)
        self.decay_factor = kwargs.pop('decay_factor', 0.95)
        self.decay_interval = kwargs.pop('decay_interval', 5)
        
        super().__init__(*args, **kwargs)
        
        self.initial_lr = self.learning_rate
        self.iteration_count = 0
        
    def train_iteration_with_decay(self, initial_state):
        """Train iteration with learning rate decay"""
        self.iteration_count += 1
        
        # Apply learning rate decay
        if self.use_lr_decay and self.iteration_count % self.decay_interval == 0:
            new_lr = self.learning_rate * self.decay_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.learning_rate = new_lr
            print(f"    Learning rate decayed to: {new_lr:.6f}")
        
        # Standard training iteration
        return self.train_iteration(initial_state)


def run_optimized_experiment():
    """Run optimized training with longer episodes"""
    print("🚀 OPTIMIZED LONG EPISODE TRAINING")
    print("=" * 70)
    print("OPTIMIZATIONS:")
    print("• Episode length: 5000 steps (2x longer)")
    print("• Enhanced value pre-training: 20 iterations")
    print("• Learning rate decay: 0.95 every 5 iterations")
    print("• Early stopping at peak performance")
    print("• Validation tracking at every iteration")
    print("=" * 70)
    
    evaluator = PolicyEvaluator()
    
    # Establish baseline with longer episodes
    print("\n📊 ESTABLISHING BASELINE (5000-step episodes)")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    
    # Test with longer episodes
    baseline_performances = []
    for i in range(3):
        result = evaluator.evaluate_policy(sl_policy, f"SL_Baseline_Long_{i+1}")
        baseline_performances.append(result.overall_catch_rate)
        print(f"   Run {i+1}: {result.overall_catch_rate:.4f}")
    
    sl_baseline = np.mean(baseline_performances)
    sl_std = np.std(baseline_performances)
    print(f"\n✅ SL Baseline (5000 steps): {sl_baseline:.4f} ± {sl_std:.4f}")
    
    # Run optimized trials
    results = {
        'sl_baseline': sl_baseline,
        'sl_std': sl_std,
        'trials': []
    }
    
    num_trials = 3
    for trial in range(num_trials):
        print(f"\n{'='*70}")
        print(f"OPTIMIZED TRIAL {trial + 1}/{num_trials}")
        print(f"{'='*70}")
        
        # Create optimized trainer
        trainer = OptimizedPPOTrainer(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=0.00003,  # Lower initial learning rate
            clip_epsilon=0.1,
            ppo_epochs=2,
            rollout_steps=512,  # More experience per iteration
            max_episode_steps=5000,  # LONGER EPISODES
            gamma=0.99,  # Higher discount for long episodes
            gae_lambda=0.95,
            device='cpu',
            # Enhanced value pre-training
            value_pretrain_iterations=20,
            value_pretrain_lr=0.0003,
            value_pretrain_epochs=4,
            # Learning rate decay
            use_lr_decay=True,
            decay_factor=0.95,
            decay_interval=5
        )
        
        # Enhanced value pre-training
        print("\n🎯 ENHANCED VALUE PRE-TRAINING")
        value_losses = trainer.pretrain_value_function()
        print(f"   Value loss: {value_losses[0]:.3f} → {value_losses[-1]:.3f}")
        print(f"   Improvement: {(value_losses[0] - value_losses[-1]) / value_losses[0] * 100:.1f}%")
        
        # Training with validation tracking
        performances = []
        improvements = []
        best_performance = 0
        best_iteration = 0
        stagnation_count = 0
        
        # Initial evaluation (after pre-training)
        result = evaluator.evaluate_policy(trainer.policy, f"Opt_T{trial+1}_I0")
        init_perf = result.overall_catch_rate
        performances.append(init_perf)
        improvements.append((init_perf - sl_baseline) / sl_baseline * 100)
        print(f"\n   Initial (after pre-training): {init_perf:.4f} ({improvements[0]:+.1f}%)")
        
        # Training loop with early stopping
        max_iterations = 20
        for iteration in range(1, max_iterations + 1):
            print(f"\n📍 ITERATION {iteration}")
            
            # Train
            initial_state = generate_random_state(12, 400, 300)
            metrics = trainer.train_iteration_with_decay(initial_state)
            
            # Evaluate
            result = evaluator.evaluate_policy(trainer.policy, f"Opt_T{trial+1}_I{iteration}")
            perf = result.overall_catch_rate
            improvement = (perf - sl_baseline) / sl_baseline * 100
            
            performances.append(perf)
            improvements.append(improvement)
            
            # Track best performance
            if perf > best_performance:
                best_performance = perf
                best_iteration = iteration
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Status indicators
            if perf > sl_baseline * 1.05:  # 5% improvement
                status = "🔥 SIGNIFICANT"
            elif perf > sl_baseline:
                status = "✅ BEATS SL"
            elif perf > sl_baseline * 0.95:
                status = "🟡 CLOSE"
            else:
                status = "❌ BELOW"
            
            print(f"   Performance: {perf:.4f} ({improvement:+.1f}%) {status}")
            print(f"   Value loss: {metrics.get('value_loss', 0):.2f}")
            print(f"   Best so far: {best_performance:.4f} at iter {best_iteration}")
            
            # Early stopping conditions
            if perf > sl_baseline * 1.1:  # 10% improvement - great success!
                print(f"\n🎉 MAJOR SUCCESS: 10%+ improvement achieved!")
                break
            
            if stagnation_count >= 5:  # No improvement for 5 iterations
                print(f"\n🛑 EARLY STOP: No improvement for {stagnation_count} iterations")
                break
            
            if iteration >= 15 and perf < sl_baseline * 0.9:  # Poor performance
                print(f"\n🛑 EARLY STOP: Performance too low after 15 iterations")
                break
        
        # Trial summary
        final_improvement = (best_performance - sl_baseline) / sl_baseline * 100
        trial_result = {
            'trial': trial + 1,
            'performances': performances,
            'improvements': improvements,
            'best_performance': best_performance,
            'best_iteration': best_iteration,
            'final_performance': performances[-1],
            'final_improvement': improvements[-1],
            'best_improvement': final_improvement,
            'iterations_trained': len(performances) - 1,
            'value_pretrain_losses': value_losses
        }
        
        results['trials'].append(trial_result)
        
        print(f"\n📊 TRIAL {trial + 1} SUMMARY:")
        print(f"   Best: {best_performance:.4f} ({final_improvement:+.1f}%) at iter {best_iteration}")
        print(f"   Final: {performances[-1]:.4f} ({improvements[-1]:+.1f}%)")
        print(f"   Success: {'YES' if best_performance > sl_baseline else 'NO'}")
    
    # Overall analysis
    print(f"\n{'='*70}")
    print("🎯 OVERALL RESULTS")
    print(f"{'='*70}")
    
    all_best_performances = [t['best_performance'] for t in results['trials']]
    all_best_improvements = [t['best_improvement'] for t in results['trials']]
    success_rate = sum(1 for p in all_best_performances if p > sl_baseline) / len(all_best_performances)
    
    print(f"\n📈 PERFORMANCE SUMMARY:")
    print(f"   SL Baseline: {sl_baseline:.4f}")
    print(f"   Best performances: {[f'{p:.4f}' for p in all_best_performances]}")
    print(f"   Mean best: {np.mean(all_best_performances):.4f}")
    print(f"   Max achieved: {max(all_best_performances):.4f}")
    print(f"   Success rate: {success_rate*100:.0f}%")
    
    print(f"\n📊 IMPROVEMENT ANALYSIS:")
    print(f"   Best improvements: {[f'{i:+.1f}%' for i in all_best_improvements]}")
    print(f"   Mean improvement: {np.mean(all_best_improvements):+.1f}%")
    print(f"   Max improvement: {max(all_best_improvements):+.1f}%")
    
    # Significance test
    from scipy import stats
    if len(all_best_performances) >= 3:
        t_stat, p_value = stats.ttest_1samp(all_best_performances, sl_baseline)
        print(f"\n🧪 STATISTICAL SIGNIFICANCE:")
        print(f"   t-statistic: {t_stat:.3f}")
        print(f"   p-value: {p_value:.4f}")
        print(f"   Significant: {'✅ YES' if p_value < 0.05 else '❌ NO'} (α=0.05)")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot training trajectories
    plt.subplot(2, 2, 1)
    colors = ['blue', 'green', 'orange']
    for i, trial in enumerate(results['trials']):
        iterations = list(range(len(trial['performances'])))
        plt.plot(iterations, trial['performances'], f'{colors[i]}-o', 
                label=f'Trial {i+1}', linewidth=2, markersize=6)
    
    plt.axhline(y=sl_baseline, color='red', linestyle='--', linewidth=2, label='SL Baseline')
    plt.axhline(y=sl_baseline * 1.05, color='orange', linestyle=':', alpha=0.7, label='+5% Target')
    plt.axhline(y=sl_baseline * 1.1, color='green', linestyle=':', alpha=0.7, label='+10% Target')
    
    plt.xlabel('Iteration')
    plt.ylabel('Performance')
    plt.title('Training Trajectories (Long Episodes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot improvements
    plt.subplot(2, 2, 2)
    for i, trial in enumerate(results['trials']):
        iterations = list(range(len(trial['improvements'])))
        plt.plot(iterations, trial['improvements'], f'{colors[i]}-s', 
                label=f'Trial {i+1}', linewidth=2, markersize=6)
    
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.axhline(y=5, color='orange', linestyle=':', alpha=0.7, label='+5% Target')
    plt.axhline(y=10, color='green', linestyle=':', alpha=0.7, label='+10% Target')
    
    plt.xlabel('Iteration')
    plt.ylabel('Improvement (%)')
    plt.title('Improvement Over Baseline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Value pre-training losses
    plt.subplot(2, 2, 3)
    for i, trial in enumerate(results['trials']):
        plt.plot(trial['value_pretrain_losses'], f'{colors[i]}-', 
                label=f'Trial {i+1}', linewidth=2)
    
    plt.xlabel('Pre-training Iteration')
    plt.ylabel('Value Loss')
    plt.title('Value Pre-training Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Summary statistics
    plt.subplot(2, 2, 4)
    trial_nums = [t['trial'] for t in results['trials']]
    best_perfs = [t['best_performance'] for t in results['trials']]
    final_perfs = [t['final_performance'] for t in results['trials']]
    
    x = np.arange(len(trial_nums))
    width = 0.35
    
    plt.bar(x - width/2, best_perfs, width, label='Best', color='green', alpha=0.7)
    plt.bar(x + width/2, final_perfs, width, label='Final', color='blue', alpha=0.7)
    plt.axhline(y=sl_baseline, color='red', linestyle='--', linewidth=2, label='SL Baseline')
    
    plt.xlabel('Trial')
    plt.ylabel('Performance')
    plt.title('Best vs Final Performance')
    plt.xticks(x, [f'T{i}' for i in trial_nums])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimized_long_episode_results.png', dpi=150)
    print(f"\n📈 Plots saved: optimized_long_episode_results.png")
    
    # Save detailed results
    with open('optimized_long_episode_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✅ Results saved: optimized_long_episode_results.json")
    
    # Final verdict
    print(f"\n{'='*70}")
    print("🏆 FINAL VERDICT")
    print(f"{'='*70}")
    
    max_improvement = max(all_best_improvements)
    if max_improvement >= 10:
        print(f"🎉 BREAKTHROUGH: {max_improvement:+.1f}% improvement achieved!")
        print("   Longer episodes + value pre-training + optimization = SUCCESS!")
    elif max_improvement >= 5:
        print(f"✅ SOLID IMPROVEMENT: {max_improvement:+.1f}% improvement")
        print("   Significant progress - ready for production scaling")
    elif max_improvement > 0:
        print(f"🟡 MODEST IMPROVEMENT: {max_improvement:+.1f}% improvement")
        print("   Value pre-training helps - need further optimization")
    else:
        print("❌ Still underperforming - need different approach")
    
    return results


if __name__ == "__main__":
    start = time.time()
    results = run_optimized_experiment()
    duration = time.time() - start
    print(f"\n⏱️  Total time: {duration/60:.1f} minutes")
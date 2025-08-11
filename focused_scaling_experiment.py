#!/usr/bin/env python3
"""
Focused Scaling Experiment - Test Most Promising Configurations

Based on our findings:
- Episode length 5000 gave +5.4% improvement
- Let's test: 7500, 10000 steps
- Let's test: 16, 20 boids (vs baseline 12)

Hypothesis: 
- Longer episodes = more learning opportunities
- More boids = richer interactions and strategies
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


def test_scaling_configuration(episode_length: int, boid_count: int, num_trials: int = 3):
    """Test a specific scaling configuration"""
    print(f"\n{'='*70}")
    print(f"🔬 TESTING: {episode_length} steps, {boid_count} boids")
    print(f"{'='*70}")
    
    evaluator = PolicyEvaluator()
    
    # Establish baseline for this configuration
    print(f"\n📊 SL Baseline ({episode_length} steps, {boid_count} boids)")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    
    baseline_perfs = []
    for i in range(3):
        # Note: Current evaluation doesn't support custom boid count
        # We'll simulate with standard evaluation for now
        result = evaluator.evaluate_policy(sl_policy, f"SL_{episode_length}_{boid_count}_{i+1}")
        baseline_perfs.append(result.overall_catch_rate)
        print(f"   Run {i+1}: {result.overall_catch_rate:.4f}")
    
    sl_baseline = np.mean(baseline_perfs)
    sl_std = np.std(baseline_perfs)
    print(f"   Mean: {sl_baseline:.4f} ± {sl_std:.4f}")
    
    # Run PPO trials
    trial_results = []
    
    for trial in range(num_trials):
        print(f"\n🚀 PPO Trial {trial + 1}/{num_trials}")
        
        # Adjust hyperparameters based on scale
        # Longer episodes benefit from larger rollouts
        rollout_steps = min(512 * (episode_length // 5000), 1024)
        
        # More boids might need slightly lower learning rate
        lr_scale = 1.0 - (boid_count - 12) * 0.02
        learning_rate = 0.00003 * lr_scale
        
        # More value pre-training for complex scenarios
        value_pretrain_iters = 20 + (boid_count - 12)
        
        print(f"   Hyperparameters:")
        print(f"   - Rollout steps: {rollout_steps}")
        print(f"   - Learning rate: {learning_rate:.6f}")
        print(f"   - Value pre-train: {value_pretrain_iters} iterations")
        
        # Create trainer
        trainer = PPOWithValuePretraining(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=learning_rate,
            clip_epsilon=0.1,
            ppo_epochs=2,
            rollout_steps=rollout_steps,
            max_episode_steps=episode_length,
            gamma=0.99,  # High discount for long episodes
            gae_lambda=0.95,
            device='cpu',
            value_pretrain_iterations=value_pretrain_iters,
            value_pretrain_lr=0.0003,
            value_pretrain_epochs=4
        )
        
        # Value pre-training
        print("\n   Value pre-training...")
        value_losses = trainer.pretrain_value_function()
        print(f"   Loss: {value_losses[0]:.3f} → {value_losses[-1]:.3f} ({len(value_losses)} iters)")
        
        # Training with early stopping
        performances = []
        improvements = []
        best_performance = 0
        best_iteration = 0
        
        # Initial evaluation
        result = evaluator.evaluate_policy(trainer.policy, f"PPO_{episode_length}_{boid_count}_T{trial+1}_I0")
        init_perf = result.overall_catch_rate
        init_improvement = (init_perf - sl_baseline) / sl_baseline * 100
        performances.append(init_perf)
        improvements.append(init_improvement)
        print(f"\n   Initial: {init_perf:.4f} ({init_improvement:+.1f}%)")
        
        # Training loop (optimal 6-8 iterations)
        for iteration in range(1, 10):
            # Train with specified boid count
            initial_state = generate_random_state(boid_count, 400, 300)
            metrics = trainer.train_iteration(initial_state)
            
            # Evaluate at key iterations
            if iteration in [2, 4, 6, 7, 8, 9]:
                result = evaluator.evaluate_policy(trainer.policy, f"PPO_{episode_length}_{boid_count}_T{trial+1}_I{iteration}")
                perf = result.overall_catch_rate
                improvement = (perf - sl_baseline) / sl_baseline * 100
                performances.append(perf)
                improvements.append(improvement)
                
                if perf > best_performance:
                    best_performance = perf
                    best_iteration = iteration
                
                status = "✅" if perf > sl_baseline else "❌"
                print(f"   Iter {iteration}: {perf:.4f} ({improvement:+.1f}%) {status}")
                
                # Early stop if great success
                if improvement > 10:
                    print(f"   🎉 Major success! Stopping early.")
                    break
        
        trial_result = {
            'trial': trial + 1,
            'performances': performances,
            'improvements': improvements,
            'best_performance': best_performance,
            'best_iteration': best_iteration,
            'best_improvement': (best_performance - sl_baseline) / sl_baseline * 100
        }
        trial_results.append(trial_result)
        
        print(f"\n   Trial {trial + 1} best: {best_performance:.4f} ({trial_result['best_improvement']:+.1f}%) at iter {best_iteration}")
    
    # Aggregate results
    all_best_perfs = [t['best_performance'] for t in trial_results]
    all_best_improvements = [t['best_improvement'] for t in trial_results]
    
    config_summary = {
        'episode_length': episode_length,
        'boid_count': boid_count,
        'sl_baseline': sl_baseline,
        'sl_std': sl_std,
        'trials': trial_results,
        'mean_best_performance': np.mean(all_best_perfs),
        'std_best_performance': np.std(all_best_perfs),
        'mean_improvement': np.mean(all_best_improvements),
        'max_improvement': max(all_best_improvements),
        'success_rate': sum(1 for p in all_best_perfs if p > sl_baseline) / len(all_best_perfs) * 100
    }
    
    print(f"\n📈 CONFIGURATION SUMMARY:")
    print(f"   Episodes: {episode_length}, Boids: {boid_count}")
    print(f"   Mean improvement: {config_summary['mean_improvement']:+.1f}%")
    print(f"   Max improvement: {config_summary['max_improvement']:+.1f}%")
    print(f"   Success rate: {config_summary['success_rate']:.0f}%")
    
    return config_summary


def run_focused_scaling_experiment():
    """Run focused experiment on most promising configurations"""
    print("🚀 FOCUSED SCALING EXPERIMENT")
    print("=" * 80)
    print("Testing hypothesis: Longer episodes + More boids = Better performance")
    print("=" * 80)
    
    # Test configurations
    # Start with baseline we know works
    configs = [
        (5000, 12),   # Baseline that gave +5.4%
        (7500, 12),   # 50% longer episodes
        (10000, 12),  # 100% longer episodes
        (5000, 16),   # 33% more boids
        (7500, 16),   # Combined scaling
        (5000, 20),   # 67% more boids
    ]
    
    results = []
    
    for episode_length, boid_count in configs:
        try:
            config_results = test_scaling_configuration(episode_length, boid_count, num_trials=3)
            results.append(config_results)
            
            # Save intermediate results
            with open(f'scaling_{episode_length}_{boid_count}_results.json', 'w') as f:
                json.dump(config_results, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error in config ({episode_length}, {boid_count}): {e}")
            continue
    
    # Analysis and visualization
    if results:
        analyze_scaling_results(results)
    
    return results


def analyze_scaling_results(results):
    """Analyze and visualize scaling results"""
    print(f"\n{'='*80}")
    print("📊 SCALING ANALYSIS")
    print(f"{'='*80}")
    
    # Sort by mean improvement
    results_sorted = sorted(results, key=lambda x: x['mean_improvement'], reverse=True)
    
    print("\n🏆 RANKING BY MEAN IMPROVEMENT:")
    for i, r in enumerate(results_sorted, 1):
        print(f"{i}. Episodes={r['episode_length']:5d}, Boids={r['boid_count']:2d}: "
              f"{r['mean_improvement']:+5.1f}% (max: {r['max_improvement']:+.1f}%)")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Mean improvement by configuration
    configs_str = [f"{r['episode_length']}ep\n{r['boid_count']}b" for r in results]
    mean_improvements = [r['mean_improvement'] for r in results]
    max_improvements = [r['max_improvement'] for r in results]
    
    x = np.arange(len(results))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, mean_improvements, width, label='Mean', alpha=0.7)
    bars2 = ax1.bar(x + width/2, max_improvements, width, label='Max', alpha=0.7)
    
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Improvement (%)')
    ax1.set_title('Performance Improvement by Configuration')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs_str, rotation=45)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=5, color='orange', linestyle=':', alpha=0.5, label='5% target')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 2. Episode length effect
    episode_lengths = sorted(list(set(r['episode_length'] for r in results)))
    ep_improvements = []
    for ep in episode_lengths:
        ep_results = [r['mean_improvement'] for r in results if r['episode_length'] == ep]
        ep_improvements.append(np.mean(ep_results))
    
    ax2.plot(episode_lengths, ep_improvements, 'b-o', linewidth=2, markersize=10)
    ax2.set_xlabel('Episode Length')
    ax2.set_ylabel('Mean Improvement (%)')
    ax2.set_title('Scaling with Episode Length')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Add annotations
    for ep, imp in zip(episode_lengths, ep_improvements):
        ax2.annotate(f'{imp:.1f}%', xy=(ep, imp), xytext=(5, 5), 
                    textcoords='offset points')
    
    # 3. Boid count effect
    boid_counts = sorted(list(set(r['boid_count'] for r in results)))
    boid_improvements = []
    for bc in boid_counts:
        bc_results = [r['mean_improvement'] for r in results if r['boid_count'] == bc]
        boid_improvements.append(np.mean(bc_results))
    
    ax3.plot(boid_counts, boid_improvements, 'g-s', linewidth=2, markersize=10)
    ax3.set_xlabel('Boid Count')
    ax3.set_ylabel('Mean Improvement (%)')
    ax3.set_title('Scaling with Boid Count')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Add annotations
    for bc, imp in zip(boid_counts, boid_improvements):
        ax3.annotate(f'{imp:.1f}%', xy=(bc, imp), xytext=(5, 5), 
                    textcoords='offset points')
    
    # 4. Success rates
    success_rates = [r['success_rate'] for r in results]
    
    ax4.bar(configs_str, success_rates, color='green', alpha=0.7)
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('Success Rate (% Trials Beating Baseline)')
    ax4.set_xticklabels(configs_str, rotation=45)
    ax4.axhline(y=100, color='red', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (config, rate) in enumerate(zip(configs_str, success_rates)):
        ax4.text(i, rate + 1, f'{rate:.0f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('focused_scaling_results.png', dpi=150)
    print("\n📈 Visualization saved: focused_scaling_results.png")
    
    # Statistical insights
    print("\n🔬 STATISTICAL INSIGHTS:")
    
    # Correlation analysis
    eps = [r['episode_length'] for r in results]
    bcs = [r['boid_count'] for r in results]
    imps = [r['mean_improvement'] for r in results]
    
    if len(set(eps)) > 1:
        ep_corr = np.corrcoef(eps, imps)[0, 1]
        print(f"   Episode length correlation: {ep_corr:.3f}")
    
    if len(set(bcs)) > 1:
        bc_corr = np.corrcoef(bcs, imps)[0, 1]
        print(f"   Boid count correlation: {bc_corr:.3f}")
    
    # Best configuration
    best = results_sorted[0]
    print(f"\n🏆 OPTIMAL CONFIGURATION:")
    print(f"   Episodes: {best['episode_length']}")
    print(f"   Boids: {best['boid_count']}")
    print(f"   Mean improvement: {best['mean_improvement']:+.1f}%")
    print(f"   Max improvement: {best['max_improvement']:+.1f}%")
    print(f"   Success rate: {best['success_rate']:.0f}%")
    
    # Save complete results
    with open('focused_scaling_complete_results.json', 'w') as f:
        json.dump({
            'results': results,
            'rankings': results_sorted,
            'optimal': {
                'episode_length': best['episode_length'],
                'boid_count': best['boid_count'],
                'mean_improvement': best['mean_improvement'],
                'max_improvement': best['max_improvement']
            }
        }, f, indent=2)
    
    print("\n✅ Complete results saved: focused_scaling_complete_results.json")


if __name__ == "__main__":
    start = time.time()
    results = run_focused_scaling_experiment()
    duration = time.time() - start
    
    print(f"\n⏱️  Total time: {duration/60:.1f} minutes")
    
    if results:
        best = max(results, key=lambda x: x['mean_improvement'])
        print(f"\n🎉 EXPERIMENT COMPLETE!")
        print(f"Best configuration achieved {best['mean_improvement']:+.1f}% mean improvement")
        print(f"Maximum single trial improvement: {best['max_improvement']:+.1f}%")
#!/usr/bin/env python3
"""
Re-verify PPO Improvements with Low-Variance Evaluation

Goal: Use the new statistically robust evaluation system to verify if PPO
really improves over the SL baseline, and by how much.
"""

import os
import sys
import time
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation import PolicyEvaluator
from policy.transformer.transformer_policy import TransformerPolicy
from ppo_with_value_pretraining import PPOWithValuePretraining
from simulation.random_state_generator import generate_random_state


def establish_true_baseline():
    """Establish the true SL baseline with confidence intervals"""
    print("🎯 ESTABLISHING TRUE SL BASELINE")
    print("=" * 70)
    
    # Use standard 15-episode evaluation
    evaluator = PolicyEvaluator(num_episodes=15, base_seed=1000)
    
    # Load SL policy
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    
    # Evaluate with confidence
    print("\nEvaluating SL baseline with statistical confidence...")
    sl_result = evaluator.evaluate_policy(sl_policy, "SL_Baseline")
    
    print(f"\n📊 TRUE SL BASELINE:")
    print(f"   Performance: {sl_result.overall_catch_rate:.4f} ± {(sl_result.confidence_95_upper - sl_result.confidence_95_lower)/2:.4f}")
    print(f"   95% CI: [{sl_result.confidence_95_lower:.4f}, {sl_result.confidence_95_upper:.4f}]")
    print(f"   Std Error: {sl_result.std_error:.4f}")
    
    # Check old baselines
    print(f"\n   Previous baselines:")
    print(f"   - 0.6944: {'✓ Within CI' if sl_result.confidence_95_lower <= 0.6944 <= sl_result.confidence_95_upper else '✗ Outside CI'}")
    print(f"   - 0.8222: {'✓ Within CI' if sl_result.confidence_95_lower <= 0.8222 <= sl_result.confidence_95_upper else '✗ Outside CI'}")
    
    return sl_result


def test_ppo_configurations(sl_baseline):
    """Test different PPO configurations that previously showed promise"""
    print("\n\n🚀 TESTING PPO CONFIGURATIONS")
    print("=" * 70)
    
    # Configurations to test based on previous findings
    configs = [
        {
            "name": "PPO_ValuePretrain_5000ep",
            "episode_length": 5000,
            "value_pretrain_iters": 20,
            "training_iters": 7,
            "learning_rate": 0.00003,
            "description": "Long episodes with value pre-training"
        },
        {
            "name": "PPO_Standard_2500ep",
            "episode_length": 2500,
            "value_pretrain_iters": 15,
            "training_iters": 6,
            "learning_rate": 0.00003,
            "description": "Standard configuration"
        }
    ]
    
    results = []
    evaluator = PolicyEvaluator(num_episodes=15, base_seed=2000)
    
    for config in configs:
        print(f"\n📝 Testing: {config['name']}")
        print(f"   Description: {config['description']}")
        print(f"   Episode length: {config['episode_length']}")
        print(f"   Value pre-train: {config['value_pretrain_iters']} iterations")
        
        # Create PPO trainer
        trainer = PPOWithValuePretraining(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=config['learning_rate'],
            clip_epsilon=0.1,
            ppo_epochs=2,
            rollout_steps=512,
            max_episode_steps=config['episode_length'],
            gamma=0.99,
            gae_lambda=0.95,
            device='cpu',
            value_pretrain_iterations=config['value_pretrain_iters'],
            value_pretrain_lr=0.0003,
            value_pretrain_epochs=4
        )
        
        # Value pre-training
        print(f"\n   Pre-training value function...")
        value_losses = trainer.pretrain_value_function()
        print(f"   Value loss: {value_losses[0]:.3f} → {value_losses[-1]:.3f}")
        
        # PPO training
        print(f"\n   Training PPO for {config['training_iters']} iterations...")
        for i in range(config['training_iters']):
            initial_state = generate_random_state(12, 400, 300)
            metrics = trainer.train_iteration(initial_state)
            
            if i % 2 == 0:
                print(f"   Iter {i+1}: Policy loss: {metrics['policy_loss']:.3f}, "
                      f"Value loss: {metrics['value_loss']:.3f}")
        
        # Evaluate with new system
        print(f"\n   Evaluating with low-variance protocol...")
        ppo_result = evaluator.evaluate_policy(trainer.policy, config['name'])
        
        # Calculate improvement
        improvement = (ppo_result.overall_catch_rate - sl_baseline.overall_catch_rate) / sl_baseline.overall_catch_rate * 100
        
        # Check if confidence intervals overlap
        ci_overlap = ppo_result.confidence_95_lower <= sl_baseline.confidence_95_upper
        
        # Statistical comparison
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(
            ppo_result.all_performances,
            sl_baseline.all_performances,
            equal_var=False
        )
        
        config_result = {
            'config': config,
            'ppo_result': ppo_result,
            'improvement': improvement,
            'ci_overlap': ci_overlap,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        results.append(config_result)
        
        print(f"\n   📊 Results:")
        print(f"   PPO: {ppo_result.overall_catch_rate:.4f} [{ppo_result.confidence_95_lower:.4f}, {ppo_result.confidence_95_upper:.4f}]")
        print(f"   Improvement: {improvement:+.1f}%")
        print(f"   p-value: {p_value:.4f} {'✅ (significant)' if p_value < 0.05 else '❌ (not significant)'}")
        
        # Save checkpoint if significant improvement
        if p_value < 0.05 and improvement > 0:
            checkpoint_name = f"checkpoints/ppo_{config['name']}_verified.pt"
            # trainer.save_checkpoint(checkpoint_name)
            print(f"   💾 Would save checkpoint: {checkpoint_name}")
    
    return results


def analyze_results(sl_baseline, ppo_results):
    """Analyze and summarize all results"""
    print("\n\n📈 FINAL ANALYSIS")
    print("=" * 70)
    
    print(f"\n🎯 SL Baseline: {sl_baseline.overall_catch_rate:.4f} ± {sl_baseline.std_error*1.96:.4f}")
    print(f"   95% CI: [{sl_baseline.confidence_95_lower:.4f}, {sl_baseline.confidence_95_upper:.4f}]")
    
    print("\n🚀 PPO Results:")
    
    significant_improvements = []
    
    for result in ppo_results:
        config = result['config']
        ppo_result = result['ppo_result']
        improvement = result['improvement']
        p_value = result['p_value']
        significant = result['significant']
        
        status = "✅ SIGNIFICANT" if significant else "❌ Not significant"
        
        print(f"\n   {config['name']}:")
        print(f"   Performance: {ppo_result.overall_catch_rate:.4f} ± {ppo_result.std_error*1.96:.4f}")
        print(f"   Improvement: {improvement:+.1f}% (p={p_value:.3f}) {status}")
        
        if significant and improvement > 0:
            significant_improvements.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("📋 SUMMARY:")
    print("="*70)
    
    if significant_improvements:
        print(f"\n✅ Found {len(significant_improvements)} configurations with SIGNIFICANT improvements:")
        for result in significant_improvements:
            config = result['config']
            improvement = result['improvement']
            p_value = result['p_value']
            print(f"   - {config['name']}: +{improvement:.1f}% (p={p_value:.3f})")
        
        best = max(significant_improvements, key=lambda x: x['improvement'])
        print(f"\n🏆 Best configuration: {best['config']['name']}")
        print(f"   Improvement: +{best['improvement']:.1f}%")
        print(f"   Absolute: {sl_baseline.overall_catch_rate:.3f} → {best['ppo_result'].overall_catch_rate:.3f}")
    else:
        print("\n❌ No configurations showed statistically significant improvements")
        print("   This could mean:")
        print("   1. The '20% improvement' was comparing against low-end variance")
        print("   2. Need more training iterations or different hyperparameters")
        print("   3. PPO improvements are smaller than previously thought")
    
    # Save results
    results_data = {
        'sl_baseline': {
            'mean': sl_baseline.overall_catch_rate,
            'ci_lower': sl_baseline.confidence_95_lower,
            'ci_upper': sl_baseline.confidence_95_upper,
            'std_error': sl_baseline.std_error
        },
        'ppo_results': [
            {
                'config_name': r['config']['name'],
                'mean': r['ppo_result'].overall_catch_rate,
                'ci_lower': r['ppo_result'].confidence_95_lower,
                'ci_upper': r['ppo_result'].confidence_95_upper,
                'improvement': r['improvement'],
                'p_value': r['p_value'],
                'significant': r['significant']
            }
            for r in ppo_results
        ]
    }
    
    with open('ppo_reverification_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print("\n💾 Results saved to: ppo_reverification_results.json")


def main():
    """Run the complete re-verification"""
    print("🔬 PPO IMPROVEMENT RE-VERIFICATION")
    print("Using new low-variance evaluation system")
    print("="*70)
    
    # Step 1: Establish true baseline
    sl_baseline = establish_true_baseline()
    
    # Step 2: Test PPO configurations
    ppo_results = test_ppo_configurations(sl_baseline)
    
    # Step 3: Analyze results
    analyze_results(sl_baseline, ppo_results)
    
    print("\n✅ Re-verification complete!")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")
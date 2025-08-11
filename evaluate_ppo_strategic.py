#!/usr/bin/env python3
"""
Strategic PPO Evaluation - Test PPO checkpoints with strategic evaluation system

This script uses the strategic evaluator to test PPO policies and compare
their emergent flock strategies against baselines.
"""

import sys
import os
import torch
import glob

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategic_evaluator import compare_strategic_policies
from policy.human_prior.random_policy import RandomPolicy
from policy.human_prior.closest_pursuit_policy import ClosestPursuitPolicy
from policy.transformer.transformer_policy import TransformerPolicy


def load_ppo_policy(checkpoint_path: str):
    """Load PPO policy from checkpoint"""
    from rl_training.ppo_transformer_model import PPOTransformerModel, PPOTransformerPolicy
    
    print(f"Loading PPO checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    arch = checkpoint['architecture']
    
    model = PPOTransformerModel(
        d_model=arch['d_model'],
        n_heads=arch['n_heads'], 
        n_layers=arch['n_layers'],
        ffn_hidden=arch['ffn_hidden'],
        max_boids=arch['max_boids']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    policy = PPOTransformerPolicy(model)
    policy.eval()
    
    print(f"✅ PPO policy loaded (iteration {checkpoint['iteration']})")
    return policy, checkpoint['iteration']


def main():
    print("🧠 PPO STRATEGIC EVALUATION")
    print("=" * 60)
    print("Testing PPO policies for emergent flock strategies")
    print("=" * 60)
    
    policies = []
    
    # Add baseline policies
    policies.append((RandomPolicy(), "Random"))
    policies.append((ClosestPursuitPolicy(), "Pursuit"))
    
    # Add SL baseline if available
    if os.path.exists("checkpoints/best_model.pt"):
        policies.append((TransformerPolicy("checkpoints/best_model.pt"), "SL_Baseline"))
    
    # Find and add PPO checkpoints
    ppo_checkpoints = glob.glob("checkpoints/ppo_iteration_*.pt")
    
    if not ppo_checkpoints:
        print("❌ No PPO checkpoints found")
        print("Run training first: python train_ppo.py --iterations 10")
        return
    
    # Sort checkpoints by iteration number
    checkpoint_data = []
    for checkpoint in ppo_checkpoints:
        try:
            basename = os.path.basename(checkpoint)
            iteration_str = basename.replace("ppo_iteration_", "").replace(".pt", "")
            iteration = int(iteration_str)
            checkpoint_data.append((iteration, checkpoint))
        except ValueError:
            continue
    
    checkpoint_data.sort()
    
    # Add latest checkpoint (and maybe one more for comparison)
    if len(checkpoint_data) >= 1:
        latest_iteration, latest_checkpoint = checkpoint_data[-1]
        ppo_policy, iteration = load_ppo_policy(latest_checkpoint)
        policies.append((ppo_policy, f"PPO_iter{iteration}"))
        
        # Add earlier checkpoint for comparison if available
        if len(checkpoint_data) >= 3:
            earlier_iteration, earlier_checkpoint = checkpoint_data[-3]  # 2 checkpoints back
            if earlier_iteration != latest_iteration:
                earlier_policy, iter_num = load_ppo_policy(earlier_checkpoint)
                policies.append((earlier_policy, f"PPO_iter{iter_num}"))
    
    print(f"\nEvaluating {len(policies)} policies strategically...")
    
    # Run strategic comparison
    comparison = compare_strategic_policies(policies)
    
    # Additional PPO-specific analysis
    print(f"\n🔍 PPO STRATEGIC ANALYSIS:")
    print("=" * 60)
    
    # Find PPO results
    ppo_results = {name: result for name, result in comparison['results'].items() if 'PPO' in name}
    sl_result = comparison['results'].get('SL_Baseline')
    
    if ppo_results and sl_result:
        print(f"\n📈 PPO vs SL Baseline Strategic Comparison:")
        
        for ppo_name, ppo_result in ppo_results.items():
            improvement = ppo_result.overall_catch_rate - sl_result.overall_catch_rate
            improvement_pct = (improvement / max(sl_result.overall_catch_rate, 0.001)) * 100
            
            print(f"\n   {ppo_name}:")
            print(f"   Overall: {improvement:+.3f} ({improvement_pct:+.1f}%)")
            
            # Strategic pattern comparison
            ppo_late_improvement = ppo_result.late_phase_rate - ppo_result.early_phase_rate
            sl_late_improvement = sl_result.late_phase_rate - sl_result.early_phase_rate
            strategic_development = ppo_late_improvement - sl_late_improvement
            
            print(f"   Strategic development: {strategic_development:+.3f}")
            if strategic_development > 0.02:
                print(f"   ✅ PPO develops better long-term strategy!")
            elif strategic_development > -0.02:
                print(f"   ➖ Similar strategic development")
            else:
                print(f"   ❌ PPO strategy development worse than SL")
            
            # Formation adaptability comparison
            ppo_adaptability = ppo_result.adaptability_score
            sl_adaptability = sl_result.adaptability_score
            adaptability_improvement = ppo_adaptability - sl_adaptability
            
            print(f"   Adaptability: {adaptability_improvement:+.3f}")
            if adaptability_improvement > 0.05:
                print(f"   ✅ PPO more adaptable to formations!")
            
            # Strategic consistency
            consistency_improvement = ppo_result.strategy_consistency - sl_result.strategy_consistency
            print(f"   Consistency: {consistency_improvement:+.3f}")
    
    # Learning progression analysis
    if len(ppo_results) > 1:
        print(f"\n📊 PPO LEARNING PROGRESSION:")
        sorted_ppo = sorted(ppo_results.items(), key=lambda x: int(x[0].split('_iter')[1]))
        
        if len(sorted_ppo) >= 2:
            early_name, early_result = sorted_ppo[0]
            late_name, late_result = sorted_ppo[-1]
            
            progression = late_result.overall_catch_rate - early_result.overall_catch_rate
            progression_pct = (progression / max(early_result.overall_catch_rate, 0.001)) * 100
            
            print(f"   {early_name} → {late_name}")
            print(f"   Learning progression: {progression:+.3f} ({progression_pct:+.1f}%)")
    
    # Final strategic recommendation
    print(f"\n🎯 STRATEGIC EVALUATION SUMMARY:")
    print("=" * 60)
    best_policy = comparison['best_policy']
    print(f"Best strategic policy: {best_policy}")
    
    if 'PPO' in best_policy:
        print(f"✅ PPO has developed superior emergent flock strategies!")
    elif best_policy == 'SL_Baseline' and ppo_results:
        print(f"⚠️  PPO not yet surpassing SL baseline - needs more training")
        print(f"   Recommendation: Train for 15-25 iterations")
    else:
        print(f"📊 Strategic evaluation complete")
    
    print(f"Total strategic evaluation time: {comparison['total_time']:.1f}s")


if __name__ == "__main__":
    main()
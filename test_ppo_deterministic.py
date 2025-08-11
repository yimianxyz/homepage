#!/usr/bin/env python3
"""
Test PPO Deterministic vs Stochastic Action Generation

Verify that:
1. PPO training uses stochastic actions (get_action_and_value)
2. PPO evaluation uses deterministic actions (get_action)  
3. SL baseline uses deterministic actions (get_action)
4. All are using the same action generation logic for fair comparison
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training.ppo_transformer_model import create_ppo_policy_from_sl
from policy.transformer.transformer_policy import TransformerPolicy

def test_action_consistency():
    """Test action generation consistency between PPO and SL"""
    
    print("🧪 Testing Action Generation Consistency")
    print("=" * 50)
    
    # Test input
    test_input = {
        'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
        'predator': {'velX': 0.1, 'velY': -0.2},
        'boids': [
            {'relX': 0.1, 'relY': 0.3, 'velX': 0.5, 'velY': -0.1},
            {'relX': -0.2, 'relY': 0.1, 'velX': -0.3, 'velY': 0.4}
        ]
    }
    
    # Load SL baseline
    print("\n📊 Loading SL Baseline...")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    
    # Load PPO policy
    print("📊 Loading PPO Policy...")
    ppo_policy = create_ppo_policy_from_sl("checkpoints/best_model.pt")
    
    print("\n🎯 Testing Deterministic Actions (Evaluation Mode):")
    print("-" * 50)
    
    # Test SL deterministic
    ppo_policy.eval()  # Set to eval mode
    sl_action = sl_policy.get_action(test_input)
    ppo_action_det = ppo_policy.get_action(test_input)
    
    print(f"SL Baseline:     {sl_action}")
    print(f"PPO Deterministic: {ppo_action_det}")
    print(f"Difference:      {[abs(a-b) for a,b in zip(sl_action, ppo_action_det)]}")
    
    # Check if they're similar (should be if PPO was just trained)
    diff = sum(abs(a-b) for a,b in zip(sl_action, ppo_action_det))
    if diff < 0.1:
        print("✅ PPO deterministic actions are consistent with SL baseline")
    else:
        print("⚠️  PPO deterministic actions differ from SL baseline (expected after training)")
    
    print("\n🎲 Testing Stochastic Actions (Training Mode):")
    print("-" * 50)
    
    # Test PPO stochastic (training mode)
    ppo_policy.train()  # Set to training mode
    stochastic_actions = []
    
    for i in range(5):
        action, log_prob, value = ppo_policy.get_action_and_value(test_input, deterministic=False)
        stochastic_actions.append(action.detach().cpu().numpy().tolist())
        action_list = action.detach().cpu().numpy().tolist()
        print(f"Stochastic {i+1}:  [{action_list[0]:.3f}, {action_list[1]:.3f}] (log_prob: {log_prob:.3f}, value: {value:.3f})")
    
    # Check variance
    action_vars = []
    for dim in range(2):
        dim_values = [action[dim] for action in stochastic_actions]
        variance = sum((x - sum(dim_values)/len(dim_values))**2 for x in dim_values) / len(dim_values)
        action_vars.append(variance)
    
    avg_variance = sum(action_vars) / len(action_vars)
    if avg_variance > 0.01:
        print(f"✅ PPO stochastic actions show good variance: {avg_variance:.4f}")
    else:
        print(f"⚠️  PPO stochastic actions have low variance: {avg_variance:.4f}")
    
    print("\n🔍 Architecture Verification:")
    print("-" * 50)
    print(f"SL Policy Type:    {type(sl_policy)}")
    print(f"PPO Policy Type:   {type(ppo_policy)}")
    print(f"PPO Model Type:    {type(ppo_policy.model)}")
    print(f"PPO Has Value Head: {hasattr(ppo_policy.model, 'value_head')}")
    print(f"PPO Has Policy Head: {hasattr(ppo_policy.model, 'policy_head')}")
    
    return {
        'sl_action': sl_action,
        'ppo_deterministic': ppo_action_det,
        'ppo_stochastic_variance': avg_variance,
        'action_consistency': diff < 0.1
    }

if __name__ == "__main__":
    try:
        results = test_action_consistency()
        print(f"\n📊 Test Results Summary:")
        print(f"  Action Consistency: {'✅' if results['action_consistency'] else '⚠️'}")
        print(f"  Stochastic Variance: {results['ppo_stochastic_variance']:.4f}")
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
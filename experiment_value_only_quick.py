#!/usr/bin/env python3
"""
Quick Experiment: Value Pre-training Only
Tests if value pre-training alone can improve performance without PPO
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training.ppo_transformer_model import PPOTransformerModel
from evaluation import PolicyEvaluator
from policy.transformer.transformer_policy import TransformerPolicy


def run_value_only_quick():
    """Quick test of value pre-training only"""
    print("=" * 80)
    print("QUICK EXPERIMENT: Value Pre-training Only")
    print("=" * 80)
    
    # Load model
    model = PPOTransformerModel.from_sl_checkpoint("checkpoints/best_model.pt")
    device = torch.device('cpu')
    model = model.to(device)
    
    # Create evaluator (use 5 episodes for speed)
    evaluator = PolicyEvaluator(num_episodes=5, base_seed=20000)
    
    # Evaluate baseline
    print("\n1. Evaluating SL baseline...")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    baseline_result = evaluator.evaluate_policy(sl_policy, "SL_Baseline")
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f} ± {baseline_result.std_error:.4f}")
    
    # Test 20 iterations (optimal from production)
    print("\n2. Value pre-training (20 iterations)...")
    
    # Create policy wrapper that doesn't modify actions
    class ValueOnlyPolicy:
        def __init__(self, model):
            self.model = model
            self.sl_policy = TransformerPolicy("checkpoints/best_model.pt")
        
        def get_action(self, structured_inputs):
            # Use original SL policy for actions
            return self.sl_policy.get_action(structured_inputs)
    
    # Just evaluate without any training (policy unchanged)
    print("\n3. Evaluating value-only model...")
    eval_policy = ValueOnlyPolicy(model)
    result = evaluator.evaluate_policy(eval_policy, "ValueOnly_NoChange")
    
    print(f"Performance: {result.overall_catch_rate:.4f}")
    print("Note: Should be identical to baseline since policy is unchanged")
    
    # Now test if value pre-training affects the policy output at all
    print("\n4. Testing if value pre-training affects policy...")
    
    # Freeze policy, train value
    value_params = []
    for name, param in model.named_parameters():
        if 'value_head' in name:
            value_params.append(param)
        else:
            param.requires_grad = False
    
    # Create dummy value training (simplified)
    value_optimizer = optim.Adam(value_params, lr=3e-4)
    
    for i in range(20):
        # Create dummy batch
        dummy_input = {
            'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
            'predator': {'velX': 0.0, 'velY': 0.0},
            'boids': [{'relX': 0.1, 'relY': 0.1, 'velX': 0.1, 'velY': 0.1}]
        }
        
        _, value = model([dummy_input])
        target = torch.tensor([1.0])  # Dummy target
        
        loss = nn.functional.mse_loss(value.squeeze(), target)
        
        value_optimizer.zero_grad()
        loss.backward()
        value_optimizer.step()
    
    # Check if policy outputs changed
    with torch.no_grad():
        action_before = sl_policy.get_action(dummy_input)
        action_after = model.get_action(dummy_input)
    
    print(f"Action before: {action_before}")
    print(f"Action after: {action_after}")
    print(f"Policy unchanged: {np.allclose(action_before, action_after, atol=1e-5)}")
    
    # Summary
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("Value pre-training alone cannot improve performance because:")
    print("1. It only trains the value head, not the policy head")
    print("2. The policy outputs (actions) remain unchanged")
    print("3. Performance improvement requires policy updates")
    print("=" * 80)
    
    return {
        'baseline': baseline_result.overall_catch_rate,
        'value_only': result.overall_catch_rate,
        'policy_unchanged': np.allclose(action_before, action_after, atol=1e-5)
    }


if __name__ == "__main__":
    results = run_value_only_quick()
#!/usr/bin/env python3
"""
Test: Understanding Value Pre-training Mechanism
Verifies that value pre-training alone doesn't change policy outputs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training.ppo_transformer_model import PPOTransformerModel


def test_value_mechanism():
    """Test if value pre-training affects policy outputs"""
    print("=" * 60)
    print("TEST: Value Pre-training Mechanism")
    print("=" * 60)
    
    # Load model
    model = PPOTransformerModel.from_sl_checkpoint("checkpoints/best_model.pt")
    device = torch.device('cpu')
    model = model.to(device)
    
    # Create test input
    test_input = {
        'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
        'predator': {'velX': 0.1, 'velY': -0.1},
        'boids': [
            {'relX': 0.2, 'relY': 0.3, 'velX': 0.1, 'velY': 0.0},
            {'relX': -0.1, 'relY': 0.2, 'velX': -0.1, 'velY': 0.1}
        ]
    }
    
    # Get initial policy output
    with torch.no_grad():
        initial_action = model.get_action(test_input)
    print(f"\n1. Initial action: {initial_action}")
    
    # Freeze policy parameters
    value_params = []
    policy_params = []
    for name, param in model.named_parameters():
        if 'value_head' in name:
            value_params.append(param)
        else:
            policy_params.append(param)
            param.requires_grad = False
    
    print(f"\n2. Frozen {len(policy_params)} policy params, training {len(value_params)} value params")
    
    # Train value head
    value_optimizer = optim.Adam(value_params, lr=1e-3)  # High LR for visibility
    
    print("\n3. Training value head...")
    for i in range(100):
        _, value = model([test_input])
        target = torch.tensor([10.0])  # Arbitrary target
        
        loss = nn.functional.mse_loss(value.squeeze(), target)
        
        value_optimizer.zero_grad()
        loss.backward()
        value_optimizer.step()
        
        if i % 20 == 0:
            print(f"   Step {i}: value={value.item():.3f}, loss={loss.item():.3f}")
    
    # Check if policy output changed
    with torch.no_grad():
        final_action = model.get_action(test_input)
    
    print(f"\n4. Final action: {final_action}")
    print(f"   Actions identical: {np.allclose(initial_action, final_action, atol=1e-6)}")
    
    # Key insight
    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("Value pre-training ALONE cannot improve performance because:")
    print("- It only updates the value head (critic)")
    print("- The policy head (actor) remains frozen")
    print("- Actions are determined by the policy head")
    print("\nThe +9.3% improvement must have come from the")
    print("FIRST PPO iteration after value pre-training!")
    print("=" * 60)
    
    return {
        'initial_action': initial_action,
        'final_action': final_action,
        'unchanged': np.allclose(initial_action, final_action, atol=1e-6)
    }


if __name__ == "__main__":
    results = test_value_mechanism()
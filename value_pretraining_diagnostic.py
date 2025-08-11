#!/usr/bin/env python3
"""
Value Pre-training Diagnostic - Find out what's going wrong

Focus on:
1. Actual value loss values during training
2. Value function predictions vs returns
3. Gradient flow during pre-training
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from ppo_with_value_pretraining import PPOWithValuePretraining
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


def diagnose_value_pretraining():
    """Diagnose what's happening with value pre-training"""
    print("🔍 VALUE PRE-TRAINING DIAGNOSTIC")
    print("=" * 60)
    
    # 1. Check standard PPO value losses
    print("\n1️⃣ STANDARD PPO - Value Function Behavior")
    standard_trainer = PPOTrainer(
        sl_checkpoint_path="checkpoints/best_model.pt",
        learning_rate=0.00005,
        device='cpu'
    )
    
    # Get initial value predictions
    print("\n   Initial value predictions (should be near 0):")
    initial_state = generate_random_state(12, 400, 300)
    with torch.no_grad():
        # Get a sample observation
        obs = standard_trainer.state_manager.get_observation(initial_state)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        _, value = standard_trainer.policy.model(obs_tensor)
        print(f"   Value prediction: {value.item():.4f}")
    
    # Train one iteration and check losses
    print("\n   Training one iteration...")
    metrics = standard_trainer.train_iteration(initial_state)
    
    print(f"   Reported metrics:")
    print(f"     Policy loss: {metrics.get('policy_loss', 'N/A')}")
    print(f"     Value loss: {metrics.get('value_loss', 'N/A')}")
    print(f"     Total loss: {metrics.get('total_loss', 'N/A')}")
    
    # 2. Check value pre-training implementation
    print("\n\n2️⃣ VALUE PRE-TRAINING - Detailed Analysis")
    pretrained_trainer = PPOWithValuePretraining(
        sl_checkpoint_path="checkpoints/best_model.pt",
        learning_rate=0.00005,
        device='cpu',
        value_pretrain_iterations=3,
        value_pretrain_lr=0.0005,
        value_pretrain_epochs=2
    )
    
    # Manual value pre-training with detailed logging
    print("\n   Manual value pre-training:")
    
    # Get value head parameters
    value_params = []
    for name, param in pretrained_trainer.policy.model.named_parameters():
        if 'value_head' in name:
            value_params.append(param)
            print(f"   Found value param: {name}, shape: {param.shape}")
    
    print(f"\n   Total value parameters: {sum(p.numel() for p in value_params)}")
    
    # Check if gradients flow
    print("\n   Checking gradient flow during pre-training...")
    
    # Collect a small batch
    initial_state = generate_random_state(12, 400, 300)
    experience_buffer = pretrained_trainer.rollout_collector.collect_rollout(
        pretrained_trainer.policy, initial_state, 64  # Small rollout
    )
    
    # Compute returns
    next_value = 0.0
    advantages, returns = experience_buffer.compute_advantages_and_returns(next_value)
    
    # Get states
    states = experience_buffer.get_stacked_observations()
    
    print(f"\n   Data shapes:")
    print(f"     States: {states.shape}")
    print(f"     Returns: {returns.shape}")
    print(f"     Returns range: [{returns.min():.2f}, {returns.max():.2f}]")
    
    # Forward pass
    batch_states = states[:32].to(pretrained_trainer.device)
    batch_returns = returns[:32].to(pretrained_trainer.device)
    
    _, values = pretrained_trainer.policy.model(batch_states)
    values = values.squeeze(-1)
    
    print(f"\n   Value predictions:")
    print(f"     Shape: {values.shape}")
    print(f"     Range: [{values.min().item():.4f}, {values.max().item():.4f}]")
    print(f"     Mean: {values.mean().item():.4f}")
    
    # Compute loss
    value_loss = nn.MSELoss()(values, batch_returns)
    print(f"\n   Value loss: {value_loss.item():.4f}")
    
    # Check gradients
    value_loss.backward()
    
    print(f"\n   Gradient check:")
    for name, param in pretrained_trainer.policy.model.named_parameters():
        if 'value_head' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"     {name}: grad_norm = {grad_norm:.6f}")
    
    # 3. Compare predictions before and after pre-training
    print("\n\n3️⃣ EFFECT OF VALUE PRE-TRAINING")
    
    # Fresh trainer for clean comparison
    fresh_trainer = PPOWithValuePretraining(
        sl_checkpoint_path="checkpoints/best_model.pt",
        learning_rate=0.00005,
        device='cpu',
        value_pretrain_iterations=5,
        value_pretrain_lr=0.0005
    )
    
    # Sample some states
    test_states = states[:5].to(fresh_trainer.device)
    test_returns = returns[:5]
    
    print("\n   Before pre-training:")
    with torch.no_grad():
        _, before_values = fresh_trainer.policy.model(test_states)
        before_values = before_values.squeeze(-1).cpu()
        
    for i in range(3):
        print(f"     State {i}: pred={before_values[i].item():.3f}, actual={test_returns[i].item():.3f}")
    
    # Pre-train
    print("\n   Running value pre-training...")
    value_losses = fresh_trainer.pretrain_value_function()
    print(f"   Loss trajectory: {[f'{loss:.2f}' for loss in value_losses]}")
    
    print("\n   After pre-training:")
    with torch.no_grad():
        _, after_values = fresh_trainer.policy.model(test_states)
        after_values = after_values.squeeze(-1).cpu()
        
    for i in range(3):
        print(f"     State {i}: pred={after_values[i].item():.3f}, actual={test_returns[i].item():.3f}")
    
    # Check if predictions improved
    before_error = ((before_values - test_returns[:5]) ** 2).mean().item()
    after_error = ((after_values - test_returns[:5]) ** 2).mean().item()
    
    print(f"\n   MSE before: {before_error:.4f}")
    print(f"   MSE after: {after_error:.4f}")
    print(f"   Improvement: {(before_error - after_error) / before_error * 100:.1f}%")
    
    # Diagnosis
    print("\n\n📊 DIAGNOSIS:")
    if after_error < before_error * 0.8:
        print("   ✅ Value pre-training IS working!")
        print("   → The poor performance might be due to:")
        print("     - Policy becoming too conservative after value training")
        print("     - Need to tune pre-training hyperparameters")
        print("     - May need fewer pre-training iterations")
    else:
        print("   ❌ Value pre-training NOT effective!")
        print("   → Issues to investigate:")
        print("     - Learning rate too low/high")
        print("     - Not enough pre-training iterations")
        print("     - Gradient clipping too aggressive")
    
    print("\n💡 RECOMMENDATIONS:")
    print("   1. Try different value learning rates (0.001, 0.0001)")
    print("   2. Experiment with pre-training iterations (10-20)")
    print("   3. Check if freezing is working properly")
    print("   4. Monitor value loss carefully during PPO training")


if __name__ == "__main__":
    diagnose_value_pretraining()
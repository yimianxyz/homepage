#!/usr/bin/env python3
"""Test PPO model output shape"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training.ppo_transformer_model import PPOTransformerModel
from simulation.random_state_generator import generate_random_state
from simulation.state_manager import StateManager

# Create model
model = PPOTransformerModel(
    d_model=128,
    n_heads=8,
    n_layers=4,
    ffn_hidden=512,
    max_boids=50,
    dropout=0.1
)

# Load checkpoint
checkpoint = torch.load("checkpoints/best_model.pt", map_location='cpu')
state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
              if not k.startswith('value_head')}
model.load_state_dict(state_dict, strict=False)

# Test with a sample input
initial_state = generate_random_state(12, 400, 300)

# Create state manager to get structured input
state_manager = StateManager()

# Create dummy policy just to init
class DummyPolicy:
    def get_action(self, s):
        return [0.0, 0.0]

state_manager.init(initial_state, DummyPolicy())
state = state_manager.get_state()
structured_input = state_manager._convert_state_to_structured_inputs(state)

print("Testing PPO model output...")
print(f"Structured input keys: {structured_input.keys()}")

# Test model forward
with torch.no_grad():
    action_logits, value = model([structured_input])
    
print(f"\nModel outputs:")
print(f"  action_logits shape: {action_logits.shape}")
print(f"  action_logits: {action_logits}")
print(f"  value shape: {value.shape}")
print(f"  value: {value}")

# Test action conversion
action = torch.tanh(action_logits)
print(f"\nAction (after tanh):")
print(f"  shape: {action.shape}")
print(f"  values: {action}")

# Show how to properly extract
if len(action.shape) == 1 and action.shape[0] == 2:
    print(f"\nProper extraction: [{float(action[0])}, {float(action[1])}]")
elif len(action.shape) == 2 and action.shape[1] == 2:
    print(f"\nProper extraction: [{float(action[0][0])}, {float(action[0][1])}]")
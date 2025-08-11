#!/usr/bin/env python3
"""
Experiment: Hyperparameter Grid Search
Tests combinations of learning rate and clip epsilon for short training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime
import os
import sys
from itertools import product

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training.ppo_transformer_model import PPOTransformerModel
from rl_training.ppo_experience_buffer import PPOExperienceBuffer, PPOExperience
from evaluation import PolicyEvaluator
from simulation.state_manager import StateManager
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


def train_ppo_iteration(model, optimizer, state_manager, clip_epsilon=0.1):
    """Run one PPO training iteration"""
    device = next(model.parameters()).device
    
    # Collect rollout
    initial_state = generate_random_state(12, 400, 300)
    buffer = PPOExperienceBuffer(gamma=0.99, gae_lambda=0.95)
    
    # Policy wrapper
    class PolicyWrapper:
        def __init__(self, model):
            self.model = model
        
        def get_action(self, structured_inputs):
            with torch.no_grad():
                action_logits = self.model(structured_inputs, return_value=False)
                return torch.tanh(action_logits).cpu().numpy().tolist()
    
    state_manager.init(initial_state, PolicyWrapper(model))
    
    # Collect experiences
    total_reward = 0
    for step in range(256):
        state = state_manager.get_state()
        structured_input = state_manager._convert_state_to_structured_inputs(state)
        
        with torch.no_grad():
            action, log_prob, value = model.get_action_and_value(structured_input)
        
        step_result = state_manager.step()
        reward = len(step_result.get('caught_boids', []))
        total_reward += reward
        done = len(step_result['boids_states']) == 0
        
        experience = PPOExperience(
            structured_input=structured_input,
            action=action.detach(),
            log_prob=log_prob.detach(),
            value=value.detach(),
            reward=reward,
            done=done
        )
        buffer.add_experience(experience)
        
        if done:
            break
    
    # PPO update
    batch_data = buffer.get_batch_data()
    if len(batch_data) == 0:
        return 0.0, 0.0, total_reward
    
    # Get batch
    structured_inputs = batch_data['structured_inputs']
    actions = batch_data['actions'].to(device)
    old_log_probs = batch_data['log_probs'].to(device)
    advantages = batch_data['advantages'].to(device)
    returns = batch_data['returns'].to(device)
    
    # PPO epochs
    policy_losses = []
    value_losses = []
    
    for _ in range(2):  # 2 PPO epochs
        # Evaluate actions
        new_log_probs, values, entropy = model.evaluate_actions(structured_inputs, actions)
        values = values.squeeze()
        
        # PPO loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = nn.functional.mse_loss(values, returns)
        entropy_loss = -0.01 * entropy.mean()
        
        total_loss = policy_loss + 0.5 * value_loss + entropy_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
    
    return np.mean(policy_losses), np.mean(value_losses), total_reward


def run_hyperparameter_grid():
    """Test grid of hyperparameters"""
    print("=" * 80)
    print("EXPERIMENT: Hyperparameter Grid Search")
    print("=" * 80)
    
    # Hyperparameter grid
    learning_rates = [1e-5, 3e-5, 1e-4]
    clip_epsilons = [0.01, 0.05, 0.1]
    n_iterations = 5  # Fixed short training
    
    # Create evaluator (3 episodes for speed)
    evaluator = PolicyEvaluator(num_episodes=3, base_seed=25000)
    
    # Evaluate baseline
    print("\n1. Evaluating SL baseline...")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    baseline_result = evaluator.evaluate_policy(sl_policy, "SL_Baseline")
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f}")
    
    # Test grid
    results = []
    state_manager = StateManager()
    
    for lr, clip_eps in product(learning_rates, clip_epsilons):
        print(f"\n{'='*60}")
        print(f"Testing LR={lr}, Clip={clip_eps}")
        print(f"{'='*60}")
        
        # Load fresh model
        model = PPOTransformerModel.from_sl_checkpoint("checkpoints/best_model.pt")
        device = torch.device('cpu')
        model = model.to(device)
        
        # Value pre-training (simplified - 10 iterations)
        print("Value pre-training...")
        value_params = [p for n, p in model.named_parameters() if 'value_head' in n]
        value_optimizer = optim.Adam(value_params, lr=3e-4)
        
        for i in range(10):
            # Simple value training
            dummy_input = {
                'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
                'predator': {'velX': 0.0, 'velY': 0.0},
                'boids': [{'relX': 0.1, 'relY': 0.1, 'velX': 0.1, 'velY': 0.1}]
            }
            _, value = model([dummy_input])
            target = torch.randn(1) * 0.1
            loss = nn.functional.mse_loss(value.squeeze(), target)
            
            value_optimizer.zero_grad()
            loss.backward()
            value_optimizer.step()
        
        # PPO training
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        print(f"Running {n_iterations} PPO iterations...")
        for i in range(n_iterations):
            p_loss, v_loss, reward = train_ppo_iteration(model, optimizer, state_manager, clip_eps)
            print(f"  Iter {i+1}: P_loss={p_loss:.4f}, V_loss={v_loss:.4f}, Reward={reward}")
        
        # Evaluate
        print("Evaluating...")
        class PolicyWrapper:
            def __init__(self, model):
                self.model = model
            
            def get_action(self, structured_inputs):
                with torch.no_grad():
                    action_logits = self.model(structured_inputs, return_value=False)
                    return torch.tanh(action_logits).cpu().numpy().tolist()
        
        eval_policy = PolicyWrapper(model)
        result = evaluator.evaluate_policy(eval_policy, f"Grid_LR{lr}_Clip{clip_eps}")
        
        improvement = (result.overall_catch_rate - baseline_result.overall_catch_rate) / baseline_result.overall_catch_rate * 100
        
        results.append({
            'learning_rate': lr,
            'clip_epsilon': clip_eps,
            'performance': result.overall_catch_rate,
            'improvement': improvement
        })
        
        print(f"Performance: {result.overall_catch_rate:.4f} ({improvement:+.1f}%)")
    
    # Summary
    print("\n" + "=" * 80)
    print("GRID SEARCH RESULTS:")
    print("=" * 80)
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f}")
    print("\n{:<10} {:<10} {:<12} {:<10}".format("LR", "Clip", "Performance", "Improvement"))
    print("-" * 50)
    
    for r in results:
        print("{:<10.0e} {:<10.2f} {:<12.4f} {:<+10.1f}%".format(
            r['learning_rate'], r['clip_epsilon'], r['performance'], r['improvement']
        ))
    
    # Find best
    if results:
        best = max(results, key=lambda x: x['performance'])
        print(f"\nBest: LR={best['learning_rate']}, Clip={best['clip_epsilon']}")
        print(f"      Performance={best['performance']:.4f} ({best['improvement']:+.1f}%)")
    
    # Save results
    with open('experiment_hyperparameter_grid_results.json', 'w') as f:
        json.dump({
            'experiment': 'hyperparameter_grid',
            'timestamp': datetime.now().isoformat(),
            'baseline': baseline_result.overall_catch_rate,
            'results': results
        }, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_hyperparameter_grid()
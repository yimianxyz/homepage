#!/usr/bin/env python3
"""
Experiment: Value Pre-training Only
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
from rl_training.ppo_experience_buffer import PPOExperienceBuffer
from evaluation import PolicyEvaluator
from simulation.state_manager import StateManager
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


def run_value_pretraining_experiment():
    """Test value pre-training only without PPO"""
    print("=" * 80)
    print("EXPERIMENT: Value Pre-training Only")
    print("Hypothesis: Value pre-training alone can improve performance")
    print("=" * 80)
    
    # Load model
    model = PPOTransformerModel.from_sl_checkpoint("checkpoints/best_model.pt")
    device = torch.device('cpu')
    model = model.to(device)
    
    # Create evaluator
    evaluator = PolicyEvaluator(num_episodes=15, base_seed=20000)
    
    # Evaluate baseline
    print("\n1. Evaluating SL baseline...")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    baseline_result = evaluator.evaluate_policy(sl_policy, "SL_Baseline")
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f} ± {baseline_result.std_error:.4f}")
    
    # Test different amounts of value pre-training
    value_iterations = [10, 20, 30, 50]
    results = []
    
    for n_iter in value_iterations:
        print(f"\n2. Testing {n_iter} value pre-training iterations...")
        
        # Reset model
        model = PPOTransformerModel.from_sl_checkpoint("checkpoints/best_model.pt").to(device)
        
        # Freeze policy parameters
        value_params = []
        for name, param in model.named_parameters():
            if 'value_head' in name:
                value_params.append(param)
            else:
                param.requires_grad = False
        
        # Value optimizer
        value_optimizer = optim.Adam(value_params, lr=3e-4)
        
        # Pre-training loop
        value_losses = []
        state_manager = StateManager()
        
        for iter_idx in range(n_iter):
            # Collect experience
            initial_state = generate_random_state(12, 400, 300)
            buffer = PPOExperienceBuffer(gamma=0.99, gae_lambda=0.95)
            
            # Create policy wrapper
            class PolicyWrapper:
                def __init__(self, model):
                    self.model = model
                
                def get_action(self, structured_inputs):
                    with torch.no_grad():
                        action_logits = self.model(structured_inputs, return_value=False)
                        return torch.tanh(action_logits).cpu().numpy().tolist()
            
            policy = PolicyWrapper(model)
            state_manager.init(initial_state, policy)
            
            # Collect short rollout
            for step in range(256):
                state = state_manager.get_state()
                structured_input = state_manager._convert_state_to_structured_inputs(state)
                
                with torch.no_grad():
                    action, _, value = model.get_action_and_value(structured_input, deterministic=True)
                
                step_result = state_manager.step()
                reward = len(step_result.get('caught_boids', []))
                done = len(step_result['boids_states']) == 0 or step >= 255
                
                from rl_training.ppo_experience_buffer import PPOExperience
                experience = PPOExperience(
                    structured_input=structured_input,
                    action=action.detach(),
                    log_prob=torch.tensor(0.0),  # Not used
                    value=value.detach(),
                    reward=reward,
                    done=done
                )
                buffer.add_experience(experience)
                
                if done:
                    break
            
            # Train value function
            batch_data = buffer.get_batch_data()
            if len(batch_data) > 0:
                inputs = batch_data['structured_inputs']
                returns = batch_data['returns'].to(device)
                
                _, values = model(inputs)
                values = values.squeeze()
                
                value_loss = nn.functional.mse_loss(values, returns)
                
                value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(value_params, 0.5)
                value_optimizer.step()
                
                value_losses.append(value_loss.item())
        
        avg_loss = np.mean(value_losses[-10:]) if value_losses else 0
        print(f"Final value loss: {avg_loss:.4f}")
        
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        
        # Evaluate
        print(f"3. Evaluating after {n_iter} iterations...")
        class EvalPolicy:
            def __init__(self, model):
                self.model = model
            
            def get_action(self, structured_inputs):
                with torch.no_grad():
                    action_logits = self.model(structured_inputs, return_value=False)
                    return torch.tanh(action_logits).cpu().numpy().tolist()
        
        eval_policy = EvalPolicy(model)
        result = evaluator.evaluate_policy(eval_policy, f"ValueOnly_{n_iter}")
        
        improvement = (result.overall_catch_rate - baseline_result.overall_catch_rate) / baseline_result.overall_catch_rate * 100
        print(f"Performance: {result.overall_catch_rate:.4f} ({improvement:+.1f}%)")
        
        results.append({
            'iterations': n_iter,
            'performance': result.overall_catch_rate,
            'improvement': improvement,
            'final_value_loss': avg_loss,
            'baseline': baseline_result.overall_catch_rate
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY:")
    print("=" * 80)
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f}")
    print("\nValue Pre-training Results:")
    for r in results:
        print(f"  {r['iterations']:2d} iterations: {r['performance']:.4f} ({r['improvement']:+.1f}%)")
    
    # Save results
    with open('experiment_value_only_results.json', 'w') as f:
        json.dump({
            'experiment': 'value_pretraining_only',
            'timestamp': datetime.now().isoformat(),
            'baseline': baseline_result.overall_catch_rate,
            'results': results
        }, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_value_pretraining_experiment()
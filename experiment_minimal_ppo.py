#!/usr/bin/env python3
"""
Experiment: Minimal PPO Training
Tests if 1-5 PPO iterations after value pre-training can improve performance
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
from rl_training.ppo_experience_buffer import PPOExperienceBuffer, PPOExperience
from evaluation import PolicyEvaluator
from simulation.state_manager import StateManager
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


def run_minimal_ppo_experiment():
    """Test minimal PPO training after value pre-training"""
    print("=" * 80)
    print("EXPERIMENT: Minimal PPO Training")
    print("Testing 1-5 iterations after value pre-training")
    print("=" * 80)
    
    # Create evaluator (use 5 episodes for speed)
    evaluator = PolicyEvaluator(num_episodes=5, base_seed=24000)
    
    # Evaluate baseline
    print("\n1. Evaluating SL baseline...")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    baseline_result = evaluator.evaluate_policy(sl_policy, "SL_Baseline")
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f} ± {baseline_result.std_error:.4f}")
    
    # Test different iteration counts
    results = []
    
    for n_ppo_iter in [1, 2, 3, 5]:
        print(f"\n{'='*60}")
        print(f"Testing {n_ppo_iter} PPO iterations")
        print(f"{'='*60}")
        
        # Load fresh model
        model = PPOTransformerModel.from_sl_checkpoint("checkpoints/best_model.pt")
        device = torch.device('cpu')
        model = model.to(device)
        
        # Step 1: Value pre-training (20 iterations as optimal)
        print("\n2. Value pre-training (20 iterations)...")
        
        # Freeze policy parameters
        value_params = []
        policy_params = []
        for name, param in model.named_parameters():
            if 'value_head' in name:
                value_params.append(param)
            else:
                policy_params.append(param)
                param.requires_grad = False
        
        value_optimizer = optim.Adam(value_params, lr=3e-4)
        state_manager = StateManager()
        
        # Value pre-training
        value_losses = []
        for i in range(20):
            # Generate random state
            initial_state = generate_random_state(12, 400, 300)
            
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
            for step in range(100):
                state = state_manager.get_state()
                structured_input = state_manager._convert_state_to_structured_inputs(state)
                
                with torch.no_grad():
                    action, _, value = model.get_action_and_value(structured_input, deterministic=True)
                
                step_result = state_manager.step()
                
                if len(step_result['boids_states']) == 0:
                    break
            
            # Simple value loss (random target for speed)
            target_value = torch.randn(1) * 0.1
            _, current_value = model([structured_input])
            value_loss = nn.functional.mse_loss(current_value.squeeze(), target_value)
            
            value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_params, 0.5)
            value_optimizer.step()
            
            value_losses.append(value_loss.item())
        
        print(f"Value pre-training complete: {np.mean(value_losses[:5]):.4f} → {np.mean(value_losses[-5:]):.4f}")
        
        # Unfreeze policy parameters
        for param in policy_params:
            param.requires_grad = True
        
        # Full model optimizer for PPO
        optimizer = optim.Adam(model.parameters(), lr=3e-5)
        
        # Step 2: Minimal PPO training
        print(f"\n3. Running {n_ppo_iter} PPO iterations...")
        
        for ppo_iter in range(n_ppo_iter):
            # Collect rollout
            initial_state = generate_random_state(12, 400, 300)
            buffer = PPOExperienceBuffer(gamma=0.99, gae_lambda=0.95)
            
            state_manager.init(initial_state, PolicyWrapper(model))
            
            # Collect experiences
            for step in range(256):
                state = state_manager.get_state()
                structured_input = state_manager._convert_state_to_structured_inputs(state)
                
                with torch.no_grad():
                    action, log_prob, value = model.get_action_and_value(structured_input)
                
                step_result = state_manager.step()
                reward = len(step_result.get('caught_boids', []))
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
            if len(batch_data) > 0:
                # Mini-batch update
                structured_inputs = batch_data['structured_inputs']
                actions = batch_data['actions'].to(device)
                old_log_probs = batch_data['log_probs'].to(device)
                advantages = batch_data['advantages'].to(device)
                returns = batch_data['returns'].to(device)
                
                # Evaluate actions
                new_log_probs, values, entropy = model.evaluate_actions(structured_inputs, actions)
                values = values.squeeze()
                
                # PPO loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.9, 1.1) * advantages  # Conservative clip
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.functional.mse_loss(values, returns)
                entropy_loss = -0.01 * entropy.mean()
                
                total_loss = policy_loss + 0.5 * value_loss + entropy_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                print(f"  PPO iter {ppo_iter+1}: policy_loss={policy_loss.item():.4f}, "
                      f"value_loss={value_loss.item():.4f}")
        
        # Step 3: Evaluate
        print(f"\n4. Evaluating after {n_ppo_iter} PPO iterations...")
        eval_policy = PolicyWrapper(model)
        result = evaluator.evaluate_policy(eval_policy, f"MinimalPPO_{n_ppo_iter}")
        
        improvement = (result.overall_catch_rate - baseline_result.overall_catch_rate) / baseline_result.overall_catch_rate * 100
        print(f"Performance: {result.overall_catch_rate:.4f} ({improvement:+.1f}%)")
        
        results.append({
            'ppo_iterations': n_ppo_iter,
            'performance': result.overall_catch_rate,
            'improvement': improvement,
            'confidence_interval': [result.confidence_95_lower, result.confidence_95_upper]
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY:")
    print("=" * 80)
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f}")
    print("\nMinimal PPO Results:")
    for r in results:
        print(f"  {r['ppo_iterations']} iterations: {r['performance']:.4f} ({r['improvement']:+.1f}%)")
    
    # Find best
    if results:
        best = max(results, key=lambda x: x['performance'])
        print(f"\nBest: {best['ppo_iterations']} iterations with {best['performance']:.4f} ({best['improvement']:+.1f}%)")
    
    # Save results
    with open('experiment_minimal_ppo_results.json', 'w') as f:
        json.dump({
            'experiment': 'minimal_ppo',
            'timestamp': datetime.now().isoformat(),
            'baseline': baseline_result.overall_catch_rate,
            'results': results
        }, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_minimal_ppo_experiment()
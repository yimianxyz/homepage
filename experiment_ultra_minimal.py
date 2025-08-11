#!/usr/bin/env python3
"""
Ultra-Minimal Experiment: Test 1-2 PPO iterations only
Based on finding that performance peaks early then degrades
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


def run_ultra_minimal():
    """Test just 1-2 PPO iterations with minimal value pre-training"""
    print("=" * 80)
    print("ULTRA-MINIMAL EXPERIMENT")
    print("Testing if 1-2 PPO iterations can improve performance")
    print("=" * 80)
    
    # Create evaluator (5 episodes for speed)
    evaluator = PolicyEvaluator(num_episodes=5, base_seed=27000)
    
    # Evaluate baseline
    print("\n1. Evaluating SL baseline...")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    baseline_result = evaluator.evaluate_policy(sl_policy, "SL_Baseline")
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f}")
    
    # Best configuration from production
    config = {
        'learning_rate': 3e-5,
        'clip_epsilon': 0.01,  # Very conservative
        'value_pretrain_iterations': 10,  # Reduced for speed
        'episode_length': 2500,  # Match evaluation
        'rollout_steps': 256,  # Shorter rollouts
    }
    
    results = []
    
    for n_ppo in [1, 2]:
        print(f"\n{'='*60}")
        print(f"Testing {n_ppo} PPO iterations")
        print(f"{'='*60}")
        
        # Load fresh model
        model = PPOTransformerModel.from_sl_checkpoint("checkpoints/best_model.pt")
        device = torch.device('cpu')
        model = model.to(device)
        
        # Quick value pre-training
        print("\n2. Quick value pre-training (10 iterations)...")
        
        # Only train value head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.value_head.parameters():
            param.requires_grad = True
        
        value_optimizer = optim.Adam(model.value_head.parameters(), lr=3e-4)
        
        # Simple value training loop
        for i in range(config['value_pretrain_iterations']):
            # Create random batch
            batch_size = 16
            dummy_inputs = []
            for _ in range(batch_size):
                dummy_inputs.append({
                    'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
                    'predator': {'velX': np.random.randn() * 0.1, 'velY': np.random.randn() * 0.1},
                    'boids': [
                        {'relX': np.random.randn() * 0.3, 'relY': np.random.randn() * 0.3, 
                         'velX': np.random.randn() * 0.1, 'velY': np.random.randn() * 0.1}
                        for _ in range(np.random.randint(1, 5))
                    ]
                })
            
            _, values = model(dummy_inputs)
            targets = torch.randn(batch_size) * 0.5 + 1.0  # Random targets around 1.0
            
            loss = nn.functional.mse_loss(values.squeeze(), targets)
            
            value_optimizer.zero_grad()
            loss.backward()
            value_optimizer.step()
        
        print(f"Value pre-training complete")
        
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        
        # Full optimizer
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # PPO training
        print(f"\n3. Running {n_ppo} PPO iterations...")
        state_manager = StateManager()
        
        for iter_idx in range(n_ppo):
            # Collect single rollout
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
            
            policy = PolicyWrapper(model)
            state_manager.init(initial_state, policy)
            
            # Collect experiences
            episode_reward = 0
            for step in range(config['rollout_steps']):
                state = state_manager.get_state()
                structured_input = state_manager._convert_state_to_structured_inputs(state)
                
                with torch.no_grad():
                    action, log_prob, value = model.get_action_and_value(structured_input)
                
                step_result = state_manager.step()
                reward = len(step_result.get('caught_boids', []))
                episode_reward += reward
                done = len(step_result['boids_states']) == 0 or step >= config['episode_length'] - 1
                
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
            
            # Single PPO update
            batch_data = buffer.get_batch_data()
            if len(batch_data) > 0:
                # Get all data
                structured_inputs = batch_data['structured_inputs']
                actions = batch_data['actions'].to(device)
                old_log_probs = batch_data['log_probs'].to(device)
                advantages = batch_data['advantages'].to(device)
                returns = batch_data['returns'].to(device)
                
                # Single epoch update
                new_log_probs, values, entropy = model.evaluate_actions(structured_inputs, actions)
                values = values.squeeze()
                
                # Conservative PPO loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - config['clip_epsilon'], 1 + config['clip_epsilon']) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.functional.mse_loss(values, returns)
                entropy_loss = -0.01 * entropy.mean()
                
                total_loss = policy_loss + 0.5 * value_loss + entropy_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                print(f"  Iter {iter_idx+1}: P_loss={policy_loss.item():.4f}, "
                      f"V_loss={value_loss.item():.4f}, Episode_reward={episode_reward}")
        
        # Evaluate
        print(f"\n4. Evaluating after {n_ppo} iterations...")
        eval_policy = PolicyWrapper(model)
        result = evaluator.evaluate_policy(eval_policy, f"UltraMinimal_PPO{n_ppo}")
        
        improvement = (result.overall_catch_rate - baseline_result.overall_catch_rate) / baseline_result.overall_catch_rate * 100
        print(f"Performance: {result.overall_catch_rate:.4f} ({improvement:+.1f}%)")
        
        results.append({
            'ppo_iterations': n_ppo,
            'performance': result.overall_catch_rate,
            'improvement': improvement,
            'confidence_interval': [result.confidence_95_lower, result.confidence_95_upper]
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f}")
    for r in results:
        print(f"{r['ppo_iterations']} PPO iter: {r['performance']:.4f} ({r['improvement']:+.1f}%)")
    
    # Save
    with open('experiment_ultra_minimal_results.json', 'w') as f:
        json.dump({
            'experiment': 'ultra_minimal',
            'timestamp': datetime.now().isoformat(),
            'baseline': baseline_result.overall_catch_rate,
            'config': config,
            'results': results
        }, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_ultra_minimal()
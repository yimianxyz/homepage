#!/usr/bin/env python3
"""
Test the best checkpoint from production training (iteration 10)
"""

import torch
import json
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training.ppo_transformer_model import PPOTransformerModel
from evaluation import PolicyEvaluator
from policy.transformer.transformer_policy import TransformerPolicy


def test_best_checkpoint():
    """Test checkpoint from iteration 10 where performance peaked"""
    print("=" * 80)
    print("TESTING BEST CHECKPOINT")
    print("Using checkpoint from iteration 10 of production training")
    print("=" * 80)
    
    # Create evaluator (use 10 episodes for better confidence)
    evaluator = PolicyEvaluator(num_episodes=10, base_seed=28000)
    
    # Evaluate baseline
    print("\n1. Evaluating SL baseline...")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    baseline_result = evaluator.evaluate_policy(sl_policy, "SL_Baseline")
    print(f"Baseline: {baseline_result.overall_catch_rate:.4f} ± {baseline_result.std_error:.4f}")
    print(f"95% CI: [{baseline_result.confidence_95_lower:.4f}, {baseline_result.confidence_95_upper:.4f}]")
    
    # Load checkpoint from iteration 10
    checkpoint_path = "checkpoints/ppo_production_20250806_095300/checkpoint_iter10.pt"
    
    if os.path.exists(checkpoint_path):
        print(f"\n2. Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint (weights_only=False for compatibility)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Create model and load weights
        model = PPOTransformerModel.from_sl_checkpoint("checkpoints/best_model.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
        print(f"Training time: {checkpoint['training_time']/60:.1f} minutes")
        
        # Create policy wrapper
        class PolicyWrapper:
            def __init__(self, model):
                self.model = model
            
            def get_action(self, structured_inputs):
                with torch.no_grad():
                    action_logits = self.model(structured_inputs, return_value=False)
                    return torch.tanh(action_logits).cpu().numpy().tolist()
        
        # Evaluate
        print("\n3. Evaluating checkpoint...")
        ppo_policy = PolicyWrapper(model)
        ppo_result = evaluator.evaluate_policy(ppo_policy, "PPO_Iter10_Checkpoint")
        
        print(f"Performance: {ppo_result.overall_catch_rate:.4f} ± {ppo_result.std_error:.4f}")
        print(f"95% CI: [{ppo_result.confidence_95_lower:.4f}, {ppo_result.confidence_95_upper:.4f}]")
        
        # Calculate improvement
        improvement = (ppo_result.overall_catch_rate - baseline_result.overall_catch_rate) / baseline_result.overall_catch_rate * 100
        
        # Check statistical significance
        is_significant = ppo_result.confidence_95_lower > baseline_result.confidence_95_upper
        
        print(f"\nImprovement: {improvement:+.1f}%")
        print(f"Statistically significant: {'YES ✓' if is_significant else 'NO ✗'}")
        
        # Also test the "best" checkpoint
        best_checkpoint_path = "checkpoints/ppo_production_20250806_095300/best_checkpoint.pt"
        if os.path.exists(best_checkpoint_path):
            print("\n4. Testing 'best' checkpoint...")
            best_checkpoint = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(best_checkpoint['model_state_dict'])
            
            best_result = evaluator.evaluate_policy(PolicyWrapper(model), "PPO_Best_Checkpoint")
            best_improvement = (best_result.overall_catch_rate - baseline_result.overall_catch_rate) / baseline_result.overall_catch_rate * 100
            
            print(f"Best checkpoint performance: {best_result.overall_catch_rate:.4f} ({best_improvement:+.1f}%)")
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY:")
        print("=" * 80)
        print(f"SL Baseline: {baseline_result.overall_catch_rate:.4f}")
        print(f"PPO Iter 10: {ppo_result.overall_catch_rate:.4f} ({improvement:+.1f}%)")
        print("Conclusion: Even the best PPO checkpoint struggles to improve on SL baseline")
        
        # Save results
        results = {
            'experiment': 'best_checkpoint_test',
            'timestamp': datetime.now().isoformat(),
            'baseline': {
                'performance': baseline_result.overall_catch_rate,
                'std_error': baseline_result.std_error,
                'confidence_interval': [baseline_result.confidence_95_lower, baseline_result.confidence_95_upper]
            },
            'ppo_iter10': {
                'performance': ppo_result.overall_catch_rate,
                'std_error': ppo_result.std_error,
                'confidence_interval': [ppo_result.confidence_95_lower, ppo_result.confidence_95_upper],
                'improvement': improvement,
                'is_significant': is_significant
            }
        }
        
        with open('test_best_checkpoint_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    else:
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        print("Please ensure production training completed successfully")
        return None


if __name__ == "__main__":
    results = test_best_checkpoint()
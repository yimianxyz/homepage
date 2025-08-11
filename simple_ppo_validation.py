#!/usr/bin/env python3
"""
Simple PPO Validation - Prove PPO works by comparing random vs trained performance

USER'S BRILLIANT INSIGHT: Start with random initialization to prove PPO works,
then apply to SL improvement with confidence.

SIMPLIFIED APPROACH:
1. Create PPO trainer starting from SL model
2. Randomly reinitialize the PPO model  
3. Train from random initialization
4. Compare final vs initial random performance
5. Prove PPO can learn significantly

This validates our PPO implementation before tackling SL improvement.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


class SimplePPOValidator:
    """Simple validator to prove PPO works from random initialization"""
    
    def __init__(self):
        self.evaluator = PolicyEvaluator()
        
        print("🧪 SIMPLE PPO VALIDATION")
        print("=" * 60)
        print("OBJECTIVE: Prove PPO implementation works")
        print("METHOD: Random init -> PPO training -> Performance comparison")
        print("=" * 60)
    
    def randomize_ppo_weights(self, trainer: PPOTrainer):
        """Randomly reinitialize all weights in PPO model"""
        print("🎲 Randomly reinitializing PPO model weights...")
        
        for name, param in trainer.policy.model.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
            else:
                nn.init.uniform_(param, -0.1, 0.1)
        
        print("✅ All PPO model weights randomly reinitialized")
    
    def run_validation_experiment(self) -> Dict[str, Any]:
        """Run simple validation: random init -> PPO training -> comparison"""
        
        print(f"\n🚀 Starting PPO validation experiment...")
        
        # Create PPO trainer (starts with SL weights)
        trainer = PPOTrainer(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=0.001,  # Higher LR for learning from scratch
            clip_epsilon=0.2,
            ppo_epochs=4,  # More epochs for better learning
            rollout_steps=512,
            max_episode_steps=2500,
            gamma=0.99,
            gae_lambda=0.95,
            device='cpu'
        )
        
        # Randomly reinitialize to start from scratch
        self.randomize_ppo_weights(trainer)
        
        # Evaluate random baseline performance
        print(f"\n📊 Step 1: Evaluate random initialized performance...")
        random_result = self.evaluator.evaluate_policy(trainer.policy, "Random_Baseline")
        random_performance = random_result.overall_catch_rate
        
        print(f"✅ Random baseline: {random_performance:.4f}")
        print(f"   (This should be very poor - random policy)")
        
        # Train with PPO
        print(f"\n🚀 Step 2: Train with PPO...")
        training_iterations = 30  # Substantial training
        performance_curve = []
        
        start_time = time.time()
        
        for iteration in range(1, training_iterations + 1):
            print(f"   Iteration {iteration}/{training_iterations}")
            
            # Training
            initial_state = generate_random_state(12, 400, 300)
            trainer.train_iteration(initial_state)
            
            # Periodic evaluation
            if iteration % 5 == 0:
                print(f"     🎯 Evaluation at iteration {iteration}...")
                result = self.evaluator.evaluate_policy(trainer.policy, f"PPO_Iter{iteration}")
                performance = result.overall_catch_rate
                improvement = ((performance - random_performance) / (random_performance + 1e-8)) * 100
                
                performance_curve.append({
                    'iteration': iteration,
                    'performance': performance,
                    'improvement_vs_random': improvement
                })
                
                status = "✅ LEARNING!" if performance > random_performance * 1.5 else "📈 Progress"
                print(f"     Result: {performance:.4f} ({improvement:+.1f}% vs random) {status}")
                
                # Early success detection
                if performance > random_performance * 3.0:
                    print(f"     🎉 MAJOR SUCCESS: 3x improvement achieved!")
        
        training_time = time.time() - start_time
        
        # Final evaluation
        print(f"\n🎯 Step 3: Final evaluation...")
        final_result = self.evaluator.evaluate_policy(trainer.policy, "PPO_Final")
        final_performance = final_result.overall_catch_rate
        final_improvement = ((final_performance - random_performance) / (random_performance + 1e-8)) * 100
        
        # Analysis
        learning_factor = final_performance / (random_performance + 1e-8)
        
        print(f"\n{'='*60}")
        print(f"🧪 PPO VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Random Baseline:    {random_performance:.4f}")
        print(f"PPO Final:          {final_performance:.4f}")
        print(f"Improvement:        {final_improvement:+.1f}%")
        print(f"Learning Factor:    {learning_factor:.1f}x")
        print(f"Training Time:      {training_time/60:.1f} minutes")
        
        # Success assessment
        if learning_factor >= 3.0:
            success_level = "EXCELLENT"
            confidence = "HIGH"
            success_message = "PPO shows excellent learning capability!"
        elif learning_factor >= 2.0:
            success_level = "GOOD"
            confidence = "HIGH"
            success_message = "PPO shows good learning capability!"
        elif learning_factor >= 1.5:
            success_level = "MODERATE"
            confidence = "MEDIUM"
            success_message = "PPO shows moderate learning capability."
        else:
            success_level = "LIMITED"
            confidence = "LOW"
            success_message = "PPO learning is limited - needs investigation."
        
        print(f"")
        print(f"🔍 ASSESSMENT:")
        print(f"   Success Level: {success_level}")
        print(f"   Confidence:    {confidence}")
        print(f"   Conclusion:    {success_message}")
        
        # Recommendations
        if learning_factor >= 2.0:
            print(f"\n🎉 VALIDATION SUCCESS!")
            print(f"   ✅ PPO implementation is working correctly")
            print(f"   ✅ PPO can learn this task from scratch")
            print(f"   ✅ Training dynamics are healthy")
            print(f"   🚀 Ready to apply PPO to SL baseline improvement!")
            
            print(f"\n🎯 RECOMMENDED NEXT STEPS:")
            print(f"   1. Use similar hyperparameters for SL improvement")
            print(f"   2. Expect slower progress (starting from better baseline)")
            print(f"   3. Focus on small, consistent improvements")
            print(f"   4. Monitor for overfitting (since SL is already good)")
            
        else:
            print(f"\n🔧 VALIDATION INCOMPLETE")
            print(f"   ⚠️  PPO learning is limited")
            print(f"   🔍 Investigate: hyperparameters, implementation, task complexity")
            print(f"   📊 Consider: different algorithms, reward engineering")
        
        # Save results
        results = {
            'random_baseline': random_performance,
            'final_performance': final_performance,
            'improvement_percent': final_improvement,
            'learning_factor': learning_factor,
            'success_level': success_level,
            'confidence': confidence,
            'performance_curve': performance_curve,
            'training_time_minutes': training_time / 60,
            'training_iterations': training_iterations,
            'hyperparameters': {
                'learning_rate': 0.001,
                'clip_epsilon': 0.2,
                'ppo_epochs': 4,
                'gamma': 0.99,
                'gae_lambda': 0.95
            }
        }
        
        with open('simple_ppo_validation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved: simple_ppo_validation_results.json")
        
        return results


def main():
    """Run simple PPO validation"""
    print("🧪 SIMPLE PPO VALIDATION EXPERIMENT")
    print("=" * 60)
    print("USER INSIGHT: Prove PPO works before improving SL baseline")
    print("APPROACH: Random init -> PPO training -> Validate learning")
    print("=" * 60)
    
    validator = SimplePPOValidator()
    results = validator.run_validation_experiment()
    
    if results['learning_factor'] >= 2.0:
        print(f"\n🎉 SUCCESS: PPO validation complete!")
        print(f"   PPO can improve performance by {results['learning_factor']:.1f}x")
        print(f"   Ready to tackle SL baseline improvement with confidence!")
    else:
        print(f"\n🔧 INVESTIGATION NEEDED")
        print(f"   PPO learning factor: {results['learning_factor']:.1f}x (target: 2.0x+)")
        print(f"   Review implementation before SL improvement attempts")
    
    return results


if __name__ == "__main__":
    main()
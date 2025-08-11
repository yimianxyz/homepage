#!/usr/bin/env python3
"""
PPO From Scratch Validation - Prove PPO Works with Random Initialization

BRILLIANT USER INSIGHT: Instead of trying to improve SL model, first prove PPO works
by training a randomly initialized model from scratch.

SCIENTIFIC APPROACH:
1. Random Init Baseline: Evaluate random initialized model (should be terrible)
2. PPO Training: Train random model with PPO for many iterations  
3. PPO vs Random: Compare trained model vs random baseline
4. Learning Analysis: Analyze PPO learning dynamics from scratch
5. Transfer Knowledge: Apply successful config to SL improvement

This proves:
- PPO implementation works
- PPO can learn this task
- Our hyperparameters are reasonable
- Training dynamics are healthy

Then we can confidently tackle SL baseline improvement!
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


class RandomInitializedTransformerPolicy:
    """Transformer policy with random initialization (same architecture as SL model)"""
    
    def __init__(self, device='cpu'):
        """Create randomly initialized transformer with same architecture as SL model"""
        self.device = device
        
        # Load SL model to get the exact architecture
        sl_policy = TransformerPolicy("checkpoints/best_model.pt")
        self.model = sl_policy.model
        
        print("🎲 Creating random initialized transformer...")
        print(f"   Architecture: {self.model}")
        
        # Reinitialize all parameters randomly
        self._reinitialize_parameters()
        
        print("✅ Random initialization complete")
        print(f"   Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _reinitialize_parameters(self):
        """Reinitialize all model parameters randomly"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    # Use Xavier/Glorot initialization for weights
                    nn.init.xavier_uniform_(param)
                else:
                    # Small random initialization for 1D weights
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                # Zero initialization for biases
                nn.init.zeros_(param)
            else:
                # Small random for other parameters
                nn.init.uniform_(param, -0.1, 0.1)
        
        print("   ✓ All parameters randomly reinitialized")
    
    def get_action(self, state: Dict) -> np.ndarray:
        """Get action from randomly initialized model (should be random/bad initially)"""
        return self.model.get_action(state)
    
    def save_checkpoint(self, path: str):
        """Save random model checkpoint"""
        torch.save(self.model.state_dict(), path)
        print(f"   Random model saved: {path}")


class PPOFromScratchValidator:
    """Validate PPO by training from random initialization"""
    
    def __init__(self):
        self.evaluator = PolicyEvaluator()
        self.random_baseline_performance = None
        self.ppo_training_curve = []
        
        print("🧪 PPO FROM SCRATCH VALIDATION")
        print("=" * 70)
        print("OBJECTIVE: Prove PPO works by training random model from scratch")
        print("HYPOTHESIS: PPO should significantly improve random baseline")
        print("APPROACH: Random init -> PPO training -> Performance comparison")
        print("=" * 70)
    
    def _reinitialize_ppo_model(self, trainer):
        """Reinitialize PPO model parameters randomly"""
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
        print("   ✓ PPO model randomly reinitialized")
    
    def establish_random_baseline(self) -> float:
        """Establish how bad a random initialized model performs"""
        print("\n📊 Step 1: Establish Random Baseline...")
        
        # Create random initialized model
        random_policy = RandomInitializedTransformerPolicy()
        
        # Save random model for reproducibility
        random_policy.save_checkpoint("checkpoints/random_init_baseline.pt")
        
        # Evaluate random model performance (should be terrible)
        print("   Evaluating random initialized model...")
        result = self.evaluator.evaluate_policy(random_policy, "Random_Baseline")
        
        self.random_baseline_performance = result.overall_catch_rate
        
        print(f"✅ Random baseline established: {self.random_baseline_performance:.4f}")
        print(f"   (This should be very low - random policy)")
        
        return self.random_baseline_performance
    
    def train_ppo_from_scratch(self, learning_rate: float = 0.001, 
                             max_iterations: int = 50) -> Dict[str, Any]:
        """Train PPO from random initialization with systematic tracking"""
        
        print(f"\n🚀 Step 2: Train PPO from Random Initialization...")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Max iterations: {max_iterations}")
        print(f"   Target: Significantly beat random baseline ({self.random_baseline_performance:.4f})")
        
        # Create PPO trainer - we'll use the SL checkpoint path but override with random weights
        trainer = PPOTrainer(
            sl_checkpoint_path="checkpoints/best_model.pt",  # Use SL for architecture
            learning_rate=learning_rate,
            clip_epsilon=0.2,
            ppo_epochs=4,  # More epochs since we're learning from scratch
            rollout_steps=512,
            max_episode_steps=2500,
            gamma=0.99,
            gae_lambda=0.95,
            device='cpu'
        )
        
        # Override with random initialization
        print("   🎲 Overriding with random initialization...")
        self._reinitialize_ppo_model(trainer)
        
        # Save the random starting point
        torch.save(trainer.policy.model.state_dict(), "checkpoints/ppo_random_start.pt")
        
        print("✅ PPO trainer initialized with random baseline")
        
        # Training loop with detailed tracking
        training_results = []
        evaluation_curve = []
        
        start_time = time.time()
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n📊 Iteration {iteration}/{max_iterations}")
            
            # Training iteration
            iteration_start = time.time()
            initial_state = generate_random_state(12, 400, 300)
            trainer.train_iteration(initial_state)
            iteration_time = time.time() - iteration_start
            
            # Regular evaluation to track learning progress
            if iteration % 5 == 0 or iteration <= 10:
                print(f"   🎯 Evaluation at iteration {iteration}...")
                eval_result = self.evaluator.evaluate_policy(
                    trainer.policy, f"PPO_FromScratch_Iter{iteration}"
                )
                
                performance = eval_result.overall_catch_rate
                improvement_vs_random = ((performance - self.random_baseline_performance) / 
                                       (self.random_baseline_performance + 1e-8)) * 100
                
                evaluation_curve.append({
                    'iteration': iteration,
                    'performance': performance,
                    'improvement_vs_random': improvement_vs_random
                })
                
                status = "✅ LEARNING!" if performance > self.random_baseline_performance * 1.5 else "📈 Progress"
                print(f"   Result: {performance:.4f} ({improvement_vs_random:+.1f}% vs random) {status}")
                
                # Early success detection
                if performance > self.random_baseline_performance * 3.0:  # 3x better than random
                    print(f"   🎉 MAJOR SUCCESS: 3x better than random baseline!")
                
                # Save checkpoint for good performance
                if performance > self.random_baseline_performance * 2.0:
                    checkpoint_path = f"checkpoints/ppo_from_scratch_success_iter{iteration}.pt"
                    torch.save(trainer.policy.model.state_dict(), checkpoint_path)
                    print(f"   💾 Success checkpoint saved: {checkpoint_path}")
            
            # Track training metrics
            training_results.append({
                'iteration': iteration,
                'training_time': iteration_time,
                'total_time': time.time() - start_time
            })
            
            print(f"   Training time: {iteration_time:.1f}s")
        
        total_training_time = time.time() - start_time
        
        # Final evaluation
        print(f"\n🎯 Final Evaluation...")
        final_result = self.evaluator.evaluate_policy(trainer.policy, "PPO_FromScratch_Final")
        final_performance = final_result.overall_catch_rate
        final_improvement = ((final_performance - self.random_baseline_performance) / 
                           (self.random_baseline_performance + 1e-8)) * 100
        
        return {
            'random_baseline': self.random_baseline_performance,
            'final_performance': final_performance,
            'final_improvement_vs_random': final_improvement,
            'learning_curve': evaluation_curve,
            'training_results': training_results,
            'total_training_time': total_training_time,
            'max_iterations': max_iterations,
            'learning_rate': learning_rate,
            'success_achieved': final_performance > self.random_baseline_performance * 2.0
        }
    
    def analyze_learning_success(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze whether PPO successfully learned from scratch"""
        
        print(f"\n🔬 Step 3: Learning Success Analysis...")
        
        random_baseline = results['random_baseline']
        final_performance = results['final_performance']
        improvement = results['final_improvement_vs_random']
        learning_curve = results['learning_curve']
        
        # Success criteria
        criteria = {
            'basic_learning': final_performance > random_baseline * 1.5,  # 50% better
            'good_learning': final_performance > random_baseline * 2.0,   # 2x better  
            'excellent_learning': final_performance > random_baseline * 3.0,  # 3x better
            'consistent_improvement': len(learning_curve) >= 3 and 
                                    learning_curve[-1]['performance'] > learning_curve[0]['performance']
        }
        
        # Learning rate analysis
        if len(learning_curve) >= 4:
            performances = [point['performance'] for point in learning_curve]
            learning_rate_quality = "good" if performances[-1] > performances[0] * 1.5 else "slow"
        else:
            learning_rate_quality = "insufficient_data"
        
        # Overall assessment
        success_score = sum(criteria.values())  # 0-4 scale
        
        if success_score >= 3:
            overall_assessment = "SUCCESS"
            confidence = "HIGH"
        elif success_score >= 2:
            overall_assessment = "PARTIAL_SUCCESS"  
            confidence = "MEDIUM"
        else:
            overall_assessment = "LIMITED_SUCCESS"
            confidence = "LOW"
        
        analysis = {
            'success_criteria': criteria,
            'success_score': success_score,
            'overall_assessment': overall_assessment,
            'confidence_level': confidence,
            'learning_rate_quality': learning_rate_quality,
            'key_findings': []
        }
        
        # Generate key findings
        if criteria['excellent_learning']:
            analysis['key_findings'].append("PPO achieved excellent learning (3x+ improvement)")
        elif criteria['good_learning']:
            analysis['key_findings'].append("PPO achieved good learning (2x+ improvement)")
        elif criteria['basic_learning']:
            analysis['key_findings'].append("PPO achieved basic learning (1.5x+ improvement)")
        else:
            analysis['key_findings'].append("PPO learning was limited (<1.5x improvement)")
        
        if criteria['consistent_improvement']:
            analysis['key_findings'].append("Learning showed consistent improvement over time")
        else:
            analysis['key_findings'].append("Learning was inconsistent or plateaued early")
        
        return analysis
    
    def run_validation_experiment(self) -> Dict[str, Any]:
        """Run complete PPO from scratch validation experiment"""
        
        print(f"\n🧪 COMPLETE PPO FROM SCRATCH VALIDATION")
        print(f"Goal: Prove PPO implementation works by training from random init")
        
        experiment_start = time.time()
        
        # Step 1: Random baseline
        random_baseline = self.establish_random_baseline()
        
        # Step 2: PPO training from scratch
        training_results = self.train_ppo_from_scratch(
            learning_rate=0.001,  # Higher LR for learning from scratch  
            max_iterations=40
        )
        
        # Step 3: Success analysis
        success_analysis = self.analyze_learning_success(training_results)
        
        total_experiment_time = time.time() - experiment_start
        
        # Comprehensive results
        validation_results = {
            'experiment_type': 'PPO_from_scratch_validation',
            'random_baseline_performance': random_baseline,
            'training_results': training_results,
            'success_analysis': success_analysis,
            'total_experiment_time_minutes': total_experiment_time / 60,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Final report
        print(f"\n{'='*80}")
        print(f"🧪 PPO FROM SCRATCH VALIDATION COMPLETE")
        print(f"{'='*80}")
        print(f"Random Baseline:     {random_baseline:.4f}")
        print(f"PPO Final:           {training_results['final_performance']:.4f}")
        print(f"Improvement:         {training_results['final_improvement_vs_random']:+.1f}%")
        print(f"Overall Assessment:  {success_analysis['overall_assessment']}")
        print(f"Confidence:          {success_analysis['confidence_level']}")
        print(f"")
        print(f"🔍 KEY FINDINGS:")
        for finding in success_analysis['key_findings']:
            print(f"   • {finding}")
        print(f"")
        print(f"Total experiment time: {total_experiment_time/60:.1f} minutes")
        
        # Recommendations based on results
        if success_analysis['overall_assessment'] == "SUCCESS":
            print(f"\n🎉 BREAKTHROUGH: PPO Implementation Validated!")
            print(f"   ✅ PPO can learn this task from scratch")
            print(f"   ✅ Implementation is working correctly")  
            print(f"   ✅ Hyperparameters are reasonable")
            print(f"   🚀 Ready to tackle SL baseline improvement with confidence!")
        elif success_analysis['overall_assessment'] == "PARTIAL_SUCCESS":
            print(f"\n📊 PARTIAL SUCCESS: PPO shows learning ability")
            print(f"   ✅ PPO can improve over random baseline")
            print(f"   ⚠️  Learning may be slow or suboptimal")
            print(f"   🔧 Consider hyperparameter tuning before SL improvement")
        else:
            print(f"\n🔧 LIMITED SUCCESS: PPO implementation needs investigation")
            print(f"   ❌ PPO barely improved over random baseline")
            print(f"   🔍 Check implementation, hyperparameters, or task complexity")
            print(f"   ⚠️  Fix these issues before attempting SL improvement")
        
        # Save results
        with open('ppo_from_scratch_validation_results.json', 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        print(f"✅ Validation results saved: ppo_from_scratch_validation_results.json")
        
        return validation_results


def main():
    """Run PPO from scratch validation experiment"""
    print("🧪 PPO FROM SCRATCH VALIDATION EXPERIMENT")
    print("=" * 70)
    print("USER INSIGHT: Prove PPO works first, then improve SL baseline")
    print("SCIENTIFIC APPROACH:")
    print("  1. Create random initialized model (terrible baseline)")
    print("  2. Train with PPO from scratch (prove learning works)")
    print("  3. Compare PPO vs random (validate improvement)")
    print("  4. Analyze learning dynamics (understand what works)")
    print("  5. Apply knowledge to SL improvement (with confidence)")
    print("=" * 70)
    
    validator = PPOFromScratchValidator()
    results = validator.run_validation_experiment()
    
    if results['success_analysis']['overall_assessment'] == "SUCCESS":
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. PPO implementation is validated ✅")
        print(f"   2. Use successful hyperparameters for SL improvement")
        print(f"   3. Apply proven training approach to beat SL baseline")
        print(f"   4. Expect similar learning dynamics but from higher starting point")
    
    return results


if __name__ == "__main__":
    main()
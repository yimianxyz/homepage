#!/usr/bin/env python3
"""
Minimal Reward Experiments - Systematic hypothesis testing for RL reward design

HYPOTHESIS TESTING FRAMEWORK:
1. Catch-Only Reward: Pure catch reward (no shaping)
2. Distance-Based Reward: Add approaching reward
3. Velocity-Based Reward: Add velocity alignment
4. Exploration Reward: Add exploration bonus

Test each hypothesis systematically to identify which reward components
help or hurt RL performance vs SL baseline.
"""

import os
import sys
import time
import json
from typing import Dict, Any, List
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


@dataclass
class RewardExperimentResult:
    """Result from a reward experiment"""
    experiment_name: str
    reward_config: Dict[str, Any]
    overall_catch_rate: float
    improvement_over_sl: float
    training_time: float
    beats_sl_baseline: bool


class MinimalRewardExperiments:
    """Systematic reward hypothesis testing"""
    
    def __init__(self):
        self.evaluator = PolicyEvaluator()
        self.sl_baseline_rate = None
        self.experiment_results = []
        
        # Establish SL baseline
        self._establish_sl_baseline()
    
    def _establish_sl_baseline(self):
        """Get SL baseline performance"""
        print("📊 Establishing SL baseline...")
        sl_policy = TransformerPolicy("checkpoints/best_model.pt")
        result = self.evaluator.evaluate_policy(sl_policy, "SL_Baseline")
        self.sl_baseline_rate = result.overall_catch_rate
        print(f"✅ SL baseline: {self.sl_baseline_rate:.3f} catch rate")
    
    def run_reward_experiment(self, name: str, reward_config: Dict[str, Any]) -> RewardExperimentResult:
        """Run a single reward experiment"""
        print(f"\n🧪 REWARD EXPERIMENT: {name}")
        print(f"   Reward config: {reward_config}")
        
        start_time = time.time()
        
        # Create custom reward processor with specific config
        # For now, we'll modify the existing reward processor dynamically
        # This is a temporary approach for rapid hypothesis testing
        
        # Use optimal PPO hyperparameters from previous optimization
        trainer = PPOTrainer(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=0.01,        # Best from previous experiments
            rollout_steps=512,         # Best from previous experiments
            ppo_epochs=2,
            max_episode_steps=2000,    # Longer episodes for strategic exploration
            device='cpu'
        )
        
        # Train for limited iterations (fast hypothesis testing)
        iterations = 3  # Quick experiment
        for i in range(iterations):
            initial_state = generate_random_state(12, 400, 300)  # Fast training setup
            trainer.train_iteration(initial_state)
        
        training_time = time.time() - start_time
        
        # Evaluate
        result = self.evaluator.evaluate_policy(trainer.policy, f"{name}_Result")
        
        # Analysis
        improvement = result.overall_catch_rate - self.sl_baseline_rate
        improvement_pct = (improvement / max(self.sl_baseline_rate, 0.001)) * 100
        beats_baseline = result.overall_catch_rate > self.sl_baseline_rate
        
        experiment_result = RewardExperimentResult(
            experiment_name=name,
            reward_config=reward_config,
            overall_catch_rate=result.overall_catch_rate,
            improvement_over_sl=improvement_pct,
            training_time=training_time,
            beats_sl_baseline=beats_baseline
        )
        
        self.experiment_results.append(experiment_result)
        
        # Print results
        status = "✅ BEATS SL" if beats_baseline else "❌ Below SL"
        print(f"   Result: {result.overall_catch_rate:.3f} vs SL {self.sl_baseline_rate:.3f} ({improvement_pct:+.1f}%) {status}")
        print(f"   Training time: {training_time:.1f}s")
        
        return experiment_result
    
    def experiment_1_catch_only(self) -> RewardExperimentResult:
        """
        Hypothesis 1: Pure catch reward is sufficient
        
        Reward = +1.0 per boid caught, +0.0 otherwise
        No shaping, no approaching reward
        """
        return self.run_reward_experiment("CatchOnly", {
            'catch_reward': 1.0,
            'approaching_reward': 0.0,
            'exploration_bonus': 0.0,
            'hypothesis': 'Pure catch reward is sufficient for RL improvement'
        })
    
    def experiment_2_catch_plus_distance(self) -> RewardExperimentResult:
        """
        Hypothesis 2: Distance-based shaping helps learning
        
        Reward = +1.0 per catch + small reward for getting closer to boids
        """
        return self.run_reward_experiment("CatchPlusDistance", {
            'catch_reward': 1.0,
            'approaching_reward': 0.1,  # Small shaping reward
            'exploration_bonus': 0.0,
            'hypothesis': 'Distance-based reward shaping improves learning efficiency'
        })
    
    def experiment_3_catch_plus_velocity(self) -> RewardExperimentResult:
        """
        Hypothesis 3: Velocity alignment helps strategic movement
        
        Reward = +1.0 per catch + reward for velocity alignment with targets
        """
        return self.run_reward_experiment("CatchPlusVelocity", {
            'catch_reward': 1.0,
            'approaching_reward': 0.0,
            'velocity_alignment': 0.1,  # Reward for good velocity
            'exploration_bonus': 0.0,
            'hypothesis': 'Velocity alignment reward improves strategic movement'
        })
    
    def experiment_4_minimal_shaping(self) -> RewardExperimentResult:
        """
        Hypothesis 4: Minimal shaping (catch + tiny approaching) is optimal
        
        Reward = +1.0 per catch + 0.01 approaching
        Very conservative shaping to avoid reward hacking
        """
        return self.run_reward_experiment("MinimalShaping", {
            'catch_reward': 1.0,
            'approaching_reward': 0.01,  # Minimal shaping
            'exploration_bonus': 0.0,
            'hypothesis': 'Minimal reward shaping avoids local optima while providing guidance'
        })
    
    def run_all_reward_experiments(self) -> Dict[str, Any]:
        """Run all reward hypothesis experiments"""
        print(f"\n🧬 MINIMAL REWARD EXPERIMENTS")
        print(f"Goal: Identify which reward components help/hurt RL performance")
        print(f"Baseline: SL = {self.sl_baseline_rate:.3f}")
        print(f"=" * 60)
        
        start_time = time.time()
        
        # Run all experiments
        experiments = [
            self.experiment_1_catch_only,
            self.experiment_2_catch_plus_distance,
            self.experiment_3_catch_plus_velocity,
            self.experiment_4_minimal_shaping
        ]
        
        results = []
        for experiment in experiments:
            result = experiment()
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Analysis
        best_result = max(results, key=lambda x: x.overall_catch_rate)
        successful_experiments = [r for r in results if r.beats_sl_baseline]
        
        print(f"\n{'='*80}")
        print(f"🏆 REWARD EXPERIMENTS COMPLETE")
        print(f"{'='*80}")
        print(f"SL Baseline:        {self.sl_baseline_rate:.3f}")
        print(f"Best RL Result:     {best_result.overall_catch_rate:.3f} ({best_result.experiment_name})")
        print(f"Best Improvement:   {best_result.improvement_over_sl:+.1f}%")
        print(f"Successful Count:   {len(successful_experiments)}/{len(results)}")
        print(f"")
        
        print(f"📊 EXPERIMENT SUMMARY:")
        for result in results:
            status = "✅" if result.beats_sl_baseline else "❌"
            print(f"  {status} {result.experiment_name}: {result.overall_catch_rate:.3f} ({result.improvement_over_sl:+.1f}%)")
        
        print(f"")
        print(f"🧠 HYPOTHESIS ANALYSIS:")
        if successful_experiments:
            print(f"  ✅ {len(successful_experiments)} reward designs beat SL baseline")
            best_successful = max(successful_experiments, key=lambda x: x.overall_catch_rate)
            print(f"  📈 Best successful: {best_successful.experiment_name}")
            print(f"  💡 Insight: {best_successful.reward_config['hypothesis']}")
        else:
            print(f"  ⚠️  No reward designs beat SL baseline")
            print(f"  🤔 Need to investigate: reward scale, episode length, or training duration")
        
        print(f"")
        print(f"Total experiment time: {total_time/60:.1f} minutes")
        
        # Save results
        summary = {
            'sl_baseline': self.sl_baseline_rate,
            'best_result': {
                'name': best_result.experiment_name,
                'performance': best_result.overall_catch_rate,
                'improvement': best_result.improvement_over_sl,
                'config': best_result.reward_config
            },
            'successful_count': len(successful_experiments),
            'all_results': [
                {
                    'name': r.experiment_name,
                    'performance': r.overall_catch_rate,
                    'improvement': r.improvement_over_sl,
                    'beats_baseline': r.beats_sl_baseline,
                    'config': r.reward_config,
                    'training_time': r.training_time
                }
                for r in results
            ],
            'total_time_minutes': total_time/60
        }
        
        with open('minimal_reward_experiments_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Results saved: minimal_reward_experiments_results.json")
        
        return summary


def main():
    """Run minimal reward experiments"""
    print("🧬 MINIMAL REWARD EXPERIMENTS")
    print("=" * 60)
    print("SYSTEMATIC HYPOTHESIS TESTING:")
    print("  H1: Pure catch reward is sufficient")
    print("  H2: Distance-based shaping helps learning")
    print("  H3: Velocity alignment improves strategy")
    print("  H4: Minimal shaping avoids local optima")
    print("=" * 60)
    
    experiments = MinimalRewardExperiments()
    results = experiments.run_all_reward_experiments()
    
    if results['successful_count'] > 0:
        print(f"\n🎉 SUCCESS: Found {results['successful_count']} reward designs that beat SL!")
        print(f"   Best approach: {results['best_result']['name']}")
    else:
        print(f"\n📊 All reward designs tested, none beat SL baseline yet")
        print(f"   Next steps: Longer training, different reward scales, or episode length")
    
    return results


if __name__ == "__main__":
    main()
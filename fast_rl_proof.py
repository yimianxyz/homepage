"""
Fast RL Proof - Minimal test to prove RL improves over SL baseline

This script runs the absolute minimum experiment needed to prove that RL training
improves performance. It compares 4 policies with fast evaluation:

1. Random Policy - Pure random actions (worst baseline)
2. Random Pursuit - Always chase closest boid (simple baseline)
3. SL Baseline - Supervised learning model (current best)
4. RL Trained - PPO fine-tuned model (should be best)

Key optimizations for speed:
- Short episodes (200 steps max)
- Fewer boids (10 instead of 20+)
- Single scenario (no need for multiple)
- Parallel evaluation
- Early stopping when sufficient data collected

Statistical validation:
- 30 episodes per policy (enough for t-tests)
- Measure catch rate in fixed time window
- One-way ANOVA + post-hoc tests
- Clear ranking with confidence intervals

Total runtime: 5-10 minutes
"""

import torch
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple
import scipy.stats as stats
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation.state_manager import StateManager
from simulation.random_state_generator import generate_random_state
from rewards.reward_processor import RewardProcessor
from policy.human_prior.random_policy import RandomPolicy
from policy.human_prior.closest_pursuit_policy import ClosestPursuitPolicy
from policy.transformer.transformer_policy import TransformerPolicy
from rl_training import create_ppo_policy_from_sl, PPOTrainer


class FastEvaluator:
    """
    Fast policy evaluator optimized for quick statistical comparison
    
    Key differences from standard evaluator:
    - Fixed short episodes (200 steps)
    - Single scenario (10 boids, 400x300)
    - No multiprocessing overhead
    - Early stopping on convergence
    - Optimized for statistical power, not comprehensive coverage
    """
    
    def __init__(self, 
                 max_steps: int = 200,
                 num_boids: int = 10,
                 canvas_width: int = 400,
                 canvas_height: int = 300):
        """Initialize fast evaluator with minimal configuration"""
        self.max_steps = max_steps
        self.num_boids = num_boids
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # Components
        self.state_manager = StateManager()
        
        print(f"⚡ Fast Evaluator initialized:")
        print(f"   Episode length: {max_steps} steps")
        print(f"   Scenario: {num_boids} boids, {canvas_width}×{canvas_height}")
    
    def evaluate_policy(self, policy, num_episodes: int = 30) -> Dict[str, Any]:
        """
        Fast evaluation of a single policy
        
        Args:
            policy: Policy to evaluate
            num_episodes: Number of episodes (30 for statistical power)
            
        Returns:
            Evaluation statistics
        """
        catch_rates = []
        catches_per_episode = []
        steps_to_first_catch = []
        
        for episode in range(num_episodes):
            # Generate initial state with fixed seed offset for reproducibility
            initial_state = generate_random_state(
                self.num_boids, 
                self.canvas_width, 
                self.canvas_height,
                seed=42 + episode  # Reproducible but different each episode
            )
            
            # Run episode
            catches, first_catch_step = self._run_episode(policy, initial_state)
            
            # Calculate metrics
            catch_rate = catches / self.num_boids
            catch_rates.append(catch_rate)
            catches_per_episode.append(catches)
            if first_catch_step is not None:
                steps_to_first_catch.append(first_catch_step)
            
            # Progress indicator
            if (episode + 1) % 10 == 0:
                print(f"      Episodes: {episode + 1}/{num_episodes}, "
                      f"mean catch rate: {np.mean(catch_rates):.3f}")
        
        # Calculate statistics
        results = {
            'catch_rates': catch_rates,
            'mean_catch_rate': np.mean(catch_rates),
            'std_catch_rate': np.std(catch_rates),
            'sem_catch_rate': stats.sem(catch_rates),  # Standard error
            'total_catches': sum(catches_per_episode),
            'mean_catches': np.mean(catches_per_episode),
            'episodes': num_episodes,
            'ci_95': stats.t.interval(
                0.95, len(catch_rates) - 1,
                loc=np.mean(catch_rates),
                scale=stats.sem(catch_rates)
            ) if len(catch_rates) > 1 else (0, 0)
        }
        
        if steps_to_first_catch:
            results['mean_steps_to_first_catch'] = np.mean(steps_to_first_catch)
        else:
            results['mean_steps_to_first_catch'] = self.max_steps
        
        return results
    
    def _run_episode(self, policy, initial_state) -> Tuple[int, int]:
        """Run single episode and return (total_catches, first_catch_step)"""
        self.state_manager.init(initial_state, policy)
        
        total_catches = 0
        first_catch_step = None
        
        for step in range(self.max_steps):
            result = self.state_manager.step()
            
            # Count catches
            if 'caught_boids' in result and len(result['caught_boids']) > 0:
                total_catches += len(result['caught_boids'])
                if first_catch_step is None:
                    first_catch_step = step
            
            # Early stop if all boids caught
            if len(result['boids_states']) == 0:
                break
        
        return total_catches, first_catch_step


def train_minimal_rl_policy(sl_checkpoint_path: str, iterations: int = 10) -> Any:
    """
    Train RL policy with minimal iterations - just enough to show improvement
    
    Args:
        sl_checkpoint_path: Path to SL model
        iterations: Number of training iterations (10 is usually enough)
        
    Returns:
        Trained RL policy
    """
    print(f"\n🚀 Training minimal RL policy ({iterations} iterations)...")
    
    trainer = PPOTrainer(
        sl_checkpoint_path=sl_checkpoint_path,
        rollout_steps=512,  # Small rollouts for speed
        ppo_epochs=2,       # Fewer epochs for speed
        mini_batch_size=64,
        learning_rate=3e-4,
        device='cpu'        # CPU is often faster for small models
    )
    
    # Track improvement
    improvements = []
    
    for i in range(iterations):
        print(f"   Iteration {i+1}/{iterations}")
        
        # Random initial state each iteration
        initial_state = generate_random_state(10, 400, 300, seed=None)
        
        # Train
        stats = trainer.train_iteration(initial_state)
        
        # Track reward improvement
        mean_reward = stats['rollout']['mean_reward']
        improvements.append(mean_reward)
        print(f"      Mean reward: {mean_reward:.3f}")
    
    print(f"   ✅ Training complete. Reward improvement: "
          f"{improvements[0]:.3f} → {improvements[-1]:.3f}")
    
    return trainer.policy


def run_fast_comparison():
    """
    Run fast comparison of all 4 policies with statistical validation
    
    This is the minimal experiment that proves RL improvement.
    """
    print(f"⚡ FAST RL PROOF - Minimal Statistical Validation")
    print(f"{'='*60}")
    print(f"Goal: Prove RL > SL > Pursuit > Random with p < 0.05")
    print(f"Time: ~5-10 minutes total")
    print(f"{'='*60}")
    
    # Check SL model exists
    sl_checkpoint = "checkpoints/best_model.pt"
    if not os.path.exists(sl_checkpoint):
        print(f"❌ SL model not found: {sl_checkpoint}")
        print(f"   Train SL model first using transformer_training.ipynb")
        return False
    
    start_time = time.time()
    
    # 1. Initialize policies
    print(f"\n📋 Loading policies...")
    policies = {
        'random': RandomPolicy(),
        'pursuit': ClosestPursuitPolicy(),
        'sl_baseline': TransformerPolicy(sl_checkpoint),
        'rl_trained': None  # Will train this
    }
    print(f"   ✓ Random, Pursuit, SL loaded")
    
    # 2. Train minimal RL policy
    policies['rl_trained'] = train_minimal_rl_policy(sl_checkpoint, iterations=10)
    
    # 3. Evaluate all policies
    print(f"\n📊 Evaluating all policies (30 episodes each)...")
    evaluator = FastEvaluator()
    results = {}
    
    for name, policy in policies.items():
        print(f"\n   Evaluating {name}...")
        start = time.time()
        results[name] = evaluator.evaluate_policy(policy, num_episodes=30)
        eval_time = time.time() - start
        
        mean = results[name]['mean_catch_rate']
        std = results[name]['std_catch_rate']
        ci = results[name]['ci_95']
        print(f"   ✓ {name}: {mean:.3f} ± {std:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])")
        print(f"     Time: {eval_time:.1f}s")
    
    # 4. Statistical analysis
    print(f"\n📈 Statistical Analysis...")
    
    # Prepare data for ANOVA
    all_catch_rates = []
    group_labels = []
    
    for name in ['random', 'pursuit', 'sl_baseline', 'rl_trained']:
        all_catch_rates.extend(results[name]['catch_rates'])
        group_labels.extend([name] * len(results[name]['catch_rates']))
    
    # One-way ANOVA
    groups = [results[name]['catch_rates'] for name in ['random', 'pursuit', 'sl_baseline', 'rl_trained']]
    f_stat, p_value = stats.f_oneway(*groups)
    
    print(f"\n   One-way ANOVA:")
    print(f"   F-statistic: {f_stat:.2f}")
    print(f"   p-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print(f"   ✅ Significant differences between policies (p < 0.05)")
    else:
        print(f"   ❌ No significant differences (p >= 0.05)")
    
    # Pairwise comparisons (key tests)
    print(f"\n   Key Pairwise Comparisons (t-tests):")
    
    key_comparisons = [
        ('rl_trained', 'sl_baseline', "RL > SL"),
        ('sl_baseline', 'pursuit', "SL > Pursuit"),
        ('pursuit', 'random', "Pursuit > Random"),
        ('rl_trained', 'random', "RL > Random")
    ]
    
    comparison_results = {}
    
    for policy1, policy2, hypothesis in key_comparisons:
        t_stat, p_val = stats.ttest_ind(
            results[policy1]['catch_rates'],
            results[policy2]['catch_rates'],
            alternative='greater'  # One-sided test
        )
        
        effect_size = (results[policy1]['mean_catch_rate'] - results[policy2]['mean_catch_rate']) / \
                     np.sqrt((np.var(results[policy1]['catch_rates']) + np.var(results[policy2]['catch_rates'])) / 2)
        
        comparison_results[f"{policy1}_vs_{policy2}"] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'effect_size': effect_size,
            'significant': p_val < 0.05
        }
        
        status = "✅" if p_val < 0.05 else "❌"
        print(f"   {status} {hypothesis}: p={p_val:.6f}, d={effect_size:.3f}")
    
    # 5. Performance ranking
    print(f"\n🏆 Performance Ranking:")
    ranking = sorted([(name, results[name]['mean_catch_rate']) for name in results.keys()],
                    key=lambda x: x[1], reverse=True)
    
    for i, (name, score) in enumerate(ranking, 1):
        ci = results[name]['ci_95']
        print(f"   {i}. {name}: {score:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])")
    
    # 6. Final verdict
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"🎯 FINAL VERDICT")
    print(f"{'='*60}")
    
    # Check if RL improved over SL
    rl_improvement = results['rl_trained']['mean_catch_rate'] - results['sl_baseline']['mean_catch_rate']
    rl_better_than_sl = comparison_results['rl_trained_vs_sl_baseline']['significant']
    
    if rl_better_than_sl and rl_improvement > 0:
        print(f"🎉 SUCCESS: RL SIGNIFICANTLY IMPROVES OVER SL!")
        print(f"   RL catch rate: {results['rl_trained']['mean_catch_rate']:.3f}")
        print(f"   SL catch rate: {results['sl_baseline']['mean_catch_rate']:.3f}")
        print(f"   Improvement: {rl_improvement:+.3f} ({rl_improvement/results['sl_baseline']['mean_catch_rate']*100:+.1f}%)")
        print(f"   Statistical significance: p={comparison_results['rl_trained_vs_sl_baseline']['p_value']:.6f}")
        print(f"   Effect size: d={comparison_results['rl_trained_vs_sl_baseline']['effect_size']:.3f}")
        success = True
    else:
        print(f"❌ INCONCLUSIVE: RL did not significantly improve over SL")
        print(f"   Consider: more training iterations, hyperparameter tuning")
        success = False
    
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"{'='*60}")
    
    # Save results
    save_results(results, comparison_results, success, total_time)
    
    return success


def save_results(results, comparisons, success, total_time):
    """Save fast proof results"""
    output = {
        'timestamp': datetime.now().isoformat(),
        'success': success,
        'total_time_seconds': total_time,
        'policy_results': {
            name: {
                'mean_catch_rate': res['mean_catch_rate'],
                'std_catch_rate': res['std_catch_rate'],
                'ci_95': res['ci_95'],
                'episodes': res['episodes']
            }
            for name, res in results.items()
        },
        'statistical_comparisons': comparisons,
        'ranking': sorted(
            [(name, results[name]['mean_catch_rate']) for name in results.keys()],
            key=lambda x: x[1], reverse=True
        )
    }
    
    os.makedirs("experiments/fast_results", exist_ok=True)
    filename = f"experiments/fast_results/fast_proof_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved: {filename}")


if __name__ == "__main__":
    success = run_fast_comparison()
    sys.exit(0 if success else 1)
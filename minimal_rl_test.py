"""
Minimal RL Test - Absolute minimum test to prove RL > SL

This is the most minimal possible test that still provides statistical proof
that RL training improves over the SL baseline.

Optimizations:
- Only compare RL vs SL (skip other policies)
- Train RL for just 5 iterations
- Evaluate on 20 episodes each
- Use paired testing (same initial states)
- Focus on binary outcome: does RL beat SL?

Total runtime: 2-3 minutes
"""

import torch
import numpy as np
from scipy import stats
import time
import os
import sys

# Add current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation.state_manager import StateManager
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy
from rl_training import PPOTrainer


def quick_evaluate(policy, test_states, max_steps=150):
    """Quickly evaluate policy on fixed test states"""
    state_manager = StateManager()
    catch_rates = []
    
    for initial_state in test_states:
        state_manager.init(initial_state, policy)
        
        catches = 0
        initial_boids = len(initial_state['boids_states'])
        
        for step in range(max_steps):
            result = state_manager.step()
            
            if 'caught_boids' in result:
                catches += len(result['caught_boids'])
            
            if len(result['boids_states']) == 0:
                break
        
        catch_rate = catches / initial_boids if initial_boids > 0 else 0
        catch_rates.append(catch_rate)
    
    return catch_rates


def minimal_rl_proof():
    """Minimal proof that RL improves over SL baseline"""
    
    print("⚡ MINIMAL RL TEST - 2 Minute Proof")
    print("="*40)
    
    # Check SL model
    sl_checkpoint = "checkpoints/best_model.pt"
    if not os.path.exists(sl_checkpoint):
        print("❌ No SL model found. Train with transformer_training.ipynb first.")
        return False
    
    start_time = time.time()
    
    # 1. Generate test states (same for both policies)
    print("\n1️⃣ Generating test scenarios...")
    test_states = []
    for i in range(20):  # 20 test episodes
        state = generate_random_state(
            num_boids=8,      # Fewer boids = faster
            canvas_width=400,
            canvas_height=300,
            seed=100 + i      # Fixed seeds for reproducibility
        )
        test_states.append(state)
    print(f"   ✓ Generated {len(test_states)} test scenarios")
    
    # 2. Evaluate SL baseline
    print("\n2️⃣ Evaluating SL baseline...")
    sl_policy = TransformerPolicy(sl_checkpoint)
    sl_scores = quick_evaluate(sl_policy, test_states)
    sl_mean = np.mean(sl_scores)
    print(f"   ✓ SL baseline: {sl_mean:.3f} ± {np.std(sl_scores):.3f}")
    
    # 3. Train minimal RL
    print("\n3️⃣ Training RL (5 iterations)...")
    trainer = PPOTrainer(
        sl_checkpoint_path=sl_checkpoint,
        rollout_steps=256,    # Very small rollouts
        ppo_epochs=2,
        learning_rate=3e-4,
        device='cpu'
    )
    
    for i in range(5):  # Just 5 iterations!
        state = generate_random_state(8, 400, 300)
        stats = trainer.train_iteration(state)
        print(f"   Iteration {i+1}: reward={stats['rollout']['mean_reward']:.3f}")
    
    # 4. Evaluate RL
    print("\n4️⃣ Evaluating RL policy...")
    rl_policy = trainer.policy
    rl_scores = quick_evaluate(rl_policy, test_states)
    rl_mean = np.mean(rl_scores)
    print(f"   ✓ RL trained: {rl_mean:.3f} ± {np.std(rl_scores):.3f}")
    
    # 5. Statistical test (paired t-test)
    print("\n5️⃣ Statistical comparison...")
    improvement = rl_mean - sl_mean
    
    # Paired t-test (most appropriate since same test states)
    t_stat, p_value = stats.ttest_rel(rl_scores, sl_scores)
    
    # Effect size
    effect_size = improvement / np.std(np.array(rl_scores) - np.array(sl_scores))
    
    print(f"   Improvement: {improvement:+.3f} ({improvement/sl_mean*100:+.1f}%)")
    print(f"   Paired t-test: t={t_stat:.2f}, p={p_value:.6f}")
    print(f"   Effect size: d={effect_size:.3f}")
    
    # 6. Result
    total_time = time.time() - start_time
    
    print(f"\n{'='*40}")
    if p_value < 0.05 and improvement > 0:
        print(f"✅ SUCCESS: RL > SL (p={p_value:.6f})")
        print(f"   Time: {total_time:.1f}s")
        return True
    else:
        print(f"❌ FAILED: No significant improvement")
        print(f"   Time: {total_time:.1f}s")
        return False


if __name__ == "__main__":
    success = minimal_rl_proof()
    sys.exit(0 if success else 1)
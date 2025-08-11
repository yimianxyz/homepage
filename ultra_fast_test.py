"""
Ultra Fast RL Test - 60 second proof of concept

The absolute fastest meaningful test that RL improves over SL.

Strategy:
- Train RL for only 3 iterations
- Test on 10 simple scenarios (5 boids each)
- Measure total catches in fixed time
- Simple before/after comparison

This sacrifices statistical rigor for speed but gives quick validation.
"""

import torch
import numpy as np
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation.state_manager import StateManager
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy
from rl_training import PPOTrainer


def ultra_fast_test():
    """60 second RL improvement test"""
    
    print("⚡ ULTRA FAST RL TEST - 60 Second Proof")
    print("="*40)
    
    if not os.path.exists("checkpoints/best_model.pt"):
        print("❌ No SL model. Run transformer_training.ipynb first.")
        return False
    
    start = time.time()
    
    # Test scenario: 5 boids, 100 steps, 10 episodes
    print("🎯 Test: 5 boids, 100 steps, 10 episodes")
    
    # Helper function
    def test_policy(policy, name):
        total_catches = 0
        for seed in range(10):
            state = generate_random_state(5, 300, 200, seed=seed)
            sm = StateManager()
            sm.init(state, policy)
            
            for _ in range(100):
                result = sm.step()
                if 'caught_boids' in result:
                    total_catches += len(result['caught_boids'])
                if len(result['boids_states']) == 0:
                    break
        
        rate = total_catches / 50  # 5 boids × 10 episodes
        print(f"   {name}: {total_catches}/50 catches ({rate:.2%})")
        return total_catches
    
    # 1. Test SL baseline
    print("\n1️⃣ SL Baseline:")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    sl_catches = test_policy(sl_policy, "SL")
    
    # 2. Train RL (ultra minimal)
    print("\n2️⃣ Training RL (3 iterations):")
    trainer = PPOTrainer(
        sl_checkpoint_path="checkpoints/best_model.pt",
        rollout_steps=128,
        ppo_epochs=1,
        device='cpu'
    )
    
    for i in range(3):
        state = generate_random_state(5, 300, 200)
        trainer.train_iteration(state)
        print(f"   ✓ Iteration {i+1} complete")
    
    # 3. Test RL
    print("\n3️⃣ RL Trained:")
    rl_catches = test_policy(trainer.policy, "RL")
    
    # 4. Result
    improvement = rl_catches - sl_catches
    elapsed = time.time() - start
    
    print(f"\n{'='*40}")
    print(f"📊 RESULT:")
    print(f"   SL: {sl_catches} catches")
    print(f"   RL: {rl_catches} catches")
    print(f"   Improvement: {improvement:+d} ({improvement/max(sl_catches,1)*100:+.0f}%)")
    print(f"   Time: {elapsed:.1f}s")
    
    if improvement > 0:
        print(f"\n✅ RL IMPROVED! (+{improvement} catches)")
        return True
    else:
        print(f"\n❌ No improvement detected")
        return False


if __name__ == "__main__":
    success = ultra_fast_test()
    sys.exit(0 if success else 1)
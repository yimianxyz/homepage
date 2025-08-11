"""
Simple Verification - Direct RL vs SL comparison

The simplest possible test to verify RL improvement over SL baseline.
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


def simple_verify():
    """Simple verification that RL improves over SL"""
    
    print("🔬 Simple RL Verification")
    print("=" * 30)
    
    if not os.path.exists("checkpoints/best_model.pt"):
        print("❌ No SL model found")
        return False
    
    start = time.time()
    
    # Test on 10 fixed scenarios
    test_scenarios = []
    for i in range(10):
        state = generate_random_state(6, 300, 200, seed=50 + i)
        test_scenarios.append(state)
    
    def evaluate_policy(policy, name):
        total_catches = 0
        state_manager = StateManager()
        
        for scenario in test_scenarios:
            state_manager.init(scenario, policy)
            catches = 0
            
            for _ in range(120):  # 120 steps
                result = state_manager.step()
                if 'caught_boids' in result:
                    catches += len(result['caught_boids'])
                if len(result['boids_states']) == 0:
                    break
            
            total_catches += catches
        
        avg_catches = total_catches / len(test_scenarios)
        print(f"   {name}: {total_catches} total catches, {avg_catches:.1f} avg")
        return total_catches
    
    # 1. Test SL baseline
    print("\n1️⃣ Testing SL baseline...")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    sl_catches = evaluate_policy(sl_policy, "SL")
    
    # 2. Train RL briefly
    print("\n2️⃣ Training RL (4 iterations)...")
    trainer = PPOTrainer(
        sl_checkpoint_path="checkpoints/best_model.pt",
        rollout_steps=200,
        ppo_epochs=1,
        device='cpu'
    )
    
    for i in range(4):
        state = generate_random_state(6, 300, 200)
        trainer.train_iteration(state)
        print(f"      Iteration {i+1} done")
    
    # 3. Test RL
    print("\n3️⃣ Testing RL policy...")
    rl_catches = evaluate_policy(trainer.policy, "RL")
    
    # 4. Compare
    improvement = rl_catches - sl_catches
    improvement_pct = (improvement / max(sl_catches, 1)) * 100
    
    elapsed = time.time() - start
    
    print(f"\n{'='*30}")
    print(f"📊 VERIFICATION RESULT:")
    print(f"   SL Baseline: {sl_catches} catches")
    print(f"   RL Trained:  {rl_catches} catches")
    print(f"   Improvement: {improvement:+d} ({improvement_pct:+.0f}%)")
    print(f"   Time: {elapsed:.1f}s")
    
    if improvement > 0:
        print(f"\n✅ VERIFIED: RL improves over SL!")
        return True
    elif improvement == 0:
        print(f"\n➖ NEUTRAL: RL equals SL performance")
        return False
    else:
        print(f"\n❌ REGRESSION: RL performs worse than SL")
        return False


if __name__ == "__main__":
    success = simple_verify()
    print(f"\nVerification {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
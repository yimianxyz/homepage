"""
Extended Verification - More training iterations to verify RL can improve

This test uses more training iterations to see if RL can eventually surpass SL.
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


def extended_verify():
    """Extended verification with more training"""
    
    print("🔬 Extended RL Verification")
    print("=" * 35)
    
    if not os.path.exists("checkpoints/best_model.pt"):
        print("❌ No SL model found")
        return False
    
    start = time.time()
    
    # Test on 5 fixed scenarios (smaller for speed)
    test_scenarios = []
    for i in range(5):
        state = generate_random_state(6, 300, 200, seed=60 + i)
        test_scenarios.append(state)
    
    def evaluate_policy(policy, name):
        total_catches = 0
        state_manager = StateManager()
        
        for scenario in test_scenarios:
            state_manager.init(scenario, policy)
            catches = 0
            
            for _ in range(150):  # 150 steps
                result = state_manager.step()
                if 'caught_boids' in result:
                    catches += len(result['caught_boids'])
                if len(result['boids_states']) == 0:
                    break
            
            total_catches += catches
        
        avg_catches = total_catches / len(test_scenarios)
        print(f"   {name}: {total_catches} total, {avg_catches:.1f} avg")
        return total_catches
    
    # 1. Test SL baseline
    print("\n1️⃣ Testing SL baseline...")
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    sl_catches = evaluate_policy(sl_policy, "SL")
    
    # 2. Train RL with more iterations
    print("\n2️⃣ Training RL (15 iterations)...")
    trainer = PPOTrainer(
        sl_checkpoint_path="checkpoints/best_model.pt",
        rollout_steps=512,  # Larger rollouts
        ppo_epochs=2,       # More epochs
        learning_rate=1e-3, # Higher learning rate
        device='cpu'
    )
    
    rewards = []
    for i in range(15):
        state = generate_random_state(6, 300, 200)
        stats = trainer.train_iteration(state)
        reward = stats['rollout']['mean_reward']
        rewards.append(reward)
        
        if (i + 1) % 5 == 0:
            print(f"      Iterations {i-4}-{i+1}: rewards {rewards[-5:]}")
    
    print(f"      Training curve: {rewards[0]:.3f} → {rewards[-1]:.3f}")
    
    # 3. Test RL multiple times to check consistency
    print("\n3️⃣ Testing RL policy...")
    rl_catches = evaluate_policy(trainer.policy, "RL")
    
    # 4. Compare
    improvement = rl_catches - sl_catches
    improvement_pct = (improvement / max(sl_catches, 1)) * 100
    
    elapsed = time.time() - start
    
    print(f"\n{'='*35}")
    print(f"📊 EXTENDED VERIFICATION:")
    print(f"   SL Baseline: {sl_catches} catches")
    print(f"   RL Trained:  {rl_catches} catches")
    print(f"   Improvement: {improvement:+d} ({improvement_pct:+.0f}%)")
    print(f"   Reward progression: {rewards[0]:.3f} → {rewards[-1]:.3f}")
    print(f"   Time: {elapsed:.1f}s")
    
    if improvement > 0:
        print(f"\n✅ SUCCESS: RL improves over SL!")
        print(f"   Longer training enables improvement")
        return True
    elif improvement == 0:
        print(f"\n➖ NEUTRAL: RL equals SL performance")
        print(f"   More training or tuning may be needed")
        return False
    else:
        print(f"\n❌ INCONCLUSIVE: RL still behind SL")
        print(f"   Consider: longer training, hyperparameter tuning")
        return False


if __name__ == "__main__":
    success = extended_verify()
    print(f"\nExtended verification {'PASSED' if success else 'NEEDS MORE WORK'}")
    sys.exit(0 if success else 1)
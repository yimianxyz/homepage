"""
Diagnostic Analysis - Why isn't RL improving over SL?

This script analyzes why our RL system isn't showing improvement and provides
actionable recommendations for debugging and fixing the issue.
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
from policy.human_prior.random_policy import RandomPolicy
from policy.human_prior.closest_pursuit_policy import ClosestPursuitPolicy
from rl_training import PPOTrainer
from rewards.reward_processor import RewardProcessor


def diagnostic_analysis():
    """Comprehensive diagnostic analysis"""
    
    print("🔍 DIAGNOSTIC ANALYSIS - Why isn't RL improving?")
    print("=" * 55)
    
    if not os.path.exists("checkpoints/best_model.pt"):
        print("❌ No SL model found")
        return
    
    # 1. Test basic policies for reference
    print("\n1️⃣ BASELINE POLICY COMPARISON")
    print("-" * 30)
    
    # Simple test scenario
    test_state = generate_random_state(5, 300, 200, seed=123)
    
    def quick_test(policy, name, max_steps=100):
        state_manager = StateManager()
        state_manager.init(test_state, policy)
        
        catches = 0
        for step in range(max_steps):
            result = state_manager.step()
            if 'caught_boids' in result:
                catches += len(result['caught_boids'])
            if len(result['boids_states']) == 0:
                break
        
        catch_rate = catches / 5  # 5 boids total
        print(f"   {name:15s}: {catches} catches ({catch_rate:.1%})")
        return catches
    
    # Test all baseline policies
    random_policy = RandomPolicy()
    pursuit_policy = ClosestPursuitPolicy()
    sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    
    random_catches = quick_test(random_policy, "Random")
    pursuit_catches = quick_test(pursuit_policy, "Pursuit")
    sl_catches = quick_test(sl_policy, "SL Baseline")
    
    # 2. Reward function analysis
    print("\n2️⃣ REWARD FUNCTION ANALYSIS")
    print("-" * 30)
    
    reward_processor = RewardProcessor()
    
    # Test reward calculation
    sample_state = {
        'context': {'canvasWidth': 0.75, 'canvasHeight': 0.5},
        'predator': {'velX': 0.1, 'velY': -0.1},
        'boids': [
            {'relX': 0.1, 'relY': 0.2, 'velX': 0.3, 'velY': -0.2},
            {'relX': -0.15, 'relY': 0.05, 'velX': -0.1, 'velY': 0.3}
        ]
    }
    
    # Test reward with no catches
    reward_input = {
        'state': sample_state,
        'action': [0.5, -0.3],
        'caughtBoids': []
    }
    reward_no_catch = reward_processor.calculate_step_reward(reward_input)
    
    # Test reward with catch
    reward_input['caughtBoids'] = [1]
    reward_with_catch = reward_processor.calculate_step_reward(reward_input)
    
    print(f"   No catch reward: {reward_no_catch['total']:.3f} (approaching: {reward_no_catch['approaching']:.3f})")
    print(f"   With catch reward: {reward_with_catch['total']:.3f} (catch: {reward_with_catch['catch']:.3f})")
    print(f"   Catch bonus: {reward_with_catch['total'] - reward_no_catch['total']:.3f}")
    
    # 3. Training dynamics analysis
    print("\n3️⃣ TRAINING DYNAMICS")
    print("-" * 30)
    
    print("   Creating PPO trainer...")
    trainer = PPOTrainer(
        sl_checkpoint_path="checkpoints/best_model.pt",
        rollout_steps=256,
        ppo_epochs=1,
        learning_rate=3e-4,
        device='cpu'
    )
    
    # Collect initial rollout to see what's happening
    print("   Collecting sample rollout...")
    from rl_training.ppo_experience_buffer import PPORolloutCollector
    collector = PPORolloutCollector(
        trainer.state_manager,
        trainer.reward_processor,
        trainer.policy,
        max_episode_steps=200
    )
    
    sample_state = generate_random_state(5, 300, 200, seed=456)
    buffer = collector.collect_rollout(sample_state, rollout_steps=50)
    
    stats = buffer.get_statistics()
    print(f"   Sample rollout: {stats['rollout_length']} steps")
    print(f"   Mean reward: {stats['mean_reward']:.3f}")
    print(f"   Reward range: [{stats['min_reward']:.3f}, {stats['max_reward']:.3f}]")
    
    # 4. Value function analysis
    print("\n4️⃣ VALUE FUNCTION CHECK")
    print("-" * 30)
    
    # Test value predictions
    trainer.policy.eval()
    with torch.no_grad():
        structured_input = trainer.state_manager._convert_state_to_structured_inputs(sample_state)
        action, _, value = trainer.policy.get_action_and_value(structured_input)
        print(f"   Value prediction: {value.item():.3f}")
        print(f"   Action: [{action[0].item():.3f}, {action[1].item():.3f}]")
    
    # 5. Diagnostic summary and recommendations
    print("\n" + "=" * 55)
    print("🎯 DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("=" * 55)
    
    print(f"\n📊 Policy Performance Hierarchy:")
    performances = [
        ("Random", random_catches),
        ("Pursuit", pursuit_catches), 
        ("SL Baseline", sl_catches)
    ]
    performances.sort(key=lambda x: x[1], reverse=True)
    for i, (name, catches) in enumerate(performances, 1):
        print(f"   {i}. {name}: {catches} catches")
    
    print(f"\n🔍 Key Findings:")
    
    # Check if SL is actually good
    if sl_catches <= random_catches:
        print(f"   ⚠️  SL baseline performs poorly (≤ random policy)")
        print(f"   ➡️  Consider retraining SL model with more data/epochs")
    elif sl_catches <= pursuit_catches:
        print(f"   ⚠️  SL baseline doesn't beat simple pursuit policy")
        print(f"   ➡️  SL model may need improvement before RL fine-tuning")
    else:
        print(f"   ✅ SL baseline beats simple policies (good starting point)")
    
    # Check reward signal
    if reward_with_catch['total'] <= reward_no_catch['total'] * 2:
        print(f"   ⚠️  Weak reward signal for catches")
        print(f"   ➡️  Consider increasing BASE_CATCH_REWARD in constants.py")
    else:
        print(f"   ✅ Strong reward signal for catches")
    
    # Check value function
    if abs(value.item()) < 0.1:
        print(f"   ⚠️  Value function predictions near zero")
        print(f"   ➡️  Value head may need better initialization or learning rate")
    else:
        print(f"   📈 Value function active (magnitude: {abs(value.item()):.3f})")
    
    print(f"\n🔧 SPECIFIC RECOMMENDATIONS:")
    print(f"   1. Increase training duration (try 50+ iterations)")
    print(f"   2. Increase learning rate (try 1e-3 or 3e-3)")
    print(f"   3. Increase rollout steps (try 1024+)")
    print(f"   4. Check reward scaling in constants.py")
    print(f"   5. Try different hyperparameters:")
    print(f"      - clip_epsilon: 0.1 or 0.3")
    print(f"      - entropy_coef: 0.1 or 0.01")
    print(f"      - value_loss_coef: 1.0")
    
    print(f"\n🚀 QUICK TEST COMMAND:")
    print(f"   python train_ppo.py --iterations 50 --rollout-steps 1024 --learning-rate 1e-3")
    
    print(f"\n💡 KEY INSIGHT:")
    if sl_catches > pursuit_catches:
        print(f"   The SL baseline is decent - RL should be able to improve it")
        print(f"   The issue is likely in PPO hyperparameters or training duration")
    else:
        print(f"   The SL baseline itself may need improvement first")
        print(f"   Consider more SL training before attempting RL fine-tuning")


if __name__ == "__main__":
    diagnostic_analysis()
"""
Quick PPO Test - Verify core functionality works

This is a minimal test to verify the PPO system is working correctly.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic PPO functionality"""
    print("🧪 Quick PPO Test")
    print("=" * 40)
    
    try:
        # Test 1: Model loading
        print("1. Testing model loading...")
        from rl_training import create_ppo_policy_from_sl
        
        policy = create_ppo_policy_from_sl("checkpoints/best_model.pt")
        print("   ✅ Model loaded successfully")
        
        # Test 2: Policy interface
        print("2. Testing policy interface...")
        test_input = {
            'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
            'predator': {'velX': 0.1, 'velY': -0.2},
            'boids': [
                {'relX': 0.1, 'relY': 0.3, 'velX': 0.5, 'velY': -0.1},
                {'relX': -0.2, 'relY': 0.1, 'velX': -0.3, 'velY': 0.4}
            ]
        }
        
        # Test standard interface
        action = policy.get_action(test_input)
        assert len(action) == 2
        assert all(-1 <= a <= 1 for a in action)
        print(f"   ✅ Standard interface: {action}")
        
        # Test PPO interface
        policy.train()
        action_tensor, log_prob, value = policy.get_action_and_value(test_input)
        print(f"   ✅ PPO interface: action={action_tensor.detach().numpy()}, log_prob={log_prob.item():.3f}, value={value.item():.3f}")
        
        # Test 3: Basic trainer creation
        print("3. Testing trainer creation...")
        from rl_training import PPOTrainer
        
        trainer = PPOTrainer(
            sl_checkpoint_path="checkpoints/best_model.pt",
            rollout_steps=10,  # Minimal for testing
            ppo_epochs=1,
            mini_batch_size=5
        )
        print("   ✅ Trainer created successfully")
        
        print("\n🎉 All tests passed!")
        print("✅ PPO system is ready for training")
        print("🚀 Run: python train_ppo.py --iterations 10")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
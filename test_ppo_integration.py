"""
PPO Integration Test - Comprehensive test of PPO system integration

This script tests the complete PPO pipeline to ensure all components work together:
1. PPO model loading from SL checkpoint
2. Experience collection using StateManager and RewardProcessor  
3. Policy training with PPO loss computation
4. Evaluation using existing PolicyEvaluator
5. Interface compatibility with simulation infrastructure

Run this before starting actual training to verify everything is working.
"""

import torch
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ppo_model_loading():
    """Test PPO model creation and SL checkpoint loading"""
    print("🧪 Testing PPO model loading...")
    
    try:
        from rl_training import create_ppo_policy_from_sl
        
        # Test model creation (this will test checkpoint loading)
        if not os.path.exists("checkpoints/best_model.pt"):
            print("  ⚠️  No SL checkpoint found at checkpoints/best_model.pt")
            print("  Creating dummy checkpoint for testing...")
            
            # Create minimal dummy checkpoint
            os.makedirs("checkpoints", exist_ok=True)
            dummy_checkpoint = {
                'architecture': {
                    'd_model': 64,
                    'n_heads': 4, 
                    'n_layers': 2,
                    'ffn_hidden': 128,
                    'max_boids': 20
                },
                'model_state_dict': {}
            }
            
            # Create dummy model to get proper state dict
            from rl_training.ppo_transformer_model import PPOTransformerModel
            dummy_model = PPOTransformerModel(64, 4, 2, 128, 20)
            dummy_checkpoint['model_state_dict'] = dummy_model.state_dict()
            
            torch.save(dummy_checkpoint, "checkpoints/best_model.pt")
            print("  ✓ Dummy checkpoint created for testing")
        
        policy = create_ppo_policy_from_sl("checkpoints/best_model.pt")
        print("  ✅ PPO model loaded successfully")
        
        # Test policy interface
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
        assert len(action) == 2, "Action should be 2D"
        assert all(-1 <= a <= 1 for a in action), "Action should be in [-1, 1]"
        print(f"  ✓ Standard interface: {action}")
        
        # Test PPO interface
        policy.train()
        action_tensor, log_prob, value = policy.get_action_and_value(test_input)
        assert action_tensor.shape == (2,), "Action tensor should be 2D"
        assert log_prob is not None, "Log prob should not be None in training mode"
        assert value is not None, "Value should not be None"
        print(f"  ✓ PPO interface: action={action_tensor.numpy()}, log_prob={log_prob.item():.3f}, value={value.item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ PPO model loading failed: {e}")
        return False


def test_experience_collection():
    """Test experience collection using existing simulation infrastructure"""
    print("\n🧪 Testing experience collection...")
    
    try:
        from rl_training import create_ppo_policy_from_sl, PPORolloutCollector
        from simulation.state_manager import StateManager
        from simulation.random_state_generator import generate_random_state
        from rewards.reward_processor import RewardProcessor
        
        # Create components
        policy = create_ppo_policy_from_sl("checkpoints/best_model.pt")
        state_manager = StateManager()
        reward_processor = RewardProcessor()
        collector = PPORolloutCollector(state_manager, reward_processor, policy, max_episode_steps=100)
        
        # Generate initial state
        initial_state = generate_random_state(5, 400, 300, seed=42)
        print(f"  ✓ Initial state created: {len(initial_state['boids_states'])} boids")
        
        # Collect short rollout
        print("  📊 Collecting test rollout (10 steps)...")
        buffer = collector.collect_rollout(initial_state, rollout_steps=10)
        
        assert len(buffer) == 10, f"Expected 10 experiences, got {len(buffer)}"
        
        # Test buffer data
        batch_data = buffer.get_batch_data()
        assert 'structured_inputs' in batch_data, "Missing structured_inputs"
        assert 'actions' in batch_data, "Missing actions"
        assert 'advantages' in batch_data, "Missing advantages"
        assert 'returns' in batch_data, "Missing returns"
        
        stats = buffer.get_statistics()
        print(f"  ✓ Rollout statistics: {stats}")
        
        print("  ✅ Experience collection successful")
        return True
        
    except Exception as e:
        print(f"  ❌ Experience collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo_training():
    """Test PPO loss computation and model update"""
    print("\n🧪 Testing PPO training...")
    
    try:
        from rl_training import PPOTrainer
        
        # Create trainer with small parameters for testing
        trainer = PPOTrainer(
            sl_checkpoint_path="checkpoints/best_model.pt",
            rollout_steps=20,  # Small for testing
            ppo_epochs=1,
            mini_batch_size=10,
            learning_rate=1e-4
        )
        
        # Get initial policy parameters for comparison
        initial_params = list(trainer.policy.model.parameters())[0].clone()
        
        # Run one training iteration
        print("  🔄 Running test training iteration...")
        from simulation.random_state_generator import generate_random_state
        initial_state = generate_random_state(3, 300, 200, seed=42)
        
        stats = trainer.train_iteration(initial_state)
        
        # Check that parameters changed
        final_params = list(trainer.policy.model.parameters())[0]
        params_changed = not torch.allclose(initial_params, final_params)
        assert params_changed, "Model parameters should change after training"
        
        # Check statistics
        assert 'rollout' in stats, "Missing rollout statistics"
        assert 'training' in stats, "Missing training statistics"
        assert stats['rollout']['rollout_length'] == 20, "Incorrect rollout length"
        
        print(f"  ✓ Training statistics: {stats['training']}")
        print("  ✅ PPO training successful")
        return True
        
    except Exception as e:
        print(f"  ❌ PPO training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_integration():
    """Test integration with existing evaluation system"""
    print("\n🧪 Testing evaluation integration...")
    
    try:
        from rl_training import create_ppo_policy_from_sl
        from evaluation.policy_evaluator import PolicyEvaluator
        
        # Create policy and evaluator
        policy = create_ppo_policy_from_sl("checkpoints/best_model.pt")
        evaluator = PolicyEvaluator()
        
        print("  🎯 Running evaluation (1 episode for speed)...")
        
        # Temporarily modify evaluator for faster testing
        import evaluation.policy_evaluator as eval_module
        original_episodes = eval_module.EPISODES_PER_SCENARIO
        original_steps = eval_module.MAX_STEPS_PER_EPISODE
        
        eval_module.EPISODES_PER_SCENARIO = 1
        eval_module.MAX_STEPS_PER_EPISODE = 50
        
        try:
            result = evaluator.evaluate_policy(policy, "PPO_Test")
            assert hasattr(result, 'overall_catch_rate'), "Missing catch rate"
            assert result.total_episodes > 0, "No episodes completed"
            
            print(f"  ✓ Evaluation result: catch_rate={result.overall_catch_rate:.3f}")
            print("  ✅ Evaluation integration successful")
            return True
            
        finally:
            # Restore original values
            eval_module.EPISODES_PER_SCENARIO = original_episodes
            eval_module.MAX_STEPS_PER_EPISODE = original_steps
        
    except Exception as e:
        print(f"  ❌ Evaluation integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_interface_compatibility():
    """Test that PPO policy is compatible with existing interfaces"""
    print("\n🧪 Testing interface compatibility...")
    
    try:
        from rl_training import create_ppo_policy_from_sl
        from simulation.state_manager import StateManager
        from simulation.random_state_generator import generate_random_state
        
        # Create policy and state manager
        policy = create_ppo_policy_from_sl("checkpoints/best_model.pt")
        state_manager = StateManager()
        
        # Test StateManager integration
        initial_state = generate_random_state(3, 300, 200, seed=42)
        state_manager.init(initial_state, policy)
        
        # Take a few steps
        for i in range(3):
            result = state_manager.step()
            assert 'boids_states' in result, "Missing boids_states"
            assert 'predator_state' in result, "Missing predator_state"
            print(f"  ✓ Step {i+1}: {len(result['boids_states'])} boids remaining")
        
        print("  ✅ Interface compatibility confirmed")
        return True
        
    except Exception as e:
        print(f"  ❌ Interface compatibility failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("🚀 PPO Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("PPO Model Loading", test_ppo_model_loading),
        ("Experience Collection", test_experience_collection),
        ("PPO Training", test_ppo_training),
        ("Evaluation Integration", test_evaluation_integration),
        ("Interface Compatibility", test_interface_compatibility),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*60}")
    print(f"🎯 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Passed: {passed}/{total}")
    
    if passed == total:
        print(f"  🎉 ALL TESTS PASSED!")
        print(f"  ✅ PPO system is ready for training")
        print(f"  🚀 Run: python train_ppo.py --iterations 10")
    else:
        print(f"  ❌ {total - passed} test(s) failed")
        print(f"  🔧 Fix issues before starting training")
    
    print(f"{'='*60}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
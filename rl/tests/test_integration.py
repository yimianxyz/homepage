"""
Integration Tests - Test the complete RL training pipeline

This script tests the integration between all components:
environment, model loading, policy, and training.
"""

import torch
import numpy as np
import sys
import os
import tempfile
import shutil

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.environment import BoidEnvironment
from rl.models import TransformerModel, TransformerModelLoader, TransformerPolicy
from rl.training import PPOTrainer, TrainingConfig
from rl.utils import set_seed, check_environment
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium import spaces


def test_environment_model_integration():
    """Test integration between environment and model"""
    print("🧪 Testing environment-model integration...")
    
    try:
        # Create environment
        env = BoidEnvironment(num_boids=5, max_steps=50, seed=42)
        
        # Create model
        model = TransformerModel(d_model=32, n_heads=4, n_layers=2, max_boids=5)
        
        # Test that model can process environment observations
        obs, info = env.reset()
        
        # Convert observation to structured input manually
        # Create a temporary policy to access the conversion method
        temp_policy = TransformerPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda x: 1e-4,
            transformer_model=model,
            max_boids=5
        )
        structured_input = temp_policy._observation_to_structured_input(torch.tensor(obs))
        
        # Test model forward pass
        model.eval()
        with torch.no_grad():
            action = model(structured_input)
        
        print("✅ Model processes environment observations correctly")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action shape: {action.shape}")
        print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
        
        # Test action is valid for environment
        action_np = action.cpu().numpy()
        assert env.action_space.contains(action_np), "Model action not in action space"
        
        # Test environment step with model action
        obs, reward, terminated, truncated, info = env.step(action_np)
        
        print("✅ Environment accepts model actions")
        print(f"  Reward: {reward:.3f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment-model integration failed: {e}")
        return False


def test_policy_environment_integration():
    """Test integration between policy and environment"""
    print("\n🧪 Testing policy-environment integration...")
    
    try:
        # Create environment
        env = BoidEnvironment(num_boids=8, max_steps=50, seed=42)
        
        # Create policy
        model = TransformerModel(d_model=32, n_heads=4, n_layers=2, max_boids=8)
        
        def dummy_lr_schedule(x):
            return 1e-4
        
        policy = TransformerPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=dummy_lr_schedule,
            transformer_model=model,
            max_boids=8
        )
        
        # Test policy with environment
        obs, info = env.reset()
        
        # Test predict method
        action, _ = policy.predict(obs, deterministic=True)
        
        print("✅ Policy predict works with environment observations")
        print(f"  Action: {action}")
        
        # Test environment step with policy action
        obs, reward, terminated, truncated, info = env.step(action)
        
        print("✅ Environment accepts policy actions")
        print(f"  Reward: {reward:.3f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Policy-environment integration failed: {e}")
        return False


def test_ppo_integration():
    """Test integration with PPO"""
    print("\n🧪 Testing PPO integration...")
    
    try:
        # Create environment
        env = BoidEnvironment(num_boids=5, max_steps=20, seed=42)
        
        # Create model
        model = TransformerModel(d_model=32, n_heads=4, n_layers=2, max_boids=5)
        
        # Create PPO agent
        policy_kwargs = {
            'transformer_model': model,
            'max_boids': 5,
        }
        
        ppo = PPO(
            policy=TransformerPolicy,
            env=env,
            learning_rate=1e-4,
            n_steps=32,
            batch_size=16,
            n_epochs=2,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device='cpu'
        )
        
        print("✅ PPO agent created successfully")
        
        # Test learning for a few steps
        ppo.learn(total_timesteps=64, progress_bar=False)
        
        print("✅ PPO learning completed")
        
        # Test trained model
        obs, info = env.reset()
        action, _ = ppo.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        print("✅ Trained PPO model works")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.3f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ PPO integration failed: {e}")
        return False


def test_trainer_config_integration():
    """Test integration with trainer and config"""
    print("\n🧪 Testing trainer-config integration...")
    
    try:
        # Create temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create minimal config
            config = TrainingConfig(
                num_boids=3,
                canvas_width=200,
                canvas_height=150,
                max_episode_steps=20,
                total_timesteps=64,
                n_steps=16,
                batch_size=8,
                n_epochs=1,
                n_envs=1,
                save_freq=32,
                eval_freq=0,  # Disable evaluation for speed
                log_dir=os.path.join(temp_dir, "logs"),
                save_dir=os.path.join(temp_dir, "models"),
                tensorboard_log=None,  # Disable tensorboard for testing
                model_checkpoint="non_existent.pt"  # Will create new model
            )
            
            print("✅ Training config created")
            
            # Create trainer
            trainer = PPOTrainer(config)
            
            print("✅ Trainer created")
            
            # Setup components (without training)
            trainer.setup_environment()
            trainer.load_model()
            trainer.setup_ppo()
            
            print("✅ Trainer setup completed")
            
            # Test very short training
            trainer.ppo_agent.learn(total_timesteps=64, progress_bar=False)
            
            print("✅ Short training completed")
            
            # Cleanup trainer
            trainer.cleanup()
            
            return True
            
        finally:
            # Cleanup temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"❌ Trainer-config integration failed: {e}")
        return False


def test_vectorized_environment():
    """Test integration with vectorized environment"""
    print("\n🧪 Testing vectorized environment integration...")
    
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        # Create vectorized environment
        def make_env():
            return BoidEnvironment(num_boids=3, max_steps=20, seed=42)
        
        vec_env = DummyVecEnv([make_env, make_env])
        
        print("✅ Vectorized environment created")
        
        # Create model and policy
        model = TransformerModel(d_model=32, n_heads=4, n_layers=2, max_boids=3)
        
        policy_kwargs = {
            'transformer_model': model,
            'max_boids': 3,
        }
        
        # Create PPO with vectorized environment
        ppo = PPO(
            policy=TransformerPolicy,
            env=vec_env,
            learning_rate=1e-4,
            n_steps=16,
            batch_size=8,
            n_epochs=1,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device='cpu'
        )
        
        print("✅ PPO with vectorized environment created")
        
        # Test short training
        ppo.learn(total_timesteps=32, progress_bar=False)
        
        print("✅ Vectorized training completed")
        
        vec_env.close()
        return True
        
    except Exception as e:
        print(f"❌ Vectorized environment integration failed: {e}")
        return False


def test_model_checkpoint_integration():
    """Test integration with model checkpoints"""
    print("\n🧪 Testing model checkpoint integration...")
    
    try:
        # Create temporary files
        temp_dir = tempfile.mkdtemp()
        
        try:
            checkpoint_path = os.path.join(temp_dir, "test_model.pt")
            
            # Create and save a model
            original_model = TransformerModel(d_model=32, n_heads=4, n_layers=2, max_boids=5)
            loader = TransformerModelLoader()
            loader.save_model(original_model, checkpoint_path)
            
            print("✅ Model checkpoint saved")
            
            # Create config that uses the checkpoint
            config = TrainingConfig(
                num_boids=5,
                max_episode_steps=20,
                total_timesteps=32,
                n_steps=16,
                batch_size=8,
                n_epochs=1,
                n_envs=1,
                model_checkpoint=checkpoint_path,
                log_dir=os.path.join(temp_dir, "logs"),
                save_dir=os.path.join(temp_dir, "models"),
                tensorboard_log=None
            )
            
            # Create trainer and test loading
            trainer = PPOTrainer(config)
            trainer.setup_environment()
            loaded_model = trainer.load_model()
            
            print("✅ Model loaded from checkpoint")
            
            # Test that loaded model works
            trainer.setup_ppo()
            trainer.ppo_agent.learn(total_timesteps=32, progress_bar=False)
            
            print("✅ Training with loaded model completed")
            
            trainer.cleanup()
            return True
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"❌ Model checkpoint integration failed: {e}")
        return False


def test_complete_pipeline():
    """Test the complete training pipeline"""
    print("\n🧪 Testing complete pipeline...")
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create minimal config for quick test
            config = TrainingConfig(
                num_boids=3,
                canvas_width=200,
                canvas_height=150,
                max_episode_steps=10,
                total_timesteps=32,
                n_steps=16,
                batch_size=8,
                n_epochs=1,
                n_envs=1,
                save_freq=16,
                eval_freq=0,
                log_dir=os.path.join(temp_dir, "logs"),
                save_dir=os.path.join(temp_dir, "models"),
                tensorboard_log=None,
                model_checkpoint="non_existent.pt"
            )
            
            # Run training
            trainer = PPOTrainer(config)
            
            # Test all setup steps
            trainer.setup_environment()
            trainer.load_model()
            trainer.setup_ppo()
            
            # Test callbacks setup
            callbacks = trainer.setup_callbacks()
            
            print("✅ Pipeline setup completed")
            
            # Test very short training
            trainer.ppo_agent.learn(
                total_timesteps=32, 
                callback=callbacks,
                progress_bar=False
            )
            
            print("✅ Complete pipeline training completed")
            
            # Test model saving
            final_model_path = os.path.join(config.save_dir, "test_final_model")
            trainer.ppo_agent.save(final_model_path)
            
            # Test model loading and evaluation
            loaded_ppo = PPO.load(final_model_path)
            
            # Test evaluation
            test_env = BoidEnvironment(num_boids=3, max_steps=10, seed=123)
            obs, info = test_env.reset()
            action, _ = loaded_ppo.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            
            print("✅ Model save/load/evaluation completed")
            
            trainer.cleanup()
            test_env.close()
            
            return True
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"❌ Complete pipeline test failed: {e}")
        return False


def run_all_integration_tests():
    """Run all integration tests"""
    print("🔬 INTEGRATION TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_environment_model_integration,
        test_policy_environment_integration,
        test_ppo_integration,
        test_trainer_config_integration,
        test_vectorized_environment,
        test_model_checkpoint_integration,
        test_complete_pipeline
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"  Passed: {passed}/{total}")
    print(f"  Success rate: {passed/total:.1%}")
    
    if passed == total:
        print("🎉 All integration tests passed!")
    else:
        print("⚠️  Some integration tests failed")
    
    return passed == total


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    
    # Run all tests
    success = run_all_integration_tests()
    
    exit(0 if success else 1)
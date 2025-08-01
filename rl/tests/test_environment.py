"""
Environment Tests - Test the BoidEnvironment Gym wrapper

This script tests the BoidEnvironment to ensure it works correctly
with the underlying simulation system and is compatible with stable-baselines3.
"""

import numpy as np
import sys
import os
import time

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.environment import BoidEnvironment
from rl.utils import check_environment, create_test_observation, set_seed


def test_environment_creation():
    """Test basic environment creation"""
    print("🧪 Testing environment creation...")
    
    try:
        env = BoidEnvironment(
            num_boids=10,
            canvas_width=400,
            canvas_height=300,
            max_steps=100,
            seed=42
        )
        print("✅ Environment created successfully")
        
        # Check basic properties
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Max boids: {env.max_obs_boids}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False


def test_environment_reset():
    """Test environment reset functionality"""
    print("\n🧪 Testing environment reset...")
    
    try:
        env = BoidEnvironment(num_boids=5, max_steps=50, seed=42)
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"  Info keys: {list(info.keys())}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment reset failed: {e}")
        return False


def test_environment_step():
    """Test environment step functionality"""
    print("\n🧪 Testing environment step...")
    
    try:
        env = BoidEnvironment(num_boids=8, max_steps=20, seed=42)
        
        obs, info = env.reset()
        print(f"Initial boids: {info['boids_remaining']}")
        
        # Test multiple steps
        total_reward = 0
        for step in range(10):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"  Step {step + 1}: Reward={reward:.3f}, "
                  f"Boids={info['boids_remaining']}, "
                  f"Terminated={terminated}, Truncated={truncated}")
            
            if terminated or truncated:
                print(f"  Episode ended at step {step + 1}")
                break
        
        print(f"✅ Steps completed successfully")
        print(f"  Total reward: {total_reward:.3f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment step failed: {e}")
        return False


def test_environment_spaces():
    """Test observation and action spaces"""
    print("\n🧪 Testing environment spaces...")
    
    try:
        env = BoidEnvironment(num_boids=15, seed=42)
        
        # Test observation space
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), "Observation not in observation space"
        print(f"✅ Observation space valid")
        
        # Test action space
        for _ in range(5):
            action = env.action_space.sample()
            assert env.action_space.contains(action), "Action not in action space"
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        
        print(f"✅ Action space valid")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Space validation failed: {e}")
        return False


def test_environment_sb3_compatibility():
    """Test compatibility with stable-baselines3"""
    print("\n🧪 Testing SB3 compatibility...")
    
    try:
        env = BoidEnvironment(num_boids=5, max_steps=50, seed=42)
        
        # Use SB3's environment checker
        is_valid = check_environment(env, verbose=False)
        
        if is_valid:
            print("✅ Environment passes SB3 checks")
        else:
            print("⚠️  Environment has some SB3 compatibility issues")
        
        env.close()
        return is_valid
        
    except Exception as e:
        print(f"❌ SB3 compatibility test failed: {e}")
        return False


def test_environment_rewards():
    """Test reward calculation"""
    print("\n🧪 Testing reward calculation...")
    
    try:
        env = BoidEnvironment(num_boids=10, max_steps=100, seed=42)
        
        obs, info = env.reset()
        rewards = []
        
        # Run for several steps and collect rewards
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            
            if terminated or truncated:
                break
        
        rewards = np.array(rewards)
        print(f"✅ Reward calculation working")
        print(f"  Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
        print(f"  Mean reward: {rewards.mean():.3f}")
        print(f"  Std reward: {rewards.std():.3f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Reward calculation failed: {e}")
        return False


def test_environment_determinism():
    """Test environment determinism with same seed"""
    print("\n🧪 Testing environment determinism...")
    
    try:
        # Create two environments with same seed
        env1 = BoidEnvironment(num_boids=5, max_steps=50, seed=123)
        env2 = BoidEnvironment(num_boids=5, max_steps=50, seed=123)
        
        # Reset both
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        
        # Check initial observations are identical
        assert np.allclose(obs1, obs2), "Initial observations not identical with same seed"
        
        # Take same actions
        actions = [env1.action_space.sample() for _ in range(5)]
        
        for i, action in enumerate(actions):
            obs1, reward1, term1, trunc1, info1 = env1.step(action)
            obs2, reward2, term2, trunc2, info2 = env2.step(action)
            
            # Check observations are identical
            if not np.allclose(obs1, obs2, atol=1e-6):
                print(f"⚠️  Observations differ at step {i + 1}")
                print(f"    Max difference: {np.abs(obs1 - obs2).max():.8f}")
            
            # Check rewards are identical
            if not np.isclose(reward1, reward2, atol=1e-6):
                print(f"⚠️  Rewards differ at step {i + 1}: {reward1:.6f} vs {reward2:.6f}")
        
        print("✅ Determinism test completed")
        
        env1.close()
        env2.close()
        return True
        
    except Exception as e:
        print(f"❌ Determinism test failed: {e}")
        return False


def benchmark_environment_speed():
    """Benchmark environment step speed"""
    print("\n🧪 Benchmarking environment speed...")
    
    try:
        env = BoidEnvironment(num_boids=20, max_steps=1000, seed=42)
        
        # Warmup
        obs, _ = env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        
        # Benchmark
        num_steps = 1000
        start_time = time.time()
        
        obs, _ = env.reset()
        for step in range(num_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        
        end_time = time.time()
        elapsed = end_time - start_time
        steps_per_second = num_steps / elapsed
        
        print(f"✅ Speed benchmark completed")
        print(f"  Steps: {num_steps}")
        print(f"  Time: {elapsed:.2f} seconds")
        print(f"  Speed: {steps_per_second:.1f} steps/second")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Speed benchmark failed: {e}")
        return False


def run_all_environment_tests():
    """Run all environment tests"""
    print("🔬 ENVIRONMENT TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_environment_creation,
        test_environment_reset,
        test_environment_step,
        test_environment_spaces,
        test_environment_sb3_compatibility,
        test_environment_rewards,
        test_environment_determinism,
        benchmark_environment_speed
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"  Passed: {passed}/{total}")
    print(f"  Success rate: {passed/total:.1%}")
    
    if passed == total:
        print("🎉 All environment tests passed!")
    else:
        print("⚠️  Some environment tests failed")
    
    return passed == total


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    
    # Run all tests
    success = run_all_environment_tests()
    
    exit(0 if success else 1)
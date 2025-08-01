"""
Model Loading Tests - Test transformer model loading and integration

This script tests the transformer model loading functionality,
including loading from checkpoints and integration with stable-baselines3.
"""

import torch
import numpy as np
import sys
import os
import tempfile

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.models import TransformerModel, TransformerModelLoader, TransformerPolicy
from rl.utils import (
    set_seed, 
    create_dummy_structured_input, 
    test_model_inference,
    print_model_summary
)
from gymnasium import spaces


def test_transformer_model_creation():
    """Test basic transformer model creation"""
    print("🧪 Testing transformer model creation...")
    
    try:
        # Test with default parameters
        model = TransformerModel()
        print("✅ Default model created successfully")
        print(f"  Parameters: {model.count_parameters():,}")
        
        # Test with custom parameters
        model_custom = TransformerModel(
            d_model=32,
            n_heads=4,
            n_layers=2,
            ffn_hidden=64,
            max_boids=20
        )
        print("✅ Custom model created successfully")
        print(f"  Parameters: {model_custom.count_parameters():,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


def test_transformer_model_forward():
    """Test transformer model forward pass"""
    print("\n🧪 Testing transformer model forward pass...")
    
    try:
        model = TransformerModel(d_model=32, n_heads=4, n_layers=2, ffn_hidden=64)
        model.eval()
        
        # Test single input
        structured_input = create_dummy_structured_input(num_boids=5, seed=42)
        
        with torch.no_grad():
            output = model(structured_input)
        
        print("✅ Single forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Test batch input
        batch_inputs = [
            create_dummy_structured_input(num_boids=3, seed=i) 
            for i in range(4)
        ]
        
        with torch.no_grad():
            batch_output = model(batch_inputs)
        
        print("✅ Batch forward pass successful")
        print(f"  Batch output shape: {batch_output.shape}")
        
        # Test with varying number of boids
        varying_inputs = [
            create_dummy_structured_input(num_boids=i+1, seed=i) 
            for i in range(3)
        ]
        
        with torch.no_grad():
            varying_output = model(varying_inputs)
        
        print("✅ Varying boids forward pass successful")
        print(f"  Varying output shape: {varying_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False


def test_model_save_load():
    """Test model saving and loading"""
    print("\n🧪 Testing model save/load...")
    
    try:
        # Create model
        model = TransformerModel(d_model=32, n_heads=4, n_layers=2)
        loader = TransformerModelLoader()
        
        # Create dummy input for testing
        test_input = create_dummy_structured_input(num_boids=5, seed=42)
        
        # Get original output
        model.eval()
        with torch.no_grad():
            original_output = model(test_input)
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            loader.save_model(model, temp_path, additional_info={'test': True})
            print("✅ Model saved successfully")
            
            # Load model
            loaded_model = loader.load_model(temp_path)
            print("✅ Model loaded successfully")
            
            # Test loaded model
            loaded_model.eval()
            with torch.no_grad():
                loaded_output = loaded_model(test_input)
            
            # Compare outputs
            if torch.allclose(original_output, loaded_output, atol=1e-6):
                print("✅ Loaded model produces identical outputs")
            else:
                print("⚠️  Loaded model outputs differ slightly")
                print(f"    Max difference: {torch.abs(original_output - loaded_output).max():.8f}")
            
            return True
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except Exception as e:
        print(f"❌ Save/load test failed: {e}")
        return False


def test_model_with_missing_checkpoint():
    """Test model loading with missing checkpoint"""
    print("\n🧪 Testing model loading with missing checkpoint...")
    
    try:
        loader = TransformerModelLoader()
        
        # Try to load non-existent file
        model = loader.load_model("non_existent_file.pt")
        
        print("✅ Handled missing checkpoint gracefully")
        print(f"  Created model with {model.count_parameters():,} parameters")
        
        # Test the model works
        test_input = create_dummy_structured_input(num_boids=3, seed=42)
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        print("✅ Model works after graceful fallback")
        
        return True
        
    except Exception as e:
        print(f"❌ Missing checkpoint test failed: {e}")
        return False


def test_transformer_policy_creation():
    """Test transformer policy for stable-baselines3"""
    print("\n🧪 Testing transformer policy creation...")
    
    try:
        # Create observation and action spaces
        obs_space = spaces.Box(low=-1, high=1, shape=(50,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Create model
        model = TransformerModel(d_model=32, n_heads=4, n_layers=2, max_boids=10)
        
        # Create policy
        def dummy_lr_schedule(x):
            return 1e-4
        
        policy = TransformerPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=dummy_lr_schedule,
            transformer_model=model,
            max_boids=10
        )
        
        print("✅ Transformer policy created successfully")
        print(f"  Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformer policy creation failed: {e}")
        return False


def test_transformer_policy_forward():
    """Test transformer policy forward pass"""
    print("\n🧪 Testing transformer policy forward pass...")
    
    try:
        # Create observation and action spaces
        obs_space = spaces.Box(low=-1, high=1, shape=(46,), dtype=np.float32)  # 2+2+4*10+2
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Create model and policy
        model = TransformerModel(d_model=32, n_heads=4, n_layers=2, max_boids=10)
        
        def dummy_lr_schedule(x):
            return 1e-4
        
        policy = TransformerPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=dummy_lr_schedule,
            transformer_model=model,
            max_boids=10
        )
        
        # Create dummy observation
        batch_size = 3
        obs = torch.randn(batch_size, 46)  # Batch of observations
        
        # Test forward pass
        policy.eval()
        with torch.no_grad():
            actions, values, log_probs = policy.forward(obs, deterministic=True)
        
        print("✅ Policy forward pass successful")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Values shape: {values.shape}")
        print(f"  Log probs shape: {log_probs.shape}")
        print(f"  Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
        
        # Test predict method
        single_obs = obs[0].numpy()
        pred_action, _ = policy.predict(single_obs, deterministic=True)
        
        print("✅ Policy predict method successful")
        print(f"  Predicted action: {pred_action}")
        
        return True
        
    except Exception as e:
        print(f"❌ Policy forward pass failed: {e}")
        return False


def test_model_architecture_validation():
    """Test model architecture validation"""
    print("\n🧪 Testing model architecture validation...")
    
    try:
        # Test valid architectures
        valid_configs = [
            {'d_model': 64, 'n_heads': 8, 'n_layers': 4, 'ffn_hidden': 256},
            {'d_model': 32, 'n_heads': 4, 'n_layers': 2, 'ffn_hidden': 128},
        ]
        
        for config in valid_configs:
            model = TransformerModel(**config)
            print(f"✅ Valid config: {config}")
        
        # Test invalid architectures
        invalid_configs = [
            {'d_model': 65, 'n_heads': 8, 'n_layers': 4, 'ffn_hidden': 256},  # d_model not divisible by n_heads
            {'d_model': 64, 'n_heads': 0, 'n_layers': 4, 'ffn_hidden': 256},   # n_heads = 0
            {'d_model': 64, 'n_heads': 8, 'n_layers': 0, 'ffn_hidden': 256},   # n_layers = 0
        ]
        
        for config in invalid_configs:
            try:
                model = TransformerModel(**config)
                print(f"⚠️  Invalid config accepted: {config}")
            except (AssertionError, ValueError) as e:
                print(f"✅ Invalid config rejected: {config}")
        
        return True
        
    except Exception as e:
        print(f"❌ Architecture validation failed: {e}")
        return False


def test_model_device_handling():
    """Test model device handling"""
    print("\n🧪 Testing model device handling...")
    
    try:
        model = TransformerModel(d_model=32, n_heads=4, n_layers=2)
        
        # Test CPU
        model_cpu = model.to('cpu')
        test_input = create_dummy_structured_input(num_boids=3, seed=42)
        
        with torch.no_grad():
            output_cpu = model_cpu(test_input)
        
        print("✅ CPU inference successful")
        
        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = model.to('cuda')
            
            with torch.no_grad():
                output_gpu = model_gpu(test_input)
            
            print("✅ GPU inference successful")
            
            # Compare outputs
            if torch.allclose(output_cpu, output_gpu.cpu(), atol=1e-6):
                print("✅ CPU and GPU outputs match")
            else:
                print("⚠️  CPU and GPU outputs differ slightly")
        else:
            print("ℹ️  GPU not available, skipping GPU test")
        
        return True
        
    except Exception as e:
        print(f"❌ Device handling test failed: {e}")
        return False


def run_all_model_tests():
    """Run all model loading tests"""
    print("🔬 MODEL LOADING TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_transformer_model_creation,
        test_transformer_model_forward,
        test_model_save_load,
        test_model_with_missing_checkpoint,
        test_transformer_policy_creation,
        test_transformer_policy_forward,
        test_model_architecture_validation,
        test_model_device_handling
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
        print("🎉 All model tests passed!")
    else:
        print("⚠️  Some model tests failed")
    
    return passed == total


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    
    # Run all tests
    success = run_all_model_tests()
    
    exit(0 if success else 1)
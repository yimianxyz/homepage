"""
Example Usage - Demonstration of RL training system

This script shows how to use the RL training system for different scenarios:
1. Quick test training
2. Loading pre-trained models
3. Custom configuration
4. Evaluation
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.training import PPOTrainer, TrainingConfig
from rl.training.config import get_quick_test_config, get_small_scale_config
from rl.environment import BoidEnvironment
from rl.models import TransformerModelLoader
from rl.utils import set_seed


def example_quick_test():
    """Example 1: Quick test training"""
    print("🚀 Example 1: Quick Test Training")
    print("=" * 40)
    
    # Use predefined quick test configuration
    config = get_quick_test_config()
    
    # Override some settings for demo
    config.total_timesteps = 1000  # Very short for demo
    config.model_checkpoint = "best_model.pt"  # Will create new if not found
    
    print(f"Configuration:")
    print(f"  Boids: {config.num_boids}")
    print(f"  Canvas: {config.canvas_width}x{config.canvas_height}")
    print(f"  Timesteps: {config.total_timesteps:,}")
    print(f"  Model: {config.model_checkpoint}")
    
    # Create and run trainer
    trainer = PPOTrainer(config)
    
    try:
        print(f"\n🏋️  Starting training...")
        trainer.train()
        print(f"✅ Training completed!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
    finally:
        trainer.cleanup()


def example_custom_config():
    """Example 2: Custom configuration"""
    print("\n🚀 Example 2: Custom Configuration")
    print("=" * 40)
    
    # Create custom configuration
    config = TrainingConfig(
        # Environment settings
        num_boids=15,
        canvas_width=600,
        canvas_height=400,
        max_episode_steps=300,
        
        # Model settings
        model_checkpoint="best_model.pt",
        model_config={
            'd_model': 48,
            'n_heads': 6,
            'n_layers': 3,
            'ffn_hidden': 192,
            'max_boids': 15,
            'dropout': 0.1
        },
        
        # Training settings
        total_timesteps=5000,
        learning_rate=2e-4,
        n_steps=512,
        batch_size=32,
        n_epochs=5,
        n_envs=2,
        
        # Saving and logging
        save_freq=1000,
        eval_freq=1000,
        log_dir="custom_logs",
        save_dir="custom_models"
    )
    
    print(f"Custom configuration created:")
    print(f"  Model architecture: {config.model_config['d_model']}×{config.model_config['n_heads']}×{config.model_config['n_layers']}")
    print(f"  Environments: {config.n_envs}")
    print(f"  Learning rate: {config.learning_rate}")


def example_environment_test():
    """Example 3: Test environment standalone"""
    print("\n🚀 Example 3: Environment Test")
    print("=" * 40)
    
    # Create environment
    env = BoidEnvironment(
        num_boids=10,
        canvas_width=400,
        canvas_height=300,
        max_steps=100,
        seed=42
    )
    
    print(f"Environment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Run a few episodes
    for episode in range(3):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Initial boids: {info['boids_remaining']}")
        
        while steps < 20:  # Limit steps for demo
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if steps % 5 == 0 or terminated or truncated:
                print(f"    Step {steps}: Boids={info['boids_remaining']}, Reward={reward:.3f}")
            
            if terminated or truncated:
                break
        
        print(f"  Episode reward: {episode_reward:.3f}")
        print(f"  Episode length: {steps}")
        print(f"  Boids caught: {info.get('total_boids_caught', 0)}")
    
    env.close()


def example_model_loading():
    """Example 4: Model loading and testing"""
    print("\n🚀 Example 4: Model Loading Test")
    print("=" * 40)
    
    from rl.models import TransformerModel
    from rl.utils import create_dummy_structured_input, test_model_inference
    
    # Try to load existing model
    loader = TransformerModelLoader()
    
    print("Attempting to load best_model.pt...")
    model = loader.load_model("best_model.pt")
    
    print(f"Model loaded:")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Architecture: {model.get_architecture_info()}")
    
    # Test model inference
    print(f"\nTesting model inference...")
    outputs = test_model_inference(model, num_tests=3, num_boids=5)
    
    for i, output in enumerate(outputs):
        if output is not None:
            print(f"  Test {i+1}: {output}")
        else:
            print(f"  Test {i+1}: Failed")


def example_evaluation():
    """Example 5: Model evaluation"""
    print("\n🚀 Example 5: Model Evaluation")
    print("=" * 40)
    
    # This would evaluate a trained model
    # For demo, we'll just show the structure
    
    config = get_quick_test_config()
    config.total_timesteps = 100  # Minimal training first
    
    trainer = PPOTrainer(config)
    
    print("This example would:")
    print("1. Load a trained model")
    print("2. Run evaluation episodes")  
    print("3. Calculate performance metrics")
    print("4. Generate evaluation report")
    
    # Uncomment to actually run evaluation:
    # trainer.evaluate("models/best_model", n_episodes=5, render=False)
    
    trainer.cleanup()


def main():
    """Main function to run examples"""
    print("🎯 RL TRAINING SYSTEM EXAMPLES")
    print("=" * 50)
    
    # Set random seed
    set_seed(42)
    
    examples = [
        ("Environment Test", example_environment_test),
        ("Model Loading Test", example_model_loading),
        ("Custom Configuration", example_custom_config),
        # ("Quick Training", example_quick_test),  # Commented out for demo
        # ("Evaluation", example_evaluation),      # Commented out for demo
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✨ Examples completed!")
    print(f"\nTo run actual training:")
    print(f"  python -m rl.training.ppo_trainer --config quick_test")
    print(f"\nTo run tests:")
    print(f"  python -m rl.tests.run_tests")


if __name__ == "__main__":
    main()
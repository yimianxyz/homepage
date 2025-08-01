#!/usr/bin/env python3
"""
Debug Training Script - Small RL training with comprehensive monitoring

This script runs a small-scale RL training session with extensive debugging
information to verify system integrity before production training.
"""

import sys
import os
import tempfile
import shutil
import numpy as np
import torch
import time
from typing import Dict, Any, List
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.environment import BoidEnvironment
from rl.models import TransformerModel, TransformerModelLoader, TransformerPolicy
from rl.training import PPOTrainer, TrainingConfig
from rl.utils import set_seed
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class DebugCallback(BaseCallback):
    """Custom callback for detailed debugging during training"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.step_rewards = []
        self.actions_history = []
        self.observations_history = []
        self.info_history = []
        
    def _on_step(self) -> bool:
        # Log every step during training
        if len(self.locals.get('rewards', [])) > 0:
            rewards = self.locals['rewards']
            actions = self.locals.get('actions', None)
            obs = self.locals.get('obs', None)
            infos = self.locals.get('infos', [])
            
            # Store step data
            self.step_rewards.extend(rewards)
            if actions is not None:
                self.actions_history.extend(actions)
            if obs is not None:
                self.observations_history.append(obs)
            self.info_history.extend(infos)
            
            # Print detailed step info every 10 steps
            if self.num_timesteps % 10 == 0:
                print(f"\n--- Step {self.num_timesteps} Debug Info ---")
                print(f"Rewards: {rewards}")
                if actions is not None:
                    print(f"Actions (first env): {actions[0] if len(actions) > 0 else 'None'}")
                if infos and len(infos) > 0 and infos[0]:
                    info = infos[0]
                    print(f"Boids remaining: {info.get('boids_remaining', 'N/A')}")
                    print(f"Total reward: {info.get('total_reward', 'N/A'):.4f}")
                    print(f"Reward breakdown: {info.get('reward_breakdown', {})}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        print(f"\n=== Rollout {len(self.episode_rewards) + 1} Complete ===")
        if hasattr(self.training_env, 'get_attr'):
            try:
                env_infos = self.training_env.get_attr('get_info')
                for i, info in enumerate(env_infos):
                    print(f"Env {i}: {info}")
            except:
                pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics"""
        return {
            'total_steps': len(self.step_rewards),
            'mean_step_reward': np.mean(self.step_rewards) if self.step_rewards else 0,
            'std_step_reward': np.std(self.step_rewards) if self.step_rewards else 0,
            'min_step_reward': np.min(self.step_rewards) if self.step_rewards else 0,
            'max_step_reward': np.max(self.step_rewards) if self.step_rewards else 0,
            'total_episodes': len(self.episode_rewards),
            'action_samples': self.actions_history[:10] if self.actions_history else [],
        }


def analyze_environment_behavior():
    """Analyze basic environment behavior before training"""
    print("🔍 ANALYZING ENVIRONMENT BEHAVIOR")
    print("=" * 60)
    
    # Create test environment
    env = BoidEnvironment(num_boids=5, max_steps=20, seed=42)
    
    print(f"Environment specs:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Max boids: {env.num_boids}")
    
    # Test random actions
    obs, info = env.reset(seed=42)
    print(f"\nInitial state:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  Initial info: {info}")
    
    total_reward = 0
    rewards_history = []
    
    for step in range(10):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        rewards_history.append(reward)
        
        print(f"  Step {step+1}: action={action}, reward={reward:.4f}, "
              f"boids={info.get('boids_remaining', 'N/A')}, "
              f"terminated={terminated}")
        
        if terminated or truncated:
            print(f"  Episode ended: terminated={terminated}, truncated={truncated}")
            break
    
    print(f"\nRandom policy summary:")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Mean reward: {np.mean(rewards_history):.4f}")
    print(f"  Reward std: {np.std(rewards_history):.4f}")
    
    env.close()
    return {
        'total_reward': total_reward,
        'mean_reward': np.mean(rewards_history),
        'reward_std': np.std(rewards_history)
    }


def test_model_loading():
    """Test model loading with detailed diagnostics"""
    print("\n🔍 TESTING MODEL LOADING")
    print("=" * 60)
    
    # Check if best_model.pt exists
    model_path = "/home/iotcat/homepage2/best_model.pt"
    
    if os.path.exists(model_path):
        print(f"✅ Found best_model.pt at {model_path}")
        
        # Try loading
        loader = TransformerModelLoader()
        try:
            model = loader.load_model(model_path)
            print(f"✅ Successfully loaded model:")
            print(f"  Parameters: {model.count_parameters():,}")
            print(f"  Architecture: {model.get_architecture_info()}")
            
            # Test model inference
            dummy_input = {
                'context': {'canvasWidth': 0.5, 'canvasHeight': 0.6},
                'predator': {'velX': 0.1, 'velY': -0.2},
                'boids': [
                    {'relX': 0.3, 'relY': 0.4, 'velX': 0.1, 'velY': 0.2},
                    {'relX': -0.2, 'relY': 0.1, 'velX': -0.1, 'velY': 0.3}
                ]
            }
            
            with torch.no_grad():
                output = model(dummy_input)
                print(f"  Test inference output: {output}")
                print(f"  Output shape: {output.shape}")
                print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Creating new model instead...")
            
    else:
        print(f"⚠️  best_model.pt not found at {model_path}")
        print("Creating new model for testing...")
    
    # Create new model
    model = TransformerModel(d_model=32, n_heads=4, n_layers=2, max_boids=5)
    print(f"✅ Created new model:")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Architecture: {model.get_architecture_info()}")
    
    return model


def run_debug_training():
    """Run small-scale training with comprehensive monitoring"""
    print("\n🚀 STARTING DEBUG TRAINING")
    print("=" * 60)
    
    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp(prefix="rl_debug_")
    print(f"Debug outputs will be saved to: {temp_dir}")
    
    try:
        # Load model
        model = test_model_loading()
        
        # Create training configuration
        config = TrainingConfig(
            # Small scale for debugging
            num_boids=3,
            canvas_width=200,
            canvas_height=150,
            max_episode_steps=30,
            
            # Short training
            total_timesteps=200,
            n_steps=32,
            batch_size=16,
            n_epochs=2,
            n_envs=1,  # Use single environment to avoid multiprocessing issues
            
            # Learning parameters
            learning_rate=3e-4,
            
            # Monitoring
            save_freq=100,
            eval_freq=100,
            eval_episodes=5,
            
            # Paths
            log_dir=os.path.join(temp_dir, "logs"),
            save_dir=os.path.join(temp_dir, "models"),
            tensorboard_log=os.path.join(temp_dir, "tensorboard"),
            
            # Model (will use the loaded model)
            model_checkpoint="debug_training"  # Signal to use provided model
        )
        
        print(f"Training configuration:")
        for key, value in config.__dict__.items():
            print(f"  {key}: {value}")
        
        # Create trainer
        trainer = PPOTrainer(config)
        
        # Manually set the model instead of loading
        trainer.transformer_model = model
        
        # Setup training components
        print("\n📋 Setting up training components...")
        trainer.setup_environment()
        print(f"✅ Environment setup complete")
        
        # Skip model loading since we're providing one
        print("✅ Using provided model")
        
        trainer.setup_ppo()
        print(f"✅ PPO agent setup complete")
        
        # Create debug callback
        debug_callback = DebugCallback(verbose=1)
        
        # Add some custom monitoring
        class VerboseCallback(BaseCallback):
            def _on_training_start(self):
                print("\n🎯 TRAINING STARTED")
                print(f"Model device: {self.model.device}")
                print(f"Environment: {self.training_env}")
                return True
                
            def _on_rollout_start(self):
                print(f"\n--- Starting rollout at step {self.num_timesteps} ---")
                return True
                
            def _on_rollout_end(self):
                print(f"--- Rollout complete at step {self.num_timesteps} ---")
                if hasattr(self.locals, 'rollout_buffer'):
                    buffer = self.locals['rollout_buffer']
                    if hasattr(buffer, 'rewards'):
                        rewards = buffer.rewards.flatten()
                        print(f"Rollout rewards: mean={rewards.mean():.4f}, "
                              f"std={rewards.std():.4f}, "
                              f"min={rewards.min():.4f}, "
                              f"max={rewards.max():.4f}")
                return True
            
            def _on_step(self) -> bool:
                return True
        
        verbose_callback = VerboseCallback()
        
        print(f"\n🎯 Starting training for {config.total_timesteps} timesteps...")
        
        # Run training with detailed monitoring
        start_time = time.time()
        
        trainer.ppo_agent.learn(
            total_timesteps=config.total_timesteps,
            callback=[debug_callback, verbose_callback],
            progress_bar=True,
            tb_log_name="debug_run"
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ TRAINING COMPLETED")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Training speed: {config.total_timesteps/training_time:.1f} steps/second")
        
        # Get training summary
        summary = debug_callback.get_summary()
        print(f"\n📊 TRAINING SUMMARY:")
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, list) and len(value) > 0:
                print(f"  {key} (first 3): {value[:3]}")
            else:
                print(f"  {key}: {value}")
        
        # Test trained model
        print(f"\n🧪 TESTING TRAINED MODEL")
        print("=" * 40)
        
        # Create test environment
        test_env = BoidEnvironment(num_boids=3, max_steps=30, seed=123)
        
        # Evaluate trained policy
        print("Evaluating trained policy...")
        mean_reward, std_reward = evaluate_policy(
            trainer.ppo_agent, 
            test_env, 
            n_eval_episodes=5,
            deterministic=True
        )
        
        print(f"Evaluation results:")
        print(f"  Mean episode reward: {mean_reward:.4f}")
        print(f"  Std episode reward: {std_reward:.4f}")
        
        # Manual episode test with detailed logging
        print(f"\nManual episode test:")
        obs, info = test_env.reset(seed=456)
        episode_reward = 0
        step_count = 0
        
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        
        for step in range(20):
            action, _states = trainer.ppo_agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            print(f"  Step {step+1}: action={action}, reward={reward:.4f}, "
                  f"boids={info.get('boids_remaining', 'N/A')}")
            
            if terminated or truncated:
                print(f"  Episode ended after {step_count} steps")
                break
        
        print(f"Final episode reward: {episode_reward:.4f}")
        
        # Save trained model
        model_save_path = os.path.join(temp_dir, "debug_trained_model")
        trainer.ppo_agent.save(model_save_path)
        print(f"Trained model saved to: {model_save_path}")
        
        # Cleanup
        test_env.close()
        trainer.cleanup()
        
        print(f"\n🎉 DEBUG TRAINING SUCCESSFUL")
        print(f"All outputs saved to: {temp_dir}")
        print(f"To examine logs: ls -la {temp_dir}")
        
        return {
            'success': True,
            'training_time': training_time,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'temp_dir': temp_dir,
            'summary': summary
        }
        
    except Exception as e:
        print(f"\n❌ DEBUG TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'temp_dir': temp_dir
        }


def main():
    """Main debug training function"""
    print("🔬 RL SYSTEM DEBUG TRAINING")
    print("=" * 80)
    print("Purpose: Verify system integrity before production training")
    print("This will run a small-scale training with comprehensive monitoring\n")
    
    # Set deterministic seed
    set_seed(42)
    
    # Step 1: Analyze environment
    env_analysis = analyze_environment_behavior()
    
    # Step 2: Run debug training
    training_result = run_debug_training()
    
    # Final summary
    print(f"\n🏁 FINAL DEBUG SUMMARY")
    print("=" * 80)
    
    if training_result['success']:
        print("✅ All systems operational!")
        print(f"✅ Environment analysis: rewards range from {env_analysis['mean_reward']:.4f}")
        print(f"✅ Training completed successfully")
        print(f"✅ Model evaluation: {training_result['mean_reward']:.4f} ± {training_result['std_reward']:.4f}")
        print(f"✅ Training speed: {200/training_result['training_time']:.1f} steps/sec")
        print(f"\n🚀 System ready for production training!")
        print(f"Debug files available at: {training_result['temp_dir']}")
    else:
        print("❌ Issues detected during debug training")
        print(f"Error: {training_result.get('error', 'Unknown error')}")
        print("🔧 Please review the logs and fix issues before production training")
    
    return training_result['success']


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
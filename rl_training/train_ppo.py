"""
PPO Training Pipeline - Fine-tune pretrained transformer with reinforcement learning

This script implements the complete PPO training pipeline that:
1. Loads pretrained transformer weights from best_model.pt
2. Creates PPO-compatible environment and policy
3. Fine-tunes the model using PPO
4. Evaluates using the existing evaluation system
5. Saves checkpoints and results

Usage:
    python rl_training/train_ppo.py --pretrained_path checkpoints/best_model.pt --total_timesteps 100000
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_training.ppo_environment import create_boids_environment
from rl_training.observation_wrapper import create_wrapped_environment, create_custom_policy_class
from rl_training.ppo_policy_wrapper import create_policy_wrapper, evaluate_ppo_model
from evaluation.policy_evaluator import PolicyEvaluator

class EvaluationCallback(BaseCallback):
    """
    Custom callback for periodic evaluation using existing evaluation system
    """
    
    def __init__(self, eval_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.evaluations = []
        self.best_catch_rate = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_model()
        return True
    
    def _evaluate_model(self):
        """Evaluate model using existing evaluation system"""
        try:
            # Create policy wrapper
            policy_wrapper = create_policy_wrapper(self.model, wrapper_type="auto")
            
            # Run evaluation
            evaluator = PolicyEvaluator()
            results = evaluator.evaluate_policy(
                policy_wrapper, 
                f"PPO-Step-{self.n_calls}"
            )
            
            # Track results
            catch_rate = results.overall_catch_rate
            self.evaluations.append({
                'step': self.n_calls,
                'catch_rate': catch_rate,
                'std_catch_rate': results.overall_std_catch_rate,
                'episodes': results.successful_episodes,
                'evaluation_time': results.evaluation_time_seconds
            })
            
            # Save best model
            if catch_rate > self.best_catch_rate:
                self.best_catch_rate = catch_rate
                best_path = os.path.join(self.logger.get_dir(), "best_ppo_model.zip")
                self.model.save(best_path)
                print(f"🎯 New best model saved: catch_rate={catch_rate:.4f}")
            
            # Log to tensorboard/logger
            self.logger.record("eval/catch_rate", catch_rate)
            self.logger.record("eval/std_catch_rate", results.overall_std_catch_rate)
            self.logger.record("eval/successful_episodes", results.successful_episodes)
            
            print(f"📊 Evaluation at step {self.n_calls}: catch_rate={catch_rate:.4f}±{results.overall_std_catch_rate:.4f}")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")

def create_training_environment(config: Dict[str, Any]) -> DummyVecEnv:
    """
    Create vectorized training environment
    
    Args:
        config: Environment configuration
        
    Returns:
        Vectorized environment
    """
    def make_env():
        return create_wrapped_environment(config)
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    print(f"✓ Training environment created:")
    print(f"  Config: {config}")
    print(f"  Vectorized: {env.num_envs} environment(s)")
    
    return env

def setup_ppo_model(
    env, 
    pretrained_path: str,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    verbose: int = 1
) -> PPO:
    """
    Setup PPO model with pretrained transformer policy
    
    Args:
        env: Training environment
        pretrained_path: Path to pretrained transformer checkpoint
        Additional PPO hyperparameters
        
    Returns:
        Configured PPO model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create custom policy class with pretrained weights
    PolicyClass = create_custom_policy_class(pretrained_path)
    
    # Create PPO model
    model = PPO(
        policy=PolicyClass,
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=verbose,
        device=device,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # Set environment wrapper reference for transformer features extractor
    if hasattr(model.policy, 'features_extractor'):
        if hasattr(env, 'envs') and len(env.envs) > 0:
            model.policy.features_extractor.set_env_wrapper(env.envs[0])
    
    print(f"✓ PPO model configured:")
    print(f"  Policy: Custom transformer policy")
    print(f"  Pretrained weights: {pretrained_path}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Steps per update: {n_steps}")
    print(f"  Batch size: {batch_size}")
    
    return model

def train_ppo_model(
    pretrained_path: str = "checkpoints/best_model.pt",
    total_timesteps: int = 100000,
    eval_freq: int = 10000,
    save_freq: int = 25000,
    log_dir: str = "ppo_logs",
    env_config: Optional[Dict[str, Any]] = None
):
    """
    Main training function
    
    Args:
        pretrained_path: Path to pretrained transformer checkpoint
        total_timesteps: Total training timesteps
        eval_freq: Evaluation frequency
        save_freq: Checkpoint save frequency
        log_dir: Logging directory
        env_config: Environment configuration
    """
    
    print("🚀 Starting PPO Training Pipeline")
    print("=" * 60)
    
    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{log_dir}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Default environment configuration
    if env_config is None:
        env_config = {
            'canvas_width': 800,
            'canvas_height': 600,
            'initial_boids': 20,
            'max_steps': 1500,
            'seed': 42
        }
    
    print(f"📁 Log directory: {log_dir}")
    print(f"🎯 Training configuration:")
    print(f"  Pretrained model: {pretrained_path}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Evaluation frequency: {eval_freq:,}")
    print(f"  Environment: {env_config}")
    
    # Verify pretrained model exists
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
    
    try:
        # Create training environment
        print(f"\n📦 Setting up environment...")
        env = create_training_environment(env_config)
        
        # Setup PPO model
        print(f"\n🤖 Setting up PPO model...")
        model = setup_ppo_model(env, pretrained_path)
        
        # Configure logging
        model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))
        
        # Create callbacks
        print(f"\n📊 Setting up callbacks...")
        callbacks = [
            EvaluationCallback(eval_freq=eval_freq, verbose=1),
            CheckpointCallback(save_freq=save_freq, save_path=log_dir, name_prefix="ppo_checkpoint")
        ]
        
        # Baseline evaluation
        print(f"\n📈 Running baseline evaluation...")
        baseline_results = evaluate_ppo_model(model, "PPO-Baseline (Pretrained)")
        baseline_catch_rate = baseline_results.overall_catch_rate
        
        print(f"🎯 Baseline performance: {baseline_catch_rate:.4f}±{baseline_results.overall_std_catch_rate:.4f}")
        
        # Save baseline results
        baseline_path = os.path.join(log_dir, "baseline_evaluation.json")
        baseline_results_dict = {
            'policy_name': baseline_results.policy_name,
            'overall_catch_rate': baseline_results.overall_catch_rate,
            'overall_std_catch_rate': baseline_results.overall_std_catch_rate,
            'total_episodes': baseline_results.total_episodes,
            'successful_episodes': baseline_results.successful_episodes,
            'evaluation_time_seconds': baseline_results.evaluation_time_seconds
        }
        
        with open(baseline_path, 'w') as f:
            json.dump(baseline_results_dict, f, indent=2)
        
        # Start training
        print(f"\n🎓 Starting PPO training...")
        print("=" * 60)
        
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            tb_log_name="ppo_transformer"
        )
        training_time = time.time() - start_time
        
        print(f"\n✅ Training completed!")
        print(f"⏱️  Training time: {training_time:.1f} seconds")
        
        # Final evaluation
        print(f"\n📈 Running final evaluation...")
        final_results = evaluate_ppo_model(model, "PPO-Final")
        final_catch_rate = final_results.overall_catch_rate
        
        improvement = final_catch_rate - baseline_catch_rate
        print(f"🎯 Final performance: {final_catch_rate:.4f}±{final_results.overall_std_catch_rate:.4f}")
        print(f"📊 Improvement: {improvement:+.4f} ({improvement/baseline_catch_rate*100:+.1f}%)")
        
        # Save final model and results
        final_model_path = os.path.join(log_dir, "final_ppo_model.zip")
        model.save(final_model_path)
        
        final_results_path = os.path.join(log_dir, "final_evaluation.json")
        final_results_dict = {
            'policy_name': final_results.policy_name,
            'overall_catch_rate': final_results.overall_catch_rate,
            'overall_std_catch_rate': final_results.overall_std_catch_rate,
            'total_episodes': final_results.total_episodes,
            'successful_episodes': final_results.successful_episodes,
            'evaluation_time_seconds': final_results.evaluation_time_seconds,
            'baseline_catch_rate': baseline_catch_rate,
            'improvement': improvement,
            'improvement_percentage': improvement/baseline_catch_rate*100,
            'training_time_seconds': training_time,
            'total_timesteps': total_timesteps
        }
        
        with open(final_results_path, 'w') as f:
            json.dump(final_results_dict, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"  Final model: {final_model_path}")
        print(f"  Evaluation: {final_results_path}")
        print(f"  Logs: {log_dir}")
        
        print(f"\n🎉 PPO training pipeline completed successfully!")
        print("=" * 60)
        
        return model, final_results
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="PPO Training Pipeline")
    
    parser.add_argument(
        "--pretrained_path", 
        type=str, 
        default="checkpoints/best_model.pt",
        help="Path to pretrained transformer checkpoint"
    )
    parser.add_argument(
        "--total_timesteps", 
        type=int, 
        default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--eval_freq", 
        type=int, 
        default=10000,
        help="Evaluation frequency"
    )
    parser.add_argument(
        "--save_freq", 
        type=int, 
        default=25000,
        help="Checkpoint save frequency"
    )
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="ppo_logs",
        help="Logging directory prefix"
    )
    parser.add_argument(
        "--canvas_width", 
        type=int, 
        default=800,
        help="Canvas width"
    )
    parser.add_argument(
        "--canvas_height", 
        type=int, 
        default=600,
        help="Canvas height"
    )
    parser.add_argument(
        "--initial_boids", 
        type=int, 
        default=20,
        help="Initial number of boids"
    )
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=1500,
        help="Maximum steps per episode"
    )
    
    args = parser.parse_args()
    
    # Environment configuration
    env_config = {
        'canvas_width': args.canvas_width,
        'canvas_height': args.canvas_height,
        'initial_boids': args.initial_boids,
        'max_steps': args.max_steps,
        'seed': 42
    }
    
    # Run training
    train_ppo_model(
        pretrained_path=args.pretrained_path,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        log_dir=args.log_dir,
        env_config=env_config
    )

if __name__ == "__main__":
    main()
"""
PPO Trainer - Main training script for reinforcement learning

This module provides the main PPO trainer that integrates all components:
environment, model loading, policy, and training loop.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Callable
import os
import time
import json
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Stable-baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv

# Local imports
from ..environment import BoidEnvironment
from ..models import TransformerModelLoader, TransformerPolicy
from .config import TrainingConfig


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback to track training metrics specific to boid environment
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.boids_caught = []
        self.success_rate = []
        
    def _on_step(self) -> bool:
        # Get info from environments
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    # Episode finished
                    self.episode_rewards.append(info.get('episode', {}).get('r', 0))
                    self.episode_lengths.append(info.get('episode', {}).get('l', 0))
                    
                    # Boid-specific metrics
                    if 'total_boids_caught' in info:
                        self.boids_caught.append(info['total_boids_caught'])
                        total_boids = info.get('total_boids', 20)
                        success = 1.0 if info['total_boids_caught'] >= total_boids * 0.8 else 0.0
                        self.success_rate.append(success)
        
        return True
    
    def _on_training_end(self) -> None:
        """Log final metrics"""
        if len(self.episode_rewards) > 0:
            print(f"\nTraining completed!")
            print(f"Average episode reward: {np.mean(self.episode_rewards[-100:]):.2f}")
            print(f"Average episode length: {np.mean(self.episode_lengths[-100:]):.1f}")
            print(f"Average boids caught: {np.mean(self.boids_caught[-100:]):.1f}")
            print(f"Success rate (80%+ boids): {np.mean(self.success_rate[-100:]):.2%}")


class PPOTrainer:
    """
    Main PPO trainer for transformer models in boid environment
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize PPO trainer
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model_loader = TransformerModelLoader()
        self.model = None
        self.env = None
        self.ppo_agent = None
        
        # Set device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        print(f"PPOTrainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Config: {config.num_boids} boids, {config.total_timesteps:,} timesteps")
        print(f"  Model checkpoint: {config.model_checkpoint}")
    
    def setup_environment(self) -> None:
        """Setup the training environment"""
        print("Setting up environment...")
        
        # Create environment creation function
        def make_env(rank: int = 0):
            def _init():
                env = BoidEnvironment(
                    num_boids=self.config.num_boids,
                    canvas_width=self.config.canvas_width,
                    canvas_height=self.config.canvas_height,
                    max_steps=self.config.max_episode_steps,
                    seed=self.config.seed + rank
                )
                return env
            return _init
        
        # Create vectorized environment
        if self.config.n_envs == 1:
            self.env = DummyVecEnv([make_env(0)])
        else:
            self.env = SubprocVecEnv([make_env(i) for i in range(self.config.n_envs)])
        
        # Optionally normalize observations
        # self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True, training=True)
        
        print(f"Created {self.config.n_envs} parallel environments")
    
    def load_model(self) -> Optional[torch.nn.Module]:
        """Load pre-trained transformer model"""
        print(f"Loading model from {self.config.model_checkpoint}...")
        
        # Check if checkpoint exists
        if not os.path.exists(self.config.model_checkpoint):
            print(f"Warning: Checkpoint {self.config.model_checkpoint} not found!")
            print("Creating new model with default configuration...")
            transformer_model = self.model_loader.create_model_from_config(
                self.config.model_config, self.device
            )
        else:
            # Load pre-trained model
            transformer_model = self.model_loader.load_model(
                self.config.model_checkpoint, self.device
            )
        
        self.model = transformer_model
        return transformer_model
    
    def setup_ppo(self) -> None:
        """Setup PPO agent with custom policy"""
        print("Setting up PPO agent...")
        
        # Create custom policy using the loaded transformer
        policy_kwargs = {
            'transformer_model': self.model,
            'max_boids': self.config.num_boids,
        }
        
        # Create PPO agent
        self.ppo_agent = PPO(
            policy=TransformerPolicy,
            env=self.env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            clip_range_vf=self.config.clip_range_vf,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.config.tensorboard_log,
            device=self.device,
            seed=self.config.seed,
            verbose=1
        )
        
        print(f"PPO agent created with {self.ppo_agent.policy.transformer.count_parameters():,} parameters")
    
    def setup_callbacks(self) -> list:
        """Setup training callbacks"""
        callbacks = []
        
        # Training metrics callback
        metrics_callback = TrainingMetricsCallback(verbose=1)
        callbacks.append(metrics_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=self.config.save_dir,
            name_prefix="ppo_boid_transformer"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        if self.config.eval_freq > 0:
            # Create evaluation environment
            eval_env = BoidEnvironment(
                num_boids=self.config.num_boids,
                canvas_width=self.config.canvas_width,
                canvas_height=self.config.canvas_height,
                max_steps=self.config.max_episode_steps,
                seed=self.config.seed + 1000
            )
            
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=self.config.save_dir,
                log_path=self.config.log_dir,
                eval_freq=self.config.eval_freq,
                n_eval_episodes=self.config.eval_episodes,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        return callbacks
    
    def train(self) -> None:
        """Run the complete training pipeline"""
        print("Starting PPO training pipeline...")
        
        # Setup all components
        self.setup_environment()
        self.load_model()
        self.setup_ppo()
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Save training configuration
        config_path = os.path.join(self.config.save_dir, "training_config.json")
        self.config.save(config_path)
        
        # Start training
        print(f"\n🚀 Starting training for {self.config.total_timesteps:,} timesteps...")
        print(f"📊 Logs: {self.config.tensorboard_log}")
        print(f"💾 Models: {self.config.save_dir}")
        
        start_time = time.time()
        
        try:
            self.ppo_agent.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                log_interval=self.config.log_interval,
                progress_bar=True
            )
            
        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            raise
        
        finally:
            training_time = time.time() - start_time
            print(f"\n⏱️  Training time: {training_time/3600:.1f} hours")
            
            # Save final model
            final_model_path = os.path.join(self.config.save_dir, "final_model")
            self.ppo_agent.save(final_model_path)
            print(f"💾 Final model saved to {final_model_path}")
            
            # Save transformer model separately
            transformer_path = os.path.join(self.config.save_dir, "final_transformer.pt")
            self.model_loader.save_model(
                self.ppo_agent.policy.transformer, 
                transformer_path,
                additional_info={
                    'training_timesteps': self.config.total_timesteps,
                    'training_time': training_time,
                    'config': self.config.to_dict()
                }
            )
            print(f"🧠 Transformer model saved to {transformer_path}")
    
    def evaluate(self, 
                 model_path: str, 
                 n_episodes: int = 10,
                 render: bool = False) -> Dict[str, float]:
        """
        Evaluate a trained model
        
        Args:
            model_path: Path to the saved model
            n_episodes: Number of episodes to evaluate
            render: Whether to render during evaluation
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        print(f"Evaluating model from {model_path}...")
        
        # Load the model
        model = PPO.load(model_path, device=self.device)
        
        # Create evaluation environment
        eval_env = BoidEnvironment(
            num_boids=self.config.num_boids,
            canvas_width=self.config.canvas_width,
            canvas_height=self.config.canvas_height,
            max_steps=self.config.max_episode_steps,
            seed=self.config.seed + 2000
        )
        
        # Run evaluation episodes
        episode_rewards = []
        episode_lengths = []
        boids_caught = []
        success_count = 0
        
        for episode in range(n_episodes):
            obs, info = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
                if render:
                    eval_env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            total_caught = info.get('total_boids_caught', 0)
            total_boids = info.get('total_boids', self.config.num_boids)
            boids_caught.append(total_caught)
            
            if total_caught >= total_boids * 0.8:  # 80% success threshold
                success_count += 1
            
            print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, Caught={total_caught}/{total_boids}")
        
        # Calculate metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_boids_caught': np.mean(boids_caught),
            'success_rate': success_count / n_episodes,
            'total_episodes': n_episodes
        }
        
        print(f"\nEvaluation Results:")
        print(f"  Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Mean length: {metrics['mean_length']:.1f}")
        print(f"  Mean boids caught: {metrics['mean_boids_caught']:.1f}")
        print(f"  Success rate: {metrics['success_rate']:.2%}")
        
        return metrics
    
    def cleanup(self):
        """Cleanup resources"""
        if self.env is not None:
            self.env.close()


def main():
    """Main training function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO on boid environment")
    parser.add_argument("--config", type=str, default="quick_test", 
                        choices=["quick_test", "small_scale", "full_scale", "large_scale"],
                        help="Training configuration preset")
    parser.add_argument("--model_checkpoint", type=str, default="best_model.pt",
                        help="Path to pre-trained model checkpoint")
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="Override total training timesteps")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run evaluation, no training")
    parser.add_argument("--eval_model", type=str, default=None,
                        help="Path to model for evaluation")
    
    args = parser.parse_args()
    
    # Get configuration
    if args.config == "quick_test":
        from .config import get_quick_test_config
        config = get_quick_test_config()
    elif args.config == "small_scale":
        from .config import get_small_scale_config
        config = get_small_scale_config()
    elif args.config == "full_scale":
        from .config import get_full_scale_config
        config = get_full_scale_config()
    else:  # large_scale
        from .config import get_large_scale_config
        config = get_large_scale_config()
    
    # Override settings
    if args.model_checkpoint:
        config.model_checkpoint = args.model_checkpoint
    if args.total_timesteps:
        config.total_timesteps = args.total_timesteps
    
    # Create trainer
    trainer = PPOTrainer(config)
    
    try:
        if args.eval_only:
            # Evaluation only
            model_path = args.eval_model or os.path.join(config.save_dir, "best_model")
            trainer.evaluate(model_path, n_episodes=20, render=False)
        else:
            # Training
            trainer.train()
    
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
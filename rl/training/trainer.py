"""
PPO Trainer - Main training loop for reinforcement learning

This trainer coordinates the PPO model, environment, and algorithm to provide
a complete RL training pipeline. It handles experience collection, model updates,
checkpointing, and evaluation.

Features:
- Complete PPO training loop
- Experience rollout collection
- Automatic checkpointing and logging
- Evaluation and monitoring
- Hyperparameter scheduling
- Graceful interruption handling
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import time
import signal
from collections import defaultdict
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.constants import CONSTANTS
from ..models.ppo_model import PPOModel, create_ppo_model
from ..algorithms.ppo import PPOAlgorithm
from .environment import BoidsEnvironment
from .utils import ExperienceBuffer, TrainingLogger, TrainingUtils

class PPOTrainer:
    """
    Complete PPO training pipeline
    
    This trainer manages the entire RL training process including model loading,
    experience collection, policy updates, and evaluation.
    """
    
    def __init__(self,
                 # Model parameters
                 supervised_checkpoint_path: str,
                 device: torch.device = None,
                 
                 # Environment parameters
                 num_boids_range: Tuple[int, int] = (10, 50),
                 canvas_size_range: Tuple[Tuple[int, int], Tuple[int, int]] = ((800, 600), (1920, 1080)),
                 max_episode_steps: int = 10000,
                 
                 # Training parameters
                 total_timesteps: int = 1000000,
                 learning_rate: float = 3e-4,
                 clip_ratio: float = 0.2,
                 value_loss_coeff: float = 0.5,
                 entropy_coeff: float = 0.01,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 10,
                 mini_batch_size: int = 64,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 
                 # Logging and evaluation
                 log_dir: str = 'rl_logs',
                 checkpoint_dir: str = 'rl_checkpoints',
                 eval_interval: int = 10000,
                 save_interval: int = 50000,
                 log_interval: int = 1000,
                 
                 # Other
                 seed: Optional[int] = None,
                 debug: bool = True):
        """
        Initialize PPO trainer
        
        Args:
            supervised_checkpoint_path: Path to supervised learning checkpoint
            device: Device for computation
            num_boids_range: Range of boids per episode
            canvas_size_range: Range of canvas sizes
            max_episode_steps: Maximum steps per episode
            total_timesteps: Total training timesteps
            learning_rate: Learning rate
            clip_ratio: PPO clip ratio
            value_loss_coeff: Value loss coefficient
            entropy_coeff: Entropy coefficient
            max_grad_norm: Maximum gradient norm
            ppo_epochs: PPO epochs per update
            mini_batch_size: Mini-batch size
            gamma: Discount factor
            gae_lambda: GAE lambda
            log_dir: Logging directory
            checkpoint_dir: Checkpoint directory
            eval_interval: Evaluation interval
            save_interval: Save interval
            log_interval: Log interval
            seed: Random seed
            debug: Enable debug logging
        """
        
        # Set random seed
        if seed is not None:
            TrainingUtils.set_seed(seed)
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug = debug
        
        # Training parameters
        self.total_timesteps = total_timesteps
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.log_interval = log_interval
        
        # Create directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if self.debug:
            print(f"🚀 Initializing Simple PPO Trainer:")
            print(f"   Device: {self.device}")
            print(f"   Total timesteps: {TrainingUtils.format_number(total_timesteps)}")
            print(f"   Single environment per rollout")  
            print(f"   Learning rate: {learning_rate}")
            print(f"   Log dir: {log_dir}")
            print(f"   Checkpoint dir: {checkpoint_dir}")
        
        # Create model
        print(f"🤖 Creating PPO model from checkpoint: {supervised_checkpoint_path}")
        self.model = create_ppo_model(supervised_checkpoint_path, self.device, debug=debug)
        if self.model is None:
            raise ValueError(f"Failed to create model from checkpoint: {supervised_checkpoint_path}")
        
        # Create algorithm
        self.ppo = PPOAlgorithm(
            model=self.model,
            lr=learning_rate,
            clip_ratio=clip_ratio,
            value_loss_coeff=value_loss_coeff,
            entropy_coeff=entropy_coeff,
            max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs,
            mini_batch_size=mini_batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=self.device,
            debug=debug
        )
        
        # Create environments
        env_kwargs = {
            'num_boids_range': num_boids_range,
            'canvas_size_range': canvas_size_range,
            'max_episode_steps': max_episode_steps
        }
        
        # Use single environment - much simpler and more robust
        self.train_env = BoidsEnvironment(**env_kwargs, debug=debug and False)  # Reduce debug noise
        self.eval_env = BoidsEnvironment(**env_kwargs, debug=False)  # Single env for evaluation
        
        # Create utilities
        self.experience_buffer = ExperienceBuffer(max_steps=max_episode_steps, debug=debug)
        self.logger = TrainingLogger(log_dir, debug=debug)
        
        # Training state
        self.timesteps_collected = 0
        self.updates_performed = 0
        self.episodes_completed = 0
        self.best_eval_reward = float('-inf')
        
        # Interruption handling
        self.interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        if self.debug:
            param_count = TrainingUtils.count_parameters(self.model)
            print(f"   Model parameters: {TrainingUtils.format_number(param_count['trainable'])}")
            print(f"   ✅ Trainer initialized successfully!")
    
    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print(f"\n⚠️  Interrupted! Saving checkpoint and exiting gracefully...")
        self.interrupted = True
    
    def collect_rollout(self) -> Dict[str, Any]:
        """
        Collect one complete episode from single environment
        Simple and robust: One rollout = One complete episode
        
        Returns:
            Rollout statistics
        """
        start_time = time.time()
        self.experience_buffer.reset()
        
        # Reset environment for fresh episode
        observation = self.train_env.reset()
        
        # Collect episode data
        episode_data = []
        step_count = 0
        max_episode_steps = self.experience_buffer.max_steps  # Use max_steps from buffer
        
        if self.debug:
            print(f"\n🎯 Starting single-environment episode")
        
        # Run complete episode
        while step_count < max_episode_steps:
            if self.interrupted:
                break
            
            # Get action and value from model
            with torch.no_grad():
                action, log_prob, value = self.model.get_action_and_value([observation])
            
            # Ensure proper tensor shapes for single environment
            if torch.is_tensor(action) and len(action.shape) > 1:
                action = action.squeeze(0)
            if torch.is_tensor(log_prob) and len(log_prob.shape) > 0:
                log_prob = log_prob.squeeze(0) 
            if torch.is_tensor(value) and len(value.shape) > 1:
                value = value.squeeze(0)
            
            # Step environment
            next_observation, reward, done, info = self.train_env.step(action)
            
            # Store step data
            episode_data.append({
                'observation': observation,
                'action': action,
                'done': done,
                'log_prob': log_prob,
                'value': value,
                'info': info
            })
            
            # Move to next step
            observation = next_observation
            step_count += 1
            self.timesteps_collected += 1
            
            # Episode complete?
            if done:
                self.episodes_completed += 1
                
                if self.debug:
                    episode_type = info['episode_end_type'].upper()
                    print(f"   🏁 {episode_type} episode complete ({step_count} steps)")
                break
        
        # Safety check
        if step_count >= max_episode_steps and not done:
            print(f"⚠️ Episode safety limit reached at {step_count} steps")
        
        rollout_time = time.time() - start_time
        
        # Process the complete episode and calculate rewards
        episode_reward, episode_length = self._process_single_episode(episode_data)
        
        # Simple rollout statistics
        stats = {
            'rollout_time': rollout_time,
            'timesteps_collected': len(self.experience_buffer),
            'episodes_completed': 1,  # Always exactly 1 episode per rollout
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'total_episodes': self.episodes_completed
        }
        
        if self.debug:
            print(f"\n📊 Simple Rollout {self.updates_performed + 1} Summary:")
            print(f"   Episode reward: {stats['episode_reward']:.4f}")
            print(f"   Episode length: {stats['episode_length']}")
            print(f"   Rollout time: {stats['rollout_time']:.1f}s")
            print(f"   Total episodes so far: {stats['total_episodes']}")
            print(f"   📦 Experience buffer: {len(self.experience_buffer)} experiences stored")
            print(f"   ═" * 60)
        
        return stats
    
    def _process_single_episode(self, episode_data: List[Dict[str, Any]]) -> Tuple[float, int]:
        """
        Process a single episode and calculate final rewards
        
        Args:
            episode_data: List of step data for the episode
            
        Returns:
            Tuple of (episode_reward, episode_length)
        """
        if not episode_data:
            return 0.0, 0
        
        # Extract reward inputs for this complete episode
        reward_inputs = []
        for step_data in episode_data:
            info = step_data['info']
            # Reconstruct reward input from stored info
            reward_input = {
                'caughtBoids': info.get('caught_boids', []),
                'state': step_data['observation'],
                'action': step_data['action'].detach().cpu().numpy().tolist() if torch.is_tensor(step_data['action']) else step_data['action']
            }
            reward_inputs.append(reward_input)
        
        # Get episode end flag from the last step
        last_step_info = episode_data[-1]['info']  
        is_episode_end = last_step_info['episode_end']  # True=success, False=timeout
        episode_end_type = last_step_info['episode_end_type']
        
        if self.debug:
            result_text = "SUCCESS" if is_episode_end else "TIMEOUT"
            reward_text = "full rewards" if is_episode_end else "limited rewards"
            print(f"   Episode: {result_text} ({len(reward_inputs)} steps) → {reward_text}")
        
        # Calculate all rewards with complete episode context
        rewards = self.train_env.reward_processor.process_rewards(
            reward_inputs,
            is_episode_end=is_episode_end  # Clear boolean flag!
        )
        
        # Calculate episode statistics for logging
        if rewards:
            total_reward = sum(r['total'] for r in rewards)
            approaching_sum = sum(r['approaching'] for r in rewards)
            retro_sum = sum(r['catchRetro'] for r in rewards)
            episode_reward = total_reward
            episode_length = len(episode_data)
            
            # Log episode with proper reward breakdown
            self.logger.log_episode(
                episode_reward=total_reward,
                episode_length=len(episode_data),
                boids_caught=last_step_info['episode_stats']['boids_caught'],
                approaching_reward=approaching_sum,
                catch_reward=retro_sum,
                episode_end_type=episode_end_type
            )
            
            if self.debug:
                print(f"     Rewards: total={total_reward:.2f}, approaching={approaching_sum:.2f}, retro={retro_sum:.2f}")
        else:
            # No rewards calculated (shouldn't happen but handle gracefully)
            episode_reward = 0.0
            episode_length = len(episode_data)
            if self.debug:
                print(f"     ⚠️ No rewards calculated for episode")
        
        # Store complete experiences with final rewards
        for step_idx, (step_data, reward_obj) in enumerate(zip(episode_data, rewards or [])):
            final_reward = reward_obj['total'] if reward_obj else 0.0
            
            self.experience_buffer.store(
                observation=step_data['observation'],
                action=step_data['action'],
                reward=final_reward,  # Final total reward from the start!
                done=step_data['done'],
                log_prob=step_data['log_prob'],
                value=step_data['value']
            )
        
        if self.debug:
            print(f"   ✅ Processed 1 episode: reward={episode_reward:.2f}, length={episode_length}")
        
        return episode_reward, episode_length
    

    
    def evaluate_model(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation statistics
        """
        if self.debug:
            print(f"🎯 Evaluating model over {num_episodes} episodes...")
        
        self.model.eval()
        eval_rewards = []
        eval_lengths = []
        eval_boids_caught = []
        
        for episode in range(num_episodes):
            if self.interrupted:
                break
            
            observation = self.eval_env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                with torch.no_grad():
                    # Use mean action (no sampling) for evaluation
                    outputs = self.model.forward([observation])
                    action = outputs['action_mean']
                
                observation, reward, done, info = self.eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done:
                    eval_rewards.append(episode_reward)
                    eval_lengths.append(episode_length)
                    eval_boids_caught.append(info['episode_stats']['boids_caught'])
                    break
        
        self.model.train()
        
        eval_stats = {
            'eval_mean_reward': np.mean(eval_rewards),
            'eval_std_reward': np.std(eval_rewards),
            'eval_min_reward': np.min(eval_rewards),
            'eval_max_reward': np.max(eval_rewards),
            'eval_mean_length': np.mean(eval_lengths),
            'eval_mean_boids_caught': np.mean(eval_boids_caught)
        }
        
        if self.debug:
            print(f"\n🎯 Evaluation Results:")
            print(f"   Episodes evaluated: {num_episodes}")
            print(f"   Mean reward: {eval_stats['eval_mean_reward']:.4f} ± {eval_stats['eval_std_reward']:.4f}")
            print(f"   Reward range: {eval_stats['eval_min_reward']:.4f} to {eval_stats['eval_max_reward']:.4f}")
            print(f"   Mean episode length: {eval_stats['eval_mean_length']:.1f}")
            print(f"   Mean boids caught per episode: {eval_stats['eval_mean_boids_caught']:.2f}")
            if eval_stats['eval_mean_boids_caught'] > 0:
                print(f"   🎉 Model is successfully catching boids!")
            else:
                print(f"   📝 Model not catching boids yet (expected for early training)")
            print(f"   ─" * 50)
        
        return eval_stats
    
    def save_checkpoint(self, is_best: bool = False) -> str:
        """Save training checkpoint"""
        checkpoint_name = f"checkpoint_{self.timesteps_collected}.pt"
        if is_best:
            checkpoint_name = "best_model.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save model
        success = self.model.save_checkpoint(
            str(checkpoint_path),
            epoch=self.updates_performed,
            timesteps=self.timesteps_collected,
            episodes=self.episodes_completed,
            best_eval_reward=self.best_eval_reward
        )
        
        if success:
            # Save algorithm state
            algo_path = str(checkpoint_path).replace('.pt', '_algo.pt')
            self.ppo.save_checkpoint(
                algo_path,
                timesteps=self.timesteps_collected,
                episodes=self.episodes_completed
            )
            
            if self.debug:
                print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop
        
        Returns:
            Training summary statistics
        """
        print(f"\n🚀 Starting PPO training!")
        print(f"Target timesteps: {TrainingUtils.format_number(self.total_timesteps)}")
        print(f"Single episode per rollout (dynamic length)")
        print(f"Approximate episodes needed: ~{self.total_timesteps // 100}")
        
        training_start_time = time.time()
        
        try:
            while self.timesteps_collected < self.total_timesteps and not self.interrupted:
                # Collect rollout
                rollout_stats = self.collect_rollout()
                
                # Get experiences
                experiences = self.experience_buffer.get_experiences()
                
                if experiences:  # Only update if we have experiences
                    # Update model
                    update_stats = self.ppo.update(experiences)
                    self.updates_performed += 1
                    
                    # Log training metrics
                    combined_stats = {**rollout_stats, **update_stats}
                    self.logger.log_step(combined_stats)
                    
                    # Evaluation
                    if self.timesteps_collected % self.eval_interval == 0:
                        # Use fewer episodes for quick evaluation
                        eval_episodes = 3 if self.total_timesteps < 1000 else 10
                        eval_stats = self.evaluate_model(num_episodes=eval_episodes)
                        combined_stats.update(eval_stats)
                        
                        # Check if best model
                        if eval_stats['eval_mean_reward'] > self.best_eval_reward:
                            self.best_eval_reward = eval_stats['eval_mean_reward']
                            self.save_checkpoint(is_best=True)
                    
                    # Save checkpoint
                    if self.timesteps_collected % self.save_interval == 0:
                        self.save_checkpoint(is_best=False)
                    
                    # Progress logging
                    if self.timesteps_collected % self.log_interval == 0:
                        progress = self.timesteps_collected / self.total_timesteps * 100
                        elapsed = time.time() - training_start_time
                        eta = elapsed / (self.timesteps_collected / self.total_timesteps) - elapsed
                        
                        current_stats = self.logger.get_current_stats()
                        print(f"\n📈 Progress: {progress:.1f}% ({TrainingUtils.format_number(self.timesteps_collected)}/{TrainingUtils.format_number(self.total_timesteps)})")
                        print(f"   Elapsed: {TrainingUtils.format_time(elapsed)}, ETA: {TrainingUtils.format_time(eta)}")
                        print(f"   Episodes: {self.episodes_completed}")
                        print(f"   Updates: {self.updates_performed}")
                        if 'avg_reward_100' in current_stats:
                            print(f"   Avg reward (100 ep): {current_stats['avg_reward_100']:.2f}")
                        print(f"   Best eval reward: {self.best_eval_reward:.2f}")
        
        except Exception as e:
            print(f"❌ Training error: {e}")
            raise
        
        finally:
            # Final save
            if not self.interrupted:
                print(f"\n🎉 Training completed!")
            else:
                print(f"\n⚠️  Training interrupted!")
            
            final_checkpoint = self.save_checkpoint()
            self.logger.save_metrics()
            
            # Final evaluation
            if not self.interrupted:
                final_eval_episodes = 3 if self.total_timesteps < 1000 else 20
                final_eval = self.evaluate_model(num_episodes=final_eval_episodes)
                print(f"📊 Final evaluation: {final_eval['eval_mean_reward']:.2f} ± {final_eval['eval_std_reward']:.2f}")
            
            # Close environments
            self.train_env.close()
            self.eval_env.close()
            
            training_time = time.time() - training_start_time
            print(f"⏱️  Total training time: {TrainingUtils.format_time(training_time)}")
            print(f"💾 Final checkpoint: {final_checkpoint}")
        
        return self.logger.get_current_stats()


if __name__ == "__main__":
    # Test trainer with minimal settings
    print("🧪 Testing PPO Trainer...")
    
    # Check if supervised checkpoint exists
    checkpoint_path = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Supervised checkpoint not found: {checkpoint_path}")
        print("Please run the supervised training first or provide a valid checkpoint path")
        exit(1)
    
    # Create trainer with minimal settings for testing
    trainer = PPOTrainer(
        supervised_checkpoint_path=checkpoint_path,
        num_boids_range=(5, 10),  # Small for testing
        canvas_size_range=((400, 300), (600, 400)),  # Small for testing
        max_episode_steps=50,  # Short for testing
        total_timesteps=1000,  # Very short for testing
        log_dir='test_rl_logs',
        checkpoint_dir='test_rl_checkpoints',
        debug=True
    )
    
    print("\n🔍 Testing single rollout...")
    rollout_stats = trainer.collect_rollout()
    print(f"✅ Rollout successful: {rollout_stats}")
    
    print("\n🔍 Testing model evaluation...")
    eval_stats = trainer.evaluate_model(num_episodes=2)
    print(f"✅ Evaluation successful: {eval_stats}")
    
    print("\n✅ PPO Trainer tests passed!") 
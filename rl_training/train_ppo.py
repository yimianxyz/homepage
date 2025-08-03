"""
PPO Training Script for Transformer Policy Finetuning

This script implements PPO training using TorchRL to finetune
a transformer policy from supervised learning checkpoint.
"""

import os
import time
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.envs import ParallelEnv, TransformedEnv
from torchrl.envs.transforms import ObservationNorm, DoubleToFloat, StepCounter
from torchrl.envs.utils import check_env_specs
from torchrl.modules import ProbabilisticActor
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_training.rl_environment import BoidsEnvironment
from rl_training.rl_policy import create_rl_modules
from rl_training.ppo_config import PPOConfig, get_default_config
from evaluation.policy_evaluator import PolicyEvaluator


class PPOTrainer:
    """Clean PPO trainer for transformer policy finetuning"""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = config.device
        
        # Setup directories
        self.setup_directories()
        
        # Create environments with validation and transforms
        self.train_env = self.create_parallel_env(config.num_envs)
        self.eval_env = self.create_eval_env(config)
        
        # Validate environment specs
        try:
            check_env_specs(self.eval_env)
            print("✓ Environment specs validation passed")
        except Exception as e:
            print(f"⚠️  Environment specs validation warning: {e}")
            print("Continuing anyway - this may cause issues during training")
        
        # Create policy modules
        self.actor_module, self.value_module = create_rl_modules(
            checkpoint_path=config.checkpoint_path,
            device=self.device,
            freeze_transformer=config.freeze_transformer,
            action_std=config.action_std,
        )
        
        # Create data collector
        self.collector = SyncDataCollector(
            self.train_env,
            self.actor_module,
            frames_per_batch=config.frames_per_batch,
            total_frames=config.total_frames,
            device=self.device,
            storing_device=self.device,
        )
        
        # Create replay buffer for PPO
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(config.frames_per_batch),
            batch_size=config.minibatch_size,
        )
        
        # Setup training components
        self.setup_training()
        
        # Setup evaluation
        self.evaluator = PolicyEvaluator()
        
        # Training statistics
        self.iteration = 0
        self.total_frames = 0
        self.training_stats = {
            'policy_losses': [],
            'value_losses': [],
            'entropy_bonuses': [],
            'mean_rewards': [],
            'eval_scores': [],
        }
        
        print(f"✓ PPO Trainer initialized")
        config.print_config()
    
    def setup_directories(self):
        """Create directories for logs and checkpoints"""
        self.experiment_dir = Path(f"experiments/{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.log_dir = self.experiment_dir / "logs"
        
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        print(f"✓ Experiment directory: {self.experiment_dir}")
    
    def create_eval_env(self, config):
        """Create single evaluation environment with transforms"""
        base_env = BoidsEnvironment(
            canvas_width=config.canvas_width,
            canvas_height=config.canvas_height,
            num_boids=config.num_boids,
            max_steps=config.max_steps_per_episode,
            max_boids_for_model=config.max_boids_for_model,
            device=self.device,
        )
        
        # Add transforms for better training stability
        transformed_env = TransformedEnv(
            base_env,
            # Add step counter for episode length tracking
            StepCounter(max_steps=config.max_steps_per_episode),
        )
        
        return transformed_env

    def create_parallel_env(self, num_envs: int):
        """Create parallel environments for training"""
        
        def make_env(seed: int):
            def _make():
                base_env = BoidsEnvironment(
                    canvas_width=self.config.canvas_width,
                    canvas_height=self.config.canvas_height,
                    num_boids=self.config.num_boids,
                    max_steps=self.config.max_steps_per_episode,
                    max_boids_for_model=self.config.max_boids_for_model,
                    device=self.device,
                    seed=seed,
                )
                
                # Add transforms for training stability
                return TransformedEnv(
                    base_env,
                    StepCounter(max_steps=self.config.max_steps_per_episode),
                )
            return _make
        
        # Create parallel environment
        parallel_env = ParallelEnv(
            num_workers=num_envs,
            create_env_fn=[make_env(i) for i in range(num_envs)],
            device=self.device,
        )
        
        return parallel_env
    
    def setup_training(self):
        """Setup PPO loss and optimizer"""
        
        # Create PPO loss module
        self.ppo_loss = ClipPPOLoss(
            actor_network=self.actor_module,
            critic_network=self.value_module,
            clip_epsilon=self.config.clip_epsilon,
            entropy_coef=self.config.entropy_coef,
            critic_coef=self.config.value_loss_coef,
            normalize_advantage=True,
        )
        
        # Create optimizer - use AdamW for better regularization
        self.optimizer = torch.optim.AdamW(
            self.ppo_loss.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5,  # Add weight decay for regularization
        )
        
        # Create advantage module
        self.advantage_module = GAE(
            gamma=self.config.discount_gamma,
            lmbda=self.config.gae_lambda,
            value_network=self.value_module,
            average_gae=True,
        )
        
        print(f"✓ Training components setup")
    
    def train(self):
        """Main training loop"""
        
        print(f"\n🚀 Starting PPO training for {self.config.total_frames:,} frames")
        print("=" * 60)
        
        start_time = time.time()
        
        # Training loop
        for i, batch in enumerate(self.collector):
            self.iteration = i
            self.total_frames = (i + 1) * self.config.frames_per_batch
            
            # Process batch
            training_stats = self.train_batch(batch)
            
            # Log progress
            if i % self.config.log_frequency == 0:
                self.log_training_progress(training_stats)
            
            # Evaluate
            if i % self.config.eval_frequency == 0 and i > 0:
                self.evaluate()
            
            # Save checkpoint
            if i % self.config.save_frequency == 0 and i > 0:
                self.save_checkpoint()
        
        # Final evaluation and save
        self.evaluate()
        self.save_checkpoint(is_final=True)
        
        total_time = time.time() - start_time
        print(f"\n✅ Training completed!")
        print(f"  Total time: {total_time/3600:.1f} hours")
        print(f"  Frames per second: {self.config.total_frames/total_time:.1f}")
    
    def train_batch(self, batch: TensorDict) -> dict:
        """Train on a single batch using PPO"""
        
        # Compute advantages
        with torch.no_grad():
            self.advantage_module(batch)
        
        # Store batch in replay buffer
        batch = batch.reshape(-1)  # Flatten batch and time dimensions
        self.replay_buffer.extend(batch)
        
        # Training stats
        policy_losses = []
        value_losses = []
        entropy_bonuses = []
        
        # PPO epochs
        for epoch in range(self.config.num_epochs):
            # Sample minibatches
            for minibatch in self.replay_buffer:
                # Compute loss
                loss_vals = self.ppo_loss(minibatch)
                loss = loss_vals["loss"]
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                
                # Monitor gradient norm before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.ppo_loss.parameters(),
                    self.config.max_grad_norm
                )
                
                self.optimizer.step()
                
                # Record stats
                policy_losses.append(loss_vals["loss_objective"].item())
                value_losses.append(loss_vals["loss_critic"].item())
                entropy_bonuses.append(loss_vals["entropy_bonus"].item())
                
                # Monitor additional metrics for debugging
                if hasattr(loss_vals, "clip_fraction"):
                    clip_fraction = loss_vals.get("clip_fraction", torch.tensor(0.0)).item()
                else:
                    clip_fraction = 0.0
        
        # Clear replay buffer
        self.replay_buffer.empty()
        
        # Compute batch statistics - fix TensorDict key access
        stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_bonus': np.mean(entropy_bonuses),
            'mean_reward': batch["reward"].mean().item(),
            'mean_episode_length': batch["done"].float().sum().item() / self.config.num_envs,
        }
        
        # Update tracking
        self.training_stats['policy_losses'].append(stats['policy_loss'])
        self.training_stats['value_losses'].append(stats['value_loss'])
        self.training_stats['entropy_bonuses'].append(stats['entropy_bonus'])
        self.training_stats['mean_rewards'].append(stats['mean_reward'])
        
        return stats
    
    def evaluate(self):
        """Evaluate current policy"""
        
        print(f"\n📊 Evaluating at iteration {self.iteration}...")
        
        # Create evaluation policy wrapper
        class EvalPolicy:
            def __init__(self, actor_module):
                self.actor_module = actor_module
            
            def get_action(self, structured_inputs):
                # Convert to tensor observation
                obs = self._structured_to_tensor(structured_inputs)
                with torch.no_grad():
                    self.actor_module.eval()
                    td = TensorDict({"observation": obs}, batch_size=())
                    td = self.actor_module(td)
                    action = td["action"].cpu().numpy()
                    self.actor_module.train()
                return action.tolist()
            
            def _structured_to_tensor(self, structured):
                # Match exact format from BoidsEnvironment._state_to_tensor
                obs = []
                
                # Context information (canvas dimensions)
                obs.extend([structured['context']['canvasWidth'], 
                           structured['context']['canvasHeight']])
                
                # Predator information (velocities)
                obs.extend([structured['predator']['velX'], 
                           structured['predator']['velY']])
                
                # Boid information - use max_boids_for_model from config
                num_boids_to_add = min(len(structured['boids']), self.actor_module[0].policy.transformer.max_boids)
                
                for i in range(num_boids_to_add):
                    boid = structured['boids'][i]
                    obs.extend([boid['relX'], boid['relY'], 
                               boid['velX'], boid['velY']])
                
                # Pad with zeros if fewer boids than max
                while len(obs) < 4 + (self.actor_module[0].policy.transformer.max_boids * 4):
                    obs.extend([0.0, 0.0, 0.0, 0.0])
                
                return torch.tensor(obs, dtype=torch.float32, device=self.actor_module.device)
        
        eval_policy = EvalPolicy(self.actor_module)
        
        # Run evaluation
        result = self.evaluator.evaluate_policy(
            eval_policy,
            policy_name=f"PPO_iter_{self.iteration}"
        )
        
        # Record results
        self.training_stats['eval_scores'].append({
            'iteration': self.iteration,
            'catch_rate': result.overall_catch_rate,
            'std': result.overall_std_catch_rate,
        })
        
        print(f"  Overall catch rate: {result.overall_catch_rate:.3f} ± {result.overall_std_catch_rate:.3f}")
    
    def log_training_progress(self, stats: dict):
        """Log training progress"""
        
        print(f"\nIteration {self.iteration}/{self.config.num_iterations} "
              f"(Frames: {self.total_frames:,}/{self.config.total_frames:,})")
        print(f"  Policy loss: {stats['policy_loss']:.4f}")
        print(f"  Value loss: {stats['value_loss']:.4f}")
        print(f"  Entropy: {stats['entropy_bonus']:.4f}")
        print(f"  Mean reward: {stats['mean_reward']:.4f}")
        print(f"  Episode length: {stats['mean_episode_length']:.1f}")
    
    def save_checkpoint(self, is_final: bool = False):
        """Save training checkpoint"""
        
        checkpoint = {
            'iteration': self.iteration,
            'total_frames': self.total_frames,
            'actor_state_dict': self.actor_module.state_dict(),
            'value_state_dict': self.value_module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'config': self.config,
        }
        
        if is_final:
            path = self.checkpoint_dir / "final_checkpoint.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_iter_{self.iteration}.pt"
        
        torch.save(checkpoint, path)
        print(f"✓ Saved checkpoint: {path}")
        
        # Also save the transformer weights separately for easy loading
        if is_final or self.iteration % 100 == 0:
            # Extract transformer architecture from loaded model
            transformer = None
            for name, module in self.actor_module.named_modules():
                if hasattr(module, 'policy') and hasattr(module.policy, 'transformer'):
                    transformer = module.policy.transformer
                    break
            
            if transformer is not None:
                # Extract transformer state dict
                transformer_checkpoint = {
                    'model_state_dict': {
                        k.replace('policy.transformer.', ''): v 
                        for k, v in self.actor_module.state_dict().items() 
                        if 'policy.transformer.' in k
                    },
                    'architecture': {
                        'd_model': transformer.d_model,
                        'n_heads': transformer.n_heads,
                        'n_layers': transformer.n_layers,
                        'ffn_hidden': transformer.ffn_hidden,
                        'max_boids': transformer.max_boids,
                    },
                    'training_type': 'RL_PPO',
                    'iteration': self.iteration,
                }
                
                transformer_path = self.checkpoint_dir / f"transformer_rl_iter_{self.iteration}.pt"
                torch.save(transformer_checkpoint, transformer_path)
                print(f"✓ Saved transformer checkpoint: {transformer_path}")
            else:
                print("⚠️  Could not extract transformer for separate checkpoint")


def main():
    """Main training entry point"""
    
    # Get configuration
    config = get_default_config()
    
    # For testing, you might want to use a smaller config
    # config = get_fast_config()
    
    # Create trainer
    trainer = PPOTrainer(config)
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
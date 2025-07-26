#!/usr/bin/env python3
"""
RL Training Main Entry Point

This script provides a command-line interface for training the PPO model
on the boids predator-prey environment.

Usage:
    python rl/main.py --checkpoint checkpoints/best_model.pt --timesteps 1000000
"""

import argparse
import sys
import os
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rl.training.trainer import PPOTrainer
from rl.training.utils import TrainingUtils

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train PPO model on boids predator-prey environment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to supervised learning checkpoint'
    )
    
    # Environment arguments
    parser.add_argument(
        '--min-boids',
        type=int,
        default=10,
        help='Minimum number of boids per episode'
    )
    parser.add_argument(
        '--max-boids',
        type=int,
        default=50,
        help='Maximum number of boids per episode'
    )
    parser.add_argument(
        '--max-episode-steps',
        type=int,
        default=10000,
        help='Maximum steps per episode'
    )
    
    # Training arguments
    parser.add_argument(
        '--timesteps',
        type=int,
        default=1000000,
        help='Total training timesteps'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--clip-ratio',
        type=float,
        default=0.2,
        help='PPO clip ratio'
    )
    parser.add_argument(
        '--value-loss-coeff',
        type=float,
        default=0.5,
        help='Value loss coefficient'
    )
    parser.add_argument(
        '--entropy-coeff',
        type=float,
        default=0.01,
        help='Entropy coefficient'
    )
    parser.add_argument(
        '--ppo-epochs',
        type=int,
        default=10,
        help='PPO epochs per update'
    )
    parser.add_argument(
        '--mini-batch-size',
        type=int,
        default=64,
        help='Mini-batch size'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor'
    )
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='GAE lambda parameter'
    )
    
    # Logging and evaluation
    parser.add_argument(
        '--log-dir',
        type=str,
        default='rl_logs',
        help='Logging directory'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='rl_checkpoints',
        help='Checkpoint directory'
    )
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=10000,
        help='Evaluation interval (timesteps)'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=50000,
        help='Save interval (timesteps)'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1000,
        help='Log interval (timesteps)'
    )
    
    # Other arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with minimal parameters'
    )
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_args()
    
    print("🎯 RL Training for Boids Predator-Prey System")
    print("=" * 50)
    
    # Validate checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("Please run supervised training first or provide a valid checkpoint path")
        return 1
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"📱 Device: {device}")
    
    # Print device info
    if args.debug:
        device_info = TrainingUtils.get_device_info()
        print(f"🔧 Device info: {device_info}")
    
    # Quick test mode
    if args.quick_test:
        print("⚡ Quick test mode enabled - using minimal parameters")
        args.timesteps = 500          # Ultra-short for speed (5 episodes max)
        args.min_boids = 2            # Minimum boids for faster episodes
        args.max_boids = 3            # Maximum boids for faster episodes  
        args.max_episode_steps = 50   # Very short episodes for speed
        args.eval_interval = 1000     # Evaluate at most once (500 < 1000)
        args.save_interval = 1000     # Save at most once (500 < 1000)
        args.log_interval = 200       # Log only 2-3 times (500/200 = 2.5)
        args.ppo_epochs = 2           # Fewer PPO epochs for speed
        args.mini_batch_size = 16     # Smaller batches for speed
        args.debug = False            # Disable debug to reduce noise
        print("   🔇 Debug disabled for clean output")
    
    # Performance recommendations for real training
    if not args.quick_test and args.timesteps >= 100000:
        print("\n💡 Performance Tips for Long Training:")
        print("   • Use fewer boids initially (10-20) then increase")
        print("   • Set eval_interval = timesteps // 50 for reasonable evaluation frequency")
        print("   • Set save_interval = timesteps // 20 for regular checkpoints")
        print("   • Consider larger mini_batch_size (128-256) for efficiency")
        print("   • Use max_episode_steps = 500-2000 depending on task complexity")
    
    # Print training configuration
    print(f"\n🚀 Training Configuration:")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Total timesteps: {TrainingUtils.format_number(args.timesteps)}")
    print(f"   Single environment per rollout")
    print(f"   Boids range: {args.min_boids}-{args.max_boids}")
    print(f"   Max episode steps: {args.max_episode_steps}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   PPO epochs: {args.ppo_epochs}")
    print(f"   Mini-batch size: {args.mini_batch_size}")
    print(f"   Log dir: {args.log_dir}")
    print(f"   Checkpoint dir: {args.checkpoint_dir}")
    
    # Create trainer
    try:
        trainer = PPOTrainer(
            supervised_checkpoint_path=args.checkpoint,
            device=device,
            num_boids_range=(args.min_boids, args.max_boids),
            max_episode_steps=args.max_episode_steps,
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            clip_ratio=args.clip_ratio,
            value_loss_coeff=args.value_loss_coeff,
            entropy_coeff=args.entropy_coeff,
            ppo_epochs=args.ppo_epochs,
            mini_batch_size=args.mini_batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            log_interval=args.log_interval,
            seed=args.seed,
            debug=args.debug
        )
        
    except Exception as e:
        print(f"❌ Failed to create trainer: {e}")
        return 1
    
    # Start training
    try:
        print(f"\n{'='*60}")
        print(f"🎯 STARTING RL TRAINING")
        print(f"{'='*60}")
        
        training_stats = trainer.train()
        
        print(f"\n{'='*60}")
        print(f"🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"📊 Final Statistics:")
        for key, value in training_stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        if args.debug:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
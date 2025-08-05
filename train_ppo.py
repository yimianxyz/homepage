"""
PPO Training Script - Simple script to train PPO agent from supervised learning baseline

This script demonstrates how to use the PPO training system to fine-tune
a supervised learning baseline using reinforcement learning.

Usage:
    python train_ppo.py [--iterations 50] [--eval-interval 5] [--checkpoint checkpoints/best_model.pt]

The script will:
1. Load the SL baseline from checkpoint
2. Initialize PPO trainer with simulation integration
3. Run PPO training for specified iterations  
4. Periodically evaluate and save checkpoints
5. Export final model for browser deployment
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer, create_ppo_policy_from_sl
from simulation.random_state_generator import generate_random_state


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent from SL baseline')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to supervised learning checkpoint')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of PPO training iterations')
    parser.add_argument('--eval-interval', type=int, default=5,
                       help='Evaluate every N iterations')
    parser.add_argument('--save-interval', type=int, default=5,
                       help='Save checkpoint every N iterations')
    parser.add_argument('--rollout-steps', type=int, default=2048,
                       help='Steps per rollout collection')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate for PPO')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--boids', type=int, default=20,
                       help='Number of boids in simulation')
    parser.add_argument('--canvas-width', type=float, default=800,
                       help='Canvas width for simulation')  
    parser.add_argument('--canvas-height', type=float, default=600,
                       help='Canvas height for simulation')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        print("Make sure you have trained a supervised learning model first.")
        print("Run the transformer_training.ipynb notebook to create one.")
        return 1
    
    print(f"🚀 PPO Training Configuration:")
    print(f"  SL checkpoint: {args.checkpoint}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Rollout steps: {args.rollout_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Device: {args.device}")
    print(f"  Simulation: {args.boids} boids, {args.canvas_width}x{args.canvas_height}")
    print()
    
    try:
        # Create PPO trainer
        print("🔧 Initializing PPO trainer...")
        trainer = PPOTrainer(
            sl_checkpoint_path=args.checkpoint,
            learning_rate=args.learning_rate,
            rollout_steps=args.rollout_steps,
            device=args.device
        )
        
        # Create output directory
        os.makedirs("checkpoints", exist_ok=True)
        
        # Run training
        print("🎯 Starting PPO training...")
        training_stats = trainer.train(
            num_iterations=args.iterations,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            initial_boids=args.boids,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height
        )
        
        # Save training statistics
        import json
        stats_path = "checkpoints/ppo_training_stats.json"
        with open(stats_path, 'w') as f:
            # Convert any tensors to lists for JSON serialization
            serializable_stats = []
            for stat in training_stats:
                serializable_stat = {}
                for key, value in stat.items():
                    if isinstance(value, dict):
                        serializable_stat[key] = {k: float(v) if hasattr(v, 'item') else v 
                                                for k, v in value.items()}
                    else:
                        serializable_stat[key] = float(value) if hasattr(value, 'item') else value
                serializable_stats.append(serializable_stat)
            
            json.dump(serializable_stats, f, indent=2)
        print(f"📊 Training statistics saved: {stats_path}")
        
        # Final evaluation
        print("🎯 Running final evaluation...")
        final_eval = trainer.evaluate_policy(num_episodes=10)
        print(f"📈 Final performance:")
        print(f"  Catch rate: {final_eval['overall_catch_rate']:.3f} ± {final_eval['overall_std_catch_rate']:.3f}")
        
        # Test policy interface
        print("🧪 Testing policy interface...")
        test_state = generate_random_state(5, 400, 300, seed=42)
        structured_input = trainer.state_manager._convert_state_to_structured_inputs(test_state)
        action = trainer.policy.get_action(structured_input)
        print(f"  Test action: [{action[0]:.3f}, {action[1]:.3f}]")
        
        print(f"\n{'='*60}")
        print(f"🎉 PPO TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"✅ Model fine-tuned with reinforcement learning")
        print(f"✅ Best model saved as: checkpoints/best_ppo_model.pt")
        print(f"✅ Training statistics saved as: {stats_path}")
        print(f"✅ Policy ready for evaluation and deployment")
        print(f"{'='*60}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
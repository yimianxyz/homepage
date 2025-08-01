"""
Simple, clean RL training script for boid catching with PPO.

This incorporates all the key learnings from our debugging:
1. Use 500 max steps for meaningful episodes
2. Use fixed seeds for reproducibility
3. Simple closest-pursuit features work well
4. Standard PPO hyperparameters are sufficient
"""

import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl.environment.boid_env import BoidEnvironment


class SimpleFeaturesExtractor(BaseFeaturesExtractor):
    """Extract simple features: closest boid info + predator velocity."""
    
    def __init__(self, observation_space: spaces.Box):
        # 6 features: distance, direction (2D), boid speed, predator velocity (2D)
        super().__init__(observation_space, features_dim=6)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        features = []
        
        for i in range(batch_size):
            obs = observations[i].cpu().numpy()
            
            # Find closest boid
            idx = 4  # Skip context(2) + predator velocity(2)
            min_dist = float('inf')
            closest_features = [0.0, 0.0, 0.0, 0.0]
            
            num_boids = (len(obs) - idx) // 4
            for j in range(num_boids):
                rel_x = obs[idx + j*4]
                rel_y = obs[idx + j*4 + 1]
                vel_x = obs[idx + j*4 + 2]
                vel_y = obs[idx + j*4 + 3]
                
                # Skip empty slots
                if rel_x == 0 and rel_y == 0:
                    continue
                
                dist = np.sqrt(rel_x**2 + rel_y**2)
                if dist < min_dist and dist > 0.001:
                    min_dist = dist
                    closest_features = [
                        min(dist, 1.0),           # Distance (clamped)
                        rel_x / dist,             # Direction X
                        rel_y / dist,             # Direction Y
                        np.sqrt(vel_x**2 + vel_y**2)  # Boid speed
                    ]
            
            # Add predator velocity
            features.append(closest_features + [obs[2], obs[3]])
        
        return torch.tensor(features, dtype=torch.float32, device=observations.device)


def train_ppo(
    num_timesteps=40000,
    max_steps=500,
    num_boids=5,
    seed=12345,
    save_path="models/ppo_simple"
):
    """
    Train a simple PPO agent to catch boids.
    
    Args:
        num_timesteps: Total training timesteps
        max_steps: Max steps per episode (500 for meaningful evaluation)
        num_boids: Number of boids in environment
        seed: Random seed for reproducibility
        save_path: Where to save the trained model
    
    Returns:
        Trained PPO model
    """
    print("🎯 SIMPLE PPO TRAINING")
    print("=" * 50)
    print(f"Timesteps: {num_timesteps}")
    print(f"Max steps: {max_steps}")
    print(f"Num boids: {num_boids}")
    print(f"Seed: {seed}")
    print()
    
    # Create environment
    env = BoidEnvironment(
        num_boids=num_boids,
        canvas_width=400,
        canvas_height=300,
        max_steps=max_steps,
        seed=seed
    )
    
    # Create PPO agent with simple features
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        n_steps=1024,          # Collect this many steps before update
        batch_size=64,         # Minibatch size
        n_epochs=5,            # Epochs per update
        gamma=0.99,            # Discount factor
        verbose=1,
        policy_kwargs={
            'features_extractor_class': SimpleFeaturesExtractor,
            'net_arch': [64, 32]  # Simple network
        },
        seed=seed
    )
    
    # Train
    print("Training...")
    model.learn(total_timesteps=num_timesteps, progress_bar=True)
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\n✅ Model saved to {save_path}")
    
    env.close()
    return model


def evaluate_model(
    model_path="models/ppo_simple",
    num_episodes=10,
    max_steps=500,
    seed=12345
):
    """
    Evaluate a trained model using our simple evaluator.
    
    Args:
        model_path: Path to saved model
        num_episodes: Episodes to evaluate
        max_steps: Max steps per episode
        seed: Random seed
    
    Returns:
        Catch rate (0.0 to 1.0)
    """
    print("\n📊 EVALUATION")
    print("=" * 50)
    
    from evaluation.evaluator import Evaluator
    from policy.human_prior.closest_pursuit_policy import ClosestPursuitPolicy
    
    # Load model
    env = BoidEnvironment(num_boids=5, canvas_width=400, canvas_height=300, max_steps=max_steps)
    model = PPO.load(model_path, env=env)
    
    # Create wrapper for evaluator
    class ModelWrapper:
        def __init__(self, ppo_model):
            self.model = ppo_model
            
        def get_action(self, structured_inputs):
            # Convert structured inputs to flat observation
            obs = []
            obs.extend([
                structured_inputs['context']['canvasWidth'],
                structured_inputs['context']['canvasHeight'],
                structured_inputs['predator']['velX'],
                structured_inputs['predator']['velY']
            ])
            
            # Add boids (pad to 5)
            boids = structured_inputs['boids']
            for i in range(5):
                if i < len(boids):
                    boid = boids[i]
                    obs.extend([boid['relX'], boid['relY'], boid['velX'], boid['velY']])
                else:
                    obs.extend([0.0, 0.0, 0.0, 0.0])
            
            obs = np.array(obs, dtype=np.float32)
            action, _ = self.model.predict(obs, deterministic=True)
            return action.tolist()
    
    # Evaluate
    evaluator = Evaluator(
        num_episodes=num_episodes,
        scenarios=['easy'],
        max_steps=max_steps,
        seed=seed
    )
    
    # Test baseline
    print("Evaluating baseline (ClosestPursuit)...")
    baseline = ClosestPursuitPolicy()
    baseline_score = evaluator.evaluate(baseline, detailed=False)
    
    # Test our model
    print("Evaluating trained model...")
    wrapper = ModelWrapper(model)
    model_score = evaluator.evaluate(wrapper, detailed=False)
    
    # Results
    improvement = (model_score - baseline_score) / baseline_score * 100 if baseline_score > 0 else 0
    
    print(f"\n📊 RESULTS:")
    print(f"  Baseline:     {baseline_score*100:.1f}%")
    print(f"  Trained:      {model_score*100:.1f}%")
    print(f"  Improvement:  {improvement:+.1f}%")
    
    env.close()
    return model_score


def main():
    """Main training and evaluation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple PPO training for boid catching")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--eval", action="store_true", help="Evaluate existing model")
    parser.add_argument("--timesteps", type=int, default=40000, help="Training timesteps")
    parser.add_argument("--episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    
    args = parser.parse_args()
    
    if not args.train and not args.eval:
        print("Please specify --train and/or --eval")
        return
    
    if args.train:
        print("🚀 STARTING TRAINING")
        print("=" * 60)
        model = train_ppo(
            num_timesteps=args.timesteps,
            seed=args.seed
        )
    
    if args.eval:
        print("\n🔍 STARTING EVALUATION")
        print("=" * 60)
        catch_rate = evaluate_model(
            num_episodes=args.episodes,
            seed=args.seed
        )
        
        if catch_rate > 0.2:
            print("\n✅ SUCCESS: Model achieves good performance!")
        else:
            print("\n📈 Model is learning but needs more training")


if __name__ == "__main__":
    main()
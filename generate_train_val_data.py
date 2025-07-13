#!/usr/bin/env python3
"""
Generate Training and Validation Data

Generates both training and validation data for supervised learning.
"""

import argparse
import os
from generate_data import generate_episodes, save_data

def main():
    parser = argparse.ArgumentParser(description='Generate training and validation data')
    parser.add_argument('--train-episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--val-episodes', type=int, default=200, help='Number of validation episodes')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per episode')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--canvas-width', type=int, default=800, help='Canvas width')
    parser.add_argument('--canvas-height', type=int, default=600, help='Canvas height')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Supervised Learning Data Generation ===")
    print(f"Training episodes: {args.train_episodes}")
    print(f"Validation episodes: {args.val_episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Canvas: {args.canvas_width}x{args.canvas_height}")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Generate training data
    print("Generating training data...")
    train_samples = generate_episodes(
        num_episodes=args.train_episodes,
        max_steps=args.max_steps,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        seed=args.seed
    )
    
    train_path = os.path.join(args.output_dir, 'train_data.pkl')
    save_data(train_samples, train_path)
    print()
    
    # Generate validation data
    print("Generating validation data...")
    val_samples = generate_episodes(
        num_episodes=args.val_episodes,
        max_steps=args.max_steps,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        seed=args.seed + 1000  # Different seed for validation
    )
    
    val_path = os.path.join(args.output_dir, 'val_data.pkl')
    save_data(val_samples, val_path)
    print()
    
    print("Data generation complete!")
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")
    print()
    print("Now you can train with:")
    print(f"python -m pytorch_training.train_supervised --train-data {train_path} --val-data {val_path}")

if __name__ == "__main__":
    main() 
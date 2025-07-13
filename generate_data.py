#!/usr/bin/env python3
"""
Generate Supervised Learning Data

Generates training data from simulation episodes using teacher policy.
Saves data to disk for later use in training.
"""

import os
import pickle
import argparse
import random
import copy
from typing import List, Dict, Any

from python_simulation import Simulation, InputProcessor
from pytorch_training.teacher_policy import TeacherPolicy

def generate_episodes(num_episodes: int, 
                     max_steps: int = 500,
                     canvas_width: int = 800, 
                     canvas_height: int = 600,
                     seed: int = None) -> List[Dict[str, Any]]:
    """Generate training samples from simulation episodes"""
    
    if seed is not None:
        random.seed(seed)
    
    # Create simulation components
    sim = Simulation(canvas_width, canvas_height)
    input_processor = InputProcessor()
    teacher_policy = TeacherPolicy()
    
    samples = []
    
    print(f"Generating {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        sim.reset()
        step_count = 0
        
        while not sim.is_episode_complete() and step_count < max_steps:
            # Get current state
            state = sim.get_state()
            
            # Process inputs
            structured_inputs = input_processor.process_inputs(
                state['boids'],
                state['predator']['position'],
                state['predator']['velocity'],
                state['canvas_width'],
                state['canvas_height']
            )
            
            # Get teacher action
            teacher_action = teacher_policy.get_normalized_action(structured_inputs)
            
            # Store sample
            samples.append({
                'inputs': copy.deepcopy(structured_inputs),
                'target': teacher_action
            })
            
            # Apply action and step
            raw_action = teacher_policy.get_action(structured_inputs)
            sim.set_predator_acceleration(raw_action[0], raw_action[1])
            sim.step()
            step_count += 1
        
        # Progress
        if (episode + 1) % 100 == 0:
            print(f"  {episode + 1}/{num_episodes} episodes completed")
    
    print(f"Generated {len(samples)} samples")
    return samples

def save_data(samples: List[Dict[str, Any]], filepath: str):
    """Save samples to disk"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(samples, f)
    
    print(f"Saved {len(samples)} samples to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Generate supervised learning data')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to generate')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per episode')
    parser.add_argument('--output', type=str, default='data/train_data.pkl', help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--canvas-width', type=int, default=800, help='Canvas width')
    parser.add_argument('--canvas-height', type=int, default=600, help='Canvas height')
    
    args = parser.parse_args()
    
    print("=== Supervised Learning Data Generation ===")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Canvas: {args.canvas_width}x{args.canvas_height}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")
    print()
    
    # Generate data
    samples = generate_episodes(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        seed=args.seed
    )
    
    # Save data
    save_data(samples, args.output)
    
    print("Data generation complete!")

if __name__ == "__main__":
    main() 
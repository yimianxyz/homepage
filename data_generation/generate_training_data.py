#!/usr/bin/env python3
"""
Training Data Generator for Supervised Learning

This script generates training data using the closest pursuit policy as the teacher.
It creates diverse scenarios with random boid counts (1-50) and canvas sizes
covering mobile to desktop devices for comprehensive training data.

Output format:
- JSON dataset with input-output pairs
- Inputs: Structured format from InputProcessor  
- Outputs: Normalized actions from ClosestPursuitPolicy
- Metadata: Canvas size, boid count, generation parameters

Usage:
    python generate_training_data.py --samples 10000 --output training_data.json
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import simulation components
from simulation.random_state_generator import RandomStateGenerator
from simulation.processors import InputProcessor
from policy.human_prior.closest_pursuit_policy import create_closest_pursuit_policy
from config.constants import CONSTANTS

class TrainingDataGenerator:
    """Generate training data using closest pursuit policy as teacher"""
    
    def __init__(self, seed: int = None):
        """
        Initialize the training data generator
        
        Args:
            seed: Random seed for reproducible generation
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        # Initialize components
        self.state_generator = RandomStateGenerator(seed)
        self.input_processor = InputProcessor()
        self.teacher_policy = create_closest_pursuit_policy()
        
        # Simple canvas size ranges - covering all possible device sizes
        self.min_canvas_width = 320    # Smallest mobile width
        self.max_canvas_width = 3840   # Largest desktop width (4K)
        self.min_canvas_height = 320   # Smallest mobile height
        self.max_canvas_height = 2160  # Largest desktop height (4K)
        
        # Boid count range
        self.min_boids = 1
        self.max_boids = 50
        
        print(f"Initialized TrainingDataGenerator:")
        print(f"  Seed: {seed}")
        print(f"  Boid range: {self.min_boids}-{self.max_boids}")
        print(f"  Canvas width range: {self.min_canvas_width}-{self.max_canvas_width}")
        print(f"  Canvas height range: {self.min_canvas_height}-{self.max_canvas_height}")
        print(f"  Teacher policy: Closest Pursuit")
    
    def generate_random_canvas_size(self) -> Tuple[int, int]:
        """
        Generate random canvas size with uniform distribution
        
        Returns:
            Tuple of (width, height)
        """
        # Generate random width and height uniformly across all possible values
        width = random.randint(self.min_canvas_width, self.max_canvas_width)
        height = random.randint(self.min_canvas_height, self.max_canvas_height)
        
        return width, height
    
    def generate_single_sample(self) -> Dict[str, Any]:
        """
        Generate a single training sample
        
        Returns:
            Dictionary containing input, output, and metadata
        """
        # Generate random parameters
        num_boids = random.randint(self.min_boids, self.max_boids)
        canvas_width, canvas_height = self.generate_random_canvas_size()
        
        # Generate random scattered state
        state = self.state_generator.generate_scattered_state(
            num_boids, canvas_width, canvas_height
        )
        
        # Convert state to structured inputs for policy
        structured_inputs = self._convert_state_to_structured_inputs(state)
        
        # Get teacher action from closest pursuit policy
        teacher_action = self.teacher_policy.get_action(structured_inputs)
        
        # Prepare sample
        sample = {
            'input': structured_inputs,
            'output': teacher_action,
            'metadata': {
                'num_boids': num_boids,
                'canvas_width': canvas_width,
                'canvas_height': canvas_height,
                'boids_density': num_boids / (canvas_width * canvas_height / 100000),  # boids per 100k pixels
                'has_valid_target': len(structured_inputs['boids']) > 0 and teacher_action != [0.0, 0.0]
            }
        }
        
        return sample
    
    def _convert_state_to_structured_inputs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert raw state to structured inputs using InputProcessor
        
        Args:
            state: Raw state from StateManager format
            
        Returns:
            Structured inputs for policy
        """
        # Extract data from state
        boids = state['boids_states']
        predator_pos = state['predator_state']['position']
        predator_vel = state['predator_state']['velocity']
        canvas_width = state['canvas_width']
        canvas_height = state['canvas_height']
        
        # Use InputProcessor to convert to structured format
        return self.input_processor.process_inputs(
            boids, predator_pos, predator_vel, canvas_width, canvas_height
        )
    

    
    def generate_dataset(self, 
                        num_samples: int, 
                        progress_interval: int = 1000) -> Dict[str, Any]:
        """
        Generate complete training dataset
        
        Args:
            num_samples: Number of samples to generate
            progress_interval: Print progress every N samples
            
        Returns:
            Complete dataset with samples and metadata
        """
        print(f"\nGenerating {num_samples:,} training samples...")
        
        samples = []
        start_time = time.time()
        
        # Statistics tracking
        boid_count_stats = {'min': float('inf'), 'max': 0, 'total': 0}
        canvas_width_stats = {'min': float('inf'), 'max': 0, 'total': 0}
        canvas_height_stats = {'min': float('inf'), 'max': 0, 'total': 0}
        valid_target_count = 0
        
        for i in range(num_samples):
            sample = self.generate_single_sample()
            samples.append(sample)
            
            # Update statistics
            metadata = sample['metadata']
            
            boid_count = metadata['num_boids']
            boid_count_stats['min'] = min(boid_count_stats['min'], boid_count)
            boid_count_stats['max'] = max(boid_count_stats['max'], boid_count)
            boid_count_stats['total'] += boid_count
            
            canvas_width = metadata['canvas_width']
            canvas_width_stats['min'] = min(canvas_width_stats['min'], canvas_width)
            canvas_width_stats['max'] = max(canvas_width_stats['max'], canvas_width)
            canvas_width_stats['total'] += canvas_width
            
            canvas_height = metadata['canvas_height']
            canvas_height_stats['min'] = min(canvas_height_stats['min'], canvas_height)
            canvas_height_stats['max'] = max(canvas_height_stats['max'], canvas_height)
            canvas_height_stats['total'] += canvas_height
            
            if metadata['has_valid_target']:
                valid_target_count += 1
            
            # Print progress
            if (i + 1) % progress_interval == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (num_samples - i - 1) / rate if rate > 0 else 0
                print(f"  Progress: {i+1:,}/{num_samples:,} ({(i+1)/num_samples*100:.1f}%) "
                      f"- {rate:.1f} samples/sec - ETA: {remaining:.1f}s")
        
        elapsed_time = time.time() - start_time
        
        # Compile dataset with metadata
        dataset = {
            'samples': samples,
            'metadata': {
                'total_samples': num_samples,
                'generation_time_seconds': elapsed_time,
                'generation_rate_samples_per_second': num_samples / elapsed_time,
                'generated_at': datetime.now().isoformat(),
                'generator_seed': self.seed,
                'teacher_policy': 'closest_pursuit',
                'simulation_constants': {
                    'boid_max_speed': CONSTANTS.BOID_MAX_SPEED,
                    'predator_max_speed': CONSTANTS.PREDATOR_MAX_SPEED,
                    'max_distance': CONSTANTS.MAX_DISTANCE,
                    'boid_neighbor_distance': CONSTANTS.BOID_NEIGHBOR_DISTANCE
                },
                'statistics': {
                    'boid_count_stats': {
                        'min': boid_count_stats['min'],
                        'max': boid_count_stats['max'],
                        'average': boid_count_stats['total'] / num_samples,
                        'range': f"{self.min_boids}-{self.max_boids}"
                    },
                    'canvas_width_stats': {
                        'min': canvas_width_stats['min'],
                        'max': canvas_width_stats['max'],
                        'average': canvas_width_stats['total'] / num_samples,
                        'range': f"{self.min_canvas_width}-{self.max_canvas_width}"
                    },
                    'canvas_height_stats': {
                        'min': canvas_height_stats['min'],
                        'max': canvas_height_stats['max'],
                        'average': canvas_height_stats['total'] / num_samples,
                        'range': f"{self.min_canvas_height}-{self.max_canvas_height}"
                    },
                    'valid_target_percentage': (valid_target_count / num_samples) * 100
                }
            }
        }
        
        print(f"\nDataset generation completed!")
        print(f"  Total samples: {num_samples:,}")
        print(f"  Generation time: {elapsed_time:.2f} seconds")
        print(f"  Generation rate: {num_samples/elapsed_time:.1f} samples/second")
        print(f"  Valid targets: {valid_target_count:,} ({valid_target_count/num_samples*100:.1f}%)")
        print(f"  Canvas width range: {canvas_width_stats['min']}-{canvas_width_stats['max']} (avg: {canvas_width_stats['total']/num_samples:.0f})")
        print(f"  Canvas height range: {canvas_height_stats['min']}-{canvas_height_stats['max']} (avg: {canvas_height_stats['total']/num_samples:.0f})")
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, Any], output_path: str) -> None:
        """
        Save dataset to JSON file
        
        Args:
            dataset: Generated dataset
            output_path: Output file path
        """
        print(f"\nSaving dataset to {output_path}...")
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Calculate file size
        file_size = Path(output_path).stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"Dataset saved successfully!")
        print(f"  File: {output_path}")
        print(f"  Size: {file_size_mb:.2f} MB ({file_size:,} bytes)")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Generate training data for supervised learning using closest pursuit policy'
    )
    parser.add_argument(
        '--samples', 
        type=int, 
        default=10000,
        help='Number of training samples to generate (default: 10000)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='training_data.json',
        help='Output JSON file path (default: training_data.json)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None,
        help='Random seed for reproducible generation (default: random)'
    )
    parser.add_argument(
        '--progress', 
        type=int, 
        default=1000,
        help='Progress reporting interval (default: 1000)'
    )
    
    args = parser.parse_args()
    
    print("🎯 Training Data Generator for Boids Simulation")
    print("=" * 50)
    print(f"Target samples: {args.samples:,}")
    print(f"Output file: {args.output}")
    print(f"Random seed: {args.seed if args.seed else 'random'}")
    print(f"Progress interval: {args.progress}")
    
    # Generate dataset
    generator = TrainingDataGenerator(seed=args.seed)
    dataset = generator.generate_dataset(args.samples, args.progress)
    
    # Save dataset
    generator.save_dataset(dataset, args.output)
    
    print("\n✅ Training data generation completed successfully!")

if __name__ == "__main__":
    main() 
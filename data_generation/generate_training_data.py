#!/usr/bin/env python3
"""
Training Data Generator for Supervised Learning (Memory-Efficient Streaming)

This script generates training data using the closest pursuit policy as the teacher.
It uses streaming writes to avoid memory issues with large datasets, allowing generation
of millions of samples without memory problems.

Features:
- Memory-efficient streaming writes
- Diverse scenarios with random boid counts (1-50) and canvas sizes
- JSON dataset with input-output pairs
- Comprehensive progress tracking and statistics

Usage:
    python generate_training_data.py --samples 1000000 --output training_data.json
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
    """Memory-efficient streaming data generator using closest pursuit policy as teacher"""
    
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
        print(f"  Mode: Streaming (memory-efficient)")
    
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
                        output_path: str,
                        progress_interval: int = 1000,
                        batch_size: int = 1000) -> None:
        """
        Generate dataset with streaming write to avoid memory issues
        
        Args:
            num_samples: Number of samples to generate
            output_path: Output file path for streaming write
            progress_interval: Print progress every N samples
            batch_size: Write to file every N samples
        """
        print(f"\nGenerating {num_samples:,} training samples (streaming mode)...")
        
        start_time = time.time()
        
        # Statistics tracking
        boid_count_stats = {'min': float('inf'), 'max': 0, 'total': 0}
        canvas_width_stats = {'min': float('inf'), 'max': 0, 'total': 0}
        canvas_height_stats = {'min': float('inf'), 'max': 0, 'total': 0}
        valid_target_count = 0
        
        # Open file for streaming write
        with open(output_path, 'w') as f:
            # Write JSON opening
            f.write('{\n  "samples": [\n')
            
            # Generate and write samples incrementally
            for i in range(num_samples):
                sample = self.generate_single_sample()
                
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
                
                # Write sample to file immediately (streaming)
                sample_json = json.dumps(sample, separators=(',', ':'))
                if i > 0:
                    f.write(',\n    ')
                else:
                    f.write('    ')
                f.write(sample_json)
                
                # Flush periodically to ensure data is written
                if (i + 1) % batch_size == 0:
                    f.flush()
                
                # Print progress
                if (i + 1) % progress_interval == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (num_samples - i - 1) / rate if rate > 0 else 0
                    print(f"  Progress: {i+1:,}/{num_samples:,} ({(i+1)/num_samples*100:.1f}%) "
                          f"- {rate:.1f} samples/sec - ETA: {remaining:.1f}s")
            
            elapsed_time = time.time() - start_time
            
            # Compile metadata
            metadata = {
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
            
            # Write JSON closing with metadata
            f.write('\n  ],\n  "metadata": ')
            json.dump(metadata, f, indent=2)
            f.write('\n}\n')
        
        print(f"\nDataset generation completed!")
        print(f"  Total samples: {num_samples:,}")
        print(f"  Generation time: {elapsed_time:.2f} seconds")
        print(f"  Generation rate: {num_samples/elapsed_time:.1f} samples/second")
        print(f"  Valid targets: {valid_target_count:,} ({valid_target_count/num_samples*100:.1f}%)")
        print(f"  Canvas width range: {canvas_width_stats['min']}-{canvas_width_stats['max']} (avg: {canvas_width_stats['total']/num_samples:.0f})")
        print(f"  Canvas height range: {canvas_height_stats['min']}-{canvas_height_stats['max']} (avg: {canvas_height_stats['total']/num_samples:.0f})")
        
        # Calculate file size
        file_size = Path(output_path).stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"\nDataset saved successfully!")
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
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=1000,
        help='Write batch size for memory efficiency (default: 1000)'
    )
    
    args = parser.parse_args()
    
    print("🎯 Training Data Generator for Boids Simulation")
    print("=" * 50)
    print(f"Target samples: {args.samples:,}")
    print(f"Output file: {args.output}")
    print(f"Random seed: {args.seed if args.seed else 'random'}")
    print(f"Progress interval: {args.progress}")
    print(f"Batch size: {args.batch_size}")
    
    # Generate dataset with streaming
    generator = TrainingDataGenerator(seed=args.seed)
    generator.generate_dataset(
        args.samples, 
        args.output, 
        args.progress, 
        args.batch_size
    )
    
    print("\n✅ Training data generation completed successfully!")

if __name__ == "__main__":
    main() 
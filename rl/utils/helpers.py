"""
Helper Functions - Utility functions for RL training

This module provides various utility functions for training,
debugging, and evaluation.
"""

import torch
import numpy as np
import random
import json
import pickle
from typing import Dict, Any, List, Optional, Tuple
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.constants import CONSTANTS


def set_seed(seed: int):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Set random seed to {seed}")


def check_environment(env: gym.Env, verbose: bool = True) -> bool:
    """
    Check if environment is compatible with stable-baselines3
    
    Args:
        env: Environment to check
        verbose: Whether to print detailed information
        
    Returns:
        valid: Whether environment is valid
    """
    try:
        check_env(env, warn=verbose)
        if verbose:
            print("✅ Environment passes all checks")
        return True
    except Exception as e:
        if verbose:
            print(f"❌ Environment check failed: {e}")
        return False


def create_test_observation(num_boids: int = 10, 
                           max_boids: int = 20,
                           seed: Optional[int] = None) -> np.ndarray:
    """
    Create a test observation for debugging
    
    Args:
        num_boids: Number of boids to include
        max_boids: Maximum boids for observation size
        seed: Random seed
        
    Returns:
        observation: Test observation array
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Observation format: [context(2) + predator(2) + boids(4*max_boids)]
    obs_size = 2 + 2 + 4 * max_boids
    observation = np.zeros(obs_size, dtype=np.float32)
    
    idx = 0
    
    # Context: [canvasWidth, canvasHeight] (normalized)
    observation[idx] = 0.4  # Canvas width / MAX_DISTANCE
    observation[idx + 1] = 0.3  # Canvas height / MAX_DISTANCE
    idx += 2
    
    # Predator: [velX, velY] (normalized)
    observation[idx] = np.random.uniform(-1, 1)
    observation[idx + 1] = np.random.uniform(-1, 1)
    idx += 2
    
    # Boids: [relX, relY, velX, velY] for each boid
    for i in range(max_boids):
        if i < num_boids:
            # Active boid
            observation[idx] = np.random.uniform(-1, 1)     # relX
            observation[idx + 1] = np.random.uniform(-1, 1) # relY
            observation[idx + 2] = np.random.uniform(-1, 1) # velX
            observation[idx + 3] = np.random.uniform(-1, 1) # velY
        else:
            # Padding (zeros)
            observation[idx:idx+4] = 0.0
        idx += 4
    
    return observation


def print_model_summary(model: torch.nn.Module, verbose: bool = True) -> Dict[str, Any]:
    """
    Print summary of model architecture and parameters
    
    Args:
        model: PyTorch model
        verbose: Whether to print detailed information
        
    Returns:
        summary: Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
    }
    
    if hasattr(model, 'get_architecture_info'):
        summary.update(model.get_architecture_info())
    
    if verbose:
        print(f"\n📊 Model Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {summary['model_size_mb']:.1f} MB")
        
        if hasattr(model, 'get_architecture_info'):
            arch_info = model.get_architecture_info()
            if 'd_model' in arch_info:
                print(f"  Architecture: {arch_info['d_model']}×{arch_info['n_heads']}×{arch_info['n_layers']}")
    
    return summary


def save_training_metrics(metrics: Dict[str, List[float]], 
                         filepath: str,
                         format: str = 'json') -> None:
    """
    Save training metrics to file
    
    Args:
        metrics: Dictionary of metric lists
        filepath: Path to save file
        format: File format ('json' or 'pickle')
    """
    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(metrics, f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Metrics saved to {filepath}")


def load_training_metrics(filepath: str,
                         format: str = 'auto') -> Dict[str, List[float]]:
    """
    Load training metrics from file
    
    Args:
        filepath: Path to metrics file
        format: File format ('json', 'pickle', or 'auto')
        
    Returns:
        metrics: Dictionary of metric lists
    """
    if format == 'auto':
        format = 'json' if filepath.endswith('.json') else 'pickle'
    
    if format == 'json':
        with open(filepath, 'r') as f:
            metrics = json.load(f)
    elif format == 'pickle':
        with open(filepath, 'rb') as f:
            metrics = pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Metrics loaded from {filepath}")
    return metrics


def create_dummy_structured_input(num_boids: int = 5,
                                 seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Create dummy structured input for testing
    
    Args:
        num_boids: Number of boids to include
        seed: Random seed
        
    Returns:
        structured_input: Dictionary with context, predator, boids
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create dummy structured input
    structured_input = {
        'context': {
            'canvasWidth': 0.4,  # Normalized
            'canvasHeight': 0.3  # Normalized
        },
        'predator': {
            'velX': np.random.uniform(-1, 1),
            'velY': np.random.uniform(-1, 1)
        },
        'boids': []
    }
    
    # Add random boids
    for i in range(num_boids):
        boid = {
            'relX': np.random.uniform(-1, 1),
            'relY': np.random.uniform(-1, 1),
            'velX': np.random.uniform(-1, 1),
            'velY': np.random.uniform(-1, 1)
        }
        structured_input['boids'].append(boid)
    
    return structured_input


def test_model_inference(model: torch.nn.Module,
                        num_tests: int = 5,
                        num_boids: int = 10) -> List[np.ndarray]:
    """
    Test model inference with dummy inputs
    
    Args:
        model: Model to test
        num_tests: Number of test runs
        num_boids: Number of boids in test inputs
        
    Returns:
        outputs: List of model outputs
    """
    model.eval()
    outputs = []
    
    print(f"Testing model inference with {num_tests} samples...")
    
    with torch.no_grad():
        for i in range(num_tests):
            # Create dummy structured input
            structured_input = create_dummy_structured_input(
                num_boids=num_boids, 
                seed=i
            )
            
            # Forward pass
            try:
                output = model(structured_input)
                if isinstance(output, torch.Tensor):
                    output = output.cpu().numpy()
                outputs.append(output)
                
                print(f"  Test {i+1}: Output shape {output.shape}, "
                      f"Range [{output.min():.3f}, {output.max():.3f}]")
                
            except Exception as e:
                print(f"  Test {i+1}: Failed with error: {e}")
                outputs.append(None)
    
    return outputs


def calculate_observation_stats(observations: List[np.ndarray]) -> Dict[str, Any]:
    """
    Calculate statistics for a list of observations
    
    Args:
        observations: List of observation arrays
        
    Returns:
        stats: Dictionary with observation statistics
    """
    if not observations:
        return {}
    
    # Stack observations
    obs_array = np.stack(observations)
    
    stats = {
        'mean': np.mean(obs_array, axis=0),
        'std': np.std(obs_array, axis=0),
        'min': np.min(obs_array, axis=0),
        'max': np.max(obs_array, axis=0),
        'shape': obs_array.shape,
        'count': len(observations)
    }
    
    return stats


def validate_action_space(actions: List[np.ndarray],
                         action_space: gym.Space) -> Dict[str, Any]:
    """
    Validate that actions are within the action space
    
    Args:
        actions: List of action arrays
        action_space: Gym action space
        
    Returns:
        validation_results: Dictionary with validation results
    """
    if not actions:
        return {'valid': False, 'reason': 'No actions provided'}
    
    valid_count = 0
    invalid_actions = []
    
    for i, action in enumerate(actions):
        if action_space.contains(action):
            valid_count += 1
        else:
            invalid_actions.append((i, action))
    
    results = {
        'valid': len(invalid_actions) == 0,
        'valid_count': valid_count,
        'invalid_count': len(invalid_actions),
        'total_count': len(actions),
        'validity_rate': valid_count / len(actions),
        'invalid_actions': invalid_actions[:5]  # Show first 5 invalid actions
    }
    
    return results


def create_evaluation_report(metrics: Dict[str, Any],
                           save_path: Optional[str] = None) -> str:
    """
    Create a formatted evaluation report
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Optional path to save the report
        
    Returns:
        report: Formatted report string
    """
    report = f"""
🎯 EVALUATION REPORT
{'='*50}

📊 Performance Metrics:
  Mean Reward: {metrics.get('mean_reward', 0):.2f} ± {metrics.get('std_reward', 0):.2f}
  Mean Episode Length: {metrics.get('mean_length', 0):.1f}
  Mean Boids Caught: {metrics.get('mean_boids_caught', 0):.1f}
  Success Rate: {metrics.get('success_rate', 0):.2%}
  Total Episodes: {metrics.get('total_episodes', 0)}

🎮 Environment Info:
  Number of Boids: {metrics.get('num_boids', 'N/A')}
  Canvas Size: {metrics.get('canvas_width', 'N/A')}x{metrics.get('canvas_height', 'N/A')}
  Max Episode Steps: {metrics.get('max_steps', 'N/A')}

🧠 Model Info:
  Model Parameters: {metrics.get('model_parameters', 'N/A'):,}
  Architecture: {metrics.get('architecture', 'N/A')}
  Device: {metrics.get('device', 'N/A')}

📅 Evaluation Details:
  Timestamp: {metrics.get('timestamp', 'N/A')}
  Evaluation Time: {metrics.get('eval_time', 'N/A')} seconds
"""
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {save_path}")
    
    return report
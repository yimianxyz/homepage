"""
Observation Wrapper - Converts between structured and flat observations

This wrapper handles the conversion between the structured observation format
used by our transformer and the flat tensor format expected by Stable Baselines3.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, List, Union
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class StructuredObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that handles structured observations for transformer policy
    
    This wrapper stores the structured observation internally and provides
    a flat observation for SB3 compatibility while allowing the policy
    to access the structured format when needed.
    """
    
    def __init__(self, env, max_boids: int = 50):
        super().__init__(env)
        
        self.max_boids = max_boids
        self.current_structured_obs = None
        
        # Define flat observation space for SB3 compatibility
        # Structure: [canvas_width, canvas_height, pred_velX, pred_velY, boid_count, boid_features...]
        # Each boid contributes 4 features: relX, relY, velX, velY
        obs_size = 5 + max_boids * 4  # context(2) + predator(2) + count(1) + boids(4 each)
        
        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        print(f"StructuredObservationWrapper initialized:")
        print(f"  Max boids: {max_boids}")
        print(f"  Flat observation size: {obs_size}")
    
    def observation(self, structured_obs: Dict[str, Any]) -> np.ndarray:
        """
        Convert structured observation to flat array
        
        Args:
            structured_obs: Structured observation from environment
            
        Returns:
            Flat observation array for SB3
        """
        # Validate structured observation
        if not isinstance(structured_obs, dict):
            raise ValueError(f"Expected dict observation, got {type(structured_obs)}")
        
        required_keys = ['context', 'predator', 'boids']
        for key in required_keys:
            if key not in structured_obs:
                raise ValueError(f"Missing required key '{key}' in structured observation")
        
        # Store structured observation for policy access
        self.current_structured_obs = structured_obs
        
        # Extract components
        context = structured_obs['context']
        predator = structured_obs['predator']
        boids = structured_obs['boids']
        
        # Validate component structure
        if 'canvasWidth' not in context or 'canvasHeight' not in context:
            raise ValueError("Invalid context structure - missing canvas dimensions")
        if 'velX' not in predator or 'velY' not in predator:
            raise ValueError("Invalid predator structure - missing velocity")
        
        # Start with context and predator info
        flat_obs = [
            float(context['canvasWidth']),
            float(context['canvasHeight']),
            float(predator['velX']),
            float(predator['velY']),
            float(len(boids))  # Number of boids
        ]
        
        # Add boid features (pad to max_boids)
        for i in range(self.max_boids):
            if i < len(boids):
                boid = boids[i]
                # Validate boid structure
                required_boid_keys = ['relX', 'relY', 'velX', 'velY']
                for key in required_boid_keys:
                    if key not in boid:
                        raise ValueError(f"Missing required boid key '{key}' in boid {i}")
                
                flat_obs.extend([
                    float(boid['relX']),
                    float(boid['relY']),
                    float(boid['velX']),
                    float(boid['velY'])
                ])
            else:
                # Padding for missing boids
                flat_obs.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(flat_obs, dtype=np.float32)
    
    def get_structured_observation(self) -> Dict[str, Any]:
        """
        Get the current structured observation
        
        Returns:
            Current structured observation
            
        Raises:
            RuntimeError: If no structured observation is available
        """
        if self.current_structured_obs is None:
            raise RuntimeError("No structured observation available - reset() or step() must be called first")
        return self.current_structured_obs

def create_wrapped_environment(env_config: Dict[str, Any] = None, max_boids: int = 50):
    """
    Create environment with observation wrapper
    
    Args:
        env_config: Environment configuration
        max_boids: Maximum number of boids for padding
        
    Returns:
        Wrapped environment
    """
    from rl_training.ppo_environment import create_boids_environment
    
    # Create base environment
    env = create_boids_environment(env_config)
    
    # Wrap with observation converter
    wrapped_env = StructuredObservationWrapper(env, max_boids=max_boids)
    
    print(f"✓ Environment wrapped with structured observation handler")
    return wrapped_env

# Custom features extractor for Stable Baselines3
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor that uses our pretrained transformer
    """
    
    def __init__(self, observation_space: gym.Space, checkpoint_path: str = "checkpoints/best_model.pt"):
        # The features dim will be set after loading the transformer
        super().__init__(observation_space, features_dim=1)
        
        # Validate checkpoint path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        from rl_training.ppo_transformer_policy import PretrainedTransformerEncoder
        
        # Load pretrained transformer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transformer_encoder = PretrainedTransformerEncoder(checkpoint_path, device)
        
        # Update features_dim after loading
        self._features_dim = self.transformer_encoder.d_model
        
        # Store reference to environment wrapper for structured observations
        self.env_wrapper = None
        
        print(f"✓ TransformerFeaturesExtractor initialized:")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Features dim: {self._features_dim}")
        print(f"  Device: {device}")
    
    def set_env_wrapper(self, env_wrapper):
        """Set reference to environment wrapper"""
        if not hasattr(env_wrapper, 'get_structured_observation'):
            raise ValueError("Environment wrapper must have get_structured_observation method")
        self.env_wrapper = env_wrapper
        print(f"✓ Environment wrapper linked to features extractor")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features using transformer encoder
        
        Args:
            observations: Flat observations from SB3
            
        Returns:
            Features tensor
        """
        if self.env_wrapper is None:
            raise RuntimeError("Environment wrapper not set - call set_env_wrapper() first")
        
        # Get structured observations from environment wrapper
        structured_obs = self.env_wrapper.get_structured_observation()
        
        # Extract features using transformer encoder
        features = self.transformer_encoder(structured_obs)
        
        # Ensure proper batch dimension
        if len(features.shape) == 1:
            # Single observation case
            batch_size = observations.shape[0] if len(observations.shape) > 1 else 1
            if batch_size > 1:
                # Repeat features for batch
                features = features.unsqueeze(0).repeat(batch_size, 1)
            else:
                features = features.unsqueeze(0)
        
        return features

class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    Custom actor-critic policy using transformer features extractor
    """
    
    def __init__(self, *args, checkpoint_path: str = "checkpoints/best_model.pt", **kwargs):
        # Validate checkpoint path before proceeding
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.checkpoint_path = checkpoint_path
        super().__init__(*args, **kwargs)
    
    def _build_mlp_extractor(self) -> None:
        """Build MLP extractor with transformer features"""
        
        # Use transformer features extractor
        self.features_extractor = TransformerFeaturesExtractor(
            self.observation_space,
            self.checkpoint_path
        )
        
        features_dim = self.features_extractor.features_dim
        
        # Build policy and value networks
        self.mlp_extractor = torch.nn.ModuleDict({
            'policy': torch.nn.Sequential(
                torch.nn.Linear(features_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU()
            ),
            'value': torch.nn.Sequential(
                torch.nn.Linear(features_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU()
            )
        })
        
        # Set latent dimensions
        self.latent_dim_pi = 64
        self.latent_dim_vf = 64
        
        print(f"✓ Custom ActorCritic policy built:")
        print(f"  Features dim: {features_dim}")
        print(f"  Policy latent dim: {self.latent_dim_pi}")
        print(f"  Value latent dim: {self.latent_dim_vf}")

def create_custom_policy_class(checkpoint_path: str = "checkpoints/best_model.pt"):
    """
    Create custom policy class with transformer features
    
    Args:
        checkpoint_path: Path to pretrained transformer checkpoint
        
    Returns:
        Policy class for use with PPO
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    # Validate checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    class ConfiguredActorCriticPolicy(CustomActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, checkpoint_path=checkpoint_path, **kwargs)
    
    return ConfiguredActorCriticPolicy

if __name__ == "__main__":
    # Test wrapper functionality
    from rl_training.ppo_environment import create_boids_environment
    
    # Create and test wrapped environment
    env_config = {
        'canvas_width': 800,
        'canvas_height': 600,
        'initial_boids': 10,
        'max_steps': 100
    }
    
    try:
        wrapped_env = create_wrapped_environment(env_config)
        
        # Test observation conversion
        obs, info = wrapped_env.reset()
        print(f"✓ Wrapped environment test:")
        print(f"  Flat observation shape: {obs.shape}")
        print(f"  Structured observation available: {wrapped_env.get_structured_observation() is not None}")
        
        # Test a step
        action = wrapped_env.action_space.sample()
        obs, reward, done, truncated, info = wrapped_env.step(action)
        print(f"  Step test successful: reward={reward:.4f}")
        
        print("✓ Observation wrapper test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
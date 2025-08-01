"""
Policy Wrapper - Stable-baselines3 integration for transformer models

This module provides a policy wrapper that integrates our custom transformer
model with stable-baselines3, allowing us to use PPO and other algorithms
with pre-trained transformer models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
from gymnasium import spaces
import sys
import os

# Stable-baselines3 imports
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .transformer_model import TransformerModel
from config.constants import CONSTANTS


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Features extractor that converts observations to structured inputs
    for the transformer model.
    """
    
    def __init__(self, 
                 observation_space: spaces.Box,
                 max_boids: int = 50):
        """
        Initialize features extractor
        
        Args:
            observation_space: The observation space
            max_boids: Maximum number of boids in observation
        """
        # The output features dimension equals d_model since we'll extract CLS token
        super().__init__(observation_space, features_dim=64)  # Will be updated by model
        
        self.max_boids = max_boids
        
        # Calculate observation structure
        # Format: [context(2) + predator(2) + boids(4*max_boids)]
        expected_size = 2 + 2 + 4 * max_boids
        
        if observation_space.shape[0] != expected_size:
            print(f"Warning: observation space size {observation_space.shape[0]} != expected {expected_size}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Convert observations to structured format and extract features
        
        Args:
            observations: Batch of observations [batch_size, obs_size]
            
        Returns:
            features: Extracted features [batch_size, features_dim]
        """
        batch_size = observations.shape[0]
        structured_inputs = []
        
        # Convert each observation to structured input
        for i in range(batch_size):
            obs = observations[i]
            structured_input = self._observation_to_structured_input(obs)
            structured_inputs.append(structured_input)
        
        # This will be called by the transformer model in the policy
        # For now, just return the observations as-is
        # The actual transformation happens in TransformerPolicy
        return observations
    
    def _observation_to_structured_input(self, obs: torch.Tensor) -> Dict[str, Any]:
        """
        Convert single observation to structured input format
        
        Args:
            obs: Single observation vector
            
        Returns:
            structured_input: Dictionary with context, predator, boids
        """
        # Parse observation components
        idx = 0
        
        # Context: [canvasWidth, canvasHeight]
        context = {
            'canvasWidth': float(obs[idx]),
            'canvasHeight': float(obs[idx + 1])
        }
        idx += 2
        
        # Predator: [velX, velY]
        predator = {
            'velX': float(obs[idx]),
            'velY': float(obs[idx + 1])
        }
        idx += 2
        
        # Boids: [relX, relY, velX, velY] for each boid
        boids = []
        for boid_idx in range(self.max_boids):
            # Extract boid data
            rel_x = float(obs[idx])
            rel_y = float(obs[idx + 1])
            vel_x = float(obs[idx + 2])
            vel_y = float(obs[idx + 3])
            
            # Only include non-zero boids (skip padding)
            if not (rel_x == 0.0 and rel_y == 0.0 and vel_x == 0.0 and vel_y == 0.0):
                boids.append({
                    'relX': rel_x,
                    'relY': rel_y,
                    'velX': vel_x,
                    'velY': vel_y
                })
            
            idx += 4
        
        return {
            'context': context,
            'predator': predator,
            'boids': boids
        }


class TransformerPolicy(BasePolicy):
    """
    Custom policy that uses a pre-trained transformer model
    Compatible with stable-baselines3 PPO and other algorithms
    """
    
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 lr_schedule: Schedule,
                 transformer_model: Optional[TransformerModel] = None,
                 model_config: Optional[Dict[str, Any]] = None,
                 max_boids: int = 50,
                 **kwargs):
        """
        Initialize transformer policy
        
        Args:
            observation_space: Observation space
            action_space: Action space  
            lr_schedule: Learning rate schedule
            transformer_model: Pre-trained transformer model
            model_config: Configuration for creating new model
            max_boids: Maximum number of boids
            **kwargs: Additional arguments
        """
        # Filter out arguments that BasePolicy doesn't accept
        base_kwargs = {}
        for key, value in kwargs.items():
            if key not in ['use_sde', 'sde_sample_freq', 'normalize_advantage', 
                          'rollout_buffer_class', 'rollout_buffer_kwargs', 'target_kl',
                          'stats_window_size', 'tensorboard_log', 'seed', 'device',
                          '_init_setup_model']:
                base_kwargs[key] = value
        
        # Initialize base policy
        super().__init__(observation_space, action_space, **base_kwargs)
        
        self.max_boids = max_boids
        self.lr_schedule = lr_schedule
        
        # Create or use provided transformer model
        if transformer_model is not None:
            self.transformer = transformer_model
            print("Using provided transformer model")
        elif model_config is not None:
            self.transformer = TransformerModel(**model_config)
            print(f"Created new transformer model with config: {model_config}")
        else:
            # Default configuration
            self.transformer = TransformerModel(max_boids=max_boids)
            print("Created transformer model with default configuration")
        
        # Update features dim based on transformer
        self._features_dim = self.transformer.d_model
        
        # Create features extractor
        self.features_extractor = TransformerFeaturesExtractor(
            observation_space, max_boids=max_boids
        )
        
        # Value network (simple MLP)
        self.value_net = nn.Sequential(
            nn.Linear(self._features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Action network is the transformer itself
        self.action_net = self.transformer
        
        # Initialize optimizer for stable-baselines3 compatibility
        self.optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=lr_schedule(1.0),  # Initial learning rate
            eps=1e-5
        )
        
        print(f"Created TransformerPolicy:")
        print(f"  Observation space: {observation_space}")
        print(f"  Action space: {action_space}")
        print(f"  Transformer parameters: {self.transformer.count_parameters():,}")
        print(f"  Max boids: {max_boids}")
    
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """Get constructor parameters for saving/loading"""
        data = super()._get_constructor_parameters()
        data.update({
            'transformer_model': self.transformer,
            'max_boids': self.max_boids,
        })
        return data
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for both action and value prediction
        
        Args:
            obs: Observations [batch_size, obs_size]
            deterministic: Whether to use deterministic policy
            
        Returns:
            actions: Predicted actions [batch_size, action_size]
            values: State values [batch_size, 1] 
            log_probs: Log probabilities [batch_size, 1]
        """
        # Convert observations to structured inputs
        batch_size = obs.shape[0]
        structured_inputs = []
        
        for i in range(batch_size):
            structured_input = self._observation_to_structured_input(obs[i])
            structured_inputs.append(structured_input)
        
        # Get actions from transformer (deterministic)
        actions = self.transformer(structured_inputs)  # [batch_size, 2]
        
        # Get values from value network
        # Use transformer's CLS token representation for value estimation
        with torch.no_grad():
            # Re-run transformer to get CLS token features
            cls_features = self._get_cls_features(structured_inputs)
        
        values = self.value_net(cls_features).squeeze(-1)  # [batch_size]
        
        # For deterministic actions, log_probs are not meaningful
        # Return zeros as placeholder
        log_probs = torch.zeros(batch_size, device=obs.device)
        
        return actions, values, log_probs
    
    def _get_cls_features(self, structured_inputs: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Get CLS token features from transformer for value estimation
        
        Args:
            structured_inputs: List of structured input dictionaries
            
        Returns:
            cls_features: CLS token features [batch_size, d_model]
        """
        batch_size = len(structured_inputs)
        device = next(self.transformer.parameters()).device
        
        # Build token sequences
        all_tokens = []
        all_masks = []
        
        for inputs in structured_inputs:
            tokens, mask = self.transformer._build_tokens(inputs, device)
            all_tokens.append(tokens)
            all_masks.append(mask)
        
        # Pad sequences
        max_len = max(len(tokens) for tokens in all_tokens)
        padded_tokens = []
        padded_masks = []
        
        for tokens, mask in zip(all_tokens, all_masks):
            pad_len = max_len - len(tokens)
            if pad_len > 0:
                padding = torch.zeros(pad_len, self.transformer.d_model, device=device)
                tokens = torch.cat([tokens, padding], dim=0)
                mask = torch.cat([mask, torch.ones(pad_len, device=device, dtype=torch.bool)])
            
            padded_tokens.append(tokens)
            padded_masks.append(mask)
        
        # Stack into batch
        token_batch = torch.stack(padded_tokens)
        mask_batch = torch.stack(padded_masks)
        
        # Pass through transformer
        transformer_output = self.transformer.transformer(token_batch, src_key_padding_mask=mask_batch)
        
        # Extract CLS tokens
        cls_features = transformer_output[:, 0, :]  # [batch_size, d_model]
        
        return cls_features
    
    def _observation_to_structured_input(self, obs: torch.Tensor) -> Dict[str, Any]:
        """
        Convert single observation to structured input format
        
        Args:
            obs: Single observation tensor
            
        Returns:
            structured_input: Dictionary with context, predator, boids
        """
        # Convert to numpy for easier manipulation
        obs_np = obs.cpu().numpy()
        
        # Parse observation components
        idx = 0
        
        # Context: [canvasWidth, canvasHeight]
        context = {
            'canvasWidth': float(obs_np[idx]),
            'canvasHeight': float(obs_np[idx + 1])
        }
        idx += 2
        
        # Predator: [velX, velY]
        predator = {
            'velX': float(obs_np[idx]),
            'velY': float(obs_np[idx + 1])
        }
        idx += 2
        
        # Boids: [relX, relY, velX, velY] for each boid
        boids = []
        for boid_idx in range(self.max_boids):
            # Extract boid data
            rel_x = float(obs_np[idx])
            rel_y = float(obs_np[idx + 1])
            vel_x = float(obs_np[idx + 2])
            vel_y = float(obs_np[idx + 3])
            
            # Only include non-zero boids (skip padding)
            if not (rel_x == 0.0 and rel_y == 0.0 and vel_x == 0.0 and vel_y == 0.0):
                boids.append({
                    'relX': rel_x,
                    'relY': rel_y,
                    'velX': vel_x,
                    'velY': vel_y
                })
            
            idx += 4
        
        return {
            'context': context,
            'predator': predator,
            'boids': boids
        }
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict values for given observations
        
        Args:
            obs: Observations [batch_size, obs_size]
            
        Returns:
            values: Predicted values [batch_size]
        """
        # Convert observations to structured inputs
        batch_size = obs.shape[0]
        structured_inputs = []
        
        for i in range(batch_size):
            structured_input = self._observation_to_structured_input(obs[i])
            structured_inputs.append(structured_input)
        
        # Get CLS features and predict values
        cls_features = self._get_cls_features(structured_inputs)
        values = self.value_net(cls_features).squeeze(-1)
        
        return values
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given observations
        
        Args:
            obs: Observations [batch_size, obs_size]
            actions: Actions to evaluate [batch_size, action_size]
            
        Returns:
            values: State values [batch_size]
            log_probs: Log probabilities [batch_size]
            entropy: Action entropy [batch_size]
        """
        # Predict values
        values = self.predict_values(obs)
        
        # For deterministic policy, log_probs and entropy are not meaningful
        # Return zeros as placeholders
        batch_size = obs.shape[0]
        log_probs = torch.zeros(batch_size, device=obs.device)
        entropy = torch.zeros(batch_size, device=obs.device)
        
        return values, log_probs, entropy
    
    def get_distribution(self, obs: torch.Tensor):
        """
        Get action distribution (not implemented for deterministic policy)
        """
        raise NotImplementedError("TransformerPolicy uses deterministic actions")
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Predict action for given observation (internal method)
        
        Args:
            observation: Observation tensor
            deterministic: Whether to use deterministic policy
            
        Returns:
            actions: Predicted actions
        """
        # Convert observations to structured inputs
        batch_size = observation.shape[0]
        structured_inputs = []
        
        for i in range(batch_size):
            structured_input = self._observation_to_structured_input(observation[i])
            structured_inputs.append(structured_input)
        
        # Get actions from transformer (deterministic)
        actions = self.transformer(structured_inputs)  # [batch_size, 2]
        
        return actions
    
    def predict(self, 
                observation: np.ndarray,
                state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = False) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Predict action for given observation
        
        Args:
            observation: Observation array
            state: RNN state (not used)
            episode_start: Episode start flags (not used)
            deterministic: Whether to use deterministic policy
            
        Returns:
            actions: Predicted actions
            state: Updated state (None for transformer)
        """
        # Convert to tensor
        obs_tensor = torch.as_tensor(observation, device=self.device).float()
        
        # Add batch dimension if needed
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            actions = self._predict(obs_tensor, deterministic=deterministic)
        
        # Convert to numpy
        actions_np = actions.cpu().numpy()
        
        # Remove batch dimension if input was single observation
        if observation.ndim == 1 and actions_np.ndim > 1:
            actions_np = actions_np[0]
        
        return actions_np, None
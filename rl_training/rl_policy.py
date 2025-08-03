"""
TorchRL Policy Wrapper for Transformer Model

This module wraps the pre-trained transformer model into a TorchRL-compatible
policy module that can be used with PPO.
"""

import torch
import torch.nn as nn
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from tensordict.nn import TensorDictModule
from typing import Optional, Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import CONSTANTS


class TransformerRLPolicy(nn.Module):
    """
    RL Policy wrapper for the pre-trained transformer model.
    
    This module:
    - Loads the transformer from a checkpoint
    - Converts flat observations to structured format
    - Outputs action distributions for PPO
    - Includes a value head for advantage estimation
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        freeze_transformer: bool = False,
        value_hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Load transformer model
        self._load_transformer(checkpoint_path)
        
        # Optionally freeze transformer weights
        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Add value head for PPO
        self.value_head = nn.Sequential(
            nn.Linear(self.transformer.d_model, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1)
        )
        
        # Move to device
        self.to(device)
        
        print(f"✓ TransformerRLPolicy initialized")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Transformer frozen: {freeze_transformer}")
        print(f"  Device: {device}")
    
    def _load_transformer(self, checkpoint_path: str) -> None:
        """Load the transformer model from checkpoint"""
        
        # Import the transformer model class
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Load checkpoint to get architecture
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        arch = checkpoint['architecture']
        
        # Import transformer model class
        from .transformer_model_definition import TransformerPredictor
        
        self.transformer = TransformerPredictor(
            d_model=arch['d_model'],
            n_heads=arch['n_heads'],
            n_layers=arch['n_layers'],
            ffn_hidden=arch['ffn_hidden'],
            max_boids=arch.get('max_boids', 50),
            dropout=0.0  # No dropout during RL
        )
        
        # Load weights
        self.transformer.load_state_dict(checkpoint['model_state_dict'])
        self.transformer.eval()  # Set to eval mode initially
        
        print(f"✓ Loaded transformer: {arch['d_model']}×{arch['n_heads']}×{arch['n_layers']}×{arch['ffn_hidden']}")
    
    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that outputs both action and value.
        
        Args:
            observation: Flat observation tensor [batch_size, obs_dim] or [obs_dim]
            
        Returns:
            action_mean: Mean of action distribution [batch_size, 2] or [2]
            value: Value estimate [batch_size, 1] or [1]
        """
        
        # Handle both batched and single observations
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = observation.shape[0]
        
        # Convert flat observations to structured format
        structured_inputs = self._tensor_to_structured_batch(observation)
        
        # Get transformer features (before final projection)
        # We need to modify this to get intermediate features
        cls_features = self._get_transformer_features(structured_inputs)
        
        # Get action from transformer's output projection
        action_mean = self.transformer.output_projection(cls_features)
        action_mean = torch.tanh(action_mean)  # Ensure [-1, 1] range
        
        # Get value from value head
        value = self.value_head(cls_features)
        
        if squeeze_output:
            action_mean = action_mean.squeeze(0)
            value = value.squeeze(0)
        
        return action_mean, value
    
    def _get_transformer_features(self, structured_inputs: list) -> torch.Tensor:
        """Get intermediate features from transformer (CLS token after all layers)"""
        
        batch_size = len(structured_inputs)
        
        # Build token sequences for each sample
        all_sequences = []
        all_masks = []
        
        for sample in structured_inputs:
            tokens = []
            
            # CLS token
            cls_token = self.transformer.cls_embedding + self.transformer.type_embeddings['cls']
            tokens.append(cls_token)
            
            # Context token
            ctx_input = torch.tensor(
                [sample['context']['canvasWidth'], sample['context']['canvasHeight']],
                dtype=torch.float32, device=self.device
            )
            ctx_token = self.transformer.ctx_projection(ctx_input) + self.transformer.type_embeddings['ctx']
            tokens.append(ctx_token)
            
            # Predator token
            predator_input = torch.tensor(
                [sample['predator']['velX'], sample['predator']['velY'], 0.0, 0.0],
                dtype=torch.float32, device=self.device
            )
            predator_token = self.transformer.predator_projection(predator_input) + self.transformer.type_embeddings['predator']
            tokens.append(predator_token)
            
            # Boid tokens
            sample_mask = [False, False, False]  # CLS, CTX, Predator are not padding
            
            for boid in sample['boids']:
                boid_input = torch.tensor(
                    [boid['relX'], boid['relY'], boid['velX'], boid['velY']],
                    dtype=torch.float32, device=self.device
                )
                boid_token = self.transformer.boid_projection(boid_input) + self.transformer.type_embeddings['boid']
                tokens.append(boid_token)
                sample_mask.append(False)
            
            # Pad to max_boids + 3
            while len(tokens) < self.transformer.max_boids + 3:
                padding_token = torch.zeros(self.transformer.d_model, device=self.device)
                tokens.append(padding_token)
                sample_mask.append(True)
            
            all_sequences.append(torch.stack(tokens))
            all_masks.append(sample_mask)
        
        # Stack sequences
        x = torch.stack(all_sequences)  # [batch_size, seq_len, d_model]
        padding_mask = torch.tensor(all_masks, dtype=torch.bool, device=self.device)
        
        # Pass through transformer layers
        for layer in self.transformer.transformer_layers:
            x = layer(x, padding_mask)
        
        # Return CLS token features
        return x[:, 0]  # [batch_size, d_model]
    
    def _tensor_to_structured_batch(self, observations: torch.Tensor) -> list:
        """Convert batch of flat tensor observations to structured format"""
        
        batch_size = observations.shape[0]
        structured_batch = []
        
        for i in range(batch_size):
            obs = observations[i]
            structured = self._tensor_to_structured(obs)
            structured_batch.append(structured)
        
        return structured_batch
    
    def _tensor_to_structured(self, observation: torch.Tensor) -> Dict[str, Any]:
        """Convert single flat tensor observation to structured format"""
        
        # Extract components from flat observation
        canvas_w_norm = observation[0].item()
        canvas_h_norm = observation[1].item()
        pred_vx_norm = observation[2].item()
        pred_vy_norm = observation[3].item()
        
        # Build structured format
        structured = {
            'context': {
                'canvasWidth': canvas_w_norm,
                'canvasHeight': canvas_h_norm
            },
            'predator': {
                'velX': pred_vx_norm,
                'velY': pred_vy_norm
            },
            'boids': []
        }
        
        # Extract boid data
        boid_start_idx = 4
        for i in range(self.transformer.max_boids):
            idx = boid_start_idx + i * 4
            
            # Check if this is padding (all zeros)
            if (observation[idx] == 0 and observation[idx+1] == 0 and 
                observation[idx+2] == 0 and observation[idx+3] == 0):
                continue
            
            boid = {
                'relX': observation[idx].item(),
                'relY': observation[idx+1].item(),
                'velX': observation[idx+2].item(),
                'velY': observation[idx+3].item()
            }
            structured['boids'].append(boid)
        
        return structured


def create_rl_modules(
    checkpoint_path: str,
    device: torch.device = torch.device("cpu"),
    freeze_transformer: bool = False,
    action_std: float = 0.2,
) -> tuple[TensorDictModule, TensorDictModule]:
    """
    Create TorchRL modules for actor and critic.
    
    Args:
        checkpoint_path: Path to transformer checkpoint
        device: Device to run on
        freeze_transformer: Whether to freeze transformer weights
        action_std: Standard deviation for action distribution
        
    Returns:
        actor_module: TensorDictModule for actor
        value_module: TensorDictModule for critic
    """
    
    # Create base policy
    policy = TransformerRLPolicy(
        checkpoint_path=checkpoint_path,
        device=device,
        freeze_transformer=freeze_transformer
    )
    
    # Create actor module
    class ActorNet(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
            self.log_std = nn.Parameter(torch.log(torch.tensor(action_std)))
        
        def forward(self, observation):
            action_mean, _ = self.policy(observation)
            # Expand log_std to match batch size
            batch_size = action_mean.shape[0] if action_mean.dim() > 1 else 1
            log_std = self.log_std.expand(batch_size, 2)
            return action_mean, log_std
    
    actor_net = ActorNet(policy)
    
    # Create actor module with TanhNormal distribution
    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )
    
    actor_module = ProbabilisticActor(
        module=actor_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
        log_prob_key="sample_log_prob",
    )
    
    # Create value module
    class ValueNet(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
        
        def forward(self, observation):
            _, value = self.policy(observation)
            return value
    
    value_net = ValueNet(policy)
    
    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )
    
    return actor_module, value_module
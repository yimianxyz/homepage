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
        # Don't set to eval mode - let the training loop control this
        
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
        
        # Get transformer features directly from flat tensor without conversion
        cls_features = self._get_transformer_features_from_tensor(observation)
        
        # Get action from transformer's output projection
        action_mean = self.transformer.output_projection(cls_features)
        action_mean = torch.tanh(action_mean)  # Ensure [-1, 1] range
        
        # Get value from value head
        value = self.value_head(cls_features)
        
        if squeeze_output:
            action_mean = action_mean.squeeze(0)
            value = value.squeeze(0)
        
        return action_mean, value
    
    def _get_transformer_features_from_tensor(self, observations: torch.Tensor) -> torch.Tensor:
        """Get transformer features directly from flat tensor observations"""
        
        batch_size = observations.shape[0]
        device = observations.device
        
        # Build token sequences for each sample in batch
        all_sequences = []
        all_masks = []
        
        for b in range(batch_size):
            obs = observations[b]  # [obs_dim]
            tokens = []
            
            # CLS token
            cls_token = self.transformer.cls_embedding + self.transformer.type_embeddings['cls']
            tokens.append(cls_token)
            
            # Context token - extract canvas dimensions (first 2 elements)
            ctx_input = obs[:2]  # [canvas_w_norm, canvas_h_norm]
            ctx_token = self.transformer.ctx_projection(ctx_input) + self.transformer.type_embeddings['ctx']
            tokens.append(ctx_token)
            
            # Predator token - extract predator velocities (next 2 elements, pad to 4D)
            pred_vel = obs[2:4]  # [pred_vx_norm, pred_vy_norm]
            pred_input = torch.cat([pred_vel, torch.zeros(2, device=device)], dim=0)  # Pad to 4D
            predator_token = self.transformer.predator_projection(pred_input) + self.transformer.type_embeddings['predator']
            tokens.append(predator_token)
            
            # Boid tokens - extract boid data (remaining elements, 4 per boid)
            sample_mask = [False, False, False]  # CLS, CTX, Predator are not padding
            boid_start = 4
            
            for i in range(self.transformer.max_boids):
                boid_idx = boid_start + i * 4
                
                if boid_idx + 3 < obs.shape[0]:
                    boid_data = obs[boid_idx:boid_idx + 4]  # [rel_x, rel_y, vel_x, vel_y]
                    
                    # Check if this is padding (all zeros) - use small epsilon for floating point comparison
                    is_padding = torch.all(torch.abs(boid_data) < 1e-6)
                    
                    if not is_padding:
                        boid_token = self.transformer.boid_projection(boid_data) + self.transformer.type_embeddings['boid']
                        tokens.append(boid_token)
                        sample_mask.append(False)
                    else:
                        # This is padding
                        padding_token = torch.zeros(self.transformer.d_model, device=device)
                        tokens.append(padding_token)
                        sample_mask.append(True)
                else:
                    # Beyond observation size - add padding
                    padding_token = torch.zeros(self.transformer.d_model, device=device)
                    tokens.append(padding_token)
                    sample_mask.append(True)
            
            # Pad to max_boids + 3 if needed
            while len(tokens) < self.transformer.max_boids + 3:
                padding_token = torch.zeros(self.transformer.d_model, device=device)
                tokens.append(padding_token)
                sample_mask.append(True)
            
            all_sequences.append(torch.stack(tokens))
            all_masks.append(sample_mask)
        
        # Stack sequences
        x = torch.stack(all_sequences)  # [batch_size, seq_len, d_model]
        padding_mask = torch.tensor(all_masks, dtype=torch.bool, device=device)
        
        # Pass through transformer layers
        for layer in self.transformer.transformer_layers:
            x = layer(x, padding_mask)
        
        # Return CLS token features
        return x[:, 0]  # [batch_size, d_model]
    
    def _tensor_to_structured_batch(self, observations: torch.Tensor) -> list:
        """Convert batch of flat tensor observations to structured format (for evaluation only)"""
        
        batch_size = observations.shape[0]
        structured_batch = []
        
        with torch.no_grad():  # Only used for evaluation, safe to detach
            for i in range(batch_size):
                obs = observations[i]
                structured = self._tensor_to_structured(obs)
                structured_batch.append(structured)
        
        return structured_batch
    
    def _tensor_to_structured(self, observation: torch.Tensor) -> Dict[str, Any]:
        """Convert single flat tensor observation to structured format (for evaluation only)"""
        
        # Extract components from flat observation (detach for evaluation)
        obs_cpu = observation.detach().cpu()
        canvas_w_norm = obs_cpu[0].item()
        canvas_h_norm = obs_cpu[1].item()
        pred_vx_norm = obs_cpu[2].item()
        pred_vy_norm = obs_cpu[3].item()
        
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
            
            if idx + 3 < len(obs_cpu):
                # Check if this is padding (all zeros) with epsilon for floating point
                boid_data = obs_cpu[idx:idx+4]
                if torch.all(torch.abs(boid_data) < 1e-6):
                    continue
                
                boid = {
                    'relX': boid_data[0].item(),
                    'relY': boid_data[1].item(),
                    'velX': boid_data[2].item(),
                    'velY': boid_data[3].item()
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
        def __init__(self, policy, action_std):
            super().__init__()
            self.policy = policy
            self.log_std = nn.Parameter(torch.log(torch.tensor(action_std)))
        
        def forward(self, observation):
            action_mean, _ = self.policy(observation)
            # Expand log_std to match batch size and convert to scale (std dev)
            batch_size = action_mean.shape[0] if action_mean.dim() > 1 else 1
            log_std = self.log_std.expand(batch_size, 2)
            scale = torch.exp(log_std)  # Convert log_std to std (scale)
            return action_mean, scale
    
    actor_net = ActorNet(policy, action_std)
    
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
"""
PPO Transformer Policy - Actor-Critic model based on pretrained transformer

This module implements a Stable Baselines3 compatible policy that uses the pretrained
transformer encoder as a shared feature extractor for both policy and value networks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Normal
from stable_baselines3.common.type_aliases import Schedule
import gym
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import CONSTANTS

class PretrainedTransformerEncoder(nn.Module):
    """
    Pretrained transformer encoder loaded from checkpoint
    
    This loads the exact same architecture as the supervised learning model
    and reuses the learned representations for RL fine-tuning.
    """
    
    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        # Load checkpoint and extract model components
        self._load_pretrained_weights()
        
        print(f"✓ Loaded pretrained transformer encoder:")
        print(f"  Architecture: {self.d_model}×{self.n_heads}×{self.n_layers}×{self.ffn_hidden}")
        print(f"  Parameters: {self._count_parameters():,}")
        print(f"  Checkpoint: {checkpoint_path}")
    
    def _load_pretrained_weights(self):
        """Load pretrained weights from checkpoint"""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Extract architecture
            arch = checkpoint['architecture']
            self.d_model = arch['d_model']
            self.n_heads = arch['n_heads']
            self.n_layers = arch['n_layers']
            self.ffn_hidden = arch['ffn_hidden']
            self.head_dim = self.d_model // self.n_heads
            
            # Extract and convert model parameters to PyTorch modules
            state_dict = checkpoint['model_state_dict']
            
            # Embeddings
            self.cls_embedding = nn.Parameter(state_dict['cls_embedding'].clone())
            
            # Type embeddings
            type_params = {}
            for key in ['cls', 'ctx', 'predator', 'boid']:
                type_params[key] = nn.Parameter(state_dict[f'type_embeddings.{key}'].clone())
            self.type_embeddings = nn.ParameterDict(type_params)
            
            # Input projections
            self.ctx_projection = nn.Linear(2, self.d_model)
            self.ctx_projection.weight.data = state_dict['ctx_projection.weight'].clone()
            self.ctx_projection.bias.data = state_dict['ctx_projection.bias'].clone()
            
            self.predator_projection = nn.Linear(4, self.d_model)
            self.predator_projection.weight.data = state_dict['predator_projection.weight'].clone()
            self.predator_projection.bias.data = state_dict['predator_projection.bias'].clone()
            
            self.boid_projection = nn.Linear(4, self.d_model)
            self.boid_projection.weight.data = state_dict['boid_projection.weight'].clone()
            self.boid_projection.bias.data = state_dict['boid_projection.bias'].clone()
            
            # Transformer layers
            self.transformer_layers = nn.ModuleList()
            for i in range(self.n_layers):
                layer = TransformerLayer(self.d_model, self.n_heads, self.ffn_hidden)
                
                # Load weights for this layer
                layer.norm1.weight.data = state_dict[f'transformer_layers.{i}.norm1.weight'].clone()
                layer.norm1.bias.data = state_dict[f'transformer_layers.{i}.norm1.bias'].clone()
                
                layer.self_attn.in_proj_weight.data = state_dict[f'transformer_layers.{i}.self_attn.in_proj_weight'].clone()
                layer.self_attn.in_proj_bias.data = state_dict[f'transformer_layers.{i}.self_attn.in_proj_bias'].clone()
                layer.self_attn.out_proj.weight.data = state_dict[f'transformer_layers.{i}.self_attn.out_proj.weight'].clone()
                layer.self_attn.out_proj.bias.data = state_dict[f'transformer_layers.{i}.self_attn.out_proj.bias'].clone()
                
                layer.norm2.weight.data = state_dict[f'transformer_layers.{i}.norm2.weight'].clone()
                layer.norm2.bias.data = state_dict[f'transformer_layers.{i}.norm2.bias'].clone()
                
                layer.ffn_gate_proj.weight.data = state_dict[f'transformer_layers.{i}.ffn_gate_proj.weight'].clone()
                layer.ffn_gate_proj.bias.data = state_dict[f'transformer_layers.{i}.ffn_gate_proj.bias'].clone()
                
                layer.ffn_up_proj.weight.data = state_dict[f'transformer_layers.{i}.ffn_up_proj.weight'].clone()
                layer.ffn_up_proj.bias.data = state_dict[f'transformer_layers.{i}.ffn_up_proj.bias'].clone()
                
                layer.ffn_down_proj.weight.data = state_dict[f'transformer_layers.{i}.ffn_down_proj.weight'].clone()
                layer.ffn_down_proj.bias.data = state_dict[f'transformer_layers.{i}.ffn_down_proj.bias'].clone()
                
                self.transformer_layers.append(layer)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained checkpoint {self.checkpoint_path}: {e}")
    
    def forward(self, structured_inputs: Union[Dict[str, Any], List[Dict[str, Any]]]) -> torch.Tensor:
        """
        Forward pass through transformer encoder
        
        Args:
            structured_inputs: Structured input dict or batch of dicts
            
        Returns:
            CLS token features [batch_size, d_model]
        """
        # Handle single sample vs batch
        if isinstance(structured_inputs, dict):
            structured_inputs = [structured_inputs]
            batch_size = 1
        else:
            batch_size = len(structured_inputs)
        
        # Build token sequences for each sample in batch
        sequences = []
        
        for sample in structured_inputs:
            tokens = []
            
            # CLS token
            cls_token = self.cls_embedding + self.type_embeddings['cls']
            tokens.append(cls_token)
            
            # Context token
            ctx_input = torch.tensor([
                sample['context']['canvasWidth'],
                sample['context']['canvasHeight']
            ], dtype=torch.float32, device=self.device)
            ctx_token = self.ctx_projection(ctx_input) + self.type_embeddings['ctx']
            tokens.append(ctx_token)
            
            # Predator token - expand to 4D
            predator_input = torch.tensor([
                sample['predator']['velX'],
                sample['predator']['velY'],
                0.0, 0.0  # padding
            ], dtype=torch.float32, device=self.device)
            predator_token = self.predator_projection(predator_input) + self.type_embeddings['predator']
            tokens.append(predator_token)
            
            # Boid tokens
            for boid in sample['boids']:
                boid_input = torch.tensor([
                    boid['relX'], boid['relY'], boid['velX'], boid['velY']
                ], dtype=torch.float32, device=self.device)
                boid_token = self.boid_projection(boid_input) + self.type_embeddings['boid']
                tokens.append(boid_token)
            
            sequences.append(torch.stack(tokens))
        
        # Pad sequences to same length
        max_len = max(seq.shape[0] for seq in sequences)
        padded_sequences = []
        attention_masks = []
        
        for seq in sequences:
            seq_len = seq.shape[0]
            
            # Pad sequence
            if seq_len < max_len:
                padding = torch.zeros(max_len - seq_len, self.d_model, device=self.device)
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq
            
            # Create attention mask (True for padding)
            mask = torch.zeros(max_len, dtype=torch.bool, device=self.device)
            mask[seq_len:] = True
            
            padded_sequences.append(padded_seq)
            attention_masks.append(mask)
        
        # Stack sequences
        x = torch.stack(padded_sequences)  # [batch_size, seq_len, d_model]
        padding_mask = torch.stack(attention_masks)  # [batch_size, seq_len]
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, padding_mask)
        
        # Extract CLS token features
        cls_features = x[:, 0]  # [batch_size, d_model]
        
        return cls_features.squeeze(0) if batch_size == 1 else cls_features
    
    def _count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())

class TransformerLayer(nn.Module):
    """Single transformer layer matching the pretrained architecture"""
    
    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(d_model)
        
        # GEGLU FFN with separate projections for export compatibility
        self.ffn_gate_proj = nn.Linear(d_model, ffn_hidden)
        self.ffn_up_proj = nn.Linear(d_model, ffn_hidden)
        self.ffn_down_proj = nn.Linear(ffn_hidden, d_model)
    
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, key_padding_mask=padding_mask)
        x = x + attn_out
        
        # FFN with residual
        normed = self.norm2(x)
        gate = torch.nn.functional.gelu(self.ffn_gate_proj(normed))
        up = self.ffn_up_proj(normed)
        ffn_out = self.ffn_down_proj(gate * up)
        x = x + ffn_out
        
        return x

class PPOTransformerPolicy(ActorCriticPolicy):
    """
    PPO policy using pretrained transformer encoder
    
    This policy uses a shared transformer encoder (initialized from pretrained weights)
    with separate heads for policy (actor) and value (critic) estimation.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: type = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        checkpoint_path: str = "checkpoints/best_model.pt"
    ):
        # Store checkpoint path for later use
        self.checkpoint_path = checkpoint_path
        
        super().__init__(
            observation_space,
            action_space, 
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs
        )
    
    def _build_mlp_extractor(self) -> None:
        """Build the shared feature extractor and policy/value networks"""
        
        # Load pretrained transformer encoder
        self.features_extractor = PretrainedTransformerEncoder(
            self.checkpoint_path,
            device=self.device
        )
        
        # Get feature dimension from transformer
        self.features_dim = self.features_extractor.d_model
        
        # Policy network (actor) - outputs action distribution parameters
        self.policy_net = nn.Sequential(
            nn.Linear(self.features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, int(np.prod(self.action_space.shape)))
        )
        
        # Value network (critic) - outputs state value
        self.value_net = nn.Sequential(
            nn.Linear(self.features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize policy and value networks
        self._init_networks()
        
        print(f"✓ PPO Transformer Policy built:")
        print(f"  Shared features: {self.features_dim}")
        print(f"  Policy net: {sum(p.numel() for p in self.policy_net.parameters())} params")
        print(f"  Value net: {sum(p.numel() for p in self.value_net.parameters())} params")
    
    def _init_networks(self):
        """Initialize policy and value networks"""
        for module in [self.policy_net, self.value_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.orthogonal_(layer.weight, 0.01)
                    torch.nn.init.constant_(layer.bias, 0.0)
    
    def extract_features(self, obs: Union[torch.Tensor, Dict[str, Any]]) -> torch.Tensor:
        """
        Extract features using the pretrained transformer encoder
        
        Args:
            obs: Observation (can be structured dict or flattened tensor)
            
        Returns:
            Feature tensor [batch_size, features_dim]
        """
        # If obs is a tensor, we need to convert it back to structured format
        # For now, assume obs is already in structured format
        if isinstance(obs, torch.Tensor):
            # This would require converting flattened obs back to structured format
            # For simplicity, we'll assume structured format is passed directly
            raise NotImplementedError("Tensor observations not yet supported - use structured format")
        
        return self.features_extractor(obs)
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for both policy and value
        
        Args:
            obs: Observations
            deterministic: Whether to use deterministic actions
            
        Returns:
            actions, values, log_probs
        """
        # Extract features
        features = self.extract_features(obs)
        
        # Get action distribution
        action_mean = self.policy_net(features)
        action_mean = torch.tanh(action_mean)  # Ensure [-1, 1] range
        
        # Create action distribution
        if self.use_sde:
            # State-dependent exploration
            action_std = torch.ones_like(action_mean) * torch.exp(self.log_std)
        else:
            # Fixed exploration
            action_std = torch.ones_like(action_mean) * torch.exp(self.log_std)
        
        distribution = Normal(action_mean, action_std)
        
        # Sample actions
        if deterministic:
            actions = action_mean
        else:
            actions = distribution.sample()
        
        # Clip actions to [-1, 1]
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # Calculate log probabilities
        log_probs = distribution.log_prob(actions).sum(dim=-1)
        
        # Get values
        values = self.value_net(features).flatten()
        
        return actions, values, log_probs
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict state values"""
        features = self.extract_features(obs)
        return self.value_net(features).flatten()
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO updates"""
        features = self.extract_features(obs)
        
        # Get action distribution
        action_mean = self.policy_net(features)
        action_mean = torch.tanh(action_mean)
        
        if self.use_sde:
            action_std = torch.ones_like(action_mean) * torch.exp(self.log_std)
        else:
            action_std = torch.ones_like(action_mean) * torch.exp(self.log_std)
        
        distribution = Normal(action_mean, action_std)
        
        # Calculate log probabilities and entropy
        log_probs = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        
        # Get values
        values = self.value_net(features).flatten()
        
        return values, log_probs, entropy

# Factory function for creating the policy
def create_ppo_transformer_policy(checkpoint_path: str = "checkpoints/best_model.pt"):
    """
    Factory function to create PPO transformer policy class
    
    Args:
        checkpoint_path: Path to pretrained transformer checkpoint
        
    Returns:
        Policy class that can be used with PPO
    """
    
    class ConfiguredPPOTransformerPolicy(PPOTransformerPolicy):
        def __init__(self, *args, **kwargs):
            kwargs['checkpoint_path'] = checkpoint_path
            super().__init__(*args, **kwargs)
    
    return ConfiguredPPOTransformerPolicy

if __name__ == "__main__":
    # Test policy creation
    import gym
    from stable_baselines3.common.env_util import make_vec_env
    
    # Create dummy environment for testing
    env = gym.make('CartPole-v1')
    
    try:
        # Test policy creation
        PolicyClass = create_ppo_transformer_policy("checkpoints/best_model.pt")
        
        print("✓ PPO Transformer Policy class created successfully")
        print("  Ready for use with Stable Baselines3 PPO")
        
    except Exception as e:
        print(f"Error creating policy: {e}")
        print("Make sure best_model.pt exists in checkpoints/")
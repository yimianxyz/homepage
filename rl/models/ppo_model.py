"""
PPO Model - Actor-Critic model for reinforcement learning

This model extends the supervised learning transformer checkpoint by adding
a critic head for value estimation while keeping the original actor (policy) head.

Architecture:
- Actor: Pre-trained transformer (from supervised learning)
- Critic: Additional value head attached to transformer features
- Both share the same transformer backbone for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.constants import CONSTANTS

class GEGLU(nn.Module):
    """GEGLU activation function used in transformer layers"""
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class TransformerLayer(nn.Module):
    """Transformer layer matching the supervised learning architecture"""
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
        gate = F.gelu(self.ffn_gate_proj(normed))
        up = self.ffn_up_proj(normed)
        ffn_out = self.ffn_down_proj(gate * up)
        x = x + ffn_out

        return x

class PPOModel(nn.Module):
    """
    PPO Actor-Critic model extending supervised learning transformer
    
    This model loads a pre-trained transformer checkpoint and adds a critic head
    for value estimation. The actor (policy) uses the original output projection,
    while the critic uses a new value projection head.
    """
    
    def __init__(self, 
                 d_model: int = 128, 
                 n_heads: int = 8, 
                 n_layers: int = 4, 
                 ffn_hidden: int = 512, 
                 max_boids: int = 50, 
                 dropout: float = 0.1,
                 debug: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_hidden = ffn_hidden
        self.max_boids = max_boids
        self.debug = debug
        
        if self.debug:
            print(f"🏗️  Initializing PPOModel:")
            print(f"   Architecture: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
            print(f"   FFN hidden: {ffn_hidden}, max_boids: {max_boids}")

        # CLS token embedding
        self.cls_embedding = nn.Parameter(torch.randn(d_model))

        # Type embeddings
        self.type_embeddings = nn.ParameterDict({
            'cls': nn.Parameter(torch.randn(d_model)),
            'ctx': nn.Parameter(torch.randn(d_model)),
            'predator': nn.Parameter(torch.randn(d_model)),
            'boid': nn.Parameter(torch.randn(d_model))
        })

        # Input projections (same as supervised learning)
        self.ctx_projection = nn.Linear(2, d_model)  # canvas_width, canvas_height
        self.predator_projection = nn.Linear(4, d_model)  # velX, velY, 0, 0 (padded to 4D)
        self.boid_projection = nn.Linear(4, d_model)  # relX, relY, velX, velY

        # Transformer layers (shared backbone)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, ffn_hidden, dropout)
            for _ in range(n_layers)
        ])

        # Actor head (policy) - same as supervised learning
        self.actor_projection = nn.Linear(d_model, 2)  # predator action [x, y]
        
        # Critic head (value function) - new for RL
        self.critic_projection = nn.Linear(d_model, 1)  # state value
        
        # Action distribution parameters
        self.log_std = nn.Parameter(torch.zeros(2))  # Learnable standard deviation
        
        if self.debug:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")

    def forward(self, structured_inputs: List[Dict[str, Any]], 
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            structured_inputs: List of structured input dicts or single dict
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
            - action_mean: Mean action values [batch_size, 2]
            - action_logstd: Log standard deviation [2]
            - value: State value [batch_size, 1]
            - features: CLS token features [batch_size, d_model] (if return_features=True)
        """
        batch_size = len(structured_inputs) if isinstance(structured_inputs, list) else 1

        # Handle single sample vs batch
        if isinstance(structured_inputs, dict):
            structured_inputs = [structured_inputs]
            batch_size = 1

        # Detailed debug output for development
        if self.debug and batch_size <= 2:
            print(f"🔍 PPOModel forward pass:")
            print(f"   Batch size: {batch_size}")
            print(f"   Sample 0 - boids: {len(structured_inputs[0]['boids'])}")
            if len(structured_inputs[0]['boids']) > 0:
                first_boid = structured_inputs[0]['boids'][0]
                print(f"   First boid: rel=({first_boid['relX']:.3f}, {first_boid['relY']:.3f}), vel=({first_boid['velX']:.3f}, {first_boid['velY']:.3f})")
            print(f"   Predator vel: ({structured_inputs[0]['predator']['velX']:.3f}, {structured_inputs[0]['predator']['velY']:.3f})")
            print(f"   Canvas: {structured_inputs[0]['context']['canvasWidth']:.1f}x{structured_inputs[0]['context']['canvasHeight']:.1f}")

        # Build token sequences for each sample in batch
        sequences = []
        masks = []

        for sample in structured_inputs:
            tokens = []

            # CLS token
            cls_token = self.cls_embedding + self.type_embeddings['cls']
            tokens.append(cls_token)

            # Context token
            ctx_input = torch.tensor(
                [sample['context']['canvasWidth'], sample['context']['canvasHeight']],
                dtype=torch.float32, device=self.cls_embedding.device
            )
            ctx_token = self.ctx_projection(ctx_input) + self.type_embeddings['ctx']
            tokens.append(ctx_token)

            # Predator token - expand to 4D
            predator_input = torch.tensor(
                [sample['predator']['velX'], sample['predator']['velY'], 0.0, 0.0],
                dtype=torch.float32, device=self.cls_embedding.device
            )
            predator_token = self.predator_projection(predator_input) + self.type_embeddings['predator']
            tokens.append(predator_token)

            # Boid tokens
            sample_mask = [False, False, False]  # CLS, CTX, Predator are not padding

            for boid in sample['boids']:
                boid_input = torch.tensor(
                    [boid['relX'], boid['relY'], boid['velX'], boid['velY']],
                    dtype=torch.float32, device=self.cls_embedding.device
                )
                boid_token = self.boid_projection(boid_input) + self.type_embeddings['boid']
                tokens.append(boid_token)
                sample_mask.append(False)

            # Pad to max_boids + 3 (CLS + CTX + Predator)
            while len(tokens) < self.max_boids + 3:
                padding_token = torch.zeros(self.d_model, device=self.cls_embedding.device)
                tokens.append(padding_token)
                sample_mask.append(True)  # Mark as padding

            sequences.append(torch.stack(tokens))
            masks.append(sample_mask)

        # Stack sequences
        x = torch.stack(sequences)  # [batch_size, seq_len, d_model]

        # Create padding mask
        padding_mask = torch.tensor(masks, dtype=torch.bool, device=x.device)

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, padding_mask)

        # Extract CLS token features
        cls_features = x[:, 0]  # [batch_size, d_model]

        # Actor output (policy)
        action_mean = torch.tanh(self.actor_projection(cls_features))  # [batch_size, 2]
        
        # Critic output (value function)
        value = self.critic_projection(cls_features)  # [batch_size, 1]
        
        result = {
            'action_mean': action_mean.squeeze(0) if batch_size == 1 else action_mean,
            'action_logstd': self.log_std,
            'value': value.squeeze(0) if batch_size == 1 else value
        }
        
        if return_features:
            result['features'] = cls_features.squeeze(0) if batch_size == 1 else cls_features
            
        # Detailed output logging for development
        if self.debug and batch_size <= 2:
            print(f"   → Action mean: {result['action_mean']}")
            print(f"   → Value prediction: {result['value']}")
            print(f"   → Action log_std: {result['action_logstd']}")
            if torch.is_tensor(result['action_mean']):
                action_magnitude = torch.norm(result['action_mean'])
                print(f"   → Action magnitude: {action_magnitude:.3f}")
            print(f"   ─" * 50)

        return result

    def get_action_and_value(self, structured_inputs: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action sample and value for given inputs
        
        Returns:
            action: Sampled action [batch_size, 2]
            action_logprob: Log probability of action [batch_size]
            value: State value [batch_size, 1]
        """
        outputs = self.forward(structured_inputs)
        
        action_mean = outputs['action_mean']
        action_std = torch.exp(outputs['action_logstd'])
        value = outputs['value']
        
        # Create action distribution
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        # Sample action
        action = action_dist.sample()
        action_logprob = action_dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions
        
        return action, action_logprob, value

    def get_action_logprob_and_value(self, structured_inputs: List[Dict[str, Any]], 
                                   actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get log probability of given actions and value
        
        Args:
            structured_inputs: Batch of structured inputs
            actions: Actions to evaluate [batch_size, 2]
            
        Returns:
            action_logprob: Log probability of actions [batch_size]
            entropy: Action distribution entropy [batch_size]
            value: State value [batch_size, 1]
        """
        outputs = self.forward(structured_inputs)
        
        action_mean = outputs['action_mean']
        action_std = torch.exp(outputs['action_logstd'])
        value = outputs['value']
        
        # Create action distribution
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        # Get log probability and entropy
        action_logprob = action_dist.log_prob(actions).sum(dim=-1)  # Sum over action dimensions
        entropy = action_dist.entropy().sum(dim=-1)  # Sum over action dimensions
        
        return action_logprob, entropy, value

    def load_supervised_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load supervised learning checkpoint and initialize the model
        
        Args:
            checkpoint_path: Path to the supervised learning checkpoint
            
        Returns:
            Success flag
        """
        try:
            if self.debug:
                print(f"🔄 Loading supervised checkpoint: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract architecture info
            if 'architecture' in checkpoint:
                arch = checkpoint['architecture']
                if self.debug:
                    print(f"   Checkpoint architecture: {arch}")
                
                # Verify architecture compatibility
                if (arch.get('d_model') != self.d_model or 
                    arch.get('n_heads') != self.n_heads or
                    arch.get('n_layers') != self.n_layers or
                    arch.get('ffn_hidden') != self.ffn_hidden):
                    print(f"⚠️  Architecture mismatch!")
                    print(f"   Expected: d_model={self.d_model}, n_heads={self.n_heads}")
                    print(f"   Found: d_model={arch.get('d_model')}, n_heads={arch.get('n_heads')}")
                    return False
            
            # Load state dict
            state_dict = checkpoint['model_state_dict']
            
            # Create mapping for renamed output projection
            our_state_dict = {}
            for key, value in state_dict.items():
                if key == 'output_projection.weight':
                    our_state_dict['actor_projection.weight'] = value
                elif key == 'output_projection.bias':
                    our_state_dict['actor_projection.bias'] = value
                else:
                    our_state_dict[key] = value
            
            # Load weights (missing keys for critic will remain randomly initialized)
            missing_keys, unexpected_keys = self.load_state_dict(our_state_dict, strict=False)
            
            if self.debug:
                print(f"   ✅ Loaded {len(state_dict)} parameters")
                if missing_keys:
                    print(f"   Missing keys (new critic components): {missing_keys}")
                if unexpected_keys:
                    print(f"   Unexpected keys: {unexpected_keys}")
                
                epoch = checkpoint.get('epoch', 'unknown')
                val_loss = checkpoint.get('val_loss', 'unknown')
                print(f"   Checkpoint epoch: {epoch}, val_loss: {val_loss}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return False

    def save_checkpoint(self, path: str, epoch: int, **kwargs) -> bool:
        """
        Save model checkpoint with metadata
        
        Args:
            path: Path to save checkpoint
            epoch: Training epoch
            **kwargs: Additional metadata
            
        Returns:
            Success flag
        """
        try:
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'epoch': epoch,
                'architecture': {
                    'd_model': self.d_model,
                    'n_heads': self.n_heads,
                    'n_layers': self.n_layers,
                    'ffn_hidden': self.ffn_hidden,
                    'max_boids': self.max_boids
                },
                'model_type': 'PPO',
                **kwargs
            }
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(checkpoint, path)
            
            if self.debug:
                print(f"✅ Saved checkpoint: {path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            return False


def create_ppo_model(checkpoint_path: str, 
                    device: torch.device = None,
                    debug: bool = True) -> Optional[PPOModel]:
    """
    Factory function to create and load PPO model from supervised checkpoint
    
    Args:
        checkpoint_path: Path to supervised learning checkpoint
        device: Device to load model on
        debug: Enable debug logging
        
    Returns:
        Loaded PPO model or None if failed
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        if debug:
            print(f"🏭 Creating PPO model from checkpoint: {checkpoint_path}")
            print(f"   Device: {device}")
        
        # Load checkpoint to get architecture
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'architecture' not in checkpoint:
            print(f"❌ No architecture info in checkpoint")
            return None
        
        arch = checkpoint['architecture']
        
        # Create model with checkpoint architecture
        model = PPOModel(
            d_model=arch['d_model'],
            n_heads=arch['n_heads'],
            n_layers=arch['n_layers'],
            ffn_hidden=arch['ffn_hidden'],
            max_boids=arch.get('max_boids', 50),
            debug=debug
        ).to(device)
        
        # Load supervised weights
        success = model.load_supervised_checkpoint(checkpoint_path)
        
        if not success:
            print(f"❌ Failed to load supervised checkpoint")
            return None
        
        if debug:
            print(f"✅ PPO model created successfully!")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error creating PPO model: {e}")
        return None


if __name__ == "__main__":
    # Test the PPO model
    print("🧪 Testing PPO Model...")
    
    # Test model creation
    model = PPOModel(debug=True)
    
    # Test forward pass
    test_input = {
        'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
        'predator': {'velX': 0.1, 'velY': -0.2},
        'boids': [
            {'relX': 0.1, 'relY': 0.3, 'velX': 0.5, 'velY': -0.1},
            {'relX': -0.2, 'relY': 0.1, 'velX': -0.3, 'velY': 0.4}
        ]
    }
    
    print("\n🔍 Testing forward pass...")
    with torch.no_grad():
        outputs = model.forward([test_input])
        print(f"✅ Forward pass successful!")
        
        action, logprob, value = model.get_action_and_value([test_input])
        print(f"✅ Action sampling successful!")
        print(f"   Action: {action}")
        print(f"   Value: {value}")
    
    print("\n✅ PPO Model tests passed!") 
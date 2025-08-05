"""
PPO Transformer Model - Extends SL transformer with value head for PPO training

This module creates a PPO-compatible transformer that:
1. Loads the supervised learning baseline from best_model.pt
2. Adds a value head for critic functionality
3. Maintains the same policy interface for seamless integration
4. Supports both action prediction (policy) and value estimation (critic)

Design: Shared transformer backbone + separate policy/value heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from typing import Dict, List, Any, Tuple
import sys

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class GEGLU(nn.Module):
    """GEGLU activation for FFN"""
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)

class TransformerLayer(nn.Module):
    """Transformer layer with multi-head attention and GEGLU FFN"""
    def __init__(self, d_model, n_heads, ffn_hidden, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model)

        # GEGLU FFN with separate projections
        self.ffn_gate_proj = nn.Linear(d_model, ffn_hidden)
        self.ffn_up_proj = nn.Linear(d_model, ffn_hidden)
        self.ffn_down_proj = nn.Linear(ffn_hidden, d_model)

    def forward(self, x, padding_mask=None):
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

class PPOTransformerModel(nn.Module):
    """
    PPO Transformer Model with shared backbone and dual heads
    
    Architecture:
    - Shared transformer backbone (loaded from SL checkpoint)
    - Policy head: outputs action logits
    - Value head: outputs state value estimate
    """
    
    def __init__(self, d_model=128, n_heads=8, n_layers=4, ffn_hidden=512, max_boids=50, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_hidden = ffn_hidden
        self.max_boids = max_boids

        # CLS token embedding
        self.cls_embedding = nn.Parameter(torch.randn(d_model))

        # Type embeddings
        self.type_embeddings = nn.ParameterDict({
            'cls': nn.Parameter(torch.randn(d_model)),
            'ctx': nn.Parameter(torch.randn(d_model)),
            'predator': nn.Parameter(torch.randn(d_model)),
            'boid': nn.Parameter(torch.randn(d_model))
        })

        # Input projections
        self.ctx_projection = nn.Linear(2, d_model)
        self.predator_projection = nn.Linear(4, d_model)
        self.boid_projection = nn.Linear(4, d_model)

        # Shared transformer backbone
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, ffn_hidden, dropout)
            for _ in range(n_layers)
        ])

        # Dual heads
        self.policy_head = nn.Linear(d_model, 2)  # Action [x, y]
        self.value_head = nn.Linear(d_model, 1)   # State value

    def forward(self, structured_inputs, return_value=True):
        """
        Forward pass through transformer
        
        Args:
            structured_inputs: Same format as existing policy interface
            return_value: Whether to compute value estimate
            
        Returns:
            If return_value=True: (action_logits, value)
            If return_value=False: action_logits
        """
        # Handle single sample vs batch
        batch_size = len(structured_inputs) if isinstance(structured_inputs, list) else 1
        if isinstance(structured_inputs, dict):
            structured_inputs = [structured_inputs]

        # Build token sequences
        sequences = []
        masks = []

        for sample in structured_inputs:
            tokens = []

            # CLS token
            cls_token = self.cls_embedding + self.type_embeddings['cls']
            tokens.append(cls_token)

            # Context token
            ctx_input = torch.tensor([
                sample['context']['canvasWidth'], 
                sample['context']['canvasHeight']
            ], dtype=torch.float32, device=self.cls_embedding.device)
            ctx_token = self.ctx_projection(ctx_input) + self.type_embeddings['ctx']
            tokens.append(ctx_token)

            # Predator token
            predator_input = torch.tensor([
                sample['predator']['velX'], 
                sample['predator']['velY'], 
                0.0, 0.0  # padding
            ], dtype=torch.float32, device=self.cls_embedding.device)
            predator_token = self.predator_projection(predator_input) + self.type_embeddings['predator']
            tokens.append(predator_token)

            # Boid tokens
            sample_mask = [False, False, False]  # CLS, CTX, Predator are not padding

            for boid in sample['boids']:
                boid_input = torch.tensor([
                    boid['relX'], boid['relY'], boid['velX'], boid['velY']
                ], dtype=torch.float32, device=self.cls_embedding.device)
                boid_token = self.boid_projection(boid_input) + self.type_embeddings['boid']
                tokens.append(boid_token)
                sample_mask.append(False)

            # Pad to max_boids + 3
            while len(tokens) < self.max_boids + 3:
                padding_token = torch.zeros(self.d_model, device=self.cls_embedding.device)
                tokens.append(padding_token)
                sample_mask.append(True)

            sequences.append(torch.stack(tokens))
            masks.append(sample_mask)

        # Stack sequences
        x = torch.stack(sequences)  # [batch_size, seq_len, d_model]
        padding_mask = torch.tensor(masks, dtype=torch.bool, device=x.device)

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, padding_mask)

        # Extract CLS token
        cls_output = x[:, 0]  # [batch_size, d_model]

        # Compute outputs
        action_logits = self.policy_head(cls_output)  # [batch_size, 2]
        
        if return_value:
            value = self.value_head(cls_output)  # [batch_size, 1]
            if batch_size == 1:
                return action_logits.squeeze(0), value.squeeze()
            return action_logits, value.squeeze(-1)
        else:
            return action_logits.squeeze(0) if batch_size == 1 else action_logits

    def get_action_and_value(self, structured_inputs, deterministic=False):
        """
        Get action and value for PPO training
        
        Args:
            structured_inputs: Standard policy input format
            deterministic: If True, return mean action; if False, sample
            
        Returns:
            action: Sampled/deterministic action [x, y]
            log_prob: Log probability of action
            value: State value estimate
        """
        action_logits, value = self.forward(structured_inputs, return_value=True)
        
        if deterministic:
            # Use mean (no sampling)
            action = torch.tanh(action_logits)
        else:
            # Sample from distribution
            # Use tanh-transformed normal distribution
            action_dist = torch.distributions.Normal(action_logits, 1.0)
            action_sample = action_dist.rsample()
            action = torch.tanh(action_sample)
            
            # Compute log probability with tanh correction
            log_prob = action_dist.log_prob(action_sample)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(-1)
            
            return action, log_prob, value
        
        return action, None, value

    def get_action(self, structured_inputs):
        """
        Standard policy interface for compatibility with existing evaluation
        
        Args:
            structured_inputs: Standard policy input format
            
        Returns:
            action: Deterministic action [x, y] in [-1, 1] range
        """
        with torch.no_grad():
            action_logits = self.forward(structured_inputs, return_value=False)
            return torch.tanh(action_logits).detach().cpu().numpy().tolist()

    def evaluate_actions(self, structured_inputs_batch, actions_batch):
        """
        Evaluate actions for PPO loss computation
        
        Args:
            structured_inputs_batch: Batch of structured inputs
            actions_batch: Batch of actions to evaluate
            
        Returns:
            log_probs: Log probabilities of actions
            values: State value estimates  
            entropy: Action entropy for exploration bonus
        """
        action_logits, values = self.forward(structured_inputs_batch, return_value=True)
        
        # Create action distribution
        action_dist = torch.distributions.Normal(action_logits, 1.0)
        
        # Convert actions back to pre-tanh space
        actions_batch = torch.clamp(actions_batch, -0.9999, 0.9999)  # Avoid numerical issues
        action_pretanh = torch.atanh(actions_batch)
        
        # Compute log probabilities
        log_probs = action_dist.log_prob(action_pretanh)
        log_probs -= torch.log(1 - actions_batch.pow(2) + 1e-6)
        log_probs = log_probs.sum(-1)
        
        # Compute entropy
        entropy = action_dist.entropy().sum(-1)
        
        return log_probs, values, entropy

    @classmethod
    def from_sl_checkpoint(cls, checkpoint_path: str):
        """
        Create PPO model from supervised learning checkpoint
        
        Args:
            checkpoint_path: Path to SL checkpoint (e.g., best_model.pt)
            
        Returns:
            PPOTransformerModel with pre-trained weights
        """
        print(f"Loading PPO model from SL checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract architecture
        arch = checkpoint['architecture']
        d_model = arch['d_model']
        n_heads = arch['n_heads']
        n_layers = arch['n_layers']
        ffn_hidden = arch['ffn_hidden']
        max_boids = arch.get('max_boids', 50)
        
        # Create model
        model = cls(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_hidden=ffn_hidden,
            max_boids=max_boids
        )
        
        # Load shared parameters (everything except the output projection)
        sl_state_dict = checkpoint['model_state_dict']
        model_state_dict = model.state_dict()
        
        # Copy shared parameters
        for name, param in sl_state_dict.items():
            if name.startswith('output_projection'):
                # Skip SL output projection, use our dual heads
                continue
            if name in model_state_dict:
                model_state_dict[name].copy_(param)
                print(f"  ✓ Loaded: {name}")
        
        # Initialize policy head with SL output weights
        if 'output_projection.weight' in sl_state_dict:
            model.policy_head.weight.data.copy_(sl_state_dict['output_projection.weight'])
            model.policy_head.bias.data.copy_(sl_state_dict['output_projection.bias'])
            print(f"  ✓ Initialized policy head from SL output projection")
        
        # Initialize value head randomly (small weights)
        nn.init.xavier_uniform_(model.value_head.weight, gain=0.01)
        nn.init.constant_(model.value_head.bias, 0)
        print(f"  ✓ Initialized value head (random small weights)")
        
        print(f"✅ PPO model created:")
        print(f"  Architecture: {d_model}×{n_heads}×{n_layers}×{ffn_hidden}")
        print(f"  Policy head: initialized from SL")
        print(f"  Value head: initialized randomly")
        
        return model


class PPOTransformerPolicy:
    """
    Wrapper class that makes PPOTransformerModel compatible with existing policy interface
    """
    
    def __init__(self, model: PPOTransformerModel):
        self.model = model
        self.device = next(model.parameters()).device
    
    def get_action(self, structured_inputs):
        """Standard policy interface for evaluation compatibility"""
        self.model.eval()
        with torch.no_grad():
            return self.model.get_action(structured_inputs)
    
    def get_action_and_value(self, structured_inputs, deterministic=False):
        """PPO interface for training"""
        return self.model.get_action_and_value(structured_inputs, deterministic)
    
    def evaluate_actions(self, structured_inputs_batch, actions_batch):
        """PPO interface for loss computation"""
        return self.model.evaluate_actions(structured_inputs_batch, actions_batch)
    
    def to(self, device):
        """Move model to device"""
        self.model = self.model.to(device)
        self.device = device
        return self
    
    def train(self):
        """Set to training mode"""
        self.model.train()
    
    def eval(self):
        """Set to evaluation mode"""
        self.model.eval()


def create_ppo_policy_from_sl(checkpoint_path: str = "checkpoints/best_model.pt") -> PPOTransformerPolicy:
    """
    Create PPO policy from supervised learning checkpoint
    
    Args:
        checkpoint_path: Path to SL checkpoint
        
    Returns:
        PPOTransformerPolicy ready for training or evaluation
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SL checkpoint not found: {checkpoint_path}")
    
    model = PPOTransformerModel.from_sl_checkpoint(checkpoint_path)
    policy = PPOTransformerPolicy(model)
    
    print(f"✅ PPO policy created from SL checkpoint")
    print(f"  Compatible with existing StateManager and evaluation")
    print(f"  Ready for PPO training")
    
    return policy


if __name__ == "__main__":
    # Test PPO model creation
    try:
        policy = create_ppo_policy_from_sl("checkpoints/best_model.pt")
        
        # Test policy interface
        test_input = {
            'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
            'predator': {'velX': 0.1, 'velY': -0.2},
            'boids': [
                {'relX': 0.1, 'relY': 0.3, 'velX': 0.5, 'velY': -0.1},
                {'relX': -0.2, 'relY': 0.1, 'velX': -0.3, 'velY': 0.4}
            ]
        }
        
        # Test standard interface
        action = policy.get_action(test_input)
        print(f"Standard interface: {action}")
        
        # Test PPO interface
        policy.train()
        action, log_prob, value = policy.get_action_and_value(test_input)
        print(f"PPO interface: action={action}, log_prob={log_prob}, value={value}")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure you have a trained SL model at checkpoints/best_model.pt")
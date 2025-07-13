"""
PyTorch Transformer Model - Exact match to JavaScript architecture

Architecture: d_model=48, n_heads=4, n_layers=3, ffn_hidden=96
Token sequence: [CLS] + [CTX] + Predator + Boids
Multi-head self-attention with GEGLU feed-forward networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Any

class GEGLU(nn.Module):
    """GEGLU activation function (Gate-Enhanced Gated Linear Units)"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim)
        self.up_proj = nn.Linear(input_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        gated = gate * up
        return self.down_proj(gated)

class TransformerEncoderLayer(nn.Module):
    """Custom transformer layer with GEGLU FFN to match JavaScript"""
    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # GEGLU feed-forward network
        self.ffn = GEGLU(d_model, ffn_hidden)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

class TransformerPredator(nn.Module):
    """Transformer-based predator matching JavaScript architecture exactly"""
    
    def __init__(self, d_model: int = 48, n_heads: int = 4, n_layers: int = 3, 
                 ffn_hidden: int = 96, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_hidden = ffn_hidden
        
        # Special tokens and type embeddings
        self.cls_embedding = nn.Parameter(torch.randn(d_model) * 0.1)
        
        self.type_embeddings = nn.ParameterDict({
            'cls': nn.Parameter(torch.randn(d_model) * 0.1),
            'ctx': nn.Parameter(torch.randn(d_model) * 0.1),
            'predator': nn.Parameter(torch.randn(d_model) * 0.1),
            'boid': nn.Parameter(torch.randn(d_model) * 0.1)
        })
        
        # Input projections (matches JavaScript dimensions)
        self.ctx_projection = nn.Linear(2, d_model)      # [w/D, h/D] -> 48D
        self.predator_projection = nn.Linear(4, d_model) # [vx/V, vy/V, 0, 0] -> 48D  
        self.boid_projection = nn.Linear(4, d_model)     # [dx/D, dy/D, dvx/V, dvy/V] -> 48D
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, ffn_hidden, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection: [CLS] token -> steering forces
        self.output_projection = nn.Linear(d_model, 2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights similar to JavaScript"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def build_tokens(self, structured_inputs_batch: List[Dict[str, Any]]):
        """Build token sequences from structured inputs batch"""
        batch_tokens = []
        batch_masks = []
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        for structured_inputs in structured_inputs_batch:
            tokens = []
            
            # [CLS] token
            cls_token = self.cls_embedding + self.type_embeddings['cls']
            tokens.append(cls_token)
            
            # [CTX] token  
            ctx_input = torch.tensor([
                structured_inputs['context']['canvasWidth'],
                structured_inputs['context']['canvasHeight']
            ], dtype=torch.float32, device=device)
            
            ctx_token = self.ctx_projection(ctx_input) + self.type_embeddings['ctx']
            tokens.append(ctx_token)
            
            # Predator token
            predator_input = torch.tensor([
                structured_inputs['predator']['velX'],
                structured_inputs['predator']['velY'],
                0.0, 0.0  # Padding to match boid dimension
            ], dtype=torch.float32, device=device)
            
            predator_token = self.predator_projection(predator_input) + self.type_embeddings['predator']
            tokens.append(predator_token)
            
            # Boid tokens
            for boid in structured_inputs['boids']:
                boid_input = torch.tensor([
                    boid['relX'], boid['relY'],
                    boid['velX'], boid['velY']
                ], dtype=torch.float32, device=device)
                
                boid_token = self.boid_projection(boid_input) + self.type_embeddings['boid']
                tokens.append(boid_token)
            
            # Stack tokens for this sequence
            sequence_tokens = torch.stack(tokens)  # [seq_len, d_model]
            batch_tokens.append(sequence_tokens)
            
            # Create attention mask (all ones since we don't mask any tokens)
            mask = torch.ones(len(tokens), dtype=torch.bool, device=device)
            batch_masks.append(mask)
        
        # Pad sequences to same length
        max_len = max(len(seq) for seq in batch_tokens)
        
        padded_tokens = []
        padded_masks = []
        
        for tokens, mask in zip(batch_tokens, batch_masks):
            seq_len = len(tokens)
            
            if seq_len < max_len:
                # Pad with zeros
                pad_length = max_len - seq_len
                padding = torch.zeros(pad_length, self.d_model, device=device)
                padded_tokens.append(torch.cat([tokens, padding], dim=0))
                
                # Extend mask with False for padded positions
                mask_padding = torch.zeros(pad_length, dtype=torch.bool, device=device)
                padded_masks.append(torch.cat([mask, mask_padding], dim=0))
            else:
                padded_tokens.append(tokens)
                padded_masks.append(mask)
        
        # Stack into batch tensors
        batch_tokens = torch.stack(padded_tokens)  # [batch_size, seq_len, d_model]
        batch_masks = torch.stack(padded_masks)    # [batch_size, seq_len]
        
        return batch_tokens, batch_masks
    
    def forward(self, structured_inputs_batch: List[Dict[str, Any]]):
        """Forward pass through transformer"""
        # Build token sequences and attention masks
        tokens, attention_mask = self.build_tokens(structured_inputs_batch)
        # tokens: [batch_size, seq_len, d_model]
        # attention_mask: [batch_size, seq_len] - True for real tokens, False for padding
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            tokens = layer(tokens, mask=~attention_mask)  # Invert mask for PyTorch (True = masked)
        
        # Extract [CLS] tokens (first token of each sequence)
        cls_tokens = tokens[:, 0, :]  # [batch_size, d_model]
        
        # Project to output actions
        actions = self.output_projection(cls_tokens)  # [batch_size, 2]
        
        # Apply tanh activation for bounded output
        return torch.tanh(actions)
    
    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_from_javascript(self, js_params: Dict[str, Any]):
        """Load parameters from JavaScript model format"""
        # This would be used to load pre-trained weights from the JS model
        # Implementation depends on the specific parameter format
        pass
    
    def export_to_javascript(self):
        """Export parameters to JavaScript model format"""
        # This would be used to export trained weights back to JS
        # Implementation depends on the specific parameter format needed
        pass

def create_model(device='cpu'):
    """Create model instance with proper device placement"""
    model = TransformerPredator()
    model = model.to(device)
    
    print(f"Created TransformerPredator model:")
    print(f"  Parameters: {model.get_num_parameters():,}")
    print(f"  Device: {device}")
    print(f"  Architecture: d_model={model.d_model}, n_heads={model.n_heads}, n_layers={model.n_layers}")
    
    return model

if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device)
    
    # Test forward pass with dummy data
    dummy_input = [{
        'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
        'predator': {'velX': 0.1, 'velY': -0.2},
        'boids': [
            {'relX': 0.1, 'relY': 0.3, 'velX': 0.5, 'velY': -0.1},
            {'relX': -0.2, 'relY': 0.1, 'velX': -0.3, 'velY': 0.4}
        ]
    }]
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Test output shape: {output.shape}")
        print(f"Test output: {output}") 
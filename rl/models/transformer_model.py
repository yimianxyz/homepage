"""
Transformer Model - PyTorch implementation for RL training

This module provides a PyTorch transformer model that matches the architecture
used in supervised learning training, allowing us to load pre-trained weights
and continue training with PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import os
import sys

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.constants import CONSTANTS


class TransformerModel(nn.Module):
    """
    PyTorch Transformer model for boid predator control
    
    This model matches the architecture used in supervised learning training
    and can load pre-trained weights from best_model.pt for continued RL training.
    """
    
    def __init__(self,
                 d_model: int = 64,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 ffn_hidden: int = 256,
                 max_boids: int = 50,
                 dropout: float = 0.1):
        """
        Initialize transformer model
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            ffn_hidden: Feed-forward hidden dimension
            max_boids: Maximum number of boids (for positional embeddings)
            dropout: Dropout rate
        """
        super().__init__()
        
        # Architecture parameters
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_hidden = ffn_hidden
        self.max_boids = max_boids
        self.dropout = dropout
        
        # Validate architecture
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.head_dim = d_model // n_heads
        
        # Normalization constants from simulation
        self.D = CONSTANTS.MAX_DISTANCE
        self.V = max(CONSTANTS.BOID_MAX_SPEED, CONSTANTS.PREDATOR_MAX_SPEED)
        
        # Token embeddings
        self.cls_embedding = nn.Parameter(torch.randn(d_model))
        
        # Type embeddings for different entity types
        self.type_embeddings = nn.Embedding(4, d_model)  # CLS, CTX, Predator, Boid
        self.register_buffer('type_cls', torch.tensor(0))
        self.register_buffer('type_ctx', torch.tensor(1))
        self.register_buffer('type_predator', torch.tensor(2))
        self.register_buffer('type_boid', torch.tensor(3))
        
        # Input projection layers
        self.ctx_projection = nn.Linear(2, d_model, bias=False)      # Canvas dimensions
        self.predator_projection = nn.Linear(2, d_model, bias=False) # Predator velocity  
        self.boid_projection = nn.Linear(4, d_model, bias=False)     # Boid rel_pos + velocity
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_hidden,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_head = nn.Linear(d_model, 2)  # [force_x, force_y]
        
        # Initialize parameters
        self._init_parameters()
        
        print(f"Created TransformerModel:")
        print(f"  Architecture: {d_model}×{n_heads}×{n_layers}×{ffn_hidden}")
        print(f"  Parameters: {self.count_parameters():,}")
        print(f"  Max boids: {max_boids}")
    
    def _init_parameters(self):
        """Initialize model parameters"""
        # Initialize embeddings
        nn.init.normal_(self.cls_embedding, std=0.02)
        nn.init.normal_(self.type_embeddings.weight, std=0.02)
        
        # Initialize projections
        nn.init.xavier_uniform_(self.ctx_projection.weight)
        nn.init.xavier_uniform_(self.predator_projection.weight)
        nn.init.xavier_uniform_(self.boid_projection.weight)
        
        # Initialize output head
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
    
    def forward(self, structured_inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass through transformer
        
        Args:
            structured_inputs: Dictionary with keys:
                - context: {canvasWidth, canvasHeight}
                - predator: {velX, velY}
                - boids: [{relX, relY, velX, velY}, ...]
                
        Returns:
            policy_outputs: Tensor of shape [batch_size, 2] with values in [-1, 1]
        """
        # Handle both single sample and batch inputs
        if not isinstance(structured_inputs, list):
            structured_inputs = [structured_inputs]
        
        batch_size = len(structured_inputs)
        device = next(self.parameters()).device
        
        # Build token sequences for the batch
        all_tokens = []
        all_masks = []
        
        for inputs in structured_inputs:
            tokens, mask = self._build_tokens(inputs, device)
            all_tokens.append(tokens)
            all_masks.append(mask)
        
        # Pad sequences to same length
        max_len = max(len(tokens) for tokens in all_tokens)
        padded_tokens = []
        padded_masks = []
        
        for tokens, mask in zip(all_tokens, all_masks):
            # Pad tokens
            pad_len = max_len - len(tokens)
            if pad_len > 0:
                padding = torch.zeros(pad_len, self.d_model, device=device)
                tokens = torch.cat([tokens, padding], dim=0)
                mask = torch.cat([mask, torch.ones(pad_len, device=device, dtype=torch.bool)])
            
            padded_tokens.append(tokens)
            padded_masks.append(mask)
        
        # Stack into batch
        token_batch = torch.stack(padded_tokens)        # [batch_size, seq_len, d_model]
        mask_batch = torch.stack(padded_masks)          # [batch_size, seq_len]
        
        # Pass through transformer
        # Note: PyTorch transformer expects inverted mask (True = ignore)
        transformer_output = self.transformer(token_batch, src_key_padding_mask=mask_batch)
        
        # Extract CLS tokens and project to output
        cls_tokens = transformer_output[:, 0, :]  # [batch_size, d_model]
        logits = self.output_head(cls_tokens)     # [batch_size, 2]
        
        # Apply tanh for bounded output [-1, 1]
        outputs = torch.tanh(logits)
        
        # Return single sample if input was single sample
        if len(structured_inputs) == 1:
            return outputs[0]
        return outputs
    
    def _build_tokens(self, inputs: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build token sequence from structured inputs
        
        Args:
            inputs: Structured inputs for single sample
            device: Device to place tensors on
            
        Returns:
            tokens: Token sequence [seq_len, d_model]
            mask: Padding mask [seq_len] (False = real token, True = padding)
        """
        tokens = []
        
        # Token 0: [CLS] - learned embedding + type embedding
        cls_token = self.cls_embedding + self.type_embeddings(self.type_cls)
        tokens.append(cls_token)
        
        # Token 1: [CTX] - context projection + type embedding
        ctx_input = torch.tensor([
            inputs['context']['canvasWidth'],   # Already normalized
            inputs['context']['canvasHeight']   # Already normalized
        ], dtype=torch.float32, device=device)
        ctx_token = self.ctx_projection(ctx_input) + self.type_embeddings(self.type_ctx)
        tokens.append(ctx_token)
        
        # Token 2: Predator - predator projection + type embedding
        predator_input = torch.tensor([
            inputs['predator']['velX'],         # Already normalized
            inputs['predator']['velY']          # Already normalized
        ], dtype=torch.float32, device=device)
        predator_token = self.predator_projection(predator_input) + self.type_embeddings(self.type_predator)
        tokens.append(predator_token)
        
        # Tokens 3+: Boids - boid projections + type embeddings
        for boid in inputs['boids']:
            boid_input = torch.tensor([
                boid['relX'],   # Already normalized
                boid['relY'],   # Already normalized
                boid['velX'],   # Already normalized
                boid['velY']    # Already normalized
            ], dtype=torch.float32, device=device)
            boid_token = self.boid_projection(boid_input) + self.type_embeddings(self.type_boid)
            tokens.append(boid_token)
        
        # Stack tokens
        token_tensor = torch.stack(tokens)  # [seq_len, d_model]
        
        # Create mask (all tokens are real, no padding needed here)
        mask = torch.zeros(len(tokens), dtype=torch.bool, device=device)
        
        return token_tensor, mask
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get model architecture information"""
        return {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'ffn_hidden': self.ffn_hidden,
            'max_boids': self.max_boids,
            'dropout': self.dropout
        }


class TransformerModelLoader:
    """
    Utility class for loading pre-trained transformer models
    """
    
    def __init__(self):
        self.loaded_models = {}
    
    def load_model(self, 
                   checkpoint_path: str,
                   device: Optional[torch.device] = None) -> TransformerModel:
        """
        Load a pre-trained transformer model from checkpoint
        
        Args:
            checkpoint_path: Path to the model checkpoint (.pt file)
            device: Device to load model on
            
        Returns:
            model: Loaded TransformerModel
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model from {checkpoint_path}")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Extract model configuration if available
            if 'config' in checkpoint:
                config = checkpoint['config']
                print(f"Found model config: {config}")
            else:
                # Use default configuration
                config = {
                    'd_model': 64,
                    'n_heads': 8,
                    'n_layers': 4,
                    'ffn_hidden': 256,
                    'max_boids': 50,
                    'dropout': 0.1
                }
                print(f"Using default config: {config}")
            
            # Create model with loaded configuration
            model = TransformerModel(**config)
            model.to(device)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Handle potential key mismatches
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            
            for key, value in state_dict.items():
                if key in model_state_dict:
                    if model_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        print(f"Shape mismatch for {key}: model {model_state_dict[key].shape} vs checkpoint {value.shape}")
                else:
                    print(f"Key {key} not found in model")
            
            # Load the filtered state dict
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
            
            model.eval()
            
            print(f"Successfully loaded model with {model.count_parameters():,} parameters")
            
            # Cache the loaded model
            self.loaded_models[checkpoint_path] = model
            
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model with default configuration...")
            
            # Fallback: create new model
            model = TransformerModel()
            model.to(device)
            return model
    
    def save_model(self, 
                   model: TransformerModel,
                   save_path: str,
                   additional_info: Optional[Dict[str, Any]] = None):
        """
        Save a transformer model
        
        Args:
            model: Model to save
            save_path: Path to save the model
            additional_info: Additional information to save
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': model.get_architecture_info(),
            'model_class': 'TransformerModel'
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
    
    def create_model_from_config(self,
                                config: Dict[str, Any],
                                device: Optional[torch.device] = None) -> TransformerModel:
        """
        Create a new model from configuration
        
        Args:
            config: Model configuration
            device: Device to place model on
            
        Returns:
            model: New TransformerModel
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = TransformerModel(**config)
        model.to(device)
        return model
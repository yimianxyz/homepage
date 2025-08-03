"""
Transformer Model Definition

This file contains the transformer architecture definition needed for RL training.
It should match exactly the architecture used in transformer_training.ipynb.
"""

import torch
import torch.nn as nn


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden, dropout=0.1):
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


class TransformerPredictor(nn.Module):
    def __init__(self, d_model=48, n_heads=4, n_layers=2, ffn_hidden=96, max_boids=50, dropout=0.1):
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
        self.ctx_projection = nn.Linear(2, d_model)  # canvas_width, canvas_height
        self.predator_projection = nn.Linear(4, d_model)  # velX, velY, 0, 0 (padded to 4D)
        self.boid_projection = nn.Linear(4, d_model)  # relX, relY, velX, velY

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, ffn_hidden, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, 2)  # predator action [x, y]

    def forward(self, structured_inputs, padding_mask=None):
        batch_size = len(structured_inputs) if isinstance(structured_inputs, list) else 1

        # Handle single sample vs batch
        if isinstance(structured_inputs, dict):
            structured_inputs = [structured_inputs]
            batch_size = 1

        # Build token sequences for each sample in batch
        sequences = []
        masks = []

        for sample in structured_inputs:
            tokens = []

            # CLS token
            cls_token = self.cls_embedding + self.type_embeddings['cls']
            tokens.append(cls_token)

            # Context token
            ctx_input = torch.tensor([sample['context']['canvasWidth'], sample['context']['canvasHeight']],
                                   dtype=torch.float32, device=self.cls_embedding.device)
            ctx_token = self.ctx_projection(ctx_input) + self.type_embeddings['ctx']
            tokens.append(ctx_token)

            # Predator token - expand to 4D
            predator_input = torch.tensor([sample['predator']['velX'], sample['predator']['velY'], 0.0, 0.0],
                                        dtype=torch.float32, device=self.cls_embedding.device)
            predator_token = self.predator_projection(predator_input) + self.type_embeddings['predator']
            tokens.append(predator_token)

            # Boid tokens
            sample_mask = [False, False, False]  # CLS, CTX, Predator are not padding

            for boid in sample['boids']:
                boid_input = torch.tensor([boid['relX'], boid['relY'], boid['velX'], boid['velY']],
                                        dtype=torch.float32, device=self.cls_embedding.device)
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
        if padding_mask is None:
            padding_mask = torch.tensor(masks, dtype=torch.bool, device=x.device)

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, padding_mask)

        # Extract CLS token and project to output
        cls_output = x[:, 0]  # [batch_size, d_model]
        action = self.output_projection(cls_output)  # [batch_size, 2]

        # Apply tanh to ensure [-1, 1] range
        action = torch.tanh(action)

        return action.squeeze(0) if batch_size == 1 else action
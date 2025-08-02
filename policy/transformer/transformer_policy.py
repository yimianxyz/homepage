"""
Transformer Policy - Python wrapper for PyTorch trained transformer models

This policy loads PyTorch checkpoints (e.g., best_model.pt) and provides the same
standard policy interface as other policies. It implements the identical transformer
architecture as the JavaScript version for cross-platform consistency.

Interface:
- Input: structured_inputs (same format as universal policy input)
- Output: normalized policy outputs [x, y] in [-1, 1] range
- The ActionProcessor handles scaling to game forces
"""

import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Any, Optional
import numpy as np

class TransformerPolicy:
    """Transformer policy that loads from PyTorch checkpoints"""
    
    def __init__(self, checkpoint_path: str):
        """
        Initialize transformer policy from PyTorch checkpoint
        
        Args:
            checkpoint_path: Path to PyTorch checkpoint (.pt file)
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cpu')  # Use CPU for inference
        
        # Load checkpoint and extract model components
        self._load_checkpoint()
        
        print(f"Created TransformerPolicy:")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Architecture: {self.d_model}×{self.n_heads}×{self.n_layers}×{self.ffn_hidden}")
        print(f"  Parameters: {self._count_parameters():,}")
        print(f"  Device: {self.device}")
    
    def _load_checkpoint(self):
        """Load PyTorch checkpoint and extract model components"""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Extract architecture
            arch = checkpoint['architecture']
            self.d_model = arch['d_model']
            self.n_heads = arch['n_heads'] 
            self.n_layers = arch['n_layers']
            self.ffn_hidden = arch['ffn_hidden']
            self.head_dim = self.d_model // self.n_heads
            
            # Extract model parameters
            state_dict = checkpoint['model_state_dict']
            
            # Embeddings
            self.cls_embedding = state_dict['cls_embedding'].numpy()
            self.type_embeddings = {
                'cls': state_dict['type_embeddings.cls'].numpy(),
                'ctx': state_dict['type_embeddings.ctx'].numpy(), 
                'predator': state_dict['type_embeddings.predator'].numpy(),
                'boid': state_dict['type_embeddings.boid'].numpy()
            }
            
            # Input projections
            self.ctx_projection = {
                'weight': state_dict['ctx_projection.weight'].numpy(),
                'bias': state_dict['ctx_projection.bias'].numpy()
            }
            self.predator_projection = {
                'weight': state_dict['predator_projection.weight'].numpy(),
                'bias': state_dict['predator_projection.bias'].numpy()
            }
            self.boid_projection = {
                'weight': state_dict['boid_projection.weight'].numpy(),
                'bias': state_dict['boid_projection.bias'].numpy()
            }
            
            # Transformer layers
            self.layers = []
            for i in range(self.n_layers):
                layer = {
                    'ln_scale': state_dict[f'transformer_layers.{i}.norm1.weight'].numpy(),
                    'ln_bias': state_dict[f'transformer_layers.{i}.norm1.bias'].numpy(),
                    'qkv_weight': state_dict[f'transformer_layers.{i}.self_attn.in_proj_weight'].numpy(),
                    'qkv_bias': state_dict[f'transformer_layers.{i}.self_attn.in_proj_bias'].numpy(),
                    'attn_out_weight': state_dict[f'transformer_layers.{i}.self_attn.out_proj.weight'].numpy(),
                    'attn_out_bias': state_dict[f'transformer_layers.{i}.self_attn.out_proj.bias'].numpy(),
                    'ffn_ln_scale': state_dict[f'transformer_layers.{i}.norm2.weight'].numpy(),
                    'ffn_ln_bias': state_dict[f'transformer_layers.{i}.norm2.bias'].numpy(),
                    'ffn_gate_weight': state_dict[f'transformer_layers.{i}.ffn_gate_proj.weight'].numpy(),
                    'ffn_gate_bias': state_dict[f'transformer_layers.{i}.ffn_gate_proj.bias'].numpy(),
                    'ffn_up_weight': state_dict[f'transformer_layers.{i}.ffn_up_proj.weight'].numpy(),
                    'ffn_up_bias': state_dict[f'transformer_layers.{i}.ffn_up_proj.bias'].numpy(),
                    'ffn_down_weight': state_dict[f'transformer_layers.{i}.ffn_down_proj.weight'].numpy(),
                    'ffn_down_bias': state_dict[f'transformer_layers.{i}.ffn_down_proj.bias'].numpy()
                }
                self.layers.append(layer)
            
            # Output projection
            self.output_weight = state_dict['output_projection.weight'].numpy()
            self.output_bias = state_dict['output_projection.bias'].numpy()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {self.checkpoint_path}: {e}")
    
    def get_action(self, structured_inputs: Dict[str, Any]) -> List[float]:
        """
        Get normalized policy outputs (compatible with ActionProcessor)
        
        Args:
            structured_inputs: Same format as universal policy input
            - context: {canvasWidth: float, canvasHeight: float}
            - predator: {velX: float, velY: float}
            - boids: [{relX: float, relY: float, velX: float, velY: float}, ...]
            
        Returns:
            Normalized policy outputs [x, y] in [-1, 1] range
        """
        try:
            # Forward pass through transformer
            outputs = self._forward(structured_inputs)
            
            # Ensure outputs are in [-1, 1] range (transformer already applies tanh)
            clamped_outputs = [
                max(-1.0, min(1.0, float(outputs[0]))),
                max(-1.0, min(1.0, float(outputs[1])))
            ]
            
            return clamped_outputs
            
        except Exception as e:
            print(f"TransformerPolicy error during forward pass: {e}")
            return [0.0, 0.0]
    
    def get_normalized_action(self, structured_inputs: Dict[str, Any]) -> List[float]:
        """
        Get normalized action in [-1, 1] range (deprecated - use get_action instead)
        
        Args:
            structured_inputs: Same format as universal policy input
            
        Returns:
            Normalized policy outputs [x, y] in [-1, 1] range
        """
        return self.get_action(structured_inputs)
    
    def _forward(self, structured_inputs: Dict[str, Any]) -> List[float]:
        """Forward pass through transformer encoder (matches JavaScript exactly)"""
        # Build token sequence
        tokens = self._build_tokens(structured_inputs)
        
        # Process through transformer layers
        for layer in self.layers:
            tokens = self._transformer_block(tokens, layer)
        
        # Extract [CLS] token and project to steering forces
        cls_token = tokens[0]
        logits = self.output_weight @ cls_token + self.output_bias
        
        # Apply tanh activation for bounded output
        return [math.tanh(logits[0]), math.tanh(logits[1])]
    
    def _build_tokens(self, structured_inputs: Dict[str, Any]) -> List[np.ndarray]:
        """Build token sequence from structured inputs (matches JavaScript exactly)"""
        tokens = []
        
        # Token 0: [CLS] - learned embedding + type embedding
        cls_token = self.cls_embedding + self.type_embeddings['cls']
        tokens.append(cls_token)
        
        # Token 1: [CTX] - context projection + type embedding
        ctx_input = np.array([
            structured_inputs['context']['canvasWidth'],
            structured_inputs['context']['canvasHeight']
        ])
        ctx_projected = self.ctx_projection['weight'] @ ctx_input + self.ctx_projection['bias']
        ctx_token = ctx_projected + self.type_embeddings['ctx']
        tokens.append(ctx_token)
        
        # Token 2: Predator - predator projection + type embedding
        predator_input = np.array([
            structured_inputs['predator']['velX'],
            structured_inputs['predator']['velY'],
            0.0,  # padding
            0.0   # padding
        ])
        predator_projected = self.predator_projection['weight'] @ predator_input + self.predator_projection['bias']
        predator_token = predator_projected + self.type_embeddings['predator']
        tokens.append(predator_token)
        
        # Tokens 3+: Boids - boid projections + type embeddings
        for boid in structured_inputs['boids']:
            boid_input = np.array([
                boid['relX'],
                boid['relY'],
                boid['velX'],
                boid['velY']
            ])
            boid_projected = self.boid_projection['weight'] @ boid_input + self.boid_projection['bias']
            boid_token = boid_projected + self.type_embeddings['boid']
            tokens.append(boid_token)
        
        return tokens
    
    def _transformer_block(self, tokens: List[np.ndarray], layer: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Single transformer block: LayerNorm → MHSA → FFN (matches JavaScript exactly)"""
        seq_len = len(tokens)
        
        # 1. Layer normalization
        normed_tokens = []
        for token in tokens:
            normed_tokens.append(self._layer_norm(token, layer['ln_scale'], layer['ln_bias']))
        
        # 2. Multi-head self-attention
        attn_output = self._multi_head_attention(normed_tokens, layer)
        
        # 3. Residual connection
        residual1 = []
        for i in range(seq_len):
            residual1.append(tokens[i] + attn_output[i])
        
        # 4. Layer norm for FFN
        ffn_normed = []
        for token in residual1:
            ffn_normed.append(self._layer_norm(token, layer['ffn_ln_scale'], layer['ffn_ln_bias']))
        
        # 5. GEGLU feed-forward
        ffn_output = []
        for token in ffn_normed:
            ffn_output.append(self._geglu(token, layer))
        
        # 6. Final residual connection
        output = []
        for i in range(seq_len):
            output.append(residual1[i] + ffn_output[i])
        
        return output
    
    def _multi_head_attention(self, tokens: List[np.ndarray], layer: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Multi-head self-attention (matches JavaScript PyTorch replication exactly)"""
        seq_len = len(tokens)
        scale = 0.25  # Exact PyTorch scale: 1 / sqrt(16) = 0.25
        
        # Step 1: QKV Projection for all tokens
        qkv_all = []
        for token in tokens:
            qkv = layer['qkv_weight'] @ token + layer['qkv_bias']
            qkv_all.append(qkv)
        
        # Step 2: Reshape QKV following PyTorch's exact approach
        Q, K, V = [], [], []
        
        # Initialize head arrays
        for h in range(self.n_heads):
            Q.append([])
            K.append([])
            V.append([])
        
        # Fill Q, K, V following PyTorch's reshape and permute operations
        for seq in range(seq_len):
            qkv = qkv_all[seq]  # [3*d_model]
            
            for h in range(self.n_heads):
                q_head = []
                k_head = []
                v_head = []
                
                for dim in range(self.head_dim):
                    # PyTorch memory layout after view(batch, seq_len, 3, n_heads, head_dim)
                    q_idx = 0 * self.d_model + h * self.head_dim + dim  # Q section
                    k_idx = 1 * self.d_model + h * self.head_dim + dim  # K section
                    v_idx = 2 * self.d_model + h * self.head_dim + dim  # V section
                    
                    q_head.append(qkv[q_idx])
                    k_head.append(qkv[k_idx])
                    v_head.append(qkv[v_idx])
                
                Q[h].append(np.array(q_head))  # Q[h][seq][dim]
                K[h].append(np.array(k_head))  # K[h][seq][dim]
                V[h].append(np.array(v_head))  # V[h][seq][dim]
        
        # Step 3: Compute attention for all heads
        attn_output = []  # [n_heads][seq_len][head_dim]
        
        for h in range(self.n_heads):
            q_head = Q[h]  # [seq_len][head_dim]
            k_head = K[h]  # [seq_len][head_dim]
            v_head = V[h]  # [seq_len][head_dim]
            
            # Compute attention scores: Q @ K.T
            scores = np.zeros((seq_len, seq_len))
            for i in range(seq_len):
                for j in range(seq_len):
                    scores[i, j] = np.dot(q_head[i], k_head[j]) * scale
            
            # Apply softmax to each row
            for i in range(seq_len):
                scores[i] = self._softmax(scores[i])
            
            # Apply attention to values: scores @ V
            head_output = []
            for i in range(seq_len):
                attended = np.zeros(self.head_dim)
                for j in range(seq_len):
                    attended += scores[i, j] * v_head[j]
                head_output.append(attended)
            
            attn_output.append(head_output)
        
        # Step 4: Concatenate heads
        concatenated = []
        for i in range(seq_len):
            token_concat = []
            # Concatenate all heads for token i
            for h in range(self.n_heads):
                token_concat.extend(attn_output[h][i])
            concatenated.append(np.array(token_concat))
        
        # Step 5: Apply output projection
        final_output = []
        for token in concatenated:
            projected = layer['attn_out_weight'] @ token + layer['attn_out_bias']
            final_output.append(projected)
        
        return final_output
    
    def _geglu(self, x: np.ndarray, layer: Dict[str, np.ndarray]) -> np.ndarray:
        """GEGLU feed-forward network (matches JavaScript exactly)"""
        # Gate projection
        gate = layer['ffn_gate_weight'] @ x + layer['ffn_gate_bias']
        
        # Up projection
        up = layer['ffn_up_weight'] @ x + layer['ffn_up_bias']
        
        # GELU activation on gate
        gate = self._gelu(gate)
        
        # Element-wise multiplication
        gated = gate * up
        
        # Down projection
        return layer['ffn_down_weight'] @ gated + layer['ffn_down_bias']
    
    def _layer_norm(self, x: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Layer normalization (matches JavaScript exactly)"""
        mean = np.mean(x)
        variance = np.mean((x - mean) ** 2)
        std = np.sqrt(variance + 1e-5)  # PyTorch's epsilon value
        return ((x - mean) / std) * scale + bias
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation (matches JavaScript exactly)"""
        max_val = np.max(x)
        exp_vals = np.exp(x - max_val)
        return exp_vals / np.sum(exp_vals)
    
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation function (matches JavaScript exactly)"""
        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        return 0.5 * x * (1 + self._erf(x / np.sqrt(2)))
    
    def _erf(self, x: np.ndarray) -> np.ndarray:
        """Error function (erf) approximation - high precision (matches JavaScript exactly)"""
        # High-precision erf approximation using Abramowitz and Stegun
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        
        sign = np.sign(x)
        x = np.abs(x)
        
        t = 1 / (1 + p * x)
        y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        
        return sign * y
    
    def _count_parameters(self) -> int:
        """Count total number of parameters"""
        count = 0
        count += self.cls_embedding.size
        for emb in self.type_embeddings.values():
            count += emb.size
        count += self.ctx_projection['weight'].size + self.ctx_projection['bias'].size
        count += self.predator_projection['weight'].size + self.predator_projection['bias'].size
        count += self.boid_projection['weight'].size + self.boid_projection['bias'].size
        
        for layer in self.layers:
            for param in layer.values():
                count += param.size
        
        count += self.output_weight.size + self.output_bias.size
        return count

def create_transformer_policy(checkpoint_path: str):
    """Create transformer policy instance from checkpoint"""
    policy = TransformerPolicy(checkpoint_path)
    print(f"TransformerPolicy created successfully:")
    print(f"  Ready for use with StateManager")
    print(f"  Compatible with ActionProcessor")
    print(f"  Output: Normalized policy outputs in [-1, 1] range")
    return policy

if __name__ == "__main__":
    # Test transformer policy
    try:
        policy = create_transformer_policy("checkpoints/best_model.pt")
        
        # Test with dummy data
        test_input = {
            'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
            'predator': {'velX': 0.1, 'velY': -0.2},
            'boids': [
                {'relX': 0.1, 'relY': 0.3, 'velX': 0.5, 'velY': -0.1},
                {'relX': -0.2, 'relY': 0.1, 'velX': -0.3, 'velY': 0.4}
            ]
        }
        
        policy_output = policy.get_action(test_input)
        
        print(f"Policy output (normalized): {policy_output}")
        print(f"Note: Use ActionProcessor to convert to game forces")
        
    except Exception as e:
        print(f"Error testing transformer policy: {e}")
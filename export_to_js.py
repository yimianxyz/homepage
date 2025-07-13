#!/usr/bin/env python3
"""
Export PyTorch Model to JavaScript

Converts trained PyTorch checkpoints to JavaScript model files for browser deployment.
"""

import torch
import json
import argparse
import os
from pathlib import Path

def load_checkpoint(checkpoint_path: str) -> dict:
    """Load PyTorch checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown')}")
    else:
        # Assume it's just the state dict
        state_dict = checkpoint
        print("Loaded raw state dict")
    
    return state_dict

def convert_tensor_to_js(tensor: torch.Tensor) -> list:
    """Convert PyTorch tensor to JavaScript array"""
    return tensor.detach().cpu().numpy().tolist()

def validate_matrix_dimensions(js_params: dict) -> dict:
    """Validate that all matrices have correct dimensions for JavaScript transformer"""
    errors = []
    total_matrices = 0
    
    # Expected dimensions for JavaScript transformer
    d_model = 48
    n_heads = 4
    ffn_hidden = 96
    
    # Input projection matrices
    matrices_to_check = [
        ("ctx_projection", js_params["ctx_projection"], [d_model, 2], "Context projection: 2D → 48D"),
        ("predator_projection", js_params["predator_projection"], [d_model, 4], "Predator projection: 4D → 48D"),
        ("boid_projection", js_params["boid_projection"], [d_model, 4], "Boid projection: 4D → 48D"),
    ]
    
    # Layer matrices
    for i, layer in enumerate(js_params["layers"]):
        matrices_to_check.extend([
            (f"layer_{i}_qkv_weight", layer["qkv_weight"], [3*d_model, d_model], f"Layer {i} QKV: 48D → 144D"),
            (f"layer_{i}_attn_out_weight", layer["attn_out_weight"], [d_model, d_model], f"Layer {i} Attn out: 48D → 48D"),
            (f"layer_{i}_ffn_gate_weight", layer["ffn_gate_weight"], [ffn_hidden, d_model], f"Layer {i} FFN gate: 48D → 96D"),
            (f"layer_{i}_ffn_up_weight", layer["ffn_up_weight"], [ffn_hidden, d_model], f"Layer {i} FFN up: 48D → 96D"),
            (f"layer_{i}_ffn_down_weight", layer["ffn_down_weight"], [d_model, ffn_hidden], f"Layer {i} FFN down: 96D → 48D"),
        ])
    
    # Output matrix
    matrices_to_check.append(
        ("output_weight", js_params["output_weight"], [2, d_model], "Output projection: 48D → 2D")
    )
    
    # Check each matrix
    for name, matrix, expected_shape, description in matrices_to_check:
        total_matrices += 1
        actual_shape = [len(matrix), len(matrix[0]) if matrix else 0]
        
        if actual_shape != expected_shape:
            errors.append(f"{name}: expected {expected_shape}, got {actual_shape} - {description}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "total_matrices": total_matrices
    }

def convert_state_dict_to_js(state_dict: dict) -> dict:
    """Convert PyTorch state dict to JavaScript transformer format"""
    print("Converting parameters to JavaScript transformer format...")
    
    # Architecture parameters (fixed for our model)
    js_params = {
        "d_model": 48,
        "n_heads": 4,
        "n_layers": 3,
        "ffn_hidden": 96
    }
    
    # Convert embeddings
    js_params["cls_embedding"] = convert_tensor_to_js(state_dict["cls_embedding"])
    
    # Type embeddings - incorporate projection biases since JS doesn't handle them separately
    js_params["type_embeddings"] = {
        "cls": convert_tensor_to_js(state_dict["type_embeddings.cls"]),
        "ctx": convert_tensor_to_js(state_dict["type_embeddings.ctx"] + state_dict["ctx_projection.bias"]),
        "predator": convert_tensor_to_js(state_dict["type_embeddings.predator"] + state_dict["predator_projection.bias"]),
        "boid": convert_tensor_to_js(state_dict["type_embeddings.boid"] + state_dict["boid_projection.bias"])
    }
    
    # Convert input projections 
    # PyTorch Linear(in, out) has weight [out, in], which matches JS expectation [out, in]
    ctx_proj_weight = state_dict["ctx_projection.weight"]  # [48, 2] - correct for JS
    ctx_proj_bias = state_dict["ctx_projection.bias"]      # [48]
    # Add bias to type embedding (JS doesn't have separate projection bias)
    js_params["ctx_projection"] = convert_tensor_to_js(ctx_proj_weight)
    
    predator_proj_weight = state_dict["predator_projection.weight"]  # [48, 4] - correct for JS
    predator_proj_bias = state_dict["predator_projection.bias"]      # [48]
    js_params["predator_projection"] = convert_tensor_to_js(predator_proj_weight)
    
    boid_proj_weight = state_dict["boid_projection.weight"]  # [48, 4] - correct for JS
    boid_proj_bias = state_dict["boid_projection.bias"]      # [48]
    js_params["boid_projection"] = convert_tensor_to_js(boid_proj_weight)
    
    # Convert transformer layers
    layers = []
    for i in range(3):  # 3 layers
        layer_prefix = f"transformer_layers.{i}"
        
        # Layer normalization parameters
        ln_scale = convert_tensor_to_js(state_dict[f"{layer_prefix}.norm1.weight"])
        ln_bias = convert_tensor_to_js(state_dict[f"{layer_prefix}.norm1.bias"])
        
        # Attention parameters
        # PyTorch uses fused QKV - PyTorch format [144, 48] matches JS expectation 
        qkv_weight = state_dict[f"{layer_prefix}.self_attn.in_proj_weight"]  # [144, 48] - correct for JS
        qkv_bias = state_dict[f"{layer_prefix}.self_attn.in_proj_bias"]      # [144]
        
        # No transpose needed for QKV - already correct [144, 48]
        qkv_weight_t = qkv_weight
        
        attn_out_weight = state_dict[f"{layer_prefix}.self_attn.out_proj.weight"].t()  # [48, 48]
        attn_out_bias = convert_tensor_to_js(state_dict[f"{layer_prefix}.self_attn.out_proj.bias"])
        
        # FFN layer norm
        ffn_ln_scale = convert_tensor_to_js(state_dict[f"{layer_prefix}.norm2.weight"])
        ffn_ln_bias = convert_tensor_to_js(state_dict[f"{layer_prefix}.norm2.bias"])
        
        # GEGLU FFN parameters
        # PyTorch Linear(in, out) has weight [out, in], which matches JS expectation for gate/up
        ffn_gate_weight = state_dict[f"{layer_prefix}.ffn.gate_proj.weight"]     # [96, 48] - correct for JS
        ffn_gate_bias = convert_tensor_to_js(state_dict[f"{layer_prefix}.ffn.gate_proj.bias"])
        
        ffn_up_weight = state_dict[f"{layer_prefix}.ffn.up_proj.weight"]         # [96, 48] - correct for JS
        ffn_up_bias = convert_tensor_to_js(state_dict[f"{layer_prefix}.ffn.up_proj.bias"])
        
        ffn_down_weight = state_dict[f"{layer_prefix}.ffn.down_proj.weight"]     # [48, 96] - correct for JS
        ffn_down_bias = convert_tensor_to_js(state_dict[f"{layer_prefix}.ffn.down_proj.bias"])
        
        layer_params = {
            "ln_scale": ln_scale,
            "ln_bias": ln_bias,
            "qkv_weight": convert_tensor_to_js(qkv_weight_t),
            "qkv_bias": convert_tensor_to_js(qkv_bias),
            "attn_out_weight": convert_tensor_to_js(attn_out_weight),
            "attn_out_bias": attn_out_bias,
            "ffn_ln_scale": ffn_ln_scale,
            "ffn_ln_bias": ffn_ln_bias,
            "ffn_gate_weight": convert_tensor_to_js(ffn_gate_weight),
            "ffn_gate_bias": ffn_gate_bias,
            "ffn_up_weight": convert_tensor_to_js(ffn_up_weight),
            "ffn_up_bias": ffn_up_bias,
            "ffn_down_weight": convert_tensor_to_js(ffn_down_weight),
            "ffn_down_bias": ffn_down_bias
        }
        
        layers.append(layer_params)
        print(f"  Converted layer {i+1}/3")
    
    js_params["layers"] = layers
    
    # Output projection  
    # PyTorch Linear(in, out) has weight [out, in], which matches JS expectation
    output_weight = state_dict["output_projection.weight"]  # [2, 48] - correct for JS
    output_bias = convert_tensor_to_js(state_dict["output_projection.bias"])
    
    js_params["output_weight"] = convert_tensor_to_js(output_weight)
    js_params["output_bias"] = output_bias
    
    print(f"✓ Converted to JavaScript transformer format")
    
    # Validate all matrix dimensions
    validation_result = validate_matrix_dimensions(js_params)
    if not validation_result["valid"]:
        print("❌ Matrix dimension validation failed!")
        for error in validation_result["errors"]:
            print(f"  - {error}")
        raise ValueError("Matrix dimensions don't match JavaScript expectations")
    else:
        print(f"✅ All {validation_result['total_matrices']} matrices have correct dimensions")
    
    return js_params

def save_js_model(js_params: dict, output_path: str):
    """Save parameters as JavaScript model file"""
    print(f"Saving JavaScript model to {output_path}...")
    
    # Create output directory if needed (only if there's a directory component)
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if there is one
        os.makedirs(output_dir, exist_ok=True)
    
    # Format as JavaScript file
    js_content = f"""// Transformer Model Parameters
// Generated from PyTorch checkpoint

window.TRANSFORMER_PARAMS = {json.dumps(js_params, indent=2)};

console.log("Loaded transformer model with", Object.keys(window.TRANSFORMER_PARAMS).length, "parameter tensors");
"""
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(js_content)
    
    print(f"✓ Saved {len(js_params)} parameter tensors to {output_path}")

def get_model_info(js_params: dict):
    """Print model information"""
    print("\nModel Information:")
    
    total_params = 0
    
    # Architecture
    print(f"  Architecture: d_model={js_params['d_model']}, n_heads={js_params['n_heads']}, n_layers={js_params['n_layers']}, ffn_hidden={js_params['ffn_hidden']}")
    
    # Embeddings
    cls_count = len(js_params['cls_embedding'])
    total_params += cls_count
    print(f"  cls_embedding: {cls_count:,} parameters")
    
    type_emb_count = 0
    for key, emb in js_params['type_embeddings'].items():
        count = len(emb)
        type_emb_count += count
        total_params += count
    print(f"  type_embeddings: {type_emb_count:,} parameters")
    
    # Input projections
    for proj_name in ['ctx_projection', 'predator_projection', 'boid_projection']:
        proj = js_params[proj_name]
        count = len(proj) * len(proj[0]) if len(proj) > 0 else 0
        total_params += count
        print(f"  {proj_name}: {count:,} parameters")
    
    # Transformer layers
    layer_total = 0
    for i, layer in enumerate(js_params['layers']):
        layer_count = 0
        # Count all layer parameters
        for key, value in layer.items():
            if isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], list):
                    # 2D array
                    count = len(value) * len(value[0])
                else:
                    # 1D array
                    count = len(value)
                layer_count += count
        layer_total += layer_count
        print(f"  layer_{i}: {layer_count:,} parameters")
    
    total_params += layer_total
    
    # Output projection
    output_weight_count = len(js_params['output_weight']) * len(js_params['output_weight'][0])
    output_bias_count = len(js_params['output_bias'])
    output_total = output_weight_count + output_bias_count
    total_params += output_total
    print(f"  output_projection: {output_total:,} parameters")
    
    print(f"  Total: {total_params:,} parameters")

def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to JavaScript')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to PyTorch checkpoint file')
    parser.add_argument('--output', type=str, default='src/config/model.js', help='Output JavaScript file path')
    parser.add_argument('--info', action='store_true', help='Show detailed model information')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    print("=== PyTorch to JavaScript Model Export ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print()
    
    try:
        # Load checkpoint
        state_dict = load_checkpoint(args.checkpoint)
        
        # Convert to JavaScript format
        js_params = convert_state_dict_to_js(state_dict)
        
        # Save JavaScript model
        save_js_model(js_params, args.output)
        
        # Show model info
        if args.info:
            get_model_info(js_params)
        
        print("\n✅ Export completed successfully!")
        print(f"You can now use the model in your browser by loading: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        raise

if __name__ == "__main__":
    main() 
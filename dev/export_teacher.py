"""Convert a teacher checkpoint (.pt) into a JS predator_nn.js weights JSON.

Teacher policy = MLP with ReLU hidden layers + tanh output + scale to
PREDATOR_MAX_FORCE. The JS NN library supports tanh natively and clips
the final output by `clipMagnitude * outputScale`, so we set
outputScale=PREDATOR_MAX_FORCE and activation='tanh' on the last layer.

Usage:
    python3 dev/export_teacher.py --ckpt path/to/best.pt --out dev/weights/teacher.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

PREDATOR_MAX_FORCE = 0.05


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    theta = ckpt['theta']
    feature_dim = ckpt['feature_dim']
    hidden_dims = ckpt['hidden_dims']
    dims = [feature_dim] + list(hidden_dims) + [2]
    layers = []
    off = 0
    for i in range(len(dims) - 1):
        in_d, out_d = dims[i], dims[i + 1]
        W = theta[off:off + in_d * out_d].view(in_d, out_d)
        off += in_d * out_d
        b = theta[off:off + out_d]
        off += out_d
        # ReLU on hidden layers, tanh on output
        act = 'relu' if i < len(dims) - 2 else 'tanh'
        layers.append({
            'inDim': in_d,
            'outDim': out_d,
            'activation': act,
            'W': W.flatten().tolist(),
            'b': b.tolist(),
        })

    out = {
        'version': 1,
        'id': f'teacher_H{"_".join(map(str, hidden_dims))}_gen{ckpt.get("gen", -1)}',
        'K': 4,
        'featureDim': feature_dim,
        # Teacher trained with raw (un-normalized) features; identity normalize.
        'inputMean': [0.0] * feature_dim,
        'inputStd': [1.0] * feature_dim,
        # tanh output is in [-1,1]; multiply by MAX_FORCE then no further clip.
        'outputScale': PREDATOR_MAX_FORCE,
        'clipMagnitude': PREDATOR_MAX_FORCE,
        'layers': layers,
        '_meta': {
            'source_ckpt': str(args.ckpt),
            'baseline_catches_at_save': ckpt.get('baseline_catches'),
            'gen': ckpt.get('gen'),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out} (featureDim={feature_dim}, hidden={hidden_dims})")


if __name__ == '__main__':
    main()

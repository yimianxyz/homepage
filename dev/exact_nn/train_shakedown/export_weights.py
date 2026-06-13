#!/usr/bin/env python3
"""Export a trained checkpoint's weights to JSON (float64 nested lists) for the
deterministic JS student scorer. Dumps every state_dict tensor by name + the
build args (arch/size/f64/head_out) so studentScores.js can reconstruct the
forward pass exactly.

  python3 export_weights.py --ckpt ckpt_l1rs_deepset_small_f64.pt --out weights.json
"""
import argparse, json
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model = sd['model']
    a = sd.get('args', {})
    weights = {k: v.double().tolist() for k, v in model.items()}
    out = {'args': {'task': a.get('task'), 'arch': a.get('arch'), 'size': a.get('size'),
                    'f64_head': a.get('f64_head')},
           'step': sd.get('step'),
           'keys': list(model.keys()),
           'shapes': {k: list(v.shape) for k, v in model.items()},
           'weights': weights}
    with open(args.out, 'w') as f:
        json.dump(out, f)
    print('exported %d tensors from %s (step %s) -> %s' % (len(weights), args.ckpt, sd.get('step'), args.out))
    for k in model.keys():
        print('  %-24s %s' % (k, list(model[k].shape)))


if __name__ == '__main__':
    main()

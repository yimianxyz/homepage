"""Screen patrol-target modes in sim_torch with the shipped NN.

Runs the same shipped weights through several auto_target modes and reports
mean catches across N seeds. Cheap exploration to find better-than-flock_centroid
patrol patterns before going to JS-verify.

Usage:
    python3 dev/screen_patrol.py --weights js/predator_weights.json \
        --seeds 64 --frames 5000
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_torch import Sim, load_weights


def run_one(weights, mode, opts, seeds, frames, device):
    sim = Sim(seeds=seeds, weights=weights, device=device,
              sequential=True, auto_target=mode, auto_target_opts=opts)
    t0 = time.time()
    if device == 'cuda' and torch.cuda.is_available():
        out = sim.run_graph(frames)
    else:
        out = sim.run(frames)
    el = time.time() - t0
    return out['mean_catches'], out['per_seed_catches'], el


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--seeds', type=int, default=64)
    p.add_argument('--seedStart', type=int, default=100)
    p.add_argument('--frames', type=int, default=5000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--report', default=None)
    a = p.parse_args()

    seeds = list(range(a.seedStart, a.seedStart + a.seeds))
    weights = load_weights(a.weights, device=a.device)

    # Variants to sweep
    variants = [
        ('flock_centroid', {}),                          # baseline
        ('weighted_centroid', {}),                       # past 16-seed +0.56
        ('predicted_centroid', {'lookahead': 5}),
        ('predicted_centroid', {'lookahead': 10}),
        ('predicted_centroid', {'lookahead': 15}),
        ('predicted_centroid', {'lookahead': 20}),
        ('weighted_predicted', {'lookahead': 5}),
        ('weighted_predicted', {'lookahead': 10}),
        ('weighted_predicted', {'lookahead': 15}),
        ('nearest_K_centroid', {'K': 4}),
        ('nearest_K_centroid', {'K': 8}),
        ('nearest_K_centroid', {'K': 16}),
        ('nearest_K_centroid', {'K': 32}),
    ]

    results = []
    base_mean = None
    base_per = None
    for mode, opts in variants:
        mean, per, el = run_one(weights, mode, opts, seeds, a.frames, a.device)
        label = mode + (('_' + '_'.join(f"{k}{v}" for k, v in opts.items())) if opts else '')
        if base_mean is None:
            base_mean, base_per = mean, per
        delta = mean - base_mean
        # paired z
        diffs = [p - bp for p, bp in zip(per, base_per)]
        mean_diff = sum(diffs) / len(diffs)
        var = sum((d - mean_diff) ** 2 for d in diffs) / max(1, len(diffs) - 1)
        sd = var ** 0.5
        z = mean_diff / (sd / (len(diffs) ** 0.5)) if sd > 0 else 0.0
        results.append({'mode': mode, 'opts': opts, 'label': label,
                         'mean': mean, 'delta_vs_base': delta, 'z': z,
                         'per_seed': per, 'elapsed_s': el})
        print(f"  {label:40s}  mean={mean:7.3f}  Δ={delta:+6.3f}  z={z:+5.2f}  ({el:5.1f}s)")

    out = {'weights': a.weights, 'seeds': seeds, 'frames': a.frames,
            'baseline_mode': 'flock_centroid', 'results': results}
    if a.report:
        with open(a.report, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"wrote {a.report}")


if __name__ == '__main__':
    main()

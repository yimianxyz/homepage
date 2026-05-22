"""High-quality eval harness for sim_torch.

  evaluate(weights_path_or_dict, seeds=range(100,116), frames=5000,
           device='cuda', use_graph=True, sequential=True)
    -> { mean_catches, per_seed_catches, elapsed_s, seeds, ... }

Defaults:
  - sequential=True   matches JS boid-update ordering (parallel mode is
                      rank-discordant with JS, ρ≈0.17)
  - use_graph=True    captures step() as a CUDA Graph and replays it,
                      cutting per-frame kernel-launch overhead
  - 16 canonical seeds (100..115) × 5000 frames = standard config

Returns a fully reproducible dict suitable for logging/JSON.
"""
import json
import time
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_torch import Sim, load_weights


def evaluate(weights_path_or_dict, seeds=tuple(range(100, 116)), frames=5000,
             device='cuda', use_graph=True, sequential=True,
             auto_target='flock_centroid'):
    if isinstance(weights_path_or_dict, (str, os.PathLike)):
        weights = load_weights(str(weights_path_or_dict), device=device)
    else:
        weights = weights_path_or_dict
    sim = Sim(seeds=list(seeds), weights=weights, device=device,
              sequential=sequential, auto_target=auto_target)
    t0 = time.time()
    if use_graph and device == 'cuda' and torch.cuda.is_available():
        out = sim.run_graph(frames)
    else:
        out = sim.run(frames)
    el = time.time() - t0
    return {
        'mean_catches': out['mean_catches'],
        'per_seed_catches': out['per_seed_catches'],
        'seeds': list(seeds),
        'frames': frames,
        'sequential': sequential,
        'use_graph': use_graph and device == 'cuda',
        'elapsed_s': el,
        'seed_fps': len(seeds) * frames / el,
    }


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('weights')
    p.add_argument('--seeds', type=int, default=16)
    p.add_argument('--seedStart', type=int, default=100)
    p.add_argument('--frames', type=int, default=5000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--no_graph', action='store_true')
    p.add_argument('--no_sequential', action='store_true')
    p.add_argument('--autoTarget', default='flock_centroid')
    p.add_argument('--report', default=None)
    a = p.parse_args()
    seeds = list(range(a.seedStart, a.seedStart + a.seeds))
    out = evaluate(a.weights, seeds=seeds, frames=a.frames, device=a.device,
                   use_graph=not a.no_graph, sequential=not a.no_sequential,
                   auto_target=a.autoTarget)
    print(json.dumps(out, indent=2))
    if a.report:
        with open(a.report, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"wrote {a.report}")

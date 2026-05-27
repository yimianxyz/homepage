"""Trace JS Oracle and sim_torch on the same seed, compare state per frame.

Runs both simulators on a single seed for N frames, dumps boid/predator
positions and catches each frame, finds the first divergence.

Usage:
    python3 dev/trace_diff.py --seed 100 --frames 50 \
        --weights js/predator_weights.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_js_trace(seed, frames, weights_path, out_path):
    """Invoke a Node helper that runs Oracle and dumps per-frame state."""
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trace_js.js')
    cmd = ['node', helper, '--seed', str(seed), '--frames', str(frames),
           '--weights', weights_path, '--out', out_path]
    subprocess.run(cmd, check=True)


def run_sim_torch_trace(seed, frames, weights_path, out_path, device='cuda'):
    """Run sim_torch on a single seed and dump state per frame."""
    from sim_torch import Sim, load_weights

    weights = load_weights(weights_path, device=device)
    sim = Sim(seeds=[seed], weights=weights, device=device,
              sequential=True, auto_target='flock_centroid')

    states = []
    for f in range(frames):
        # capture pre-step state
        pp = sim.pred_pos[0].cpu().tolist()
        pv = sim.pred_vel[0].cpu().tolist()
        bp = sim.boid_pos[0].cpu().tolist()
        bv = sim.boid_vel[0].cpu().tolist()
        ba = sim.boid_alive[0].cpu().tolist()
        pa = sim.pred_auto[0].cpu().tolist()
        catches = sim.catches[0].item()
        states.append({
            'frame': f,
            'pred_pos': pp, 'pred_vel': pv,
            'pred_auto': pa,
            'catches': catches,
            'n_alive': sum(1 for x in ba if x),
            # sampled boid (first alive)
            'boid0_pos': bp[0] if bp else None,
            'boid0_vel': bv[0] if bv else None,
        })
        sim.step()

    Path(out_path).write_text(json.dumps(states, indent=2))


def diff_traces(js_path, sim_path, atol=1e-3):
    js = json.load(open(js_path))
    sim = json.load(open(sim_path))
    n = min(len(js), len(sim))
    first_diff = None
    for i in range(n):
        a = js[i]
        b = sim[i]
        if a['catches'] != b['catches']:
            return ('catches', i, a['catches'], b['catches'])
        for k in ['pred_pos', 'pred_vel', 'pred_auto']:
            av, bv = a[k], b[k]
            for j in range(len(av)):
                if abs(av[j] - bv[j]) > atol:
                    return (k + '[' + str(j) + ']', i, av[j], bv[j])
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--frames', type=int, default=50)
    p.add_argument('--weights', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--js_out', default='/tmp/trace_js.json')
    p.add_argument('--sim_out', default='/tmp/trace_sim.json')
    a = p.parse_args()

    # JS trace
    print('Running JS trace...')
    run_js_trace(a.seed, a.frames, a.weights, a.js_out)
    print('Running sim_torch trace...')
    run_sim_torch_trace(a.seed, a.frames, a.weights, a.sim_out, a.device)

    diff = diff_traces(a.js_out, a.sim_out)
    if diff is None:
        print(f"OK: traces match for {a.frames} frames at seed {a.seed}")
    else:
        key, frame, jv, sv = diff
        print(f"DIVERGE at frame {frame}, {key}: JS={jv} SIM={sv}")


if __name__ == '__main__':
    main()

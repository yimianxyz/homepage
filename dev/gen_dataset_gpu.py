"""GPU-native dataset generator: rule rollouts capturing (features, target).

Mirrors dev/gen_dataset.js: rolls out the chosen rule policy on N seeds,
captures the rule's (features, steering) every frame, and writes a .bin
file in the same format gen_dataset.js produces (so train_one.js or
train_one_gpu.py can consume it).

Speedup: ~50-100x over Node.js (which forks 4 worker_threads on CPU).

Usage:
    python3 dev/gen_dataset_gpu.py --rule rule_v3 --mode score_minus_dist \
        --distW 0.05 --alpha 5 \
        --seeds 80 --seedStart 0 --frames 5000 --autoTarget flock_centroid \
        --out datasets/rule_v3_smd_a5_80seeds_5000f.bin
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_torch import Sim, build_features, fast_limit, PREDATOR_MAX_SPEED
import rule_torch as rt


class CaptureRuleSim(Sim):
    """RuleSim that also writes (feats, steering) to pre-allocated buffers each frame."""
    def __init__(self, seeds, kind, opts, frames, feat_dim,
                 auto_target='flock_centroid', device='cuda'):
        weights = {'featureDim': feat_dim,
                    'inputMean': None, 'inputStd': None,
                    'outputScale': 1.0, 'clipMagnitude': 0.0, 'layers': []}
        super().__init__(seeds=seeds, weights=weights, device=device,
                          sequential=True, auto_target=auto_target)
        self.rule_kind = kind
        self.rule_opts = opts
        self._rule_buffers = rt.make_rule_buffers(self.B, self.device,
                                                    dtype=torch.float64)
        self.frames = frames
        self.feat_dim = feat_dim
        # Pre-allocate capture buffers on GPU
        # (B, T, feat_dim) and (B, T, 2)
        self.cap_feats = torch.zeros(self.B, frames, feat_dim,
                                       dtype=torch.float32, device=device)
        self.cap_target = torch.zeros(self.B, frames, 2,
                                       dtype=torch.float32, device=device)
        self._frame = 0

    def _step_predator(self):
        self._update_auto_target()
        feats = build_features(
            self.pred_pos, self.pred_vel,
            self.boid_pos, self.boid_vel, self.boid_alive,
            self.pred_auto, self.feat_dim, self.device,
        )
        steering = rt.predator_steering(feats, self.rule_kind, self.rule_opts,
                                          buffers=self._rule_buffers).double()

        # Capture BEFORE we mutate sim state. Use float32 for compactness.
        if self._frame < self.frames:
            self.cap_feats[:, self._frame] = feats.float()
            self.cap_target[:, self._frame] = steering.float()
            self._frame += 1

        new_vx = self.pred_vel[:, 0] + steering[:, 0]
        new_vy = self.pred_vel[:, 1] + steering[:, 1]
        new_vx, new_vy = fast_limit(new_vx, new_vy, PREDATOR_MAX_SPEED)
        self.pred_vel[:, 0] = new_vx
        self.pred_vel[:, 1] = new_vy
        self.pred_pos[:, 0] += new_vx
        self.pred_pos[:, 1] += new_vy
        self.pred_pos[:, 0] = torch.where(self.pred_pos[:, 0] > self._wrap_w_max,
                                          self._wrap_neg20, self.pred_pos[:, 0])
        self.pred_pos[:, 0] = torch.where(self.pred_pos[:, 0] < self._wrap_neg20,
                                          self._wrap_w_max, self.pred_pos[:, 0])
        self.pred_pos[:, 1] = torch.where(self.pred_pos[:, 1] > self._wrap_h_max,
                                          self._wrap_neg20, self.pred_pos[:, 1])
        self.pred_pos[:, 1] = torch.where(self.pred_pos[:, 1] < self._wrap_neg20,
                                          self._wrap_h_max, self.pred_pos[:, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rule', default='rule_v3',
                   choices=['rule_v1', 'rule_v2', 'rule_v3', 'rule_v4', 'rule_v5'])
    p.add_argument('--mode', default='score_minus_dist')
    p.add_argument('--distW', type=float, default=0.05)
    p.add_argument('--alpha', type=float, default=0.0)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--seeds', type=int, default=80)
    p.add_argument('--seedStart', type=int, default=0)
    p.add_argument('--frames', type=int, default=5000)
    p.add_argument('--autoTarget', default='flock_centroid')
    p.add_argument('--featureDim', type=int, default=45)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    p.add_argument('--meta', default=None)
    args = p.parse_args()

    seeds = list(range(args.seedStart, args.seedStart + args.seeds))
    opts = {'mode': args.mode, 'distW': args.distW, 'alpha': args.alpha,
            'steps': args.steps}

    t0 = time.time()
    sim = CaptureRuleSim(seeds=seeds, kind=args.rule, opts=opts,
                          frames=args.frames, feat_dim=args.featureDim,
                          auto_target=args.autoTarget,
                          device=args.device)
    # We can't use run_graph here because graph-capture mode wraps step() in
    # a fixed-call recording, which would clash with our per-frame buffer
    # writes (the frame index is mutated each call). Use the un-graphed
    # run() — still GPU-fast.
    for _ in range(args.frames):
        sim.step()
    elapsed = time.time() - t0
    print(json.dumps({'phase': 'rollout_done', 'elapsed_s': elapsed,
                       'seed_fps': args.seeds * args.frames / elapsed}))

    # Reshape (B, T, *) -> (B*T, *) then column-stack feats|target -> .bin
    feats = sim.cap_feats.cpu().numpy()
    targs = sim.cap_target.cpu().numpy()
    B, T, FD = feats.shape
    rows = feats.reshape(B * T, FD)
    targs_r = targs.reshape(B * T, 2)
    out_arr = np.concatenate([rows, targs_r], axis=1).astype(np.float32)
    out_arr.tofile(args.out)

    meta = {
        'n': B * T,
        'featureDim': FD,
        'outputDim': 2,
        'rowFloats': FD + 2,
        'seeds': seeds,
        'framesPerSeed': T,
        'numBoids': sim.N,
        'rule': args.rule,
        'mode': args.mode,
        'distW': args.distW,
        'alpha': args.alpha,
        'steps': args.steps,
        'autoTarget': args.autoTarget,
        'generatedAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'elapsedMs': int(elapsed * 1000),
        'source': 'gen_dataset_gpu.py',
    }
    meta_path = args.meta or args.out.replace('.bin', '.meta.json')
    Path(meta_path).write_text(json.dumps(meta, indent=2))
    print(json.dumps({'phase': 'done', 'out': args.out, 'meta': meta_path,
                       'n': B * T, 'elapsedMs': int(elapsed * 1000)}))


if __name__ == '__main__':
    main()

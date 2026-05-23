"""Evaluate rule policies (v1/v2/v3/v4) in sim_torch on GPU.

Same sim_torch sequential+graph eval used for the NN, but with the
predator's steering coming from a Python port of dev/policy_spec.js's
rule policies (dev/rule_torch.py). Lets us screen many rule variants
fast on the L4 without going through Node.

Usage:
    python3 dev/eval_rule_torch.py --rule v3 --mode score_minus_dist \
        --distW 0.05 --alpha 0 --seeds 16 --frames 5000 --device cuda
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_torch import Sim, build_features
import rule_torch as rt


class RuleSim(Sim):
    """sim_torch Sim with predator stepping via rule_torch instead of nn_forward."""
    def __init__(self, seeds, kind, opts=None, **kw):
        # Initialise with a dummy weights dict so the parent Sim doesn't
        # try to use it during init. We replace _step_predator below.
        # We need featureDim=45 so all rule slots are populated.
        kw['weights'] = {'featureDim': 45,
                          'inputMean': None, 'inputStd': None,
                          'outputScale': 1.0, 'clipMagnitude': 0.0,
                          'layers': []}
        super().__init__(seeds=seeds, **kw)
        self.rule_kind = kind
        self.rule_opts = opts or {}

    def _step_predator(self):
        self._update_auto_target()
        feats = build_features(
            self.pred_pos, self.pred_vel,
            self.boid_pos, self.boid_vel, self.boid_alive,
            self.pred_auto, 45, self.device,
        )
        steering = rt.predator_steering(feats, self.rule_kind, self.rule_opts).double()

        # Same predator integration as Sim._step_predator
        new_vx = self.pred_vel[:, 0] + steering[:, 0]
        new_vy = self.pred_vel[:, 1] + steering[:, 1]
        from sim_torch import fast_limit
        new_vx, new_vy = fast_limit(new_vx, new_vy,
                                      torch.tensor(2.5, dtype=torch.float64, device=self.device).item())
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
    p.add_argument('--rule', default='v1',
                   choices=['v1', 'rule', 'v2', 'v3', 'v4'])
    p.add_argument('--mode', default='score_minus_dist')
    p.add_argument('--distW', type=float, default=0.05)
    p.add_argument('--alpha', type=float, default=0.0)
    p.add_argument('--seeds', type=int, default=16)
    p.add_argument('--seedStart', type=int, default=100)
    p.add_argument('--frames', type=int, default=5000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--no_graph', action='store_true')
    p.add_argument('--report', default=None)
    args = p.parse_args()

    seeds = list(range(args.seedStart, args.seedStart + args.seeds))
    kind = 'rule_' + args.rule if not args.rule.startswith('rule_') else args.rule
    if kind in ('rule_rule', 'rule_v1'):
        kind = 'rule_v1'
    opts = {'mode': args.mode, 'distW': args.distW, 'alpha': args.alpha}

    sim = RuleSim(seeds=seeds, kind=kind, opts=opts,
                   device=args.device, sequential=True,
                   auto_target='flock_centroid')

    t0 = time.time()
    if args.no_graph or args.device != 'cuda' or not torch.cuda.is_available():
        out = sim.run(args.frames)
    else:
        out = sim.run_graph(args.frames)
    el = time.time() - t0

    result = {
        'rule': kind,
        'opts': opts,
        'seeds': seeds,
        'frames': args.frames,
        'mean_catches': out['mean_catches'],
        'per_seed_catches': out['per_seed_catches'],
        'elapsed_s': el,
        'seed_fps': args.seeds * args.frames / el,
    }
    print(json.dumps(result, indent=2))
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()

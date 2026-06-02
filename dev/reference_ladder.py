"""E1 — reactive-vs-lookahead reference ladder.

Question: how much of the planner's catch edge is *reactive* (a smooth pursuit
of the nearest catchable cluster) vs *genuine lookahead* (the K-rollout argmax)?
Answer sets the floor on how small a distilled net can be: if a fixed reactive
heuristic already matches the planner, a tiny net trivially captures it.

All controllers drive the SAME analytic steering (planner_probe._step_with_target):
force = chase(nearest) if a boid is within POLICY_R else seek(target). They differ
ONLY in the patrol target they emit each frame — the exact distillable lever.

Controllers:
  e3d        : target = production E3D evolved-patrol target            (lower anchor)
  nearest    : target = nearest live boid, lead-adjusted (cand1)        (pure reactive pursuit)
  gateR      : nearest-lead if nearest-boid dist < R else E3D, swept R  (reactive + patrol fallback)
  planner    : K-rollout argmax committed target                        (ceiling)

Usage:
  python3 reference_ladder.py --n 256 --frames 5000 --seedStart 200000 \
      --K 16 --H 120 --D 8 --gateR 60,90,120,160,220
"""

import argparse
import json
import time
import numpy as np
import torch

import sim_torch as st
from sim_torch import Sim, PREDATOR_MAX_SPEED
import planner_probe as pp


def _nearest_lead_target(sim):
    """Nearest live boid, lead-adjusted (== cand1 of planner_probe), (B,2)."""
    cand = pp._candidate_targets(sim, 2)   # [E3D, nearest-lead]
    return cand[:, 1].clone()


def _nearest_dist(sim):
    dx = sim.boid_pos[..., 0] - sim.pred_pos[:, None, 0]
    dy = sim.boid_pos[..., 1] - sim.pred_pos[:, None, 1]
    d2 = dx * dx + dy * dy
    d2 = torch.where(sim.boid_alive, d2, torch.full_like(d2, float('inf')))
    return torch.sqrt(d2.min(dim=1).values)   # (B,)


def run_controller(kind, seeds, frames, device, R=None):
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D))
    for _ in range(frames):
        if kind == 'e3d':
            tgt = pp._e3d_target(sim)
        elif kind == 'nearest':
            tgt = _nearest_lead_target(sim)
        elif kind == 'gateR':
            e3d = pp._e3d_target(sim)
            near = _nearest_lead_target(sim)
            nd = _nearest_dist(sim).unsqueeze(1)
            tgt = torch.where(nd < R, near, e3d)
        else:
            raise ValueError(kind)
        pp._step_with_target(sim, tgt)
    return sim.catches.cpu().numpy()


def summarize(name, catches):
    mean = float(catches.mean())
    se = float(catches.std(ddof=1) / np.sqrt(len(catches)))
    return dict(controller=name, mean=round(mean, 3), se=round(se, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--seedStart', type=int, default=200000)
    ap.add_argument('--frames', type=int, default=5000)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--H', type=int, default=120)
    ap.add_argument('--D', type=int, default=8)
    ap.add_argument('--gateR', default='60,90,120,160,220')
    ap.add_argument('--weights', default='js/predator_weights.json')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    pp.WEIGHTS = st.load_weights(args.weights, device=device)
    pp.DENSE_LAMBDA = 0.0

    seeds = list(range(args.seedStart, args.seedStart + args.n))
    results = []
    t0 = time.time()

    results.append(summarize('e3d', run_controller('e3d', seeds, args.frames, device)))
    print(json.dumps(results[-1]), flush=True)

    results.append(summarize('nearest', run_controller('nearest', seeds, args.frames, device)))
    print(json.dumps(results[-1]), flush=True)

    for R in [float(x) for x in args.gateR.split(',') if x]:
        r = summarize(f'gateR_{R:g}', run_controller('gateR', seeds, args.frames, device, R=R))
        results.append(r)
        print(json.dumps(r), flush=True)

    planner_catches = pp.run_planner(seeds, args.frames, device, args.K, args.H, args.D)
    results.append(summarize('planner', planner_catches))
    print(json.dumps(results[-1]), flush=True)

    elapsed = time.time() - t0
    out = dict(n=args.n, seedStart=args.seedStart, frames=args.frames,
               K=args.K, H=args.H, D=args.D, device=device,
               elapsed=round(elapsed, 1), ladder=results)
    print('=== LADDER ===')
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(out, fh, indent=2)


if __name__ == '__main__':
    main()

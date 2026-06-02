"""Closed-loop eval of a TinyMLP coordinate-regression net (E2+).

The net emits a patrol target each frame; the predator drives the SAME analytic
steering as the planner (chase-in-range else seek-target). Reports mean catches
over n seeds x frames — the OUTCOME metric (not supervised loss). Feature build
reuses planner_probe.planner_obs so train/inference parity is exact.

Usage:
  python3 eval_coord.py --net net_coord.pt --n 256 --frames 5000 --seedStart 200000
"""

import argparse
import json
import time
import numpy as np
import torch

import sim_torch as st
from sim_torch import Sim
import planner_probe as pp
from distill_coord import TinyMLP


def run_net(net_blob, seeds, frames, device):
    M = net_blob['M']
    mu = net_blob['mu'].to(device); sd = net_blob['sd'].to(device)
    model = TinyMLP(net_blob['in_dim'], net_blob['hidden']).to(device)
    model.load_state_dict(net_blob['state']); model.eval()
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D))
    PS = 200.0
    with torch.no_grad():
        for _ in range(frames):
            e3d_rel = pp._e3d_target(sim) - sim.pred_pos
            ob = pp.planner_obs(sim, M, e3d_rel).float().to(device)
            tgt_rel = model((ob - mu) / sd)               # (B,2) in /PS units
            tgt = sim.pred_pos + tgt_rel.double() * PS
            pp._step_with_target(sim, tgt)
    return sim.catches.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--seedStart', type=int, default=200000)
    ap.add_argument('--frames', type=int, default=5000)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--weights', default='js/predator_weights.json')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    pp.WEIGHTS = st.load_weights(args.weights, device=device)
    pp.DENSE_LAMBDA = 0.0
    blob = torch.load(args.net, map_location=device)
    seeds = list(range(args.seedStart, args.seedStart + args.n))
    t0 = time.time()
    catches = run_net(blob, seeds, args.frames, device)
    res = dict(net=args.net, nparams=blob.get('nparams'), n=args.n,
               frames=args.frames, seedStart=args.seedStart,
               mean=round(float(catches.mean()), 3),
               se=round(float(catches.std(ddof=1) / np.sqrt(len(catches))), 3),
               elapsed=round(time.time() - t0, 1), device=device)
    print(json.dumps(res))
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(res, fh)


if __name__ == '__main__':
    main()

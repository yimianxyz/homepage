"""Closed-loop eval of the E3 candidate-pointer net. Each frame: build the same
K candidates the planner ranks (planner_probe._candidate_targets), score them,
argmax, commit to that candidate's point. Discrete selection -> no blur. The
analytic chase+E3D steering is reused. Reports OUTCOME (catches).

Usage: python3 eval_pointer.py --net net_ptr.pt --K 16 --n 256 --frames 5000
"""

import argparse
import json
import time
import numpy as np
import torch

import sim_torch as st
from sim_torch import Sim
import planner_probe as pp
from distill_pointer import PointerNet

PS, VS = 200.0, 6.0


def cand_features(sim, cand_abs, K):
    """(B,K,4) [rx/PS, ry/PS, dist/PS, is_e3d] matching dataset data[3]."""
    rel = cand_abs - sim.pred_pos[:, None, :]
    dist = torch.sqrt((rel ** 2).sum(2, keepdim=True))
    is_e3d = torch.zeros((sim.B, K, 1), dtype=rel.dtype, device=sim.device)
    is_e3d[:, 0, 0] = 1.0
    return torch.cat([rel / PS, dist / PS, is_e3d], dim=2)


def run(net_blob, seeds, frames, device, K):
    model = PointerNet(hidden=net_blob['hidden']).to(device)
    model.load_state_dict(net_blob['state']); model.eval()
    cmu = net_blob['cmu'].to(device); csd = net_blob['csd'].to(device)
    xmu = net_blob['xmu'].to(device); xsd = net_blob['xsd'].to(device)
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D))
    rows = torch.arange(sim.B, device=device)
    with torch.no_grad():
        for _ in range(frames):
            cand_abs = pp._candidate_targets(sim, K)             # (B,K,2)
            cf = cand_features(sim, cand_abs, K).float()
            n_alive = sim.boid_alive.float().sum(1) / sim.N
            ctx = torch.stack([sim.pred_vel[:, 0].float() / VS,
                               sim.pred_vel[:, 1].float() / VS,
                               n_alive.float()], dim=1)
            cfn = (cf - cmu) / csd
            ctxn = (ctx - xmu) / xsd
            score = model(cfn, ctxn)                             # (B,K)
            pick = score.argmax(1)
            tgt = cand_abs[rows, pick]
            pp._step_with_target(sim, tgt)
    return sim.catches.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--K', type=int, default=16)
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
    catches = run(blob, seeds, args.frames, device, args.K)
    res = dict(net=args.net, nparams=blob.get('nparams'), loss=blob.get('loss'),
               argmax_acc=blob.get('best', {}).get('acc'), K=args.K, n=args.n,
               frames=args.frames, mean=round(float(catches.mean()), 3),
               se=round(float(catches.std(ddof=1) / np.sqrt(len(catches))), 3),
               elapsed=round(time.time() - t0, 1))
    print(json.dumps(res))
    if args.out:
        json.dump(res, open(args.out, 'w'))


if __name__ == '__main__':
    main()

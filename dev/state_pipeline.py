"""End-to-end TD-MPC state-value pipeline: gen data -> train Vs -> eval configs.
Single process (no intermediate transfer). Eval is fast (no ballistic).

  python3 state_pipeline.py --n 64 --frames 5000 --K 16 --H 120 --D 8 --W 120 \
      --device cuda --weights ../js/predator_weights.json
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn

import state_value as sv
import planner_probe as pp
import sim_torch as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--seedStart', type=int, default=210000)   # held-out for train
    ap.add_argument('--evalSeedStart', type=int, default=200000)
    ap.add_argument('--evalN', type=int, default=64)
    ap.add_argument('--frames', type=int, default=5000)
    ap.add_argument('--evalFrames', type=int, default=5000)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--H', type=int, default=120)
    ap.add_argument('--D', type=int, default=8)
    ap.add_argument('--W', type=int, default=120)
    ap.add_argument('--hidden', type=int, default=48)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--twopass', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--weights', default='../js/predator_weights.json')
    ap.add_argument('--configs', default='60:120,40:120,60:32,60:16,80:120',
                    help='comma list of Hs:M for eval')
    args = ap.parse_args()
    dev = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    pp.TWO_PASS = args.twopass
    pp.WEIGHTS = st.load_weights(args.weights, device=dev)

    t0 = time.time()
    obs, vt, pcat = sv.run_log_state(list(range(args.seedStart, args.seedStart + args.n)),
                                     args.frames, dev, args.K, args.H, args.D, args.W)
    print(json.dumps(dict(stage='data', rows=int(obs.shape[0]), fdim=int(obs.shape[1]),
                          planner_mean=float(pcat.mean()), secs=round(time.time() - t0, 1))), flush=True)

    X = torch.from_numpy(obs).float(); Y = torch.from_numpy(vt).float()
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(X.shape[0], generator=g)
    nval = max(1, X.shape[0] // 10); vi, ti = perm[:nval], perm[nval:]
    Xtr = ((X[ti] - mu) / sd).to(dev); Ytr = Y[ti].to(dev)
    Xva = ((X[vi] - mu) / sd).to(dev); Yva = Y[vi].to(dev)
    net = sv.StateValueNet(X.shape[1], args.hidden, 2).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    bs = 8192
    for ep in range(args.epochs):
        net.train(); idx = torch.randperm(Xtr.shape[0], device=dev)
        for s in range(0, Xtr.shape[0], bs):
            b = idx[s:s + bs]; opt.zero_grad()
            loss = nn.functional.smooth_l1_loss(net(Xtr[b]), Ytr[b]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        vmae = (net(Xva) - Yva).abs().mean().item()
    nparams = sum(p.numel() for p in net.parameters())
    print(json.dumps(dict(stage='train', nparams=nparams, val_mae=round(vmae, 4))), flush=True)

    blob = dict(state={k: v.cpu() for k, v in net.state_dict().items()}, mu=mu, sd=sd,
                fdim=X.shape[1], hidden=args.hidden, depth=2)
    torch.save(blob, 'net_vs.pt')
    dv = sv.DeployVs(blob, dev)
    seeds = list(range(args.evalSeedStart, args.evalSeedStart + args.evalN))
    e3d = pp.run_e3d(seeds, args.evalFrames, dev)
    for cfg in args.configs.split(','):
        hs, m = [int(x) for x in cfg.split(':')]
        te = time.time()
        c = sv.run_state_student(seeds, args.evalFrames, dev, dv, args.K, args.D, hs, roll_M=m)
        print(json.dumps(dict(stage='eval', Hs=hs, roll_M=m, K=args.K,
                              student_mean=round(float(c.mean()), 3),
                              e3d=round(float(e3d.mean()), 2),
                              eval_secs=round(time.time() - te, 1))), flush=True)
    print('STATE_PIPELINE_DONE')


if __name__ == '__main__':
    main()

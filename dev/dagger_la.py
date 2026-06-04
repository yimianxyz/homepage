"""DAgger-on-lookahead: the key test for a SIMPLER design at 66.4.

Generate relabel data from the LOOKAHEAD student's own visited states (planner
relabels each decision), retrain the value net on it, then eval whether PRUNED
deploy (top-2/3 candidates) now reaches the target. If DAgger fixes the prior V's
poor candidate ranking, pruning becomes lossless -> an 8x-cheaper rollout at 66.4.

  python3 dagger_la.py --net net_v2_absval.pt --n 64 --seedStart 211000 --device cuda
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F

import feat_planner as fp
import planner_probe as pp
import sim_torch as st
from train_value import ValueNet
from eval_value import Deploy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--seedStart', type=int, default=211000)
    ap.add_argument('--frames', type=int, default=5000)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--H', type=int, default=120)
    ap.add_argument('--D', type=int, default=8)
    ap.add_argument('--Hs', type=int, default=60)
    ap.add_argument('--base', default='', help='optional base dataset to merge')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--weights', default='../js/predator_weights.json')
    args = ap.parse_args()
    dev = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    pp.WEIGHTS = st.load_weights(args.weights, device=dev)
    blob = torch.load(args.net, map_location='cpu'); model = Deploy(blob, dev)

    t0 = time.time()
    feat, ctx, gain, cat = fp.run_dagger_lookahead(
        list(range(args.seedStart, args.seedStart + args.n)),
        args.frames, dev, model, args.K, args.H, args.D, args.Hs)
    print(json.dumps(dict(stage='dagger_data', rows=int(feat.shape[0]),
                          student_mean=float(cat.mean()), secs=round(time.time() - t0, 1))), flush=True)
    F0 = torch.from_numpy(feat).float(); X0 = torch.from_numpy(ctx).float(); G0 = torch.from_numpy(gain).float()
    if args.base:
        b = torch.load(args.base, map_location='cpu')
        F0 = torch.cat([b['feat'].float(), F0], 0); X0 = torch.cat([b['ctx'].float(), X0], 0)
        G0 = torch.cat([b['gain'].float(), G0], 0)
    torch.save(dict(feat=F0, ctx=X0, gain=G0), 'ds_dag_la.pt')

    N, K, FC = F0.shape; FCTX = X0.shape[1]
    fmu, fsd = F0.reshape(-1, FC).mean(0), F0.reshape(-1, FC).std(0).clamp(min=1e-6)
    xmu, xsd = X0.mean(0), X0.std(0).clamp(min=1e-6)
    fn = ((F0 - fmu) / fsd).to(dev); xn = ((X0 - xmu) / xsd).to(dev); g = G0.to(dev)
    net = ValueNet(FC, FCTX, 48, 2).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    for ep in range(150):
        idx = torch.randperm(N, device=dev)
        for s in range(0, N, 8192):
            b2 = idx[s:s + 8192]; opt.zero_grad()
            F.smooth_l1_loss(net(fn[b2], xn[b2]), g[b2]).backward(); opt.step()
    dblob = dict(state={k: v.cpu() for k, v in net.state_dict().items()}, fmu=fmu, fsd=fsd,
                 xmu=xmu, xsd=xsd, fc=FC, fctx=FCTX, hidden=48, depth=2, nparams=sum(p.numel() for p in net.parameters()))
    torch.save(dblob, 'net_dagla.pt')
    print(json.dumps(dict(stage='trained', nparams=dblob['nparams'])), flush=True)

    dm = Deploy(dblob, dev)
    seeds = list(range(200000, 200024))
    for kr, m in [(16, 120), (3, 120), (2, 120)]:
        c = fp.run_value_lookahead_cheap(seeds, 1500, dev, dm, args.K, args.D, args.Hs, m, K_roll=kr)
        print(json.dumps(dict(stage='eval', K_roll=kr, roll_M=m, catches=round(float(c.mean()), 3),
                              note='ref full-net@1500f=15.1')), flush=True)
    print('DAGLA_DONE')


if __name__ == '__main__':
    main()

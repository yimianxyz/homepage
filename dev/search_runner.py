"""Automated architecture search: for each net-variant IDEA, train a value net on
the (feat,gain) data, then fast-eval the lookahead student at full + pruned deploy
configs. Goal: the SIMPLEST net/config that still hits ~66.4 catches. Logs every
idea to a jsonl so the search is a resumable, git-tracked trace.

Idea string: "hidden:depth:loss"  (loss in {absval,value,listnet,value+listnet,...}, '+' joined).
Deploy configs evaluated per net: full (16,120,Hs) + prune (Kr,120,Hs) + Mcut (16,M,Hs).

  python3 search_runner.py --data ds_feat_v2.pt --ideas 16:2:absval 24:2:absval ... \
      --evalN 24 --evalFrames 1500 --Hs 60 --out search_results.jsonl
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import feat_planner as fp
import planner_probe as pp
import sim_torch as st
from train_value import ValueNet, load_any
from eval_value import Deploy


def train_net(feat, ctx, gain, hidden, depth, losses, device, epochs=140):
    N, K, FC = feat.shape; FCTX = ctx.shape[1]
    fmu, fsd = feat.reshape(-1, FC).mean(0), feat.reshape(-1, FC).std(0).clamp(min=1e-6)
    xmu, xsd = ctx.mean(0), ctx.std(0).clamp(min=1e-6)
    fn = ((feat - fmu) / fsd).to(device); xn = ((ctx - xmu) / xsd).to(device); g = gain.to(device)
    net = ValueNet(FC, FCTX, hidden, depth).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    bs = 8192

    def loss_fn(score, gn):
        tot = 0.0
        if 'absval' in losses: tot = tot + F.smooth_l1_loss(score, gn)
        if 'value' in losses:
            tot = tot + F.smooth_l1_loss(score - score.mean(1, keepdim=True), gn - gn.mean(1, keepdim=True))
        if 'listnet' in losses:
            tot = tot + F.kl_div(F.log_softmax(score, 1), F.softmax(gn / 0.5, 1), reduction='batchmean')
        return tot
    for ep in range(epochs):
        idx = torch.randperm(N, device=device)
        for s in range(0, N, bs):
            b = idx[s:s + bs]; opt.zero_grad()
            loss_fn(net(fn[b], xn[b]), g[b]).backward(); opt.step()
    nparams = sum(p.numel() for p in net.parameters())
    blob = dict(state={k: v.cpu() for k, v in net.state_dict().items()}, fmu=fmu, fsd=fsd,
                xmu=xmu, xsd=xsd, fc=FC, fctx=FCTX, hidden=hidden, depth=depth, nparams=nparams)
    return blob, nparams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ideas', nargs='+', required=True)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--D', type=int, default=8)
    ap.add_argument('--Hs', type=int, default=60)
    ap.add_argument('--evalN', type=int, default=24)
    ap.add_argument('--evalFrames', type=int, default=1500)
    ap.add_argument('--seedStart', type=int, default=200000)
    ap.add_argument('--deploys', default='16:120,2:120,16:16',
                    help='comma list of K_roll:M deploy configs to test per net')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--weights', default='../js/predator_weights.json')
    ap.add_argument('--out', default='search_results.jsonl')
    args = ap.parse_args()
    dev = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    pp.WEIGHTS = st.load_weights(args.weights, device=dev)
    feat, ctx, gain = load_any(args.data)
    seeds = list(range(args.seedStart, args.seedStart + args.evalN))
    e3d = pp.run_e3d(seeds, args.evalFrames, dev)
    deploys = [tuple(int(x) for x in d.split(':')) for d in args.deploys.split(',')]
    fh = open(args.out, 'a')
    for idea in args.ideas:
        hid, dep, loss = idea.split(':')
        hid, dep = int(hid), int(dep); losses = loss.split('+')
        t0 = time.time()
        blob, nparams = train_net(feat, ctx, gain, hid, dep, losses, dev)
        model = Deploy(blob, dev)
        for (kr, m) in deploys:
            te = time.time()
            c = fp.run_value_lookahead_cheap(seeds, args.evalFrames, dev, model, args.K,
                                             args.D, args.Hs, m, bias0=0.0, K_roll=kr)
            rec = dict(idea=idea, hidden=hid, depth=dep, loss=loss, nparams=nparams,
                       K_roll=kr, roll_M=m, Hs=args.Hs, evalN=args.evalN, evalFrames=args.evalFrames,
                       catches=round(float(c.mean()), 3), e3d=round(float(e3d.mean()), 2),
                       train_s=round(time.time() - t0, 1), eval_s=round(time.time() - te, 1))
            print(json.dumps(rec), flush=True); fh.write(json.dumps(rec) + '\n'); fh.flush()
    print('SEARCH_DONE')


if __name__ == '__main__':
    main()

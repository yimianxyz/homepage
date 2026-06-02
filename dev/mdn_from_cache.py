"""mdn_from_cache — train the continuous-target mixture-density net from a cached
distill_v5 dataset (ds*.pt), NO planner re-gen.

WHY: candidate-argmax classification hits a near-tie precision wall (~0.25 decisive
acc, capacity-independent) because among 'decisive' frames the gain gap is a single
catch whose timing is a chaotic sub-frame quantity — not robustly predictable from
current state. BUT the planner's COMMITTED TARGET point g*(s)=cand[argmax gain] is a
clean deterministic function of state; the only obstruction to regressing it is
MULTIMODALITY (near-tie states have two ~equally-good targets, whose mean is the
centroid == baseline). A mixture-density head predicts the modes and picks one —
picking EITHER near-tied target is fine in closed loop, so this sidesteps the wall.
This is also the purest end-to-end shape: net -> 2D target -> production's analytic chase.

The committed offset/PS is recoverable from the cache: CF[...,:2] == (cand-pred)/PS,
so y = CF[arange, GN.argmax(1), :2].

Usage:
  python3 mdn_from_cache.py --load_data ds512.pt --d 256 --layers 6 --comps 8 \
      --epochs 300 --eval_n 512 --out mdnc.json
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn

import sim_torch as st
import planner_probe as pp
from distill_v5_mdn import MDNNet, mdn_nll, eval_cl, num_params, PS, BASE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load_data', required=True)
    ap.add_argument('--d', type=int, default=256)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--hid', type=int, default=512)
    ap.add_argument('--comps', type=int, default=8)
    ap.add_argument('--D', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--va_frac', type=float, default=0.1)
    ap.add_argument('--eval_seedStart', type=int, default=200000)
    ap.add_argument('--eval_n', type=int, default=512)
    ap.add_argument('--eval_frames', type=int, default=1500)
    ap.add_argument('--weights', default='predator_weights.json')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    pp.WEIGHTS = st.load_weights(args.weights, device=device)
    eval_seeds = list(range(args.eval_seedStart, args.eval_seedStart + args.eval_n))
    t0 = time.time()

    blob = torch.load(args.load_data, map_location='cpu')
    BF, MK, PSt, CF, GN = blob['data']
    gmean = blob['gmean']
    n = BF.shape[0]
    # committed target offset/PS = candidate (at argmax gain) rel/PS, i.e. CF[...,:2]
    y = CF[torch.arange(n), GN.argmax(1), :2].contiguous()          # (n,2)
    # how multimodal is the supervision? frac of frames where >1 candidate is ~tied-best
    gmax = GN.max(1, keepdim=True).values
    near = (GN >= gmax - 1e-6).float().sum(1)                       # #tied-best
    print(f"[mdnc] rows={n} planner_mean={gmean:.2f} d={args.d} L={args.layers} "
          f"comps={args.comps} mean_tied_best={near.mean():.2f} "
          f"multimodal_frac={float((near>1).float().mean()):.3f}", flush=True)

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    nv = int(n * args.va_frac)
    vi, ti = perm[:nv], perm[nv:]
    BFt, MKt, PSt_t, yt = [t[ti].to(device) for t in (BF, MK, PSt, y)]
    BFv, MKv, PSv, yv = [t[vi].to(device) for t in (BF, MK, PSt, y)]
    nt = BFt.shape[0]

    net = MDNNet(args.d, args.layers, args.heads, args.hid, args.comps).to(device)
    print(f"  params={num_params(net)}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    history = []
    for ep in range(args.epochs):
        net.train()
        pm = torch.randperm(nt, device=device)
        tot = 0.0
        for j in range(0, nt, args.bs):
            b = pm[j:j + args.bs]
            pi, mu, ls = net(BFt[b].float(), MKt[b], PSt_t[b].float())
            loss = mdn_nll(pi, mu, ls, yt[b].float())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            tot += loss.item() * len(b)
        sched.step()
        if ep % 25 == 0 or ep == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                pi, mu, ls = net(BFv.float(), MKv, PSv.float())
                vnll = mdn_nll(pi, mu, ls, yv.float()).item()
                # eval offset error of the picked mode vs committed target (in px)
                k = pi.argmax(1)
                best = mu[torch.arange(mu.shape[0], device=device), k] * PS
                perr = (best - yv.float() * PS).norm(dim=1).mean().item()
            print(f"    ep{ep}: nll={tot/nt:.4f} va_nll={vnll:.4f} va_pick_err_px={perr:.1f}",
                  flush=True)
    catches = eval_cl(net, eval_seeds, args.eval_frames, device, args.D)
    mean = float(catches.mean()); se = float(catches.std(ddof=1) / np.sqrt(len(catches)))
    print(f"[mdnc] EVAL mean={mean:.3f}±{se:.3f} vs_base={100*(mean-BASE)/BASE:+.1f}% "
          f"of_planner={100*mean/21.40:.1f}% {time.time()-t0:.0f}s", flush=True)
    history.append(dict(mean=mean, se=se, of_planner=100 * mean / 21.40))
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(dict(args=vars(args), params=num_params(net),
                           planner_mean=gmean, history=history), fh)
    print("DONE", json.dumps(history), flush=True)


if __name__ == '__main__':
    main()

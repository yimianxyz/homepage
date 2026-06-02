"""E2 — tiny coordinate-regression distillation, derived from existing data.

Reframe (see dev/reports/distill_rethink.md): the planner's per-frame target
choice is multimodal & near-arbitrary on ~65% of frames, so action-cloning by
classification / value-regression is ill-posed. Goal = OUTCOME-equivalence
(same catches). Approach: regress the committed TARGET COORDINATE with a tiny
MLP, and on AMBIGUOUS (near-tied) frames substitute a smooth canonical target so
the regression is single-moded. Net is tiny (browser-fast); the analytic chase +
E3D steering are reused unchanged — the net only emits the patrol target.

Features are reconstructed EXACTLY from the full-obs dataset (ds1024_dense08.pt)
to match planner_probe.planner_obs at inference (no dataset regeneration):
  [pred_vx/VS, pred_vy/VS,                 # 2
   e3d_rel_x/PS, e3d_rel_y/PS,             # 2
   (rx/PS, ry/PS, rvx/VS, rvy/VS) * M,     # 4M nearest live boids
   frac_alive,                             # 1
   cent_rx/PS, cent_ry/PS]                 # 2
Output: target_rel (x,y) in /PS units (predator-relative, *200 -> px).

Usage:
  python3 distill_coord.py --data datasets/ds1024_dense08.pt --M 6 \
      --ambig nearest --margin 0.10 --hidden 32 --epochs 60 --out net_coord.pt
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn

PS, VS = 200.0, 6.0


def build_obs_label(data, M, ambig, margin_thr):
    """From full-obs tensors -> (X[N,2+4M+3], Y[N,2], meta). No new rollouts."""
    boid = data[0].float()        # (N,120,5) [dx/PS, dy/PS, vx/VS, vy/VS, alive]
    pred = data[2].float()        # (N,5) [vx/VS, vy/VS, frac_alive, cd, size]
    cand = data[3].float()        # (N,16,4) [rx/PS, ry/PS, dist/PS, is_e3d]
    gain = data[4].double()       # (N,16)
    N = boid.shape[0]
    dxs, dys = boid[..., 0], boid[..., 1]
    alive = boid[..., 4] > 0.5
    d2 = dxs * dxs + dys * dys
    d2 = torch.where(alive, d2, torch.full_like(d2, float('inf')))
    order = torch.argsort(d2, dim=1)[:, :M]            # M nearest live
    rows = torch.arange(N).unsqueeze(1)
    nb = boid[rows, order]                              # (N,M,5)
    al = (nb[..., 4] > 0.5).float().unsqueeze(2)
    boid_block = (nb[..., :4] * al).reshape(N, 4 * M)   # dead -> 0
    pred_vel = pred[:, :2]
    frac_alive = pred[:, 2:3]
    e3d_rel = cand[:, 0, :2]                            # cand0 = E3D, /PS
    # alive centroid rel (in /PS units)
    af = alive.float()
    n_alive = af.sum(1, keepdim=True).clamp(min=1)
    cent = (boid[..., :2] * af.unsqueeze(2)).sum(1) / n_alive   # (N,2) /PS
    X = torch.cat([pred_vel, e3d_rel, boid_block, frac_alive, cent], dim=1)

    # committed target (argmax gain) and decisiveness margin
    srt_g, srt_i = gain.sort(1, descending=True)
    win = srt_i[:, 0]
    margin = (srt_g[:, 0] - srt_g[:, 1]).float()
    committed = cand[rows.squeeze(1), win, :2]          # (N,2) /PS

    # ambiguous-frame smooth prior
    if ambig == 'committed':
        Y = committed
    else:
        # nearest live boid lead-adjusted == cand index 1 (dist-sorted in gen)
        nearest_lead = cand[:, 1, :2]
        prior = nearest_lead if ambig == 'nearest' else e3d_rel
        decisive = (margin > margin_thr).unsqueeze(1)
        Y = torch.where(decisive, committed, prior)
    meta = dict(N=int(N), decisive_frac=float((margin > margin_thr).float().mean()))
    return X, Y, meta


class TinyMLP(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='datasets/ds1024_dense08.pt')
    ap.add_argument('--M', type=int, default=6)
    ap.add_argument('--ambig', choices=['committed', 'nearest', 'e3d'], default='nearest')
    ap.add_argument('--margin', type=float, default=0.10)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='net_coord.pt')
    args = ap.parse_args()

    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    blob = torch.load(args.data, map_location='cpu')
    X, Y, meta = build_obs_label(blob['data'], args.M, args.ambig, args.margin)
    print(json.dumps(dict(meta=meta, in_dim=X.shape[1], ambig=args.ambig,
                          margin=args.margin, hidden=args.hidden)), flush=True)

    N = X.shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    nval = int(N * args.val_frac)
    vi, ti = perm[:nval], perm[nval:]
    # standardize inputs
    mu = X[ti].mean(0); sd = X[ti].std(0).clamp(min=1e-6)
    Xn = (X - mu) / sd
    Xtr, Ytr = Xn[ti].to(device), Y[ti].to(device)
    Xva, Yva = Xn[vi].to(device), Y[vi].to(device)

    model = TinyMLP(X.shape[1], args.hidden).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.SmoothL1Loss()
    ntr = Xtr.shape[0]
    best_val = 1e9
    for ep in range(args.epochs):
        model.train()
        idx = torch.randperm(ntr, device=device)
        for s in range(0, ntr, args.bs):
            b = idx[s:s + args.bs]
            opt.zero_grad()
            loss = lossf(model(Xtr[b]), Ytr[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = lossf(model(Xva), Yva).item()
        best_val = min(best_val, vloss)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(json.dumps(dict(epoch=ep, val_smoothl1=round(vloss, 5))), flush=True)

    torch.save(dict(state=model.state_dict(), mu=mu, sd=sd, M=args.M,
                    in_dim=X.shape[1], hidden=args.hidden, ambig=args.ambig,
                    margin=args.margin, nparams=nparams, best_val=best_val),
               args.out)
    print(json.dumps(dict(saved=args.out, nparams=nparams, best_val=round(best_val, 5))))


if __name__ == '__main__':
    main()

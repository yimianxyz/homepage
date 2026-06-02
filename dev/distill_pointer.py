"""E3 — candidate-pointer net (discrete selection, no free-space blur).

The planner's committed target is ALWAYS one of K analytic candidates (E3D +
M nearest boid-leads). Instead of regressing a free-space coordinate (which
averages multimodal targets into a meaningless midpoint -- see E2 raw=27.2), a
tiny shared MLP scores each candidate and we ARGMAX. Output is therefore always
a valid candidate point -> no blur. Per the outcome-equivalence reframe, even
modest argmax accuracy can still match the planner closed-loop IF the near-tied
alternatives the net picks are themselves good (the audit says they are). This
is the experiment the prior "0.55 argmax wall" despair skipped.

Per-candidate features come straight from ds1024_dense08.pt data[3]
([rx/PS, ry/PS, dist/PS, is_e3d]); global context from data[2]
([vx/VS, vy/VS, frac_alive]). Target = per-candidate gain (data[4]).

Loss options:
  listnet : KL(softmax(score) || softmax(gain/tau))  -- rank-matching
  value   : SmoothL1(score, centered gain)           -- value regression
  cls     : cross-entropy on argmax(gain)            -- (baseline; expected weak)

Usage:
  python3 distill_pointer.py --data datasets/ds1024_dense08.pt \
      --loss listnet --tau 0.1 --hidden 24 --epochs 80 --out net_ptr.pt
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


def build(data):
    cand = data[3].float()        # (N,16,4) [rx/PS, ry/PS, dist/PS, is_e3d]
    pred = data[2].float()        # (N,5)
    gain = data[4].float()        # (N,16)
    ctx = pred[:, :3]             # vx/VS, vy/VS, frac_alive
    return cand, ctx, gain


class PointerNet(nn.Module):
    """Shared per-candidate scorer. in = [cand_feat(4), ctx(3)] -> scalar."""
    def __init__(self, cand_dim=4, ctx_dim=3, hidden=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cand_dim + ctx_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, cand, ctx):
        # cand (B,K,4), ctx (B,3) -> scores (B,K)
        K = cand.shape[1]
        c = ctx.unsqueeze(1).expand(-1, K, -1)
        x = torch.cat([cand, c], dim=2)
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='datasets/ds1024_dense08.pt')
    ap.add_argument('--loss', choices=['listnet', 'value', 'cls'], default='listnet')
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--hidden', type=int, default=24)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='net_ptr.pt')
    args = ap.parse_args()

    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    blob = torch.load(args.data, map_location='cpu')
    cand, ctx, gain = build(blob['data'])
    N, K = gain.shape
    # standardize candidate + ctx features (per-dim, over the candidate axis too)
    cflat = cand.reshape(-1, 4)
    cmu, csd = cflat.mean(0), cflat.std(0).clamp(min=1e-6)
    xmu, xsd = ctx.mean(0), ctx.std(0).clamp(min=1e-6)
    candn = (cand - cmu) / csd
    ctxn = (ctx - xmu) / xsd

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    nval = int(N * args.val_frac); vi, ti = perm[:nval], perm[nval:]
    cand_tr, ctx_tr, gain_tr = candn[ti].to(device), ctxn[ti].to(device), gain[ti].to(device)
    cand_va, ctx_va, gain_va = candn[vi].to(device), ctxn[vi].to(device), gain[vi].to(device)

    model = PointerNet(hidden=args.hidden).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def loss_fn(score, gn):
        if args.loss == 'listnet':
            return F.kl_div(F.log_softmax(score, 1),
                            F.softmax(gn / args.tau, 1), reduction='batchmean')
        if args.loss == 'value':
            gc = gn - gn.mean(1, keepdim=True)
            sc = score - score.mean(1, keepdim=True)
            return F.smooth_l1_loss(sc, gc)
        return F.cross_entropy(score, gn.argmax(1))

    def argmax_acc(score, gn):
        return (score.argmax(1) == gn.argmax(1)).float().mean().item()

    ntr = cand_tr.shape[0]
    best = dict(val=1e9, acc=0.0)
    for ep in range(args.epochs):
        model.train()
        idx = torch.randperm(ntr, device=device)
        for s in range(0, ntr, args.bs):
            b = idx[s:s + args.bs]
            opt.zero_grad()
            loss = loss_fn(model(cand_tr[b], ctx_tr[b]), gain_tr[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            sv = model(cand_va, ctx_va)
            vloss = loss_fn(sv, gain_va).item()
            acc = argmax_acc(sv, gain_va)
        if vloss < best['val']:
            best = dict(val=vloss, acc=acc)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(json.dumps(dict(epoch=ep, val=round(vloss, 5), argmax_acc=round(acc, 3))), flush=True)

    torch.save(dict(state=model.state_dict(), cmu=cmu, csd=csd, xmu=xmu, xsd=xsd,
                    hidden=args.hidden, loss=args.loss, tau=args.tau,
                    nparams=nparams, best=best), args.out)
    print(json.dumps(dict(saved=args.out, nparams=nparams, best=best)))


if __name__ == '__main__':
    main()

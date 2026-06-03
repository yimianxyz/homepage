"""Train the tiny per-candidate value net on feat_planner data.

Input: an .npz/.pt with feat (N,K,FC), ctx (N,FCTX), gain (N,K) (true H-frame
catch count per candidate). A SHARED MLP scores each candidate from
[cand_feat(FC), ctx(FCTX)] -> scalar; deploy picks argmax (optionally + short
rollout). This is distill_pointer.py but with the rich pursuit features that the
old 4-dim net lacked.

Losses (combinable):
  value   : SmoothL1(centered score, centered gain)  -- value regression
  listnet : KL(softmax(score) || softmax(gain/tau))  -- rank matching
  cls     : cross-entropy on argmax(gain)
We report argmax accuracy + top-tie accuracy (did the net pick a candidate whose
TRUE gain equals the max gain -- the outcome-relevant metric, since many maxes
are ties).

Usage:
  python3 train_value.py --data /tmp/feat_train.npz --loss value,listnet \
      --hidden 48 --epochs 120 --out net_value.pt
"""
import argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    def __init__(self, fc, fctx, hidden=48, depth=2):
        super().__init__()
        layers = [nn.Linear(fc + fctx, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, feat, ctx):
        # feat (B,K,FC), ctx (B,FCTX) -> (B,K)
        K = feat.shape[1]
        c = ctx.unsqueeze(1).expand(-1, K, -1)
        x = torch.cat([feat, c], dim=2)
        return self.net(x).squeeze(-1)


def load_any(path):
    if path.endswith('.npz'):
        d = np.load(path)
        return (torch.from_numpy(d['feat']).float(),
                torch.from_numpy(d['ctx']).float(),
                torch.from_numpy(d['gain']).float())
    blob = torch.load(path, map_location='cpu')
    return blob['feat'].float(), blob['ctx'].float(), blob['gain'].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--loss', default='value,listnet')   # comma list
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--hidden', type=int, default=48)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='net_value.pt')
    args = ap.parse_args()
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    losses = args.loss.split(',')

    parts = [load_any(p) for p in args.data.split(',')]
    feat = torch.cat([p[0] for p in parts], 0)
    ctx = torch.cat([p[1] for p in parts], 0)
    gain = torch.cat([p[2] for p in parts], 0)
    if len(parts) > 1:
        print(json.dumps(dict(merged=[p.split('/')[-1] for p in args.data.split(',')],
                              rows=[int(p[0].shape[0]) for p in parts])))
    N, K, FC = feat.shape
    FCTX = ctx.shape[1]
    # standardize features over (N*K) and ctx over N
    fflat = feat.reshape(-1, FC)
    fmu, fsd = fflat.mean(0), fflat.std(0).clamp(min=1e-6)
    xmu, xsd = ctx.mean(0), ctx.std(0).clamp(min=1e-6)
    featn = (feat - fmu) / fsd
    ctxn = (ctx - xmu) / xsd

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    nval = max(1, int(N * args.val_frac)); vi, ti = perm[:nval], perm[nval:]
    f_tr, x_tr, g_tr = featn[ti].to(device), ctxn[ti].to(device), gain[ti].to(device)
    f_va, x_va, g_va = featn[vi].to(device), ctxn[vi].to(device), gain[vi].to(device)

    model = ValueNet(FC, FCTX, args.hidden, args.depth).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def loss_fn(score, gn):
        tot = 0.0
        if 'value' in losses:
            gc = gn - gn.mean(1, keepdim=True)
            sc = score - score.mean(1, keepdim=True)
            tot = tot + F.smooth_l1_loss(sc, gc)
        if 'absval' in losses:
            # CALIBRATED absolute-gain regression: score ~ true catch count, so it
            # can be combined with real rollout catch counts in a value bootstrap.
            tot = tot + F.smooth_l1_loss(score, gn)
        if 'listnet' in losses:
            tot = tot + F.kl_div(F.log_softmax(score, 1),
                                 F.softmax(gn / args.tau, 1), reduction='batchmean')
        if 'cls' in losses:
            tot = tot + 0.3 * F.cross_entropy(score, gn.argmax(1))
        return tot

    def metrics(score, gn):
        pick = score.argmax(1)
        mx = gn.max(1, keepdim=True).values
        picked_gain = gn.gather(1, pick[:, None]).squeeze(1)
        top_tie = (picked_gain >= mx.squeeze(1) - 1e-6).float().mean().item()  # picked a true-max
        exact = (pick == gn.argmax(1)).float().mean().item()
        # realized vs E3D-default: how much MORE gain than always-cand0
        gain_pick = picked_gain.mean().item()
        gain_e3d = gn[:, 0].mean().item()
        gain_oracle = mx.squeeze(1).mean().item()
        return exact, top_tie, gain_pick, gain_e3d, gain_oracle

    ntr = f_tr.shape[0]
    best = dict(top_tie=0.0, ep=-1, state=None)
    for ep in range(args.epochs):
        model.train()
        idx = torch.randperm(ntr, device=device)
        for s in range(0, ntr, args.bs):
            b = idx[s:s + args.bs]
            opt.zero_grad()
            loss = loss_fn(model(f_tr[b], x_tr[b]), g_tr[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            sv = model(f_va, x_va)
            exact, top_tie, gp, ge, go = metrics(sv, g_va)
        if top_tie > best['top_tie']:
            best = dict(top_tie=top_tie, exact=exact, gain_pick=gp, ep=ep,
                        state={k: v.cpu().clone() for k, v in model.state_dict().items()})
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(json.dumps(dict(ep=ep, exact=round(exact, 3), top_tie=round(top_tie, 3),
                                  gain_pick=round(gp, 3), gain_e3d=round(ge, 3),
                                  gain_oracle=round(go, 3))), flush=True)

    torch.save(dict(state=best['state'], fmu=fmu, fsd=fsd, xmu=xmu, xsd=xsd,
                    fc=FC, fctx=FCTX, hidden=args.hidden, depth=args.depth,
                    nparams=nparams, best={k: best[k] for k in ('top_tie', 'exact', 'gain_pick', 'ep')}),
               args.out)
    print(json.dumps(dict(saved=args.out, nparams=nparams,
                          best_top_tie=round(best['top_tie'], 3), best_ep=best['ep'])))


if __name__ == '__main__':
    main()

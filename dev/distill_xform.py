"""Transformer / set-attention ranker over the K candidates.

The MLP value net scores each candidate independently: V(feat_k, ctx). This
scores all K candidates JOINTLY via self-attention, so each candidate is judged
RELATIVE to the others -- the natural inductive bias for "which candidate is the
planner's pick." Trained on the same strict gen data (feat,ctx,gain).

Decisiveness weighting: ~72% of rows have E3D (cand 0) as the argmax and ~46%
have no catch at all, so the rare DECISIVE rows (a non-E3D candidate strictly
beats E3D) are what matter. We up-weight them so the ranker learns them instead
of defaulting to E3D.

Metrics match train_value.py (exact = picks the planner's argmax; gain_pick vs
gain_e3d vs gain_oracle) so results are directly comparable to the MLP.

  python3 distill_xform.py --data ds_strict_vm2.pt --d 32 --layers 2 --heads 4 \
      --loss listnet,cls --decisive_w 4 --epochs 200 --device cuda --out net_xform.pt
"""
import argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_value import load_any


class XformRanker(nn.Module):
    def __init__(self, fc, fctx, d=32, heads=4, layers=2, ff=2):
        super().__init__()
        self.fc, self.fctx = fc, fctx
        self.tok = nn.Linear(fc + fctx, d)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=ff * d,
                                         batch_first=True, dropout=0.0, activation='gelu')
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, 1)

    def forward(self, feat, ctx):
        B, K, _ = feat.shape
        x = torch.cat([feat, ctx[:, None, :].expand(B, K, self.fctx)], dim=-1)
        x = self.tok(x)
        x = self.enc(x)                      # self-attention across the K candidates
        return self.head(x).squeeze(-1)      # (B,K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--loss', default='listnet,cls')
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--d', type=int, default=32)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--decisive_w', type=float, default=1.0,
                    help='extra weight on rows where a non-E3D candidate strictly beats E3D')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='net_xform.pt')
    args = ap.parse_args()
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    losses = args.loss.split(',')

    parts = [load_any(p) for p in args.data.split(',')]
    feat = torch.cat([p[0] for p in parts], 0)
    ctx = torch.cat([p[1] for p in parts], 0)
    gain = torch.cat([p[2] for p in parts], 0)
    N, K, FC = feat.shape
    FCTX = ctx.shape[1]
    fflat = feat.reshape(-1, FC)
    fmu, fsd = fflat.mean(0), fflat.std(0).clamp(min=1e-6)
    xmu, xsd = ctx.mean(0), ctx.std(0).clamp(min=1e-6)
    featn = (feat - fmu) / fsd
    ctxn = (ctx - xmu) / xsd

    # per-row weight: decisive rows (a non-E3D candidate strictly beats cand0) up-weighted
    best = gain.max(1).values
    decisive = (best > gain[:, 0] + 1e-6)
    w = torch.ones(N) + (args.decisive_w - 1.0) * decisive.float()

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    nval = max(1, int(N * args.val_frac)); vi, ti = perm[:nval], perm[nval:]
    f_tr, x_tr, g_tr, w_tr = featn[ti].to(device), ctxn[ti].to(device), gain[ti].to(device), w[ti].to(device)
    f_va, x_va, g_va = featn[vi].to(device), ctxn[vi].to(device), gain[vi].to(device)

    model = XformRanker(FC, FCTX, args.d, args.heads, args.layers).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def loss_fn(score, gn, w):
        tot = 0.0
        if 'value' in losses:
            gc = gn - gn.mean(1, keepdim=True); sc = score - score.mean(1, keepdim=True)
            tot = tot + (F.smooth_l1_loss(sc, gc, reduction='none').mean(1) * w).mean()
        if 'absval' in losses:
            tot = tot + (F.smooth_l1_loss(score, gn, reduction='none').mean(1) * w).mean()
        if 'listnet' in losses:
            kl = F.kl_div(F.log_softmax(score, 1), F.softmax(gn / args.tau, 1), reduction='none').sum(1)
            tot = tot + (kl * w).mean()
        if 'cls' in losses:
            ce = F.cross_entropy(score, gn.argmax(1), reduction='none')
            tot = tot + (ce * w).mean()
        return tot

    @torch.no_grad()
    def metrics(f, x, gn):
        score = model(f, x)
        pick = score.argmax(1)
        mx = gn.max(1, keepdim=True).values
        pg = gn.gather(1, pick[:, None]).squeeze(1)
        return (dict(exact=float((pick == gn.argmax(1)).float().mean()),
                     top_tie=float((pg >= mx.squeeze(1) - 1e-6).float().mean()),
                     gain_pick=float(pg.mean()), gain_e3d=float(gn[:, 0].mean()),
                     gain_oracle=float(mx.squeeze(1).mean())))

    ntr = f_tr.shape[0]
    best_exact, best_state = -1, None
    for ep in range(args.epochs):
        model.train()
        idx = torch.randperm(ntr, device=device)
        for s in range(0, ntr, args.bs):
            b = idx[s:s + args.bs]
            loss = loss_fn(model(f_tr[b], x_tr[b]), g_tr[b], w_tr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 0 or ep == args.epochs - 1:
            model.eval(); m = metrics(f_va, x_va, g_va)
            print(json.dumps(dict(ep=ep, **{k: round(v, 4) for k, v in m.items()})), flush=True)
            if m['exact'] > best_exact:
                best_exact = m['exact']
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(dict(arch='xform', state=best_state, fc=FC, fctx=FCTX, d=args.d, heads=args.heads,
                    layers=args.layers, fmu=fmu, fsd=fsd, xmu=xmu, xsd=xsd, nparams=nparams),
               args.out)
    print(json.dumps(dict(saved=args.out, nparams=nparams, best_exact=round(best_exact, 4))))


if __name__ == '__main__':
    main()

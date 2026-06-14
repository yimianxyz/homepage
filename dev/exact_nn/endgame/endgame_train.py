#!/usr/bin/env python3
"""Phase-2 (simplified direction) PURE endgame NN — maximize standalone egBoid
agreement with prod, predicting from RAW kinematics / cheap closed-form geometry
(eg_features, 18-dim; NOT the exact O(N·TMAX) scan-t). The egBoid decision is
argmin over present boids of the per-boid predicted scan-t (smooth separable
geometry) — a GENUINE learnable target, no fallback in the decision path.

Objective = argmin cross-entropy (the decision, dominant) + scan-t Huber (aux/
calibration). Held-out by NATURAL seed (the deployable distribution): val = the
top val-frac of natural seeds; train = all scatter + the remaining natural seeds.
A-S erf GELU == prod cp_erf == egboidPick deploy (bit-faithful argmin).

  python3 endgame_train.py --data packed_eg.json.gz --nat-src 2 --steps 12000 \
      --h 128 --w-ce 3.0 --out eg_weights_pure.json
"""
import argparse, gzip, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_AS = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]


def as_erf(x):
    s = torch.sign(x); ax = x.abs()
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((_AS[4] * t + _AS[3]) * t) + _AS[2]) * t + _AS[1]) * t + _AS[0]) * t * torch.exp(-ax * ax)
    return s * y


class ASGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + as_erf(x * 0.7071067811865476))


class MLP(nn.Module):
    def __init__(self, nf, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nf, h), ASGELU(), nn.Linear(h, h), ASGELU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def agree(model, feat, label, mask, egIdx, bs=16384):
    model.eval(); n = feat.shape[0]; ok = 0; mn = 0.0; md = 0.0
    with torch.no_grad():
        for lo in range(0, n, bs):
            f, m, eg, lb = feat[lo:lo+bs], mask[lo:lo+bs], egIdx[lo:lo+bs], label[lo:lo+bs]
            p = model(f); d = (p - lb) * m
            mn += float((d*d).sum()); md += float(m.sum())
            pm = torch.where(m > 0.5, p, torch.full_like(p, 1e9))
            ok += int((pm.argmin(1) == eg).sum())
    model.train()
    return ok / max(n, 1), mn / max(md, 1.0)


def percell(model, feat, label, mask, egIdx, cell, names):
    out = {}
    for c in sorted(set(cell.tolist())):
        sel = cell == c
        if sel.sum() == 0: continue
        a, _ = agree(model, feat[sel], label[sel], mask[sel], egIdx[sel])
        out[names[c] if c < len(names) else str(c)] = (a, int(sel.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--nat-src', type=int, default=2, help='src index of the NATURAL dir (deployable val)')
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--h', type=int, default=128)
    ap.add_argument('--w-ce', type=float, default=3.0)
    ap.add_argument('--ce-scale', type=float, default=3.0)
    ap.add_argument('--w-mse', type=float, default=1.0)
    ap.add_argument('--nat-val-frac', type=float, default=0.2)
    ap.add_argument('--nat-batch-frac', type=float, default=0.0, help='fraction of each batch drawn from natural-train (oversample the deployable distribution)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='eg_weights_pure.json')
    ap.add_argument('--ckpt', default='eg_pure.pt')
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    d = json.loads(gzip.open(args.data, 'rt').read())
    feat = np.asarray(d['feat'], np.float32); label = np.asarray(d['label'], np.float32)
    mask = np.asarray(d['mask'], np.float32); egIdx = np.asarray(d['egIdx'], np.int64)
    seed = np.asarray(d['seed'], np.int64); src = np.asarray(d['src'], np.int64)
    cellnames = sorted(set(d['cell'])); cidx = {c: i for i, c in enumerate(cellnames)}
    cell = np.asarray([cidx[c] for c in d['cell']], np.int64)
    nf = feat.shape[2]
    # held-out NATURAL val: top val-frac of natural seeds (whole games on one side)
    nat = src == args.nat_src
    nat_seeds = np.unique(seed[nat])
    thr = nat_seeds[int(len(nat_seeds) * (1 - args.nat_val_frac))] if len(nat_seeds) else 1 << 62
    held = nat_seeds[nat_seeds >= thr]                 # held-out natural seeds
    in_held = np.isin(seed, held)
    val = nat & in_held                                 # natural commits at held-out seeds
    tr = ~in_held                                       # everything else (excl. ALL commits at held seeds → leakage-clean)
    dev = torch.device(args.device)
    T = lambda a, t=torch.float32: torch.as_tensor(a, dtype=t, device=dev)
    ftr, ltr, mtr, etr = T(feat[tr]), T(label[tr]), T(mask[tr]), T(egIdx[tr], torch.long)
    # natural-train index pool (for oversampling the deployable distribution)
    tr_is_nat = (src[tr] == args.nat_src)
    nat_tr_idx = torch.as_tensor(np.where(tr_is_nat)[0], device=dev)
    sca_tr_idx = torch.as_tensor(np.where(~tr_is_nat)[0], device=dev)
    natv = val
    fv, lv, mv, ev, cv = T(feat[natv]), T(label[natv]), T(mask[natv]), T(egIdx[natv], torch.long), cell[natv]
    # natural TRAIN subset (sanity) + scatter-only
    print('[eg-pure] commits %d | train %d (scatter %d + nat %d) | NAT-VAL %d (%d seeds), nfeat=%d, egDerivedMismatch=%s' %
          (len(feat), int(tr.sum()), int((tr & (src != args.nat_src)).sum()), int((tr & nat).sum()),
           int(natv.sum()), len(np.unique(seed[natv])), nf, d.get('egDerivedMismatch')), flush=True)

    model = MLP(nf, args.h).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    g = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    ntr = ftr.shape[0]; t0 = time.time(); best = -1.0; best_sd = None
    n_nat_b = int(args.bs * args.nat_batch_frac) if len(nat_tr_idx) else 0
    n_sca_b = args.bs - n_nat_b
    for step in range(args.steps):
        if n_nat_b > 0:
            ni = nat_tr_idx[torch.randint(0, len(nat_tr_idx), (n_nat_b,), generator=g).to(dev)]
            si = sca_tr_idx[torch.randint(0, len(sca_tr_idx), (n_sca_b,), generator=g).to(dev)]
            idx = torch.cat([ni, si])
        else:
            idx = torch.randint(0, ntr, (args.bs,), generator=g).to(dev)
        f, l, m, eg = ftr[idx], ltr[idx], mtr[idx], etr[idx]
        p = model(f)
        dd = (p - l) * m
        mse = (dd * dd).sum() / m.sum().clamp(min=1)     # TIGHT scan-t regression (calibration → argmin)
        logits = (-p * args.ce_scale).masked_fill(m < 0.5, float('-inf'))
        ce = F.cross_entropy(logits, eg)                 # argmin decision (the egBoid)
        loss = args.w_mse * mse + args.w_ce * ce
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step + 1) % 1000 == 0 or step + 1 == args.steps:
            va, vmse = agree(model, fv, lv, mv, ev)
            ta, _ = agree(model, ftr, ltr, mtr, etr)
            print('[eg-pure] step %5d loss %.4f ce %.3f | train-agree %.4f | NAT-VAL agree %.4f (rmse %.1ff) | %.0fs' %
                  (step + 1, float(loss), float(ce), ta, va, (vmse**0.5)*100, time.time()-t0), flush=True)
            if va > best:
                best = va; best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    # restore best-natural-val checkpoint
    if best_sd is not None:
        model.load_state_dict(best_sd)
    va, vmse = agree(model, fv, lv, mv, ev)
    pc = percell(model, fv, lv, mv, ev, cv, cellnames)
    print('[eg-pure] BEST NAT-VAL agree %.4f | per-cell %s' % (va, {k: '%.4f(%d)' % v for k, v in pc.items()}), flush=True)
    sd = model.state_dict()
    json.dump({'arch': 'mlp', 'h': args.h, 'nfeat': nf, 'keys': list(sd.keys()),
               'shapes': {k: list(v.shape) for k, v in sd.items()},
               'weights': {k: v.double().cpu().tolist() for k, v in sd.items()},
               'nat_val_agree': va, 'nat_val_percell': {k: v[0] for k, v in pc.items()},
               'objective': 'argmin-CE+Huber, cheap-geom (no exact scan-t)', 'steps': args.steps},
              open(args.out, 'w'))
    torch.save({'model': sd, 'args': vars(args), 'nat_val_agree': va}, args.ckpt)
    print('[eg-pure] DONE nat_val_agree=%.4f -> %s' % (va, args.out), flush=True)


if __name__ == '__main__':
    main()

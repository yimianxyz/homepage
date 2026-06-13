#!/usr/bin/env python3
"""L1e scan-t regressor (task #18). Per-boid MLP f(features)->scan-t (the egBoid
commit is argmin over the <=5 present boids; margin = 2nd-min - min scan-t). The
target is per-boid SEPARABLE, so this is a clean scalar regression — unlike D1's
chaotic rollout. Reads the JS-packed file (features computed in eg_features.js, so
they match the deploy egboidPick bit-for-bit). Reports per-boid MSE + per-COMMIT
egBoid argmin-agreement (the deploy metric), exports weights JSON for the JS deploy.

  python3 eg_train.py --data packed_eg.json.gz --steps 8000 --out eg_weights.json
"""
import argparse, gzip, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TMAX_N = 14.0  # TMAX/100 — unreachable sentinel (scan-t label for null)


def load(path):
    d = json.loads(gzip.open(path, 'rt').read())
    feat = np.asarray(d['feat'], dtype=np.float32)        # (C,5,F)
    label = np.asarray(d['label'], dtype=np.float32)      # (C,5)
    mask = np.asarray(d['mask'], dtype=np.float32)        # (C,5)
    egIdx = np.asarray(d['egIdx'], dtype=np.int64)        # (C,)
    seed = np.asarray(d['seed'], dtype=np.int64)
    cell = np.asarray(d['cell'])
    return feat, label, mask, egIdx, seed, cell, d


class MLP(nn.Module):
    def __init__(self, nf, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nf, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):                  # (B,5,F) -> (B,5)
        return self.net(x).squeeze(-1)


def agree(model, feat, label, mask, egIdx, bs=8192):
    model.eval()
    n = feat.shape[0]; ok = 0; mse_num = 0.0; mse_den = 0.0
    with torch.no_grad():
        for lo in range(0, n, bs):
            f = feat[lo:lo+bs]; m = mask[lo:lo+bs]; eg = egIdx[lo:lo+bs]; lb = label[lo:lo+bs]
            p = model(f)
            d = (p - lb) * m
            mse_num += float((d*d).sum()); mse_den += float(m.sum())
            pm = torch.where(m > 0.5, p, torch.full_like(p, 1e9))
            am = pm.argmin(1)
            ok += int((am == eg).sum())
    model.train()
    return ok / n, mse_num / max(mse_den, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--h', type=int, default=64)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='eg_weights.json')
    ap.add_argument('--ckpt', default='eg_ckpt.pt')
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    feat, label, mask, egIdx, seed, cell, raw = load(args.data)
    nf = feat.shape[2]
    # val split by seed (held-out games)
    useeds = np.unique(seed); thr = useeds[int(len(useeds) * (1 - args.val_frac))]
    tr = seed < thr; va = ~tr
    dev = torch.device(args.device)
    def T(a): return torch.from_numpy(a).to(dev)
    ftr, ltr, mtr, etr = T(feat[tr]), T(label[tr]), T(mask[tr]), T(egIdx[tr])
    fva, lva, mva, eva = T(feat[va]), T(label[va]), T(mask[va]), T(egIdx[va])
    print('[l1e] commits %d (%d train / %d val), nfeat=%d, egDerivedMismatch=%d' %
          (feat.shape[0], int(tr.sum()), int(va.sum()), nf, raw.get('egDerivedMismatch', -1)), flush=True)

    model = MLP(nf, args.h).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ntr = ftr.shape[0]
    g = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, ntr, (args.bs,), generator=g).to(dev)
        f, l, m = ftr[idx], ltr[idx], mtr[idx]
        p = model(f)
        d = (p - l) * m
        loss = (d * d).sum() / m.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step + 1) % 1000 == 0 or step + 1 == args.steps:
            va_ag, va_mse = agree(model, fva, lva, mva, eva)
            tr_ag, tr_mse = agree(model, ftr, ltr, mtr, etr)
            print('[l1e] step %5d loss %.4f | train agree %.4f mse %.4f | val agree %.4f mse %.4f (rmse %.2f frames)' %
                  (step + 1, float(loss), tr_ag, tr_mse, va_ag, va_mse, (va_mse ** 0.5) * 100), flush=True)

    va_ag, va_mse = agree(model, fva, lva, mva, eva)
    # export weights (float64 lists) for the JS deploy
    sd = model.state_dict()
    weights = {k: v.double().cpu().tolist() for k, v in sd.items()}
    json.dump({'arch': 'mlp', 'h': args.h, 'nfeat': nf, 'keys': list(sd.keys()),
               'shapes': {k: list(v.shape) for k, v in sd.items()}, 'weights': weights,
               'val_agree': va_ag, 'val_mse': va_mse, 'steps': args.steps},
              open(args.out, 'w'))
    torch.save({'model': sd, 'args': vars(args), 'val_agree': va_ag}, args.ckpt)
    print('[l1e] DONE val_agree=%.4f val_rmse=%.2f frames -> %s' % (va_ag, (va_mse**0.5)*100, args.out), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Decisive genuineness ablation (lead 3-lab review): does the endgame NN learn the
egBoid geometry, or lean on the FED wrap-aware analytic reach-time?

On the SAME held-out NATURAL commits, compares three deciders:
  RAW-NN    : NN trained on raw-kinematics features (eg_features_raw, NO reach-time)
  FULL-NN   : NN trained with the analytic reach-time fed (eg_features)        [as-shipped]
  ANALYTIC  : argmin of the wrap-aware analytic reach-time (full feat col 12), NO NN
Reports per-method egBoid agreement + bootstrap 95% CI, and the bootstrap CI of the
lifts (FULL-NN − ANALYTIC, RAW-NN − ANALYTIC). raw/full packs share commit order
(same dirs/files/lines), so index i aligns across them.

  python3 eg_ablation.py --raw-pack packed_raw.json.gz --full-pack packed_full.json.gz \
      --raw-ckpt eg_raw.pt --full-ckpt eg_full.pt --nat-src 2 --nat-val-frac 0.4
"""
import argparse, gzip, json
import numpy as np
import torch
import torch.nn as nn

_AS = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
def as_erf(x):
    s = torch.sign(x); ax = x.abs(); t = 1.0/(1.0+0.3275911*ax)
    return s*(1.0-(((((_AS[4]*t+_AS[3])*t)+_AS[2])*t+_AS[1])*t+_AS[0])*t*torch.exp(-ax*ax))
class ASGELU(nn.Module):
    def forward(self, x): return 0.5*x*(1.0+as_erf(x*0.7071067811865476))
class MLP(nn.Module):
    def __init__(s, nf, h): super().__init__(); s.net=nn.Sequential(nn.Linear(nf,h),ASGELU(),nn.Linear(h,h),ASGELU(),nn.Linear(h,1))
    def forward(s, x): return s.net(x).squeeze(-1)

def load(p):
    d = json.loads(gzip.open(p, 'rt').read())
    return (np.asarray(d['feat'], np.float32), np.asarray(d['mask'], np.float32),
            np.asarray(d['egIdx'], np.int64), np.asarray(d['seed'], np.int64),
            np.asarray(d['src'], np.int64), np.asarray(d['cell']))

def nn_correct(ckpt, feat, mask, egIdx, dev):
    ck = torch.load(ckpt, map_location=dev)
    m = MLP(feat.shape[2], ck['args']['h']).to(dev); m.load_state_dict(ck['model']); m.eval()
    with torch.no_grad():
        f = torch.as_tensor(feat, device=dev); mk = torch.as_tensor(mask, device=dev)
        p = m(f); pm = torch.where(mk > 0.5, p, torch.full_like(p, 1e9))
        return (pm.argmin(1).cpu().numpy() == egIdx).astype(np.int32)

def boot_ci(x, B=20000, seed=0):
    rng = np.random.default_rng(seed); n = len(x)
    means = x[rng.integers(0, n, size=(B, n))].mean(1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

def boot_diff_ci(a, b, B=20000, seed=0):
    rng = np.random.default_rng(seed); n = len(a); idx = rng.integers(0, n, size=(B, n))
    d = a[idx].mean(1) - b[idx].mean(1)
    return (a.mean()-b.mean()), np.percentile(d, 2.5), np.percentile(d, 97.5)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-pack', required=True); ap.add_argument('--full-pack', required=True)
    ap.add_argument('--raw-ckpt', required=True); ap.add_argument('--full-ckpt', required=True)
    ap.add_argument('--nat-src', type=int, default=2); ap.add_argument('--nat-val-frac', type=float, default=0.4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args(); dev = torch.device(args.device)
    rf, rm, reg, rseed, rsrc, rcell = load(args.raw_pack)
    ff, fm, feg, fseed, fsrc, fcell = load(args.full_pack)
    assert np.array_equal(reg, feg) and np.array_equal(rseed, fseed), 'packs not aligned!'
    # held-out natural seeds (same split the trainer used)
    nat = rsrc == args.nat_src; nat_seeds = np.unique(rseed[nat])
    thr = nat_seeds[int(len(nat_seeds) * (1 - args.nat_val_frac))]
    held = nat_seeds[nat_seeds >= thr]; val = nat & np.isin(rseed, held)
    print('held-out NATURAL commits n=%d (%d seeds)' % (int(val.sum()), len(held)))
    c_raw = nn_correct(args.raw_ckpt, rf[val], rm[val], reg[val], dev)
    c_full = nn_correct(args.full_ckpt, ff[val], fm[val], feg[val], dev)
    # analytic baseline = argmin wrap-aware reach-time = full feat col 12 (wa0/100)
    wa = np.where(fm[val] > 0.5, ff[val][:, :, 12], 1e9)
    c_ana = (wa.argmin(1) == feg[val]).astype(np.int32)
    for name, c in (('RAW-NN  (no reach-time fed)', c_raw), ('FULL-NN (reach-time fed) ', c_full), ('ANALYTIC (argmin wa, no NN)', c_ana)):
        m, lo, hi = boot_ci(c); print('  %s : %.4f  [95%% CI %.4f, %.4f]  (n=%d)' % (name, m, lo, hi, len(c)))
    print('LIFTS (bootstrap 95% CI of the paired difference):')
    for name, a, b in (('FULL-NN − ANALYTIC', c_full, c_ana), ('RAW-NN  − ANALYTIC', c_raw, c_ana), ('RAW-NN  − FULL-NN ', c_raw, c_full)):
        d, lo, hi = boot_diff_ci(a, b); sig = 'significant' if (lo > 0 or hi < 0) else 'NOT significant (CI spans 0)'
        print('  %s : %+.4f  [95%% CI %+.4f, %+.4f]  -> %s' % (name, d, lo, hi, sig))

if __name__ == '__main__':
    main()

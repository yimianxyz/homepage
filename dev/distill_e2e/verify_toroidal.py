"""Decisive test of the toroidal-features hypothesis. The setds 'force' label is the production
chase-NN output; in PATROL (no boid within R) it equals seekStep(autoTarget - predPos, predVel),
and that offset is NON-toroidal (buildPredatorFeatures/computeEvolvedTarget use raw position diffs).
But set_obs stored TOROIDAL rel_pos. This script proves the gap on the EXISTING val set:

  A) cos(seekStep(auto-ppos NON-toroidal), force) on patrol -> should be ~1.0 (label is non-tor seek)
  B) cos(seekStep(torus(auto-ppos)),       force) on patrol -> should be ~0.92 (the net's ceiling)
  C) %patrol frames where |auto-ppos| exceeds HALF in some axis (where torus != raw)

  python3 verify_toroidal.py setds_densA_val.pt --device cuda
"""
import argparse, torch
HALF, WORLD, MAXSPEED, MAXFORCE = 840.0, 1680.0, 2.5, 0.05

def fast_mag(x, y):
    ax, ay = x.abs(), y.abs()
    return torch.maximum(ax, ay) * 0.96 + torch.minimum(ax, ay) * 0.398

def seek_step(dx, dy, vx, vy):
    m = fast_mag(dx, dy); nz = m > 0
    dx0 = torch.where(nz, dx * MAXSPEED / (m + 1e-12), torch.zeros_like(dx))
    dy0 = torch.where(nz, dy * MAXSPEED / (m + 1e-12), torch.zeros_like(dy))
    sx, sy = dx0 - vx, dy0 - vy
    sm = fast_mag(sx, sy); over = sm > MAXFORCE
    f = torch.where(over, MAXFORCE / (sm + 1e-12), torch.ones_like(sm))
    return torch.stack([sx * f, sy * f], 1)

def report(name, seek, Y, patrol):
    yn = Y / (Y.norm(dim=1, keepdim=True) + 1e-9)
    pn = seek / (seek.norm(dim=1, keepdim=True) + 1e-9)
    cos = (pn * yn).sum(1).clamp(-1, 1)[patrol]
    print(f"  {name:22s} patrol cos_med={cos.median():.4f} cos_mean={cos.mean():.4f} "
          f"%>.99={(cos>0.99).float().mean()*100:5.1f} %>.999={(cos>0.999).float().mean()*100:5.1f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    d = torch.load(a.inp, map_location='cpu')
    if 'auto' not in d or 'ppos' not in d:
        print("MISSING auto/ppos keys:", list(d.keys())); return
    dev = a.device
    Y = d['force'].float().to(dev)
    P = d['pvel'].float().to(dev)
    auto = d['auto'].float().to(dev); ppos = d['ppos'].float().to(dev)
    D = d['d1']
    pv = P * MAXSPEED if P.abs().max() < 2.0 else P
    off_raw = auto - ppos                                  # NON-toroidal autoTarget offset
    off_tor = (off_raw + HALF) % WORLD - HALF              # toroidal min-image
    chase = (torch.isfinite(D) & (D < 80.0)).to(dev); patrol = ~chase
    print(f"{a.inp}  frames={Y.shape[0]}  patrol={int(patrol.sum())} chase={int(chase.sum())}")
    report("seek(non-toroidal)", seek_step(off_raw[:,0], off_raw[:,1], pv[:,0], pv[:,1]), Y, patrol)
    report("seek(toroidal)",     seek_step(off_tor[:,0], off_tor[:,1], pv[:,0], pv[:,1]), Y, patrol)
    diff = (off_raw - off_tor).abs().max(dim=1).values > 1.0
    pd = diff[patrol].float().mean() * 100
    print(f"  %patrol frames where torus!=raw (wrap matters): {pd:5.1f}")
    # magnitude of autoTarget offset distribution on patrol
    mag = off_raw.norm(dim=1)[patrol]
    print(f"  patrol |auto-ppos| median={mag.median():.1f} p90={mag.quantile(0.9):.1f} "
          f"max={mag.max():.1f}  (>{HALF:.0f} => beyond torus half-image)")

if __name__ == '__main__':
    main()

"""Oracle ceiling for the 'compute target -> seek' approach, on a GRID val set (which stores
auto + ppos). Answers: if a net computed the EXACT E3D patrol target, how well would the
resulting seek force match the production force? And does production patrol force == pure
seek-to-target, or does the chase NN add structure a target-seeker can't reach?

For each frame:
  seek_auto = exact production seekStep(auto - ppos, vel)   [fastMag + fastLimit, matches JS]
Report cosine(seek_auto, production_force) in d1 buckets. If patrol buckets ~1.0 -> the force
IS seek-to-target and the only job is computing the target. If ~0.9 -> there's an irreducible
gap and >99% needs reconstructing the chase NN's patrol behavior too.

  python3 patrol_oracle.py --val dataset_gb9_val.pt --device cuda
"""
import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MAXSPEED, MAXFORCE, HALF = 2.5, 0.05, 840.0


def fast_mag(x, y):
    ax, ay = x.abs(), y.abs()
    return torch.maximum(ax, ay) * 0.96 + torch.minimum(ax, ay) * 0.398


def seek_step(dx, dy, vx, vy):
    m = fast_mag(dx, dy)
    nz = m > 0
    dx0 = torch.where(nz, dx * MAXSPEED / (m + 1e-12), torch.zeros_like(dx))
    dy0 = torch.where(nz, dy * MAXSPEED / (m + 1e-12), torch.zeros_like(dy))
    sx, sy = dx0 - vx, dy0 - vy
    sm = fast_mag(sx, sy)
    over = sm > MAXFORCE
    f = torch.where(over, MAXFORCE / (sm + 1e-12), torch.ones_like(sm))
    return torch.stack([sx * f, sy * f], 1)


def cosrep(pred, tgt, mask, label):
    pn = pred / (pred.norm(dim=1, keepdim=True) + 1e-12)
    tn = tgt / (tgt.norm(dim=1, keepdim=True) + 1e-12)
    cos = (pn * tn).sum(1).clamp(-1, 1)[mask]
    if cos.numel() == 0:
        print(f"  {label:14s} (none)"); return
    ang = torch.rad2deg(torch.arccos(cos))
    print(f"  {label:14s} n={cos.numel():7d} cos_med={cos.median():.4f} cos_mean={cos.mean():.4f} "
          f"%>.99={(cos>0.99).float().mean()*100:5.1f} %>.999={(cos>0.999).float().mean()*100:5.1f} "
          f"ang_med={ang.median():5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    d = torch.load(a.val, map_location=a.device)
    obs, force, d1 = d['obs'].to(a.device), d['force'].to(a.device), d['d1'].to(a.device)
    auto, ppos = d['auto'].to(a.device), d['ppos'].to(a.device)
    vel = obs[:, :2] * MAXSPEED                          # raw_obs stores vel/maxspeed in ch 0,1
    off = auto - ppos
    off = (off + HALF) % (2 * HALF) - HALF               # torus min-image
    seek = seek_step(off[:, 0], off[:, 1], vel[:, 0], vel[:, 1])
    chase = (torch.isfinite(d1) & (d1 < 80.0))
    patrol = ~chase
    print(f"# oracle: seek-to-EXACT-target vs production force   frac_chase={chase.float().mean():.3f}")
    cosrep(seek, force, torch.ones_like(chase, dtype=torch.bool), 'all')
    cosrep(seek, force, patrol, 'patrol(all)')
    cosrep(seek, force, chase, 'chase')
    for lo, hi in [(80, 120), (120, 200), (200, 400), (400, 1e9)]:
        m = torch.isfinite(d1) & (d1 >= lo) & (d1 < hi)
        cosrep(seek, force, m, f'patrol{lo}-{hi if hi<1e9 else "inf"}')


if __name__ == '__main__':
    main()

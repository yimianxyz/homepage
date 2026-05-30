"""Augment a setds dataset with per-boid NEIGHBORHOOD-CENTROID features. E3D's target blends
the density-weighted centroid toward the densest boid's *neighborhood* centroid (nbhd=0.461) --
a two-level pairwise quantity a single gate-pool can't compute at runtime. We compute it ONCE
here (B x N x N pairwise, toroidal) and append it as features, so a cheap deepsets/gate can read
the densest boid's neighborhood centroid directly. This moves the pairwise work into data-gen.

For each frame, each alive boid i: neighbors = alive boids j with toroidal |pos_i-pos_j| < R.
Append 2 cols: (mean_j rel_pos_j - rel_pos_i)/HALF  = vector from boid i to its neighborhood
centroid, normalized. (Zero if no neighbors.) rel_pos is boid-relative-to-predator in feats[:, :, :2]*HALF.

  python3 augment_nbhd.py IN.pt OUT.pt --r 178 --device cuda
"""
import argparse, torch
HALF, WORLD = 840.0, 1680.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--r', type=float, default=178.0)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    d = torch.load(a.inp, map_location='cpu')
    F = d['feats']; M = d['mask']                      # (B,N,Fd) , (B,N)
    B, N, Fd = F.shape
    dev = a.device
    out_cols = torch.empty(B, N, 2)
    bs = 2048
    for s in range(0, B, bs):
        f = F[s:s+bs].to(dev); m = (M[s:s+bs] > 0.5).to(dev)   # (b,N,Fd),(b,N)
        pos = f[..., :2] * HALF                                 # (b,N,2) rel to predator, world units
        off = pos[:, :, None, :] - pos[:, None, :, :]           # (b,N,N,2) i->j
        off = (off + HALF) % WORLD - HALF                       # toroidal min-image
        dist = torch.sqrt((off ** 2).sum(-1) + 1e-9)            # (b,N,N)
        neigh = (dist < a.r) & m[:, None, :] & m[:, :, None]    # j is neighbor of i, both alive
        neigh = neigh & ~torch.eye(N, dtype=torch.bool, device=dev)[None]  # exclude self
        cnt = neigh.sum(-1, keepdim=True).clamp(min=1)          # (b,N,1)
        # mean neighbor position (relative to predator), via mean of (pos_i + off_ij) = pos_i + mean off
        mean_off = (off * neigh[..., None]).sum(2) / cnt        # (b,N,2) vector i->centroid
        has = neigh.any(-1, keepdim=True).float()
        col = (mean_off / HALF) * has                            # normalized, 0 if no neighbors
        out_cols[s:s+bs] = col.cpu()
        del f, m, pos, off, dist, neigh, mean_off, col
    F2 = torch.cat([F, out_cols], dim=-1)
    d2 = dict(d); d2['feats'] = F2
    meta = dict(d.get('meta', {})); meta['nbhd_centroid_r'] = a.r
    d2['meta'] = meta
    torch.save(d2, a.out)
    print(f"{a.inp} -> {a.out}  feats {Fd} -> {F2.shape[-1]}  (r={a.r})  B={B} N={N}")

if __name__ == '__main__':
    main()

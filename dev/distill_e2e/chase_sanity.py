"""Is densA2k chase data learnable? For chase frames, the production force == seek to the
nearest boid. Reconstruct seek-to-nearest from feats and compare to the label force. High
cosine -> data is fine (net just failing); low cosine -> feature/label MISALIGNMENT in the
dataset (data-generation bug), which would explain why every net collapses to cos~0.1 on chase."""
import torch, sys
HALF, BOID_MAX, MAXSPEED, MAXFORCE = 840.0, 6.0, 2.5, 0.05

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

for path in sys.argv[1:]:
    d = torch.load(path, map_location="cpu")
    F = d["feats"].float(); M = d["mask"].float(); P = d["pvel"].float(); Y = d["force"].float(); D = d["d1"]
    chase = (torch.isfinite(D) & (D < 80.0))
    rel = F[..., :2] * HALF                         # rel_pos in world units
    dist = F[..., 4] * HALF
    dist = torch.where(M > 0.5, dist, torch.full_like(dist, 1e9))
    nn_idx = dist.argmin(dim=1)                      # nearest boid per frame
    bi = torch.arange(F.shape[0])
    off = rel[bi, nn_idx]                            # offset to nearest boid
    pv = P * MAXSPEED if P.abs().max() < 2.0 else P  # pvel may be normalized
    seek = seek_step(off[:, 0], off[:, 1], pv[:, 0], pv[:, 1])
    sn = seek / (seek.norm(dim=1, keepdim=True) + 1e-9)
    yn = Y / (Y.norm(dim=1, keepdim=True) + 1e-9)
    cos = (sn * yn).sum(1).clamp(-1, 1)
    print(f"{path}  pvel_absmax={P.abs().max():.3f}")
    print(f"  chase seek-to-nearest vs label: cos_med={cos[chase].median():.4f} "
          f"%>.9={(cos[chase]>0.9).float().mean()*100:.1f} n={int(chase.sum())}")

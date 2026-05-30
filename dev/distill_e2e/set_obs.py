"""Set-observation encoding for the end-to-end predator NN (architectural alt to raw_obs).

Instead of quantizing boids into a predator-centric density histogram (raw_obs, which
loses exact positions and cannot represent each boid's OWN local neighbourhood density),
this hands the net the raw boid SET: every alive boid as a (rel_pos, rel_vel, dist) row,
plus a validity mask. A permutation-invariant set network (set_net.SetNet) can then learn
the density-weighted cluster centroid that drives PATROL directly from exact positions.

Returns, all float32:
  feats : (B, N, 5)  per-boid [rel_pos_x/HALF, rel_pos_y/HALF,
                               rel_vel_x/BOID_MAX, rel_vel_y/BOID_MAX, dist/HALF]
                     torus min-image, predator-centric; zeros for dead boids
  mask  : (B, N) float  1.0 alive / 0.0 dead
  pvel  : (B, 2)  predator velocity / PREDATOR_MAX_SPEED
  d1    : (B,)    nearest-alive torus distance (raw units; inf if none)
"""
import torch

CANVAS_W = 1680.0
CANVAS_H = 1680.0
HALF_W = CANVAS_W / 2.0
HALF_H = CANVAS_H / 2.0
PREDATOR_MAX_SPEED = 2.5
BOID_MAX = 6.0
FEAT_DIM = 5


def _torus_offset(boid_xy, pred_xy, half):
    d = boid_xy - pred_xy
    return (d + half) % (2.0 * half) - half


def _local_density(bp, alive, radii):
    """Per-boid local density = #alive neighbours within each radius (exact, torus).
    bp (B,N,2), alive (B,N) bool, radii list -> (B,N,len(radii)) float, normalised /N.
    This materialises the PAIRWISE quantity production weights patrol by — the thing a
    softmax-attention pool averages away and a histogram only quantises. Given it as an
    input feature, a plain DeepSets can form the density^pow-weighted centroid directly."""
    B, N, _ = bp.shape
    ddx = _torus_offset(bp[:, :, None, 0], bp[:, None, :, 0], HALF_W)   # (B,N,N)
    ddy = _torus_offset(bp[:, :, None, 1], bp[:, None, :, 1], HALF_H)
    pdist = torch.sqrt(ddx * ddx + ddy * ddy)                          # boid-boid torus dist
    am = alive.to(bp.dtype)                                            # (B,N)
    within = (pdist[..., None] < torch.tensor(radii, device=bp.device, dtype=bp.dtype)) \
        .to(bp.dtype) * am[:, None, :, None]                           # (B,N,N,R), mask cols by alive
    dens = within.sum(dim=2) / float(N)                                # (B,N,R) neighbour fraction
    return dens * am[:, :, None]                                       # dead rows -> 0


def set_obs(pred_pos, pred_vel, boid_pos, boid_vel, boid_alive, density_radii=None):
    """pred_* (B,2), boid_* (B,N,2), boid_alive (B,N) bool. -> feats,mask,pvel,d1.
    If density_radii (list of raw-unit radii) is given, appends per-boid local density
    columns to feats so FEAT_DIM = 5 + len(density_radii)."""
    f = torch.float32
    pv = pred_vel.to(f)
    bp = boid_pos.to(f)
    bv = boid_vel.to(f)
    pp = pred_pos.to(f)
    alive = boid_alive

    dx = _torus_offset(bp[..., 0], pp[:, None, 0], HALF_W)        # (B,N)
    dy = _torus_offset(bp[..., 1], pp[:, None, 1], HALF_H)
    dist = torch.sqrt(dx * dx + dy * dy)

    rvx = bv[..., 0] - pv[:, None, 0]
    rvy = bv[..., 1] - pv[:, None, 1]
    m = alive.to(f)
    cols = [dx / HALF_W, dy / HALF_H, rvx / BOID_MAX, rvy / BOID_MAX, dist / HALF_W]
    feats = torch.stack(cols, dim=2) * m.unsqueeze(2)            # (B,N,5), dead->0
    if density_radii:
        dens = _local_density(bp, alive, density_radii)          # (B,N,R)
        feats = torch.cat([feats, dens], dim=2)                  # (B,N,5+R)

    inf = torch.full_like(dist, float('inf'))
    d1 = torch.where(alive, dist, inf).min(dim=1).values
    pvel = pv / PREDATOR_MAX_SPEED
    return feats, m, pvel, d1

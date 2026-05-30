"""Raw-observation encoding for the end-to-end predator NN.

Design (Occam): the predator is translation-equivariant on a toroidal world, so
position is NOT an input — only predator velocity and the *relative* configuration
of boids matter. Two complementary views of "all boids", both predator-centric and
torus-wrapped:

  * coarse DENSITY GRID  (G x G)  -> global cluster structure that drives PATROL
  * K-NEAREST boids (rel pos+vel) -> fine local structure that drives the CHASE

The production policy blends seek-cluster (patrol) and seek-nearest (chase) via an
in-range gate; these two views carry exactly the information each regime needs,
with no hand-built features in between.

obs layout (default G=9, K=8, vel=True -> dim = 2 + 4*8 + 81 + 162 = 277):
  [0:2]            predator velocity / PREDATOR_MAX_SPEED
  [2 : 2+4K]       K nearest alive boids, by torus distance:
                     rel_pos/HALF (2), rel_vel/BOID_MAX (2)  -- zeros if <K alive
  next G*G         soft (bilinear) density of alive boids, torus-centred, /N_BOIDS
  next 2*G*G       (vel only) bilinear MOMENTUM field Sum(w * boid_vel)/ (N*BOID_MAX)
                     -- carries cluster motion for the patrol travel-time LEAD term
"""
import math
import torch

CANVAS_W = 1680.0
CANVAS_H = 1680.0
HALF_W = CANVAS_W / 2.0
HALF_H = CANVAS_H / 2.0
PREDATOR_MAX_SPEED = 2.5
BOID_MAX = 6.0
R_MAX = (HALF_W * HALF_W + HALF_H * HALF_H) ** 0.5     # corner distance
R0 = 20.0                                              # inner log-radius floor


def obs_dim(G=9, K=8, vel=True, polar=None):
    if polar is not None:
        nr, nt = polar
        cells = nr * nt
        return 2 + 4 * K + cells + (2 * cells if vel else 0)
    return 2 + 4 * K + G * G + (2 * G * G if vel else 0)


def _torus_offset(boid_xy, pred_xy, half):
    # min-image relative offset on a torus of size 2*half
    d = boid_xy - pred_xy
    return (d + half) % (2.0 * half) - half


def raw_obs(pred_pos, pred_vel, boid_pos, boid_vel, boid_alive, G=9, K=8, vel=True, polar=None):
    """All float32. Shapes: pred_* (B,2), boid_* (B,N,2), boid_alive (B,N) bool.
    Returns (obs (B, obs_dim), d1 (B,) nearest-alive torus distance).
    polar=(nr,nt): replace the Cartesian density/momentum grid with a predator-centric
    LOG-POLAR histogram (nr log-radial x nt angular bins, angle wraps). Angular bins align
    directly with the output direction, so 'densest angular sector' = patrol direction."""
    B, N, _ = boid_pos.shape
    dev = boid_pos.device
    f = torch.float32

    pv = pred_vel.to(f)
    bp = boid_pos.to(f)
    bv = boid_vel.to(f)
    pp = pred_pos.to(f)
    alive = boid_alive

    dx = _torus_offset(bp[..., 0], pp[:, None, 0], HALF_W)        # (B,N)
    dy = _torus_offset(bp[..., 1], pp[:, None, 1], HALF_H)
    dist = torch.sqrt(dx * dx + dy * dy)
    inf = torch.full_like(dist, float('inf'))
    dist_m = torch.where(alive, dist, inf)

    # ---- self velocity ----
    out = [pv / PREDATOR_MAX_SPEED]                               # (B,2)

    # ---- K nearest alive ----
    k = min(K, N)
    dk, idx = torch.topk(dist_m, k, dim=1, largest=False)         # (B,k)
    present = torch.isfinite(dk)                                  # fewer than k alive -> False
    rdx = torch.gather(dx, 1, idx)
    rdy = torch.gather(dy, 1, idx)
    rvx = torch.gather(bv[..., 0], 1, idx) - pv[:, None, 0]
    rvy = torch.gather(bv[..., 1], 1, idx) - pv[:, None, 1]
    p = present.to(f)
    knn = torch.stack([rdx / HALF_W * p, rdy / HALF_H * p,
                       rvx / BOID_MAX * p, rvy / BOID_MAX * p], dim=2)  # (B,k,4)
    out.append(knn.reshape(B, k * 4))
    if k < K:                                                     # pad if K>N (never for N=120)
        out.append(torch.zeros((B, (K - k) * 4), device=dev, dtype=f))

    # ---- LOG-POLAR density/momentum grid (bilinear; radial clamp, angular wrap) ----
    if polar is not None:
        nr, nt = polar
        aw = alive.to(f)
        bvx, bvy = bv[..., 0], bv[..., 1]
        r = torch.sqrt(dx * dx + dy * dy)                        # (B,N) torus distance
        theta = torch.atan2(dy, dx)                              # (B,N) in [-pi, pi]
        # log-radial cell coord in [-0.5, nr-0.5]
        lr = (torch.log(torch.clamp(r, min=R0)) - math.log(R0)) / (math.log(R_MAX) - math.log(R0))
        rb = lr * nr - 0.5
        tb = (theta + math.pi) / (2.0 * math.pi) * nt - 0.5      # angular coord, wraps mod nt
        r0i = torch.floor(rb); t0i = torch.floor(tb)
        fr = rb - r0i; ft = tb - t0i
        cells = nr * nt
        grid = torch.zeros((B, cells), device=dev, dtype=f)
        mvx = torch.zeros((B, cells), device=dev, dtype=f)
        mvy = torch.zeros((B, cells), device=dev, dtype=f)
        for cr, wr in ((r0i, 1 - fr), (r0i + 1, fr)):
            ri = torch.clamp(cr.long(), 0, nr - 1)               # radial: clamp at edges
            for ct, wt in ((t0i, 1 - ft), (t0i + 1, ft)):
                ti = ct.long() % nt                              # angular: wrap (periodic)
                flat = ri * nt + ti                              # (B,N)
                w = (wr * wt * aw)
                grid.scatter_add_(1, flat, w)
                if vel:
                    mvx.scatter_add_(1, flat, w * bvx)
                    mvy.scatter_add_(1, flat, w * bvy)
        out.append(grid / float(N))
        if vel:
            out.append(mvx / (N * BOID_MAX))
            out.append(mvy / (N * BOID_MAX))
        obs = torch.cat(out, dim=1)
        d1 = torch.where(torch.isfinite(dk[:, 0]), dk[:, 0],
                         torch.full((B,), float('inf'), device=dev, dtype=f))
        return obs, d1

    # ---- coarse soft density grid (bilinear scatter), torus-centred ----
    # cell coords in [0, G-1]; cell width = 2*half / G
    gx = (dx + HALF_W) / (2.0 * HALF_W) * G - 0.5                 # (B,N)
    gy = (dy + HALF_H) / (2.0 * HALF_H) * G - 0.5
    x0 = torch.floor(gx); y0 = torch.floor(gy)
    fx = gx - x0; fy = gy - y0
    grid = torch.zeros((B, G * G), device=dev, dtype=f)
    mvx = torch.zeros((B, G * G), device=dev, dtype=f)
    mvy = torch.zeros((B, G * G), device=dev, dtype=f)
    aw = alive.to(f)
    bvx, bvy = bv[..., 0], bv[..., 1]
    for cx, wx in ((x0, 1 - fx), (x0 + 1, fx)):
        for cy, wy in ((y0, 1 - fy), (y0 + 1, fy)):
            ci = (cx.long() % G)
            cj = (cy.long() % G)
            flat = ci * G + cj                                   # (B,N) wrap grid edges (torus)
            w = (wx * wy * aw)
            grid.scatter_add_(1, flat, w)
            if vel:
                mvx.scatter_add_(1, flat, w * bvx)
                mvy.scatter_add_(1, flat, w * bvy)
    out.append(grid / float(N))
    if vel:
        out.append(mvx / (N * BOID_MAX))
        out.append(mvy / (N * BOID_MAX))

    obs = torch.cat(out, dim=1)
    d1 = torch.where(torch.isfinite(dk[:, 0]), dk[:, 0],
                     torch.full((B,), float('inf'), device=dev, dtype=f))
    return obs, d1

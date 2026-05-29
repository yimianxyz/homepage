"""Egocentric observation for an end-to-end predator policy.

The existing distilled net only sees the K=4 nearest boids + a hand-computed
patrol target (seek_auto_xy). To let a policy discover global behaviors
(find the densest cluster, lead it, herd) WITHOUT a hand-crafted target, we
give it an egocentric polar density+flow grid over ALL boids:

  - A angular sectors x R radial rings around the predator
  - per bin: boid count (density), and mean boid velocity (flow)
  - plus predator's own velocity and size

This is fully vectorized / GPU- and CUDA-graph-friendly (scatter_add over
boids into fixed-size bins; no Python loop, no dynamic shapes).

obs_dim = A*R (count) + A*R (mean vx) + A*R (mean vy) + 2 (pred vel) + 1 (size)
        = 3*A*R + 3
For A=8, R=3 -> 75.
"""
import math
import torch


def build_obs_egocentric(pred_pos, pred_vel, boid_pos, boid_vel, boid_alive,
                         A=8, R=3, ring_edges=(40.0, 120.0, 1e9),
                         vnorm=6.0, dnorm=120.0, device=None):
    """
    pred_pos (B,2), pred_vel (B,2), boid_pos (B,N,2), boid_vel (B,N,2),
    boid_alive (B,N) bool. Returns obs (B, 3*A*R+3) float32.

    ring_edges: upper distance bound of each of the R rings (last = inf).
    Counts are normalized by N; velocities by vnorm; pred vel by vnorm.
    """
    B, N, _ = boid_pos.shape
    dev = device or boid_pos.device
    dt = torch.float32

    dx = (boid_pos[..., 0] - pred_pos[:, None, 0])   # (B,N)
    dy = (boid_pos[..., 1] - pred_pos[:, None, 1])
    dist = torch.sqrt(dx * dx + dy * dy + 1e-9)
    ang = torch.atan2(dy, dx)                         # [-pi, pi]

    # angular bin in [0, A)
    abin = torch.floor((ang + math.pi) / (2 * math.pi) * A).long()
    abin = torch.clamp(abin, 0, A - 1)
    # radial bin in [0, R): first ring with dist < edge
    rbin = torch.zeros_like(abin)
    edges = list(ring_edges)
    for r in range(R - 1):
        rbin = rbin + (dist >= edges[r]).long()
    rbin = torch.clamp(rbin, 0, R - 1)

    bin_idx = (rbin * A + abin)                       # (B,N) in [0, A*R)
    AR = A * R
    alive_f = boid_alive.to(dt)

    counts = torch.zeros((B, AR), dtype=dt, device=dev)
    counts.scatter_add_(1, bin_idx, alive_f)

    svx = torch.zeros((B, AR), dtype=dt, device=dev)
    svy = torch.zeros((B, AR), dtype=dt, device=dev)
    svx.scatter_add_(1, bin_idx, (boid_vel[..., 0] * alive_f).to(dt))
    svy.scatter_add_(1, bin_idx, (boid_vel[..., 1] * alive_f).to(dt))
    denom = torch.clamp(counts, min=1.0)
    mvx = svx / denom
    mvy = svy / denom

    counts_n = counts / float(N)
    mvx_n = mvx / vnorm
    mvy_n = mvy / vnorm
    pv = pred_vel.to(dt) / vnorm                      # (B,2)
    size = (torch.zeros((B, 1), dtype=dt, device=dev))  # placeholder; caller can fill

    obs = torch.cat([counts_n, mvx_n, mvy_n, pv, size], dim=1)
    return obs


AUG_EXTRA = 8  # seek(3) + nearest boid(5)


def build_obs_augmented(pred_pos, pred_vel, boid_pos, boid_vel, boid_alive,
                        seek_xy, A=8, R=3, ring_edges=(40.0, 120.0, 1e9),
                        vnorm=6.0, dnorm=120.0, seeknorm=300.0, cooldown=None):
    """Egocentric grid + injected hand-crafted signals so PPO starts from the
    known-good nearest_cluster behavior and learns to improve on it:
      - seek vector to the patrol target (unit dir 2 + norm dist 1)
      - nearest live boid (unit dir 2 + norm dist 1 + its velocity 2)
      - (optional) feed-cooldown remaining fraction (1) — info the deployed
        policy ignores: 1.0 just after a catch (cannot eat), 0.0 when ready.
    Returns (B, 75 + 8 [+1 if cooldown]) — 83 (or 84) for A=8,R=3.
    """
    base = build_obs_egocentric(pred_pos, pred_vel, boid_pos, boid_vel,
                                boid_alive, A, R, ring_edges, vnorm, dnorm)
    dt = base.dtype
    # seek vector to patrol target
    sdx = (seek_xy[:, 0] - pred_pos[:, 0]).to(dt)
    sdy = (seek_xy[:, 1] - pred_pos[:, 1]).to(dt)
    sdist = torch.sqrt(sdx * sdx + sdy * sdy + 1e-9)
    seek = torch.stack([sdx / sdist, sdy / sdist,
                        torch.clamp(sdist / seeknorm, 0.0, 4.0)], dim=1)
    # nearest live boid
    bdx = (boid_pos[..., 0] - pred_pos[:, None, 0])
    bdy = (boid_pos[..., 1] - pred_pos[:, None, 1])
    bd = torch.sqrt(bdx * bdx + bdy * bdy + 1e-9)
    bd = torch.where(boid_alive, bd, torch.full_like(bd, 1e18))
    ni = bd.argmin(dim=1)                          # (B,)
    r = torch.arange(pred_pos.shape[0], device=pred_pos.device)
    ndx = bdx[r, ni]; ndy = bdy[r, ni]; ndist = bd[r, ni].clamp_max(1e9)
    nd_safe = ndist.clamp_min(1e-6)
    nvx = boid_vel[r, ni, 0]; nvy = boid_vel[r, ni, 1]
    nearest = torch.stack([(ndx / nd_safe).to(dt), (ndy / nd_safe).to(dt),
                           torch.clamp(ndist / dnorm, 0.0, 4.0).to(dt),
                           (nvx / vnorm).to(dt), (nvy / vnorm).to(dt)], dim=1)
    parts = [base, seek, nearest]
    if cooldown is not None:
        parts.append(cooldown.to(dt).view(-1, 1))
    return torch.cat(parts, dim=1)


if __name__ == '__main__':
    # CPU shape/sanity test
    torch.manual_seed(0)
    B, N = 4, 120
    pp = torch.rand(B, 2) * 1000
    pv = (torch.rand(B, 2) - 0.5) * 5
    bp = torch.rand(B, N, 2) * 1000
    bv = (torch.rand(B, N, 2) - 0.5) * 10
    ba = torch.rand(B, N) > 0.1
    obs = build_obs_egocentric(pp, pv, bp, bv, ba)
    A, R = 8, 3
    print('obs shape:', tuple(obs.shape), 'expected', (B, 3 * A * R + 3))
    print('counts sum per env (should ~= alive frac):',
          obs[:, :A * R].sum(1).tolist(), 'alive:', ba.float().mean(1).mul(N / N).tolist())
    print('finite:', bool(torch.isfinite(obs).all()))
    print('OK')

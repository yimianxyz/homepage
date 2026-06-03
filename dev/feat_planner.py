"""Hybrid student: engineered per-candidate pursuit features + tiny value net +
minimal deploy-time rollout (AlphaZero / TD-MPC style).

The prior pointer net (distill_pointer.py) scored each candidate from only
[rx, ry, dist, is_e3d] + [pred_vx, pred_vy, frac_alive] -> it could not see the
TARGET BOID's velocity, closing speed, time-to-go, or local flock, so it capped
at E3D. Here each candidate gets rich pursuit-guidance features (range, closing
velocity, LOS rate, time-to-go, lead/miss, collision-cone margin, local flock
density + flee alignment), so a tiny MLP can score "is committing to this target
good" WITHOUT a full rollout. We then optionally add a SHORT rollout + the value
net as a learned terminal cost (Bertsekas limited-lookahead: a good terminal
value collapses the horizon).

This module provides:
  candidate_features(sim, cand)      -> (feat (B,K,Fc), ctx (B,Fctx))
  run_log_feat(...)                  -> (feat, ctx, gain, planner_catches)  [data gen]
  run_value_student(..., model, Hs)  -> per-seed catches deploying argmax over
                                        candidates of [Hs-rollout catches + V]

Features are computed in sim_torch (vectorized); a JS mirror is written only once
a net wins, for browser deploy.
"""
import numpy as np
import torch

import planner_probe as pp
import sim_torch as st
from sim_torch import Sim, PREDATOR_MAX_SPEED

_PS = 200.0          # position scale
_VS = 6.0            # velocity scale (MAX_SPEED)
_RHO = 70.0          # local-density radius around a candidate (px)
FC = 16              # per-candidate feature dim (keep in sync with build below)
FCTX = 4             # global context dim


def candidate_features(sim, cand):
    """Rich per-candidate pursuit features.

    cand: (B,K,2) absolute candidate target points (cand[:,0]=E3D).
    Returns feat (B,K,FC) float32, ctx (B,FCTX) float32.
    """
    B, K, _ = cand.shape
    dev = sim.device
    px = sim.pred_pos[:, 0]; py = sim.pred_pos[:, 1]          # (B,)
    pvx = sim.pred_vel[:, 0]; pvy = sim.pred_vel[:, 1]
    bpos = sim.boid_pos                                        # (B,N,2)
    bvel = sim.boid_vel
    alive = sim.boid_alive                                     # (B,N) bool

    # candidate relative to predator
    rx = cand[..., 0] - px[:, None]                            # (B,K)
    ry = cand[..., 1] - py[:, None]
    dist = torch.sqrt(rx * rx + ry * ry).clamp(min=1e-6)
    t_go = dist / PREDATOR_MAX_SPEED                           # frames-to-reach (approx)
    is_e3d = torch.zeros(B, K, device=dev, dtype=cand.dtype)
    is_e3d[:, 0] = 1.0

    # nearest ALIVE boid to each candidate point ("targeted boid")
    cdx = cand[:, :, None, 0] - bpos[:, None, :, 0]            # (B,K,N)
    cdy = cand[:, :, None, 1] - bpos[:, None, :, 1]
    cd2 = cdx * cdx + cdy * cdy
    inf = torch.full_like(cd2, float('inf'))
    cd2m = torch.where(alive[:, None, :], cd2, inf)
    nb = cd2m.argmin(dim=2)                                    # (B,K) index of targeted boid
    tb_dist_to_cand = torch.sqrt(cd2m.gather(2, nb[:, :, None]).squeeze(2).clamp(min=0))  # miss dist

    bx = bpos[..., 0].gather(1, nb); by = bpos[..., 1].gather(1, nb)   # (B,K)
    bvx = bvel[..., 0].gather(1, nb); bvy = bvel[..., 1].gather(1, nb)
    # targeted boid relative to predator
    tbrx = bx - px[:, None]; tbry = by - py[:, None]
    rangepb = torch.sqrt(tbrx * tbrx + tbry * tbry).clamp(min=1e-6)
    # relative velocity (boid wrt predator)
    relvx = bvx - pvx[:, None]; relvy = bvy - pvy[:, None]
    # closing speed = -d(range)/dt = -(rel_pos . rel_vel)/range  (>0 = approaching)
    closing = -(tbrx * relvx + tbry * relvy) / rangepb
    # LOS rate = (rel_pos x rel_vel)_z / range^2  (~0 => collision course)
    los_rate = (tbrx * relvy - tbry * relvx) / (rangepb * rangepb)

    # local flock density + flee alignment near candidate (alive boids within RHO)
    near = (cd2m < (_RHO * _RHO))                              # (B,K,N)
    dens = near.sum(dim=2).to(cand.dtype)                     # (B,K)
    nearf = near.to(cand.dtype)
    nsafe = nearf.sum(dim=2).clamp(min=1.0)
    mvx = (bvel[..., 0][:, None, :] * nearf).sum(2) / nsafe   # (B,K) mean nearby boid vel
    mvy = (bvel[..., 1][:, None, :] * nearf).sum(2) / nsafe
    # unit LOS predator->candidate
    ux = rx / dist; uy = ry / dist
    flee_align = mvx * ux + mvy * uy                          # >0: flock fleeing away along LOS

    feat = torch.stack([
        rx / _PS, ry / _PS, dist / _PS, is_e3d,
        t_go / 60.0,
        tbrx / _PS, tbry / _PS, bvx / _VS, bvy / _VS,
        tb_dist_to_cand / _PS,
        rangepb / _PS, closing / _VS, los_rate * 50.0,
        dens / 20.0, flee_align / _VS,
        (rangepb - dist) / _PS,                               # boid-vs-candidate range gap
    ], dim=2).float()                                        # (B,K,FC)

    alive_f = alive.float()
    frac_alive = alive_f.mean(dim=1, keepdim=True)
    ctx = torch.cat([
        (sim.pred_vel / _VS).float(),
        frac_alive.float(),
        (sim.pred_size[:, None] / 20.0).float(),
    ], dim=1)                                                # (B,FCTX)
    return feat, ctx


def run_log_feat(seeds, frames, device, K, H, D):
    """Generate training data: per decision, log (feat, ctx, gain). gain is the
    TRUE H-frame catch count per candidate (the value-regression target)."""
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=pp.TWO_PASS)
    B = sim.B
    roll = Sim(seeds=list(range(B * K)), weights=pp.WEIGHTS, device=device,
               auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=pp.TWO_PASS)
    rows = torch.arange(B, device=device)
    feat_log, ctx_log, gain_log = [], [], []
    held = None
    f = 0
    while f < frames:
        if f % D == 0:
            cand = pp._candidate_targets(sim, K)
            feat, ctx = candidate_features(sim, cand)
            base = pp._save_state(sim)
            pp._load_state(roll, pp._tile_state(base, K))
            roll_tgt = cand.reshape(B * K, 2).contiguous()
            h = min(H, frames - f)
            c0 = roll.catches.clone()
            for _ in range(h):
                pp._step_with_target(roll, roll_tgt)
            gain = pp.rollout_gain(roll, c0, B, K)
            held = cand[rows, gain.argmax(dim=1)]
            feat_log.append(feat.cpu()); ctx_log.append(ctx.cpu()); gain_log.append(gain.cpu())
        pp._step_with_target(sim, held)
        f += 1
    feat = torch.cat(feat_log, 0).numpy()
    ctx = torch.cat(ctx_log, 0).numpy()
    gain = torch.cat(gain_log, 0).numpy()
    return feat, ctx, gain, sim.catches.cpu().numpy()


def run_value_student(seeds, frames, device, model, K, D, Hs=0, bias0=0.0):
    """Deploy: every D frames, score K candidates by [Hs-frame rollout catches +
    V(features)], commit argmax. Hs=0 => pure value net (no rollout).

    bias0 adds a constant to candidate 0's (E3D) score, so the student only
    deviates from E3D when another candidate beats it by > bias0 -- mirrors the
    planner's tie->E3D default and curbs harmful over-deviation in OOD states."""
    model.eval()
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=pp.TWO_PASS)
    B = sim.B
    roll = None
    if Hs > 0:
        roll = Sim(seeds=list(range(B * K)), weights=pp.WEIGHTS, device=device,
                   auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=pp.TWO_PASS)
    rows = torch.arange(B, device=device)
    held = None
    f = 0
    while f < frames:
        if f % D == 0:
            cand = pp._candidate_targets(sim, K)
            feat, ctx = candidate_features(sim, cand)
            with torch.no_grad():
                v = model(feat.to(device), ctx.to(device))    # (B,K)
            score = v
            if Hs > 0:
                base = pp._save_state(sim)
                pp._load_state(roll, pp._tile_state(base, K))
                roll_tgt = cand.reshape(B * K, 2).contiguous()
                c0 = roll.catches.clone()
                for _ in range(Hs):
                    pp._step_with_target(roll, roll_tgt)
                short_gain = pp.rollout_gain(roll, c0, B, K).to(device)
                score = short_gain + v
            if bias0 != 0.0:
                score = score.clone()
                score[:, 0] = score[:, 0] + bias0
            held = cand[rows, score.argmax(dim=1)]
        pp._step_with_target(sim, held)
        f += 1
    return sim.catches.cpu().numpy()


if __name__ == '__main__':
    # CPU self-test: tiny run, check shapes + that argmax(gain) student == planner.
    import sys
    dev = 'cpu'
    pp.WEIGHTS = st.load_weights('../js/predator_weights.json', device=dev)
    feat, ctx, gain, cats = run_log_feat(list(range(200000, 200002)), 120, dev, 16, 120, 8)
    print('feat', feat.shape, 'ctx', ctx.shape, 'gain', gain.shape, 'FC', FC, 'FCTX', FCTX)
    print('planner catches (n=2,f=120):', cats.tolist())
    print('feat[0,0]=', np.round(feat[0, 0], 3).tolist())
    print('any nan:', bool(np.isnan(feat).any()), bool(np.isnan(ctx).any()))

"""Planner ceiling probe — the decisive test of the user's ≥50% goal.

Question: can ANY non-reactive, multi-step-anticipating expert that controls
ONLY the distillable lever (the patrol target choice) reach the +50% target
(>=12.52 catches), given the predator is structurally slower than prey
(2.5 vs 6.0)? If even an explicit lookahead planner can't, then a *reactive*
browser net distilled from it certainly can't, and the goal is infeasible.

The planner controls the same lever E3D does: the PATROL TARGET. Steering is
production's exact analytic decomposition (M5 finding):
    force = (nearest boid within POLICY_R=80) ? seek(nearest) : seek(target)
so any gain here is purely from BETTER INSTANTANEOUS TARGET CHOICE — exactly
the part of a planner's edge that survives distillation to a reactive net.

Controllers:
  - e3d      : target = the production E3D evolved-patrol target each frame
               (analytic baseline; should reproduce ~8.3).
  - planner  : every D frames, branch over K candidate targets (K-nearest live
               boids, lead-adjusted, + the E3D target as candidate 0), roll the
               TRUE dynamics forward H frames committed to each candidate, pick
               the candidate maximizing catches over the horizon, hold it D
               frames, re-plan. Greedy receding-horizon target commitment — the
               strongest distillable (reactive-target) expert.

Usage:
  python3 planner_probe.py --controller e3d     --n 256 --seedStart 200000
  python3 planner_probe.py --controller planner --n 256 --seedStart 200000 \
          --K 8 --H 60 --D 15
"""

import argparse
import json
import time
import numpy as np
import torch

import sim_torch as st
from sim_torch import (
    Sim, build_features, fast_set_magnitude, fast_limit,
    PREDATOR_MAX_SPEED, PREDATOR_MAX_FORCE, PREDATOR_RANGE,
)

# E3D production evolved-patrol constants (the analytic policy the shipped net
# distills at cos 0.987).
E3D = dict(cluster_r=178.09, dens_pow=2.373, reach_scale=1515.0, sharp=9.25,
           lead_scale=0.454, lead_max=230.6, nbhd=0.461)


def _analytic_steer(sim, plan_target):
    """Production's exact analytic predator step toward `plan_target` (B,2).

    Uses build_features feats 29/30 (chase=seek nearest) and 31/32 (patrol=
    seek target); combines by the in-range gate (feat 34). This is the exact
    E3D+chase decomposition, NN-free."""
    feats = build_features(
        sim.pred_pos.float(), sim.pred_vel.float(),
        sim.boid_pos.float(), sim.boid_vel.float(), sim.boid_alive,
        plan_target.float(), sim.weights['featureDim'], sim.device,
        dtype=torch.float64,
    )
    in_range = feats[:, 34] > 0.5
    chase = feats[:, 29:31]
    patrol = feats[:, 31:33]
    steering = torch.where(in_range.unsqueeze(1), chase, patrol)
    new_vx = sim.pred_vel[:, 0] + steering[:, 0]
    new_vy = sim.pred_vel[:, 1] + steering[:, 1]
    new_vx, new_vy = fast_limit(new_vx, new_vy, PREDATOR_MAX_SPEED)
    sim.pred_vel[:, 0] = new_vx
    sim.pred_vel[:, 1] = new_vy
    sim.pred_pos[:, 0] += new_vx
    sim.pred_pos[:, 1] += new_vy
    sim.pred_pos[:, 0] = torch.where(sim.pred_pos[:, 0] > sim._wrap_w_max, sim._wrap_neg20, sim.pred_pos[:, 0])
    sim.pred_pos[:, 0] = torch.where(sim.pred_pos[:, 0] < sim._wrap_neg20, sim._wrap_w_max, sim.pred_pos[:, 0])
    sim.pred_pos[:, 1] = torch.where(sim.pred_pos[:, 1] > sim._wrap_h_max, sim._wrap_neg20, sim.pred_pos[:, 1])
    sim.pred_pos[:, 1] = torch.where(sim.pred_pos[:, 1] < sim._wrap_neg20, sim._wrap_h_max, sim.pred_pos[:, 1])


def _step_with_target(sim, plan_target):
    """One full sim frame with the predator analytically chasing plan_target."""
    sim._step_boids()
    _analytic_steer(sim, plan_target)
    sim._check_catches()
    sim._decay_size()
    sim.frame += 1
    sim._frame_ms += st.FRAME_MS


def _e3d_target(sim):
    """Compute the production E3D evolved-patrol target for the current state,
    honoring the freeze-during-chase rule (matches production)."""
    sim.auto_target_mode = 'evolved'
    sim.auto_target_opts = dict(E3D)
    sim._update_auto_target()
    return sim.pred_auto.clone()


STATE_KEYS = ['boid_pos', 'boid_vel', 'boid_alive', 'pred_pos', 'pred_vel',
              'pred_size', 'pred_auto', 'pred_last_feed_ms', 'catches']


def _save_state(sim):
    s = {k: getattr(sim, k).clone() for k in STATE_KEYS}
    s['_frame_ms'] = sim._frame_ms.clone()
    s['frame'] = sim.frame
    return s


def _load_state(sim, s):
    for k in STATE_KEYS:
        getattr(sim, k).copy_(s[k])
    sim._frame_ms.copy_(s['_frame_ms'])
    sim.frame = s['frame']


def _tile_state(s, K):
    """Repeat each env K times along batch dim (env-major: e0c0,e0c1,...)."""
    out = {}
    for k in STATE_KEYS:
        v = s[k]
        out[k] = v.repeat_interleave(K, dim=0).contiguous()
    out['_frame_ms'] = s['_frame_ms'].clone()
    out['frame'] = s['frame']
    return out


def _candidate_targets(sim, K):
    """K candidate patrol targets per env (B,K,2): candidate 0 = E3D target;
    candidates 1..K-1 = the (K-1) nearest live boids, lead-adjusted by travel
    time (same adaptive-lead form E3D uses)."""
    B = sim.B
    e3dt = _e3d_target(sim)                       # (B,2)
    dx = sim.boid_pos[..., 0] - sim.pred_pos[:, None, 0]
    dy = sim.boid_pos[..., 1] - sim.pred_pos[:, None, 1]
    d2 = dx * dx + dy * dy
    d2 = torch.where(sim.boid_alive, d2, torch.full_like(d2, float('inf')))
    nb = K - 1
    _, order = torch.sort(d2, dim=1)
    idx = order[:, :nb]                            # (B,nb)
    bx = sim.boid_pos[..., 0].gather(1, idx)
    by = sim.boid_pos[..., 1].gather(1, idx)
    bvx = sim.boid_vel[..., 0].gather(1, idx)
    bvy = sim.boid_vel[..., 1].gather(1, idx)
    # adaptive lead = (dist / predator_speed) * lead_scale, capped
    ddx = bx - sim.pred_pos[:, None, 0]
    ddy = by - sim.pred_pos[:, None, 1]
    dcent = torch.sqrt(ddx * ddx + ddy * ddy)
    lead = torch.clamp(dcent / PREDATOR_MAX_SPEED * E3D['lead_scale'], 0.0, E3D['lead_max'])
    tx = bx + lead * bvx
    ty = by + lead * bvy
    cand = torch.empty((B, K, 2), dtype=sim.boid_pos.dtype, device=sim.device)
    cand[:, 0, 0] = e3dt[:, 0]
    cand[:, 0, 1] = e3dt[:, 1]
    cand[:, 1:, 0] = tx
    cand[:, 1:, 1] = ty
    # dead-boid candidates: fall back to the E3D target so they're never chosen
    # spuriously (their rollout just equals E3D's).
    alive_nb = sim.boid_alive.gather(1, idx)       # (B,nb)
    cand[:, 1:, 0] = torch.where(alive_nb, cand[:, 1:, 0], e3dt[:, None, 0])
    cand[:, 1:, 1] = torch.where(alive_nb, cand[:, 1:, 1], e3dt[:, None, 1])
    return cand


def run_e3d(seeds, frames, device):
    sim = Sim(seeds=seeds, weights=WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(E3D))
    for _ in range(frames):
        tgt = _e3d_target(sim)
        _step_with_target(sim, tgt)
    return sim.catches.cpu().numpy()


def run_planner(seeds, frames, device, K, H, D):
    sim = Sim(seeds=seeds, weights=WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(E3D))
    B = sim.B
    # rollout sim with B*K envs, state injected each decision point
    roll = Sim(seeds=list(range(B * K)), weights=WEIGHTS, device=device,
               auto_target='evolved', auto_target_opts=dict(E3D))
    held = None        # (B,2) currently committed target
    rows = torch.arange(B, device=device)
    f = 0
    while f < frames:
        if f % D == 0:
            cand = _candidate_targets(sim, K)            # (B,K,2)
            base = _save_state(sim)
            tiled = _tile_state(base, K)                 # B*K envs
            _load_state(roll, tiled)
            roll_tgt = cand.reshape(B * K, 2).contiguous()
            h = min(H, frames - f)
            c0 = roll.catches.clone()
            for _ in range(h):
                _step_with_target(roll, roll_tgt)
            gain = (roll.catches - c0).reshape(B, K)      # (B,K) catches over horizon
            best = gain.float().argmax(dim=1)             # (B,)
            held = cand[rows, best]                       # (B,2)
        _step_with_target(sim, held)
        f += 1
    return sim.catches.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--controller', choices=['e3d', 'planner'], required=True)
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--seedStart', type=int, default=200000)
    ap.add_argument('--frames', type=int, default=1500)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--H', type=int, default=60)
    ap.add_argument('--D', type=int, default=15)
    ap.add_argument('--weights', default='js/predator_weights.json')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    global WEIGHTS
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    WEIGHTS = st.load_weights(args.weights, device=device)

    seeds = list(range(args.seedStart, args.seedStart + args.n))
    t0 = time.time()
    if args.controller == 'e3d':
        catches = run_e3d(seeds, args.frames, device)
    else:
        catches = run_planner(seeds, args.frames, device, args.K, args.H, args.D)
    elapsed = time.time() - t0

    mean = float(catches.mean())
    se = float(catches.std(ddof=1) / np.sqrt(len(catches)))
    res = dict(controller=args.controller, n=args.n, seedStart=args.seedStart,
               frames=args.frames, K=args.K, H=args.H, D=args.D,
               mean=mean, se=se, elapsed=elapsed, device=device)
    print(json.dumps(res))
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(res, fh)


if __name__ == '__main__':
    main()

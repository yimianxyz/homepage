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

# Dense-gain tie-breaker. The integer catch-count over the horizon is ~86%
# all-tie (most candidates yield the same #catches over H), so argmax defaults
# to index-0 (E3D) on those frames and the supervised target is a chaotic
# arbitrary pick that no reactive net can fit. Adding a small continuation
# bonus in [0, DENSE_LAMBDA] (predator->nearest-live-boid proximity at horizon
# end) makes the per-candidate value strictly tie-free and smoothly rankable:
# a real 1-catch advantage (=1.0) always dominates the bonus (<1), while among
# integer-tied candidates the better-positioned one wins. This both (a) makes
# the teacher distillable and (b) may RAISE the ceiling, since tied frames now
# pick a forward-positioned target instead of defaulting to baseline E3D.
DENSE_LAMBDA = 0.0
# TWO_PASS: when True, every Sim built here (closed-loop + rollout) runs the
# live browser's two-flock-pass dynamics (see Sim._step_boids_twopass). The
# planner thus plans AND acts under the same two-pass world. Default False =
# single-pass (matches Oracle + ds1024_dense08.pt teacher data).
TWO_PASS = False
_PS = 200.0


def continuation_bonus(roll):
    """(B*K,) in [0,1]: 1 when predator is touching a live boid, ->0 at >= _PS px."""
    dx = roll.boid_pos[..., 0] - roll.pred_pos[:, None, 0]
    dy = roll.boid_pos[..., 1] - roll.pred_pos[:, None, 1]
    d = torch.sqrt(dx * dx + dy * dy)
    d = torch.where(roll.boid_alive, d, torch.full_like(d, float('inf')))
    mind = d.min(dim=1).values
    return (1.0 - (mind / _PS).clamp(0.0, 1.0))


def rollout_gain(roll, c0, B, K):
    """Per-candidate teacher value (B,K). Integer catches over the horizon plus,
    when DENSE_LAMBDA>0, a continuous proximity tie-breaker."""
    gain = (roll.catches - c0).reshape(B, K).float()
    if DENSE_LAMBDA > 0.0:
        gain = gain + DENSE_LAMBDA * continuation_bonus(roll).reshape(B, K)
    return gain


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


def planner_obs(sim, M, e3d_rel):
    """Target-free, predator-relative observation for the distill net (B,F).

    NON-toroidal rel coords (matches js/policy_features.js). Layout:
      [pred_vx, pred_vy,                       # 2
       e3d_target_rel_x, e3d_target_rel_y,     # 2  (production already computes this)
       (rel_x, rel_y, rel_vx, rel_vy) * M,     # 4M nearest live boids (dead -> 0)
       frac_alive,                             # 1
       cent_rel_x, cent_rel_y]                 # 2  alive centroid rel
    Positions scaled by 1/200, velocities by 1/6 (MAX_SPEED)."""
    B = sim.B
    PS, VS = 200.0, 6.0
    dx = sim.boid_pos[..., 0] - sim.pred_pos[:, None, 0]
    dy = sim.boid_pos[..., 1] - sim.pred_pos[:, None, 1]
    d2 = dx * dx + dy * dy
    d2m = torch.where(sim.boid_alive, d2, torch.full_like(d2, float('inf')))
    _, order = torch.sort(d2m, dim=1)
    idx = order[:, :M]
    rx = dx.gather(1, idx); ry = dy.gather(1, idx)
    rvx = sim.boid_vel[..., 0].gather(1, idx)
    rvy = sim.boid_vel[..., 1].gather(1, idx)
    al = sim.boid_alive.gather(1, idx).double()
    rx = rx * al; ry = ry * al; rvx = rvx * al; rvy = rvy * al
    alive_f = sim.boid_alive.double()
    n_alive = alive_f.sum(dim=1, keepdim=True)
    n_safe = torch.where(n_alive > 0, n_alive, torch.ones_like(n_alive))
    cx = (sim.boid_pos[..., 0] * alive_f).sum(dim=1, keepdim=True) / n_safe
    cy = (sim.boid_pos[..., 1] * alive_f).sum(dim=1, keepdim=True) / n_safe
    cent_rx = (cx[:, 0] - sim.pred_pos[:, 0]).unsqueeze(1)
    cent_ry = (cy[:, 0] - sim.pred_pos[:, 1]).unsqueeze(1)
    boid_block = torch.stack([rx / PS, ry / PS, rvx / VS, rvy / VS], dim=2).reshape(B, 4 * M)
    feat = torch.cat([
        sim.pred_vel / VS,
        e3d_rel / PS,
        boid_block,
        (n_alive / sim.N),
        cent_rx / PS, cent_ry / PS,
    ], dim=1)
    return feat


def run_planner_log(seeds, frames, device, K, H, D, M):
    """Run the planner and log (obs, target_rel) every frame for distillation."""
    sim = Sim(seeds=seeds, weights=WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(E3D), two_pass=TWO_PASS)
    B = sim.B
    roll = Sim(seeds=list(range(B * K)), weights=WEIGHTS, device=device,
               auto_target='evolved', auto_target_opts=dict(E3D), two_pass=TWO_PASS)
    held = None
    rows = torch.arange(B, device=device)
    obs_log = []
    tgt_log = []
    f = 0
    while f < frames:
        if f % D == 0:
            cand = _candidate_targets(sim, K)
            base = _save_state(sim)
            tiled = _tile_state(base, K)
            _load_state(roll, tiled)
            roll_tgt = cand.reshape(B * K, 2).contiguous()
            h = min(H, frames - f)
            c0 = roll.catches.clone()
            for _ in range(h):
                _step_with_target(roll, roll_tgt)
            gain = rollout_gain(roll, c0, B, K)
            best = gain.argmax(dim=1)
            held = cand[rows, best]
        # log obs (e3d target as feature) + chosen target (predator-relative)
        e3d_rel = _e3d_target(sim) - sim.pred_pos
        ob = planner_obs(sim, M, e3d_rel)
        tgt_rel = held - sim.pred_pos
        obs_log.append(ob.float().cpu())
        tgt_log.append(tgt_rel.float().cpu())
        _step_with_target(sim, held)
        f += 1
    obs = torch.cat(obs_log, dim=0).numpy()
    tgt = torch.cat(tgt_log, dim=0).numpy()
    return obs, tgt, sim.catches.cpu().numpy()


def run_planner_log_cand(seeds, frames, device, K, H, D, M):
    """Run the planner, logging PER DECISION: (obs, candidate offsets, gains).

    For the pointer-net distillation: the net scores the SAME K candidates the
    planner ranks and picks argmax, so we log candidate offsets (predator-rel)
    and the rollout gain per candidate (the value to regress / argmax over)."""
    sim = Sim(seeds=seeds, weights=WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(E3D), two_pass=TWO_PASS)
    B = sim.B
    roll = Sim(seeds=list(range(B * K)), weights=WEIGHTS, device=device,
               auto_target='evolved', auto_target_opts=dict(E3D), two_pass=TWO_PASS)
    held = None
    rows = torch.arange(B, device=device)
    obs_log, cand_log, gain_log = [], [], []
    f = 0
    while f < frames:
        if f % D == 0:
            cand = _candidate_targets(sim, K)                  # (B,K,2) absolute
            e3d_rel = cand[:, 0] - sim.pred_pos                 # E3D target rel
            ob = planner_obs(sim, M, e3d_rel)                  # (B,F)
            cand_rel = cand - sim.pred_pos[:, None, :]          # (B,K,2)
            base = _save_state(sim)
            tiled = _tile_state(base, K)
            _load_state(roll, tiled)
            roll_tgt = cand.reshape(B * K, 2).contiguous()
            h = min(H, frames - f)
            c0 = roll.catches.clone()
            for _ in range(h):
                _step_with_target(roll, roll_tgt)
            gain = rollout_gain(roll, c0, B, K)    # (B,K)
            best = gain.argmax(dim=1)
            held = cand[rows, best]
            obs_log.append(ob.float().cpu())
            cand_log.append(cand_rel.float().cpu())
            gain_log.append(gain.cpu())
        _step_with_target(sim, held)
        f += 1
    obs = torch.cat(obs_log, dim=0).numpy()
    cand = torch.cat(cand_log, dim=0).numpy()
    gain = torch.cat(gain_log, dim=0).numpy()
    return obs, cand, gain, sim.catches.cpu().numpy()


def run_e3d(seeds, frames, device):
    sim = Sim(seeds=seeds, weights=WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(E3D), two_pass=TWO_PASS)
    for _ in range(frames):
        tgt = _e3d_target(sim)
        _step_with_target(sim, tgt)
    return sim.catches.cpu().numpy()


def run_planner(seeds, frames, device, K, H, D):
    sim = Sim(seeds=seeds, weights=WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(E3D), two_pass=TWO_PASS)
    B = sim.B
    # rollout sim with B*K envs, state injected each decision point
    roll = Sim(seeds=list(range(B * K)), weights=WEIGHTS, device=device,
               auto_target='evolved', auto_target_opts=dict(E3D), two_pass=TWO_PASS)
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
            gain = rollout_gain(roll, c0, B, K)      # (B,K) catches (+dense bonus)
            best = gain.argmax(dim=1)                     # (B,)
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
    ap.add_argument('--dense', type=float, default=0.0,
                    help='DENSE_LAMBDA proximity tie-breaker on gain (0=pure integer)')
    ap.add_argument('--twopass', action='store_true',
                    help='use the live browser two-flock-pass dynamics')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    global WEIGHTS, DENSE_LAMBDA, TWO_PASS
    DENSE_LAMBDA = args.dense
    TWO_PASS = args.twopass
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
               two_pass=TWO_PASS, mean=mean, se=se, elapsed=elapsed, device=device)
    print(json.dumps(res))
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(res, fh)


if __name__ == '__main__':
    main()

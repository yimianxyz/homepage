"""Cheap-rollout planner — a DEPLOYABLE reactive policy, no net.

The true-dynamics planner (planner_probe.run_planner) reaches 14-22 catches by
ranking K candidate targets via H-frame TRUE flocking rollouts. v2/v3 showed that
edge is NOT distillable into a reactive net: the per-candidate value is a function
of the full 120-boid future, and a net reading only the M-nearest boids can fit
only ~45% of the decisive picks even on the training set (insufficient observation,
not distribution shift).

Different idea: keep the planner's argmax-over-candidates STRUCTURE, but replace the
expensive O(N^2) true-flocking rollout with a CHEAP O(N) rollout the browser can run
every frame. Two cheap boid models for the H-frame candidate ranking:
  - cv   : constant velocity (boids coast; O(1) per boid)
  - cvpa : constant velocity + predator avoidance only (drops the O(N^2) boid-boid
           cohesion/separation/alignment, keeps the O(N) flee-from-predator term that
           drives escape — the dynamic that makes catching hard)
The REAL game still runs true dynamics; only the policy's internal ranking rollout is
cheap. If cheap-rollout ranking ~ true-rollout ranking, this recovers the planner's
gain as a genuinely reactive, deployable policy (just a few cheap sims per decision).

Usage (VM):
  python3 cheap_planner.py --cheap_mode cvpa --K 8 --H 60 --D 15 \
      --n 512 --seedStart 200000 --weights predator_weights.json --out cheap.json
"""
import argparse, json, time
import numpy as np
import torch

import sim_torch as st
from sim_torch import (Sim, fast_limit, MAX_SPEED, MAX_FORCE, PREDATOR_RANGE,
                       PREDATOR_TURN_FACTOR, EPSILON)
import planner_probe as pp


def _step_boids_cheap(sim, mode):
    """Cheap boid advance. mode='cv' constant velocity; 'cvpa' adds the O(N)
    predator-avoidance term (vectorised over all boids), matching the true
    avoidance in sim_torch._compute_single_boid_acceleration (lines 542-556)."""
    vx = sim.boid_vel[..., 0]
    vy = sim.boid_vel[..., 1]
    if mode == 'cvpa':
        pdx = sim.boid_pos[..., 0] - sim.pred_pos[:, None, 0]      # (B,N) self - pred
        pdy = sim.boid_pos[..., 1] - sim.pred_pos[:, None, 1]
        pdist = torch.sqrt(pdx * pdx + pdy * pdy) + EPSILON
        in_pr = pdist < PREDATOR_RANGE
        fm_safe = torch.where(pdist > 0, pdist, torch.ones_like(pdist))
        avx = pdx / fm_safe
        avy = pdy / fm_safe
        strength = (PREDATOR_RANGE - pdist) / PREDATOR_RANGE
        avx = avx * strength * PREDATOR_TURN_FACTOR
        avy = avy * strength * PREDATOR_TURN_FACTOR
        avx, avy = fast_limit(avx, avy, MAX_FORCE * 1.5)
        avx = torch.where(in_pr, avx, torch.zeros_like(avx))
        avy = torch.where(in_pr, avy, torch.zeros_like(avy))
        vx = vx + avx
        vy = vy + avy
        vx, vy = fast_limit(vx, vy, MAX_SPEED)
        sim.boid_vel[..., 0] = vx
        sim.boid_vel[..., 1] = vy
    sim.boid_pos[..., 0] += vx
    sim.boid_pos[..., 1] += vy
    neg_b, pos_mw, pos_mh = sim._wrap_neg_b, sim._wrap_b_w_max, sim._wrap_b_h_max
    sim.boid_pos[..., 0] = torch.where(sim.boid_pos[..., 0] > pos_mw, neg_b, sim.boid_pos[..., 0])
    sim.boid_pos[..., 0] = torch.where(sim.boid_pos[..., 0] < neg_b, pos_mw, sim.boid_pos[..., 0])
    sim.boid_pos[..., 1] = torch.where(sim.boid_pos[..., 1] > pos_mh, neg_b, sim.boid_pos[..., 1])
    sim.boid_pos[..., 1] = torch.where(sim.boid_pos[..., 1] < neg_b, pos_mh, sim.boid_pos[..., 1])


def _step_with_target_cheap(sim, target, mode):
    _step_boids_cheap(sim, mode)
    pp._analytic_steer(sim, target)
    sim._check_catches()
    sim._decay_size()
    sim.frame += 1
    sim._frame_ms += st.FRAME_MS


def _pred_min_dist(roll):
    """Min distance predator->nearest ALIVE boid, per env (B*K,). Cheap O(N)."""
    dx = roll.boid_pos[..., 0] - roll.pred_pos[:, None, 0]
    dy = roll.boid_pos[..., 1] - roll.pred_pos[:, None, 1]
    d2 = dx * dx + dy * dy
    d2 = torch.where(roll.boid_alive, d2, torch.full_like(d2, float('inf')))
    return torch.sqrt(d2.min(dim=1).values)


@torch.no_grad()
def run_cheap_planner(seeds, frames, device, K, H, D, mode, score='catches'):
    """Real game = TRUE dynamics; candidate ranking = CHEAP rollout.

    score='catches'  : rank by catches over the horizon (sparse; ties -> cand 0).
    score='combined' : catches*1000 - (closest predator->boid approach over the
                       horizon). Catch-count still dominates, but among the (many)
                       no-catch candidates the one the predator gets CLOSEST to wins
                       — breaks the all-tie degeneracy that flattened the net distill."""
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D))
    B = sim.B
    roll = Sim(seeds=list(range(B * K)), weights=pp.WEIGHTS, device=device,
               auto_target='evolved', auto_target_opts=dict(pp.E3D))
    held = None
    rows = torch.arange(B, device=device)
    f = 0
    while f < frames:
        if f % D == 0:
            cand = pp._candidate_targets(sim, K)
            base = pp._save_state(sim)
            tiled = pp._tile_state(base, K)
            pp._load_state(roll, tiled)
            roll_tgt = cand.reshape(B * K, 2).contiguous()
            h = min(H, frames - f)
            c0 = roll.catches.clone()
            closest = _pred_min_dist(roll)
            for _ in range(h):
                _step_with_target_cheap(roll, roll_tgt, mode)
                if score == 'combined':
                    closest = torch.minimum(closest, _pred_min_dist(roll))
            gain = (roll.catches - c0).float()
            if score == 'combined':
                val = (gain * 1000.0 - closest).reshape(B, K)
            else:
                val = gain.reshape(B, K)
            best = val.argmax(dim=1)
            held = cand[rows, best]
        pp._step_with_target(sim, held)        # TRUE dynamics in the real game
        f += 1
    return sim.catches.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cheap_mode', choices=['cv', 'cvpa'], default='cvpa')
    ap.add_argument('--score', choices=['catches', 'combined'], default='catches')
    ap.add_argument('--n', type=int, default=512)
    ap.add_argument('--seedStart', type=int, default=200000)
    ap.add_argument('--frames', type=int, default=1500)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--H', type=int, default=60)
    ap.add_argument('--D', type=int, default=15)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--weights', default='predator_weights.json')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    pp.WEIGHTS = st.load_weights(args.weights, device=device)
    seeds = list(range(args.seedStart, args.seedStart + args.n))
    t0 = time.time()
    catches = run_cheap_planner(seeds, args.frames, device, args.K, args.H, args.D,
                                args.cheap_mode, args.score)
    mean = float(catches.mean()); se = float(catches.std(ddof=1) / np.sqrt(len(catches)))
    res = dict(cheap_mode=args.cheap_mode, score=args.score, n=args.n, seedStart=args.seedStart,
               frames=args.frames, K=args.K, H=args.H, D=args.D,
               mean=mean, se=se, pct_vs_baseline=100.0 * (mean - 8.3447) / 8.3447,
               elapsed=time.time() - t0, device=device)
    print(json.dumps(res), flush=True)
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(res, fh)


if __name__ == '__main__':
    main()

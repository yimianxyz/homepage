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
from sim_torch import Sim, PREDATOR_MAX_SPEED, PREDATOR_MAX_FORCE

_PS = 200.0          # position scale
_VS = 6.0            # velocity scale (MAX_SPEED)
_RHO = 70.0          # local-density radius around a candidate (px)
_HB = 90             # ballistic 2-body intercept horizon (frames)
FC = 19              # per-candidate feature dim (keep in sync with build below)
FCTX = 4             # global context dim


def _ballistic_intercept(px, py, vx, vy, bx, by, bvx, bvy, catch_d, H_b=_HB):
    """Cheap 2-body pursuit: predator (max-speed seek, no flock) vs a
    constant-velocity boid. Returns (t_catch_norm, min_dist, caught) per element.
    Because the predator is SLOWER than prey, a fleeing boid is never caught
    (t_catch->H_b) while a crossing/approaching one is intercepted -- exactly the
    'is this target catchable' signal the value net needs and instantaneous
    features can't express."""
    px, py, vx, vy = px.clone(), py.clone(), vx.clone(), vy.clone()
    bx, by = bx.clone(), by.clone()
    caught = torch.zeros_like(px, dtype=torch.bool)
    t_catch = torch.full_like(px, float(H_b))
    mind = torch.full_like(px, float('inf'))
    for t in range(H_b):
        dx = bx - px; dy = by - py
        d = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-6)
        mind = torch.minimum(mind, d)
        newly = (d < catch_d) & (~caught)
        t_catch = torch.where(newly, torch.full_like(t_catch, float(t)), t_catch)
        caught = caught | newly
        # seek: desired velocity at max speed toward boid; steering clamped to max force
        desx = dx / d * PREDATOR_MAX_SPEED - vx
        desy = dy / d * PREDATOR_MAX_SPEED - vy
        sm = torch.sqrt(desx * desx + desy * desy).clamp(min=1e-6)
        sc = torch.clamp(PREDATOR_MAX_FORCE / sm, max=1.0)
        vx = vx + desx * sc; vy = vy + desy * sc
        spd = torch.sqrt(vx * vx + vy * vy).clamp(min=1e-6)
        vsc = torch.clamp(PREDATOR_MAX_SPEED / spd, max=1.0)
        vx = vx * vsc; vy = vy * vsc
        px = px + vx; py = py + vy
        bx = bx + bvx; by = by + bvy
    return t_catch / H_b, mind, caught.float()


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

    # 2-body ballistic intercept of the targeted boid (the decisive catchability
    # signal: predator is slower than prey, so a fleeing boid is uncatchable).
    catch_d = (sim.pred_size * 0.7)[:, None].expand(B, K)
    t_catch, bmin_d, caught = _ballistic_intercept(
        px[:, None].expand(B, K), py[:, None].expand(B, K),
        pvx[:, None].expand(B, K), pvy[:, None].expand(B, K),
        bx, by, bvx, bvy, catch_d)

    feat = torch.stack([
        rx / _PS, ry / _PS, dist / _PS, is_e3d,
        t_go / 60.0,
        tbrx / _PS, tbry / _PS, bvx / _VS, bvy / _VS,
        tb_dist_to_cand / _PS,
        rangepb / _PS, closing / _VS, los_rate * 50.0,
        dens / 20.0, flee_align / _VS,
        (rangepb - dist) / _PS,                               # boid-vs-candidate range gap
        t_catch, bmin_d / _PS, caught,                        # ballistic intercept (FC 17-19)
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


def run_value_lookahead(seeds, frames, device, model, K, D, Hs, bias0=0.0):
    """Depth-1 lookahead + value bootstrap (the AlphaZero/Bellman backup): roll
    each of K candidates Hs frames with TRUE dynamics, then score each by
    (catches during Hs) + max_j V(candidate j at the TERMINAL state). argmax.
    The short rollout resolves near-term catches the features can't see; the
    max-V bootstrap stands in for the beyond-horizon tail. Needs a CALIBRATED
    (absval) value net so V is in catch units and combines with the rollout
    catch counts."""
    model.eval()
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=pp.TWO_PASS)
    B = sim.B
    roll = Sim(seeds=list(range(B * K)), weights=pp.WEIGHTS, device=device,
               auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=pp.TWO_PASS)
    rows = torch.arange(B, device=device)
    held = None
    f = 0
    while f < frames:
        if f % D == 0:
            cand = pp._candidate_targets(sim, K)                 # (B,K,2)
            base = pp._save_state(sim)
            pp._load_state(roll, pp._tile_state(base, K))
            roll_tgt = cand.reshape(B * K, 2).contiguous()
            c0 = roll.catches.clone()
            for _ in range(Hs):
                pp._step_with_target(roll, roll_tgt)
            c_near = (roll.catches - c0).reshape(B, K).float()   # (B,K) catches during Hs
            tcand = pp._candidate_targets(roll, K)               # (B*K, K, 2) at terminal
            tfeat, tctx = candidate_features(roll, tcand)
            with torch.no_grad():
                tv = model(tfeat.to(device), tctx.to(device))    # (B*K, K)
            boot = tv.max(dim=1).values.reshape(B, K)            # state value at terminal
            score = c_near + boot
            if bias0 != 0.0:
                score = score.clone(); score[:, 0] = score[:, 0] + bias0
            held = cand[rows, score.argmax(dim=1)]
        pp._step_with_target(sim, held)
        f += 1
    return sim.catches.cpu().numpy()


def run_value_lookahead_cheap(seeds, frames, device, model, K, D, Hs, roll_M, bias0=0.0,
                              K_roll=0, prune_by='v', no_value=False):
    """Browser-affordable lookahead: the rollout only simulates the M nearest boids
    to each predator (rest frozen) -> O(M) instead of O(120) rollout cost.

    K_roll>0: AlphaZero policy-prior pruning -- score all K candidates by V at the
    CURRENT state (cheap, no rollout), and only the top-K_roll get the (expensive)
    rollout score (c_near + bootstrap); the rest keep just their prior V. Cuts
    rollout cost ~K/K_roll x. (sim_torch still rolls all K to MEASURE the pruned
    decision's catches; the deploy cost saving is realized in JS.)
    score = (catches during cheap Hs rollout) + max_j V(terminal)."""
    model.eval()
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=pp.TWO_PASS)
    B = sim.B
    roll = Sim(seeds=list(range(B * K)), weights=pp.WEIGHTS, device=device,
               auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=pp.TWO_PASS)
    rows = torch.arange(B, device=device)
    held = None
    f = 0
    while f < frames:
        if f % D == 0:
            cand = pp._candidate_targets(sim, K)
            base = pp._save_state(sim)
            pp._load_state(roll, pp._tile_state(base, K))
            roll_tgt = cand.reshape(B * K, 2).contiguous()
            # M nearest boids to each env's predator (at rollout start) = active
            dx = roll.boid_pos[..., 0] - roll.pred_pos[:, None, 0]
            dy = roll.boid_pos[..., 1] - roll.pred_pos[:, None, 1]
            d2 = dx * dx + dy * dy
            d2 = torch.where(roll.boid_alive, d2, torch.full_like(d2, float('inf')))
            order = torch.argsort(d2, dim=1)
            active = torch.zeros_like(roll.boid_alive)
            active.scatter_(1, order[:, :roll_M], torch.ones_like(order[:, :roll_M], dtype=active.dtype))
            frozen = ~active
            c0 = roll.catches.clone()
            for _ in range(Hs):
                sp = roll.boid_pos.clone(); sv = roll.boid_vel.clone()
                roll._step_boids()
                roll.boid_pos[frozen] = sp[frozen]
                roll.boid_vel[frozen] = sv[frozen]
                pp._analytic_steer(roll, roll_tgt)
                roll._check_catches()
                # FIDELITY FIX: the planner's rollout (_step_with_target) also decays
                # predator size + advances frame/time each step; omitting these made
                # the rollout over-count catches (predator never shrinks) and mis-rank
                # candidates -> cheap(K16,Hs120,no_value) was 0.55x the planner, not 1x.
                roll._decay_size(); roll.frame += 1; roll._frame_ms += st.FRAME_MS
            c_near = (roll.catches - c0).reshape(B, K).float()
            if no_value:
                roll_score = c_near                                   # no terminal bootstrap
            else:
                tcand = pp._candidate_targets(roll, K)
                tfeat, tctx = candidate_features(roll, tcand)
                with torch.no_grad():
                    tv = model(tfeat.to(device), tctx.to(device))
                roll_score = c_near + tv.max(dim=1).values.reshape(B, K)
            if 0 < K_roll < K:
                f0, x0 = candidate_features(sim, cand)
                vprior = None
                if (not no_value) or prune_by != 'ball':
                    with torch.no_grad():
                        vprior = model(f0.to(device), x0.to(device))  # (B,K) prior V
                if prune_by == 'ball':
                    # rank candidates by ballistic catchability: caught flag (feat 18)
                    # minus normalized time-to-catch (feat 16). Higher = catch sooner.
                    pscore = (f0[:, :, 18] - f0[:, :, 16]).to(device)
                else:
                    pscore = vprior
                thr = torch.topk(pscore, K_roll, dim=1).values[:, -1:]
                is_top = pscore >= thr                                # top-K_roll get rollout
                # no_value: non-rolled candidates score 0 -> argmax falls back to
                # cand0 (E3D) when the single rolled candidate yields no catch.
                nonroll = torch.zeros_like(roll_score) if no_value else vprior
                score = torch.where(is_top, roll_score, nonroll)
            else:
                score = roll_score
            if bias0 != 0.0:
                score = score.clone(); score[:, 0] = score[:, 0] + bias0
            held = cand[rows, score.argmax(dim=1)]
        pp._step_with_target(sim, held)
        f += 1
    return sim.catches.cpu().numpy()


def run_dagger_lookahead(seeds, frames, device, model, K, H, D, Hs):
    """DAgger for the LOOKAHEAD student (the correct one): the lookahead+bootstrap
    student drives the episode (its own state distribution); at each decision the
    PLANNER's true H-frame rollout relabels (feat, ctx, gain). Two rollouts/decision
    (student Hs + planner H) -> slow but one-time. Fixes dist-shift for the deployed
    policy. Returns (feat, ctx, gain, student_catches)."""
    model.eval()
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
            roll_tgt = cand.reshape(B * K, 2).contiguous()
            pp._load_state(roll, pp._tile_state(base, K))
            c0 = roll.catches.clone()
            for _ in range(min(H, frames - f)):
                pp._step_with_target(roll, roll_tgt)
            gain = pp.rollout_gain(roll, c0, B, K)
            feat_log.append(feat.cpu()); ctx_log.append(ctx.cpu()); gain_log.append(gain.cpu())
            pp._load_state(roll, pp._tile_state(base, K))
            c0 = roll.catches.clone()
            for _ in range(Hs):
                pp._step_with_target(roll, roll_tgt)
            c_near = (roll.catches - c0).reshape(B, K).float()
            tfeat, tctx = candidate_features(roll, pp._candidate_targets(roll, K))
            with torch.no_grad():
                tv = model(tfeat.to(device), tctx.to(device))
            held = cand[rows, (c_near + tv.max(dim=1).values.reshape(B, K)).argmax(dim=1)]
        pp._step_with_target(sim, held)
        f += 1
    feat = torch.cat(feat_log, 0).numpy(); ctx = torch.cat(ctx_log, 0).numpy()
    gain = torch.cat(gain_log, 0).numpy()
    return feat, ctx, gain, sim.catches.cpu().numpy()


def run_dagger_feat(seeds, frames, device, model, K, H, D, bias0=0.0):
    """DAgger relabel: the STUDENT drives the episode (predator follows the
    student's committed target, visiting the student's own state distribution),
    but at every decision we ALSO run the planner's true H-frame rollout to
    relabel (feat, ctx, gain). Returns (feat, ctx, gain, student_catches).

    Fixes the v1<E3D distribution-shift: the value net gets trained on the OOD
    states it actually wanders into, with correct planner targets there."""
    model.eval()
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
            feat_log.append(feat.cpu()); ctx_log.append(ctx.cpu()); gain_log.append(gain.cpu())
            with torch.no_grad():
                v = model(feat.to(device), ctx.to(device))
            score = v
            if bias0 != 0.0:
                score = score.clone(); score[:, 0] = score[:, 0] + bias0
            held = cand[rows, score.argmax(dim=1)]
        pp._step_with_target(sim, held)
        f += 1
    feat = torch.cat(feat_log, 0).numpy()
    ctx = torch.cat(ctx_log, 0).numpy()
    gain = torch.cat(gain_log, 0).numpy()
    return feat, ctx, gain, sim.catches.cpu().numpy()


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

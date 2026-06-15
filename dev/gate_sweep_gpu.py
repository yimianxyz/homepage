#!/usr/bin/env python3
"""Batched GPU throughput surface for the planner/endgame gate search (#5).

Runs the DEPLOYED policy in sim_torch — planner (feat_planner._planner_held, the
validated run_value_lookahead_cheap decision: K16/K_roll4/Hs90/D16, ballistic prune,
value-net bootstrap) for N>gate + the raw-kinematics endgame NN (egboidPickRaw) for
N<=gate with prod's torus-scan AIM — under a configurable gate (count / density /
horizon, incl. screen-scaled reach). Metric = THROUGHPUT (catches/effective-frame) +
never-clear sanity. Wide map: screen matrix x rule x many seeds, batched.

NON-decisive (JS/side-b decides); the GPU<->JS rank-correlation cross-check
(gate_search.js) validates it before any ranking is trusted.

  python3 gate_sweep_gpu.py --rule count --param 5 --cell 1024x768:120 --seeds 128 \
      --seed0 700000 --frames 6000 --device cuda

Faithfulness notes:
  - endgame NN forward (eg_scan_t) is bit-exact vs JS egboidPickRaw (max|dt|~1e-12).
  - planner decision reuses the validated _planner_held -> identical to the parity port.
  - endgame steer replicates intercept(): desired=fast_set_magnitude(aim,sM); steer=
    desired-vel; fast_limit(MAX_FORCE); then the SAME predator integration as the
    planner's _analytic_steer (vel+=force; fast_limit(MAX_SPEED); pos+=vel; torus wrap).
  - planner TARGET is decided pre-step (matches run_value_lookahead_cheap); both the
    planner FORCE and the endgame aim are computed POST-boid-step (matches the deploy's
    force() being called after the boid step, per the validated _step_with_target).
"""
import argparse, json, os
import numpy as np
import torch
import sim_torch as st
from sim_torch import build_features, fast_set_magnitude, fast_limit
import planner_probe as pp
import feat_planner as fp
from eval_value import Deploy

SM = st.PREDATOR_MAX_SPEED          # 2.5
MF = st.PREDATOR_MAX_FORCE
BORDER = 10
HS_REACH = 90                       # planner rollout horizon (for the horizon rule)

# ---- endgame NN (egboidPickRaw): raw-kinematics features -> scan-t, argmin = egBoid ----
def eg_features_raw(pred_pos, pred_vel, pred_size, boid_pos, boid_vel, W, Hc, device):
    # pred_pos (B,2), boid_pos (B,N,2) -> (B,N,15)
    PX, PY = W + 2 * BORDER, Hc + 2 * BORDER
    rx = boid_pos[..., 0] - pred_pos[:, None, 0]
    ry = boid_pos[..., 1] - pred_pos[:, None, 1]
    rwx = rx - PX * torch.round(rx / PX)
    rwy = ry - PY * torch.round(ry / PY)
    d0 = torch.sqrt(rwx * rwx + rwy * rwy); ds = torch.clamp(d0, min=1e-6)
    bvx, bvy = boid_vel[..., 0], boid_vel[..., 1]
    radial = (rwx * bvx + rwy * bvy) / ds
    tangent = (rwx * bvy - rwy * bvx) / ds
    bspeed = torch.sqrt(bvx * bvx + bvy * bvy)
    B, N = rx.shape
    Wt = torch.full((B, N), W / 2560.0, device=device, dtype=rx.dtype)
    Ht = torch.full((B, N), Hc / 1440.0, device=device, dtype=rx.dtype)
    pvx = (pred_vel[:, None, 0] / 6.0).expand(B, N)
    pvy = (pred_vel[:, None, 1] / 6.0).expand(B, N)
    ps = (pred_size[:, None] / 20.0).expand(B, N)
    return torch.stack([rwx / 200, rwy / 200, d0 / 200, rx / 200, ry / 200,
                        bvx / 6, bvy / 6, bspeed / 6, radial / 6, tangent / 6,
                        Wt, Ht, pvx, pvy, ps], dim=-1)

_AS = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]

def load_eg(path, device):
    J = json.load(open(path)); W = J['weights']
    return {k: torch.tensor(W[k], dtype=torch.float64, device=device) for k in
            ['net.0.weight', 'net.0.bias', 'net.2.weight', 'net.2.bias', 'net.4.weight', 'net.4.bias']}

def eg_scan_t(feat, egw):  # feat (B,N,15) -> (B,N) predicted scan-t (frames)
    def gelu(x):
        z = x * 0.7071067811865476; sz = torch.sign(z); az = z.abs(); tt = 1.0 / (1.0 + 0.3275911 * az)
        erf = sz * (1.0 - (((((_AS[4] * tt + _AS[3]) * tt) + _AS[2]) * tt + _AS[1]) * tt + _AS[0]) * tt * torch.exp(-az * az))
        return 0.5 * x * (1.0 + erf)
    h = gelu(feat @ egw['net.0.weight'].T + egw['net.0.bias'])
    h = gelu(h @ egw['net.2.weight'].T + egw['net.2.bias'])
    return (h @ egw['net.4.weight'].T + egw['net.4.bias']).squeeze(-1) * 100.0


def _endgame_aim(pred_pos, eb, ev, PX, PY, dev):
    """intercept() scan(): earliest torus point of egBoid the predator can reach ->
    return the WRAPPED aim displacement (B,2). Perpendicular cut-off onto the boid's
    line if no reachable t. Constant-velocity straight-line projection, TMAX=1400."""
    B = pred_pos.shape[0]; TMAX = 1400
    px, py = pred_pos[:, 0], pred_pos[:, 1]
    found = torch.zeros(B, dtype=torch.bool, device=dev)
    aimx = torch.zeros(B, dtype=pred_pos.dtype, device=dev); aimy = torch.zeros_like(aimx)
    for t in range(0, TMAX + 1):
        ddx = (eb[:, 0] + ev[:, 0] * t - px); ddy = (eb[:, 1] + ev[:, 1] * t - py)
        ddx = ddx - PX * torch.round(ddx / PX); ddy = ddy - PY * torch.round(ddy / PY)
        reach = (ddx * ddx + ddy * ddy) <= (SM * t) * (SM * t)
        hit = reach & (~found)
        aimx = torch.where(hit, ddx, aimx); aimy = torch.where(hit, ddy, aimy)
        found = found | reach
        if bool(found.all()):
            break
    # perpendicular cut-off fallback (no reachable t): aim onto the boid's velocity line
    cdx = (eb[:, 0] - px); cdy = (eb[:, 1] - py)
    cdx = cdx - PX * torch.round(cdx / PX); cdy = cdy - PY * torch.round(cdy / PY)
    bs = torch.clamp(torch.sqrt(ev[:, 0] ** 2 + ev[:, 1] ** 2), min=1e-6)
    ux, uy = ev[:, 0] / bs, ev[:, 1] / bs; along = cdx * ux + cdy * uy
    fbx, fby = cdx - along * ux, cdy - along * uy
    aimx = torch.where(found, aimx, fbx); aimy = torch.where(found, aimy, fby)
    return aimx, aimy


A_REF = (1024 + 2 * BORDER) * (768 + 2 * BORDER)   # side-b density ref area (1044x788)


def _reach_time(pred_pos, boid_pos, boid_vel, boid_alive, PX, PY):
    """Wrap-aware analytic 2-body intercept time per boid (B,N): smallest t>0 solving
    |minImage(r + v*t)| = SM*t. No real positive root / dead boid -> +inf. This is the
    horizon rule's wa0 (side-b def: enter endgame when even the soonest boid's wa0 > H)."""
    rx = boid_pos[..., 0] - pred_pos[:, None, 0]; ry = boid_pos[..., 1] - pred_pos[:, None, 1]
    rwx = rx - PX * torch.round(rx / PX); rwy = ry - PY * torch.round(ry / PY)
    vx = boid_vel[..., 0]; vy = boid_vel[..., 1]
    a = vx * vx + vy * vy - SM * SM
    b = 2.0 * (rwx * vx + rwy * vy)
    c = rwx * rwx + rwy * rwy
    INF = torch.full_like(c, float('inf'))
    disc = b * b - 4.0 * a * c
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    pos = lambda x: torch.where(x > 1e-9, x, INF)
    safe_a = torch.where(a.abs() > 1e-12, a, torch.ones_like(a))
    r1 = (-b - sq) / (2.0 * safe_a); r2 = (-b + sq) / (2.0 * safe_a)
    quad = torch.minimum(pos(r1), pos(r2))
    safe_b = torch.where(b.abs() > 1e-12, b, torch.ones_like(b))
    lin = pos(-c / safe_b)
    t = torch.where(a.abs() > 1e-12, quad, lin)
    t = torch.where(disc < 0, INF, t)
    t = torch.where(boid_alive, t, INF)
    return t


def _gate_step(sim, planner_held, inEnd, egidx, committed, egw, W, H, PX, PY, dev, rows, fd):
    """One frame: step boids, then steer each env by the planner force (N>gate) OR the
    endgame force (N<=gate, latched), integrate the predator once, check catches/decay.
    Both forces are computed from the POST-boid-step state. egidx is the committed egBoid
    (commit-and-hold); committed marks envs that have an egBoid yet. Returns (egidx, committed)."""
    sim._step_boids()
    # --- planner force (chase-nearest-in-range / patrol seek), exactly _analytic_steer ---
    feats = build_features(sim.pred_pos.float(), sim.pred_vel.float(),
                           sim.boid_pos.float(), sim.boid_vel.float(), sim.boid_alive,
                           planner_held.float(), fd, dev, dtype=torch.float64)
    in_range = feats[:, 34] > 0.5
    pforce = torch.where(in_range.unsqueeze(1), feats[:, 29:31], feats[:, 31:33])  # (B,2)
    fx = pforce[:, 0].clone(); fy = pforce[:, 1].clone()
    # --- endgame force (intercept torus-aim), only where latched ---
    if bool(inEnd.any()):
        # NN pick on entry (uncommitted) OR when the held egBoid died (commit-and-hold).
        held_dead = ~sim.boid_alive[rows, egidx]
        need = inEnd & (~committed | held_dead)
        if bool(need.any()):
            feat = eg_features_raw(sim.pred_pos, sim.pred_vel, sim.pred_size,
                                   sim.boid_pos, sim.boid_vel, W, H, dev)
            t_pred = eg_scan_t(feat, egw)
            t_pred = torch.where(sim.boid_alive, t_pred, torch.full_like(t_pred, 1e18))
            egidx = torch.where(need, t_pred.argmin(1), egidx)
            committed = committed | need
        eb = sim.boid_pos[rows, egidx]; ev = sim.boid_vel[rows, egidx]          # (B,2)
        aimx, aimy = _endgame_aim(sim.pred_pos, eb, ev, PX, PY, dev)
        desx, desy = fast_set_magnitude(aimx, aimy, SM)
        ex, ey = fast_limit(desx - sim.pred_vel[:, 0], desy - sim.pred_vel[:, 1], MF)
        fx = torch.where(inEnd, ex, fx); fy = torch.where(inEnd, ey, fy)
    # --- shared predator integration (vel += force; clamp speed; move; torus wrap) ---
    nvx, nvy = fast_limit(sim.pred_vel[:, 0] + fx, sim.pred_vel[:, 1] + fy, SM)
    sim.pred_vel[:, 0] = nvx; sim.pred_vel[:, 1] = nvy
    sim.pred_pos[:, 0] += nvx; sim.pred_pos[:, 1] += nvy
    sim.pred_pos[:, 0] = torch.where(sim.pred_pos[:, 0] > sim._wrap_w_max, sim._wrap_neg20, sim.pred_pos[:, 0])
    sim.pred_pos[:, 0] = torch.where(sim.pred_pos[:, 0] < sim._wrap_neg20, sim._wrap_w_max, sim.pred_pos[:, 0])
    sim.pred_pos[:, 1] = torch.where(sim.pred_pos[:, 1] > sim._wrap_h_max, sim._wrap_neg20, sim.pred_pos[:, 1])
    sim.pred_pos[:, 1] = torch.where(sim.pred_pos[:, 1] < sim._wrap_neg20, sim._wrap_h_max, sim.pred_pos[:, 1])
    sim._check_catches()
    sim._decay_size(); sim.frame += 1; sim._frame_ms += st.FRAME_MS
    return egidx, committed


def _build_sims(W, H, boids, seed_list, dev, K=16):
    """Build the main sim (one env per seed in seed_list) + the B*K rollout sim. The
    rollout seeds are arbitrary (overwritten by _load_state each planner decision)."""
    st.CANVAS_W = float(W); st.CANVAS_H = float(H)                  # screen-matrix: set globals BEFORE Sim
    B = len(seed_list)
    sim = st.Sim(seeds=list(seed_list), weights=pp.WEIGHTS, device=dev, num_boids=boids,
                 auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=pp.TWO_PASS)
    sim.always_recompute_target = True
    roll = st.Sim(seeds=list(range(B * K)), weights=pp.WEIGHTS, device=dev, num_boids=boids,
                  auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=pp.TWO_PASS)
    roll.always_recompute_target = True
    return sim, roll


def _simulate(sim, roll, model, egw, rule, param_vec, frames, W, H, dev):
    """Run `frames` of the gated policy; param_vec (B,) is the PER-ENV gate threshold
    (lets one batched sim cover many gate params at once). Returns (catches, clearFrame)
    numpy arrays of length B."""
    B = sim.B; rows = torch.arange(B, device=dev)
    fd = pp.WEIGHTS['featureDim']
    K, D, Hs, Kroll, roll_M = 16, 16, 90, 4, sim.boid_pos.shape[1]
    PX, PY = W + 2 * BORDER, H + 2 * BORDER
    inEnd = torch.zeros(B, dtype=torch.bool, device=dev)
    egidx = torch.zeros(B, dtype=torch.long, device=dev)
    committed = torch.zeros(B, dtype=torch.bool, device=dev)
    planner_held = sim.pred_auto.clone()
    clear_frame = torch.full((B,), -1, dtype=torch.long, device=dev)
    f = 0
    while f < frames:
        nalive = sim.boid_alive.sum(1)                              # (B,)
        # gate ENTER condition (latched once true; no exit-hysteresis -> N monotone).
        # rule defs MATCH side-b (#6) for the GPU<->JS cross-check.
        if rule == 'count':                       # enter when N <= T
            cond = nalive <= param_vec
        elif rule == 'density':                   # area-scaled count: Td = round(Tref * A/A_ref)
            Td = torch.round(param_vec * (PX * PY / A_REF))
            cond = nalive <= Td
        elif rule == 'horizon':                   # enter when soonest reachable boid's wa0 > H
            wa0 = _reach_time(sim.pred_pos, sim.boid_pos, sim.boid_vel, sim.boid_alive, PX, PY)
            cond = wa0.min(dim=1).values > param_vec
        else:
            raise SystemExit('bad rule')
        inEnd = inEnd | (cond & (nalive > 0))
        # PLANNER target decision every D frames (pre-step; held constant between)
        if f % D == 0:
            planner_held = fp._planner_held(sim, roll, model, K, Hs, roll_M,
                                            K_roll=Kroll, prune_by='ball', no_value=False)
        egidx, committed = _gate_step(sim, planner_held, inEnd, egidx, committed, egw, W, H, PX, PY, dev, rows, fd)
        f += 1
        just = (clear_frame < 0) & (sim.boid_alive.sum(1) == 0)
        clear_frame = torch.where(just, torch.full_like(clear_frame, f), clear_frame)
        if f % 64 == 0 and bool((clear_frame >= 0).all()):   # all envs extinct -> done
            break
    return sim.catches.cpu().numpy().astype(float), clear_frame.cpu().numpy()


def _agg(catches, cf, frames):
    cleared = cf >= 0; eff = np.where(cleared, cf, frames)
    return {'throughput': float(catches.sum() / eff.sum()), 'clearRate': float(cleared.mean()),
            'medCatches': float(np.median(catches)),
            'medClearFrames': float(np.median(cf[cleared])) if cleared.any() else -1}


def run_one(W, H, boids, rule, param, seed0, n_seeds, frames, dev, model, egw, dump=''):
    """One (screen, rule, param) config -> throughput + clear stats."""
    sim, roll = _build_sims(W, H, boids, list(range(seed0, seed0 + n_seeds)), dev)
    param_vec = torch.full((sim.B,), float(param), device=dev)
    catches, cf = _simulate(sim, roll, model, egw, rule, param_vec, frames, W, H, dev)
    out = {'cell': f'{W}x{H}', 'boids': boids, 'rule': rule, 'param': param,
           'seeds': n_seeds, 'frames': frames, **_agg(catches, cf, frames)}
    if dump:
        json.dump({'seeds': list(range(seed0, seed0 + n_seeds)), 'catches': catches.tolist(),
                   'clearFrame': cf.tolist(), 'cleared': (cf >= 0).tolist(),
                   **{k: out[k] for k in ('cell', 'rule', 'param')}}, open(dump, 'w'))
    return out


def run_screen_packed(W, H, boids, rule, params, n_seeds, frames, dev, model, egw, seed0=700000):
    """ALL gate params for one screen+rule in a SINGLE batched sim (env = param-group x
    seed). The high-N planner phase is identical across params; packing amortizes the
    Python frame-loop (latency-bound at small env counts) -> ~Gx speedup. Each env uses
    its own seed (NOTE: seeds are shared across param-groups -> PAIRED across params).
    Returns list of per-param result dicts."""
    G = len(params); B = G * n_seeds
    # env e = group g (g=e//n_seeds) x seed s (s=e%n_seeds); seeds TILED so each group
    # sees the same n_seeds initial states -> paired across params.
    seed_list = list(range(seed0, seed0 + n_seeds)) * G
    sim, roll = _build_sims(W, H, boids, seed_list, dev)
    grp = torch.arange(B, device=dev) // n_seeds                    # (B,) param-group index
    pvals = torch.tensor([float(p) for p in params], device=dev)
    param_vec = pvals[grp]
    catches, cf = _simulate(sim, roll, model, egw, rule, param_vec, frames, W, H, dev)
    g_np = grp.cpu().numpy()
    res = []
    for gi, p in enumerate(params):
        m = g_np == gi
        res.append({'cell': f'{W}x{H}', 'boids': boids, 'rule': rule, 'param': p,
                    'seeds': n_seeds, 'frames': frames, **_agg(catches[m], cf[m], frames)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rule', default='count'); ap.add_argument('--param', type=float, default=5)
    ap.add_argument('--cell', default='1024x768:120'); ap.add_argument('--seeds', type=int, default=128)
    ap.add_argument('--seed0', type=int, default=700000); ap.add_argument('--frames', type=int, default=6000)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--egw', default='eg_weights_raw.json')
    ap.add_argument('--valnet', default='net_strict.pt')
    ap.add_argument('--predw', default='predator_weights.json')
    ap.add_argument('--dump', default='')      # optional: dump per-seed catches/clear for rank-corr
    ap.add_argument('--configs', default='')   # JSON list of {cell,rule,param[,seeds,frames,seed0]} -> JSONL
    ap.add_argument('--out', default='')        # JSONL output path for --configs mode
    ap.add_argument('--packed', action='store_true')  # group configs by (cell,rule) -> one batched sim
    a = ap.parse_args(); dev = a.device
    pw = a.predw if os.path.exists(a.predw) else os.path.expanduser('~/' + a.predw)
    pp.WEIGHTS = st.load_weights(pw, device=dev)
    vn = a.valnet if os.path.exists(a.valnet) else os.path.expanduser('~/' + a.valnet)
    model = Deploy(torch.load(vn, map_location='cpu'), dev); model.eval()
    egw = load_eg(a.egw, dev)
    if a.configs:                               # MULTI-CONFIG in-process (amortize model load)
        cfgs = json.load(open(a.configs))
        fh = open(a.out, 'w') if a.out else None
        import time
        if a.packed:                            # group by (cell,rule,seeds,frames,seed0) -> 1 batched sim
            from collections import OrderedDict
            groups = OrderedDict()
            for c in cfgs:
                key = (c['cell'], c['rule'], c.get('seeds', a.seeds), c.get('frames', a.frames), c.get('seed0', a.seed0))
                groups.setdefault(key, []).append(c['param'])
            done = 0
            for gi, ((cell_s, rule, n_seeds, frames, seed0), params) in enumerate(groups.items()):
                cell, boids = cell_s.split(':'); W, H = map(int, cell.split('x')); boids = int(boids)
                t0 = time.time()
                res = run_screen_packed(W, H, boids, rule, params, n_seeds, frames, dev, model, egw, seed0=seed0)
                sec = round(time.time() - t0, 1)
                for out in res:
                    out['sec'] = round(sec / len(params), 1)
                    line = json.dumps(out); done += 1
                    print(f'[{done}/{len(cfgs)}] {line}', flush=True)
                    if fh: fh.write(line + '\n'); fh.flush()
                print(f'  (group {gi+1}/{len(groups)} {cell_s} {rule} x{len(params)}params took {sec}s)', flush=True)
            if fh: fh.close()
            return
        for i, c in enumerate(cfgs):
            cell, boids = c['cell'].split(':'); W, H = map(int, cell.split('x')); boids = int(boids)
            t0 = time.time()
            out = run_one(W, H, boids, c['rule'], c['param'], c.get('seed0', a.seed0),
                          c.get('seeds', a.seeds), c.get('frames', a.frames), dev, model, egw)
            out['sec'] = round(time.time() - t0, 1)
            line = json.dumps(out)
            print(f'[{i+1}/{len(cfgs)}] {line}', flush=True)
            if fh: fh.write(line + '\n'); fh.flush()
        if fh: fh.close()
        return
    cell, boids = a.cell.split(':'); W, H = map(int, cell.split('x')); boids = int(boids)
    print(json.dumps(run_one(W, H, boids, a.rule, a.param, a.seed0, a.seeds, a.frames, dev, model, egw, dump=a.dump)))


if __name__ == '__main__':
    main()

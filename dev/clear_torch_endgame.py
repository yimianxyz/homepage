#!/usr/bin/env python3
"""GPU few-boid endgame sim with a vectorized TERI predator, for large-scale
validation + param tuning of the endgame interceptor across thousands of envs in
parallel (the long run-to-extinction episodes are exactly where batched-GPU wins).

Reuses sim_torch's boid dynamics (flee+flock) for parity; replaces the predator with
TERI (Torus Earliest-Reachable Lead Intercept + bubble-commit). Boids are scattered
(the endgame regime) and run to extinction; reports mean frames-to-clear + clear-rate.

  python3 dev/clear_torch_endgame.py --device cuda --K 1 --B 2048 --maxFrames 4000 --sweep
"""
import sys, argparse, time
import numpy as np
import torch
import sim_torch as st
from sim_torch import Sim, mulberry32_seq, PREDATOR_MAX_SPEED, PREDATOR_MAX_FORCE, BORDER_OFFSET


class EndgameSim(Sim):
    def __init__(self, seeds, weights, num_boids, device, teri):
        self.teri = teri
        super().__init__(seeds=seeds, weights=weights, num_boids=num_boids,
                         auto_target='evolved', auto_target_opts=dict(), device=device, two_pass=True)
        # persistent TERI commit state
        B = self.B
        self.eg_tgt = torch.zeros((B,), dtype=torch.long, device=device)
        self.eg_frozen = torch.zeros((B,), dtype=torch.bool, device=device)
        self.eg_aim = torch.zeros((B, 2), dtype=torch.float64, device=device)

    def _initialize(self):
        super()._initialize()
        d = self.device
        # scatter boids to random positions (endgame regime), keep unit init velocity
        for bi, seed in enumerate(self.seeds):
            r = mulberry32_seq(seed ^ 0x5bd1e995, 2 * self.N)
            self.boid_pos[bi, :, 0] = torch.tensor(r[0::2] * st.CANVAS_W, dtype=torch.float64, device=d)
            self.boid_pos[bi, :, 1] = torch.tensor(r[1::2] * st.CANVAS_H, dtype=torch.float64, device=d)

    def _step_predator(self):
        teri = self.teri
        PX = st.CANVAS_W + 2 * BORDER_OFFSET
        PY = st.CANVAS_H + 2 * BORDER_OFFSET
        sM = PREDATOR_MAX_SPEED
        SLACK, FREEZE_R, DT, TMAX = teri['SLACK'], teri['FREEZE_R'], teri['DT'], teri['TMAX']
        B, K = self.B, self.N
        d = self.device
        px = self.pred_pos[:, 0:1]; py = self.pred_pos[:, 1:2]          # (B,1)
        bpx = self.boid_pos[:, :, 0]; bpy = self.boid_pos[:, :, 1]      # (B,K)
        bvx = self.boid_vel[:, :, 0]; bvy = self.boid_vel[:, :, 1]
        alive = self.boid_alive
        tgrid = torch.arange(0, TMAX + 1, DT, dtype=torch.float64, device=d)  # (T,)
        T = tgrid.shape[0]

        def minimg(a, P):
            return a - P * torch.round(a / P)

        # future boid positions B+t*v  -> (B,K,T)
        fx = bpx[:, :, None] + bvx[:, :, None] * tgrid[None, None, :]
        fy = bpy[:, :, None] + bvy[:, :, None] * tgrid[None, None, :]
        ddx = minimg(fx - px[:, :, None], PX)
        ddy = minimg(fy - py[:, :, None], PY)
        dist = torch.sqrt(ddx * ddx + ddy * ddy)                       # (B,K,T)
        feas = (dist <= sM * tgrid[None, None, :] * SLACK) & alive[:, :, None]
        # earliest feasible t index per (B,K): first True, else T (infeasible)
        feas_i = torch.where(feas, torch.arange(T, device=d)[None, None, :], torch.full((1, 1, T), T, device=d, dtype=torch.long))
        first = feas_i.min(dim=2).values                                # (B,K)
        tstar = torch.where(first < T, tgrid[first.clamp(max=T - 1)], torch.full_like(first, TMAX + 1, dtype=torch.float64))
        tstar = torch.where(alive, tstar, torch.full_like(tstar, 1e18))
        # target = smallest feasible t* (alive); commit unless current target alive
        tgt_alive = torch.gather(alive, 1, self.eg_tgt[:, None]).squeeze(1)
        need = ~tgt_alive
        new_tgt = tstar.argmin(dim=1)
        self.eg_tgt = torch.where(need, new_tgt, self.eg_tgt)
        self.eg_frozen = torch.where(need, torch.zeros_like(self.eg_frozen), self.eg_frozen)
        ti = self.eg_tgt                                                # (B,)
        rows = torch.arange(B, device=d)
        # current min-image to committed target
        cdx = minimg(bpx[rows, ti] - px.squeeze(1), PX)
        cdy = minimg(bpy[rows, ti] - py.squeeze(1), PY)
        curdist = torch.sqrt(cdx * cdx + cdy * cdy)
        # aim from the scan for the committed target (earliest feasible delta)
        fi = first[rows, ti].clamp(max=T - 1)
        aimx = ddx[rows, ti, fi]; aimy = ddy[rows, ti, fi]
        has_feas = first[rows, ti] < T
        # perpendicular cut-off fallback when no feasible root
        bs = torch.sqrt(bvx[rows, ti] ** 2 + bvy[rows, ti] ** 2).clamp(min=1e-6)
        ux = bvx[rows, ti] / bs; uy = bvy[rows, ti] / bs
        along = cdx * ux + cdy * uy
        pxk = cdx - along * ux; pyk = cdy - along * uy
        aimx = torch.where(has_feas, aimx, pxk); aimy = torch.where(has_feas, aimy, pyk)
        # freeze: keep stored aim when committed inside the bubble
        keep = (curdist < FREEZE_R) & self.eg_frozen
        self.eg_aim[:, 0] = torch.where(keep, self.eg_aim[:, 0], aimx)
        self.eg_aim[:, 1] = torch.where(keep, self.eg_aim[:, 1], aimy)
        self.eg_frozen = torch.where(~keep, curdist < FREEZE_R, self.eg_frozen)
        # steer: desired = unit(aim)*sM ; force = limit(desired - vel, PMF)
        am = torch.sqrt(self.eg_aim[:, 0] ** 2 + self.eg_aim[:, 1] ** 2).clamp(min=1e-9)
        desx = self.eg_aim[:, 0] / am * sM; desy = self.eg_aim[:, 1] / am * sM
        fx2 = desx - self.pred_vel[:, 0]; fy2 = desy - self.pred_vel[:, 1]
        fm = torch.sqrt(fx2 * fx2 + fy2 * fy2)
        sc = torch.where(fm > PREDATOR_MAX_FORCE, PREDATOR_MAX_FORCE / fm.clamp(min=1e-9), torch.ones_like(fm))
        nvx = self.pred_vel[:, 0] + fx2 * sc; nvy = self.pred_vel[:, 1] + fy2 * sc
        spd = torch.sqrt(nvx * nvx + nvy * nvy)
        vsc = torch.where(spd > sM, sM / spd.clamp(min=1e-9), torch.ones_like(spd))
        self.pred_vel[:, 0] = nvx * vsc; self.pred_vel[:, 1] = nvy * vsc
        self.pred_pos[:, 0] += self.pred_vel[:, 0]; self.pred_pos[:, 1] += self.pred_vel[:, 1]
        self.pred_pos[:, 0] = torch.where(self.pred_pos[:, 0] > self._wrap_w_max, self._wrap_neg20, self.pred_pos[:, 0])
        self.pred_pos[:, 0] = torch.where(self.pred_pos[:, 0] < self._wrap_neg20, self._wrap_w_max, self.pred_pos[:, 0])
        self.pred_pos[:, 1] = torch.where(self.pred_pos[:, 1] > self._wrap_h_max, self._wrap_neg20, self.pred_pos[:, 1])
        self.pred_pos[:, 1] = torch.where(self.pred_pos[:, 1] < self._wrap_neg20, self._wrap_h_max, self.pred_pos[:, 1])

    def run_clear(self, max_frames):
        d = self.device
        clr = torch.full((self.B,), max_frames, dtype=torch.long, device=d)
        for f in range(max_frames):
            self.step()
            any_alive = self.boid_alive.any(dim=1)
            newly = (clr == max_frames) & ~any_alive
            clr = torch.where(newly, torch.full_like(clr, f + 1), clr)
            if not bool(any_alive.any()):
                break
        cleared = (clr < max_frames)
        return clr.float().mean().item(), cleared.float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--K', type=int, default=1)
    ap.add_argument('--B', type=int, default=2048)
    ap.add_argument('--maxFrames', type=int, default=4000)
    ap.add_argument('--W', type=int, default=390)
    ap.add_argument('--H', type=int, default=844)
    ap.add_argument('--weights', default='weights/predator_weights_v6.json')
    ap.add_argument('--sweep', action='store_true')
    args = ap.parse_args()
    dev = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    st.CANVAS_W = float(args.W); st.CANVAS_H = float(args.H)
    weights = st.load_weights(args.weights, device=dev)
    seeds = list(range(300000, 300000 + args.B))

    def run(teri):
        sim = EndgameSim(seeds, weights, args.K, dev, teri)
        t0 = time.time()
        ttc, cr = sim.run_clear(args.maxFrames)
        return ttc, cr, time.time() - t0

    base = dict(SLACK=0.97, FREEZE_R=110.0, DT=4, TMAX=1400)
    if not args.sweep:
        ttc, cr, el = run(base)
        print(f'TERI K={args.K} B={args.B} {args.W}x{args.H}: TTC={ttc:.0f} clear={cr*100:.1f}%  ({el:.1f}s)')
        return
    # Direct old-vs-new comparison at GPU scale: the shipped TERI (freeze=110, dt=4)
    # vs the new interceptor (no freeze, dt=1), plus two ablations to attribute the gain.
    NAMED = [
        ('old_teri  (fz110 dt4)', dict(SLACK=0.97, FREEZE_R=110.0, DT=4, TMAX=1400)),
        ('new       (fz0   dt1)', dict(SLACK=1.0,  FREEZE_R=0.0,   DT=1, TMAX=1400)),
        ('ablate-dt (fz110 dt1)', dict(SLACK=1.0,  FREEZE_R=110.0, DT=1, TMAX=1400)),
        ('ablate-fz (fz0   dt4)', dict(SLACK=1.0,  FREEZE_R=0.0,   DT=4, TMAX=1400)),
    ]
    print(f'GPU endgame old-vs-new, K={args.K}, B={args.B}, {args.W}x{args.H}:')
    for name, t in NAMED:
        ttc, cr, el = run(t)
        print(f'  {name}: TTC={ttc:.0f} clear={cr*100:.1f}%  ({el:.1f}s)', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Generate the GPU throughput-surface config list (3 split rules x screen matrix),
split across VM1/VM2. ALIGNED to side-b (#6) for the GPU<->JS cross-check:
  - startBoids=28 (late-game scatter: the N=120->28 planner phase is config-identical
    and cancels; this isolates the gate-discriminating regime ~4x faster on fp64-L4).
  - count   : enter endgame when N <= T                         (param T)
  - density : area-scaled count Td = round(Tref * A/A_ref), A=(W+20)(Hc+20)  (param Tref)
  - horizon : enter when soonest boid's analytic reach-time wa0 > H  (param H frames)
  - seeds from 270000 (side-b's space).
Run: gate_sweep_gpu.py --configs FILE --packed (all params of a (screen,rule) -> 1 sim;
early-exit stops when every env is extinct, so the generous frame caps just bound the
slowest seed).
"""
import json

START_BOIDS = 28
# screen matrix: cell, frames-cap (~1.5x slowest extinction of a 28-boid late game;
# early-exit trims small screens, so over-budget is cheap).
SCREENS = [
    ('390x844', 5000),    ('414x896', 5000),
    ('768x1024', 8000),   ('820x1180', 9000),
    ('1280x800', 8000),   ('1440x900', 10000),
    ('1512x982', 11000),  ('1680x1050', 12000),
    ('1920x1080', 14000), ('2560x1440', 18000),
]
COUNT_T = [3, 4, 5, 6, 7, 8, 10, 12]      # per-screen T* locator (defs match JS+side-b)
DENS_TREF = [3, 5, 7, 10]                 # area-scaled count refs (the auto-capture candidate)
HORIZON_H = [40, 90, 140]                 # reach-time thresholds (side-b sweeps 40,90)
SEEDS = 32   # keep count group (8 params) at 8*32=256 envs = the L4 fp64 latency-bound
             # sweet spot (~36fps); 512 envs saturates -> ~5fps -> 10x slower. 32 paired
             # seeds is ample for a coarse RELATIVE map (side-b's decisive farm + JS does CIs).
SEED0 = 270000

def cfgs_for(cell, frames):
    out = []
    for T in COUNT_T:
        out.append(dict(cell=f'{cell}:{START_BOIDS}', rule='count', param=T, seeds=SEEDS, seed0=SEED0, frames=frames))
    for tr in DENS_TREF:
        out.append(dict(cell=f'{cell}:{START_BOIDS}', rule='density', param=tr, seeds=SEEDS, seed0=SEED0, frames=frames))
    for h in HORIZON_H:
        out.append(dict(cell=f'{cell}:{START_BOIDS}', rule='horizon', param=h, seeds=SEEDS, seed0=SEED0, frames=frames))
    return out

vm1, vm2 = [], []
for i, (cell, frames) in enumerate(SCREENS):
    (vm1 if i % 2 == 0 else vm2).extend(cfgs_for(cell, frames))

json.dump(vm1, open('cfgs_vm1.json', 'w'))
json.dump(vm2, open('cfgs_vm2.json', 'w'))
ngroups = lambda L: len({(c['cell'], c['rule']) for c in L})
print(f'vm1={len(vm1)} configs ({ngroups(vm1)} packed groups), vm2={len(vm2)} configs ({ngroups(vm2)} packed groups)')
print(f'screens={len(SCREENS)} count={len(COUNT_T)} density={len(DENS_TREF)} horizon={len(HORIZON_H)} startBoids={START_BOIDS}')

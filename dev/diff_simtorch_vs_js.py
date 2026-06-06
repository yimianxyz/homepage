"""Inject an IDENTICAL state (dumped from the real boid.js game by
diff_rollout_vs_game.js --dump) into sim_torch, run the SAME fixed-target game,
and compare per-frame predator pos + catches. This isolates whether sim_torch's
GAME (the 14.64 source) diverges from boid.js's GAME (the 7.0 deploy truth) in
the catch regime -- the root cause of the GPU<->browser 2x gap.
  node diff_rollout_vs_game.js --seed S --warm W --H H --bigsize --dump /tmp/st.json
  python3 diff_simtorch_vs_js.py /tmp/st.json
"""
import sys, json
import numpy as np
import torch
import sim_torch as st
from sim_torch import Sim
import planner_probe as pp

st.FAST_TWO_PASS = False
dev = 'cpu'
pp.WEIGHTS = st.load_weights('../js/predator_weights.json', device=dev)

J = json.load(open(sys.argv[1]))
init, trajB, T = J['init'], J['trajB'], J['T']
H = J['H']
nlive = len(init['bx'])
print(f"seed={J['seed']} warm={J['warm']} H={H} nlive={nlive} "
      f"CANVAS_W={st.CANVAS_W} catchesB={J['catchesB']}")

sim = Sim(seeds=[J['seed']], weights=pp.WEIGHTS, device=dev,
          auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=True)
N = sim.N
# inject the exact JS state (alive boids 0..nlive-1, rest dead)
sim.boid_alive[0, :] = False
sim.boid_alive[0, :nlive] = True
sim.boid_pos[0, :nlive, 0] = torch.tensor(init['bx'], dtype=torch.float64)
sim.boid_pos[0, :nlive, 1] = torch.tensor(init['by'], dtype=torch.float64)
sim.boid_vel[0, :nlive, 0] = torch.tensor(init['bvx'], dtype=torch.float64)
sim.boid_vel[0, :nlive, 1] = torch.tensor(init['bvy'], dtype=torch.float64)
sim.pred_pos[0, 0] = init['px']; sim.pred_pos[0, 1] = init['py']
sim.pred_vel[0, 0] = init['pvx']; sim.pred_vel[0, 1] = init['pvy']
sim.pred_size[0] = init['psize']
sim.pred_last_feed_ms[0] = init['lastFeed']
sim._frame_ms = torch.tensor(float(init['nowMs']), dtype=torch.float64, device=dev)
sim.frame = J['warm']
sim.catches[0] = 0

Tt = torch.tensor([[T['x'], T['y']]], dtype=torch.float64, device=dev)
rows = []
for f in range(H):
    pp._step_with_target(sim, Tt)
    rows.append(dict(px=float(sim.pred_pos[0, 0]), py=float(sim.pred_pos[0, 1]),
                     sz=float(sim.pred_size[0]), c=int(sim.catches[0])))

# compare
first_pos_div = -1
first_catch_div = -1
for f in range(H):
    a, b = rows[f], trajB[f]
    dp = abs(a['px'] - b['px']) + abs(a['py'] - b['py'])
    if first_pos_div < 0 and dp > 1e-4:
        first_pos_div = f
    if first_catch_div < 0 and a['c'] != b['c']:
        first_catch_div = f
catchesST = rows[-1]['c']
print(f"catchesST(sim_torch)={catchesST}  catchesB(boid.js)={trajB[-1]['c']}  "
      f"firstPosDiv={first_pos_div}  firstCatchDiv={first_catch_div}")
lo = max(0, (H - 8) if first_pos_div < 0 else first_pos_div - 2)
hi = min(H, lo + 12)
print("frame |  ST.px   ST.py  ST.c |  JS.px   JS.py  JS.c | dpos")
for f in range(lo, hi):
    a, b = rows[f], trajB[f]
    dp = abs(a['px'] - b['px']) + abs(a['py'] - b['py'])
    print(f"{f:5d} | {a['px']:7.2f} {a['py']:7.2f} {a['c']:4d} | "
          f"{b['px']:7.2f} {b['py']:7.2f} {b['c']:4d} | {dp:.3e}")

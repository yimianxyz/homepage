"""Compare GPU _candidate_targets vs the browser candidates() on the IDENTICAL
injected state (dumped with planDbg.cands). The decision diff flagged a 1356-unit
per-candidate mismatch -> find exactly which candidates differ and why (sort order,
lead adjustment, dead-boid handling, wraparound)."""
import sys, json, os
import torch
import sim_torch as st
from sim_torch import Sim
import planner_probe as pp

dev = 'cpu'
pp.WEIGHTS = st.load_weights('../js/predator_weights.json', device=dev)
J = json.load(open(sys.argv[1])); init = J['init']; jc = J['planDbg']['cands']
nlive = len(init['bx'])
sim = Sim(seeds=[J['seed']], weights=pp.WEIGHTS, device=dev,
          auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=True)
sim.boid_alive[0, :] = False; sim.boid_alive[0, :nlive] = True
sim.boid_pos[0, :nlive, 0] = torch.tensor(init['bx'], dtype=torch.float64)
sim.boid_pos[0, :nlive, 1] = torch.tensor(init['by'], dtype=torch.float64)
sim.boid_vel[0, :nlive, 0] = torch.tensor(init['bvx'], dtype=torch.float64)
sim.boid_vel[0, :nlive, 1] = torch.tensor(init['bvy'], dtype=torch.float64)
sim.pred_pos[0, 0] = init['px']; sim.pred_pos[0, 1] = init['py']
sim.pred_vel[0, 0] = init['pvx']; sim.pred_vel[0, 1] = init['pvy']
sim.pred_size[0] = init['psize']
gc = pp._candidate_targets(sim, 16)[0].tolist()
print(f"nlive={nlive} pred=({init['px']:.3f},{init['py']:.3f})")
print(" k |    GPU.x    GPU.y |     JS.x     JS.y |  |dx|+|dy|")
for k in range(16):
    d = abs(gc[k][0]-jc[k][0]) + abs(gc[k][1]-jc[k][1])
    flag = '  <-- DIFF' if d > 1.0 else ''
    print(f"{k:2d} | {gc[k][0]:9.3f} {gc[k][1]:9.3f} | {jc[k][0]:9.3f} {jc[k][1]:9.3f} | {d:9.3f}{flag}")

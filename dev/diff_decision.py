"""Compare ONE cheap-policy decision between sim_torch (run_value_lookahead_cheap)
and the browser (planCheap), on the IDENTICAL injected state dumped by
diff_rollout_vs_game.js --dump (which includes planDbg = the JS decision internals).
Pinpoints which part of the decision diverges: candidates, ballistic top1, rollout
catches, value prior, final score, or pick."""
import sys, json, os
import torch
import sim_torch as st
from sim_torch import Sim
import planner_probe as pp
import feat_planner as fp
from eval_value import Deploy

st.FAST_TWO_PASS = False
dev = 'cpu'
pp.WEIGHTS = st.load_weights('../js/predator_weights.json', device=dev)
J = json.load(open(sys.argv[1]))
init, dbg = J['init'], J['planDbg']
nlive = len(init['bx'])
_np = next(p for p in ['/tmp/net_strict.pt', os.path.expanduser('~/net_strict.pt')] if os.path.exists(p))
model = Deploy(torch.load(_np, map_location='cpu'), dev)

sim = Sim(seeds=[J['seed']], weights=pp.WEIGHTS, device=dev,
          auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=True)
sim.boid_alive[0, :] = False; sim.boid_alive[0, :nlive] = True
sim.boid_pos[0, :nlive, 0] = torch.tensor(init['bx'], dtype=torch.float64)
sim.boid_pos[0, :nlive, 1] = torch.tensor(init['by'], dtype=torch.float64)
sim.boid_vel[0, :nlive, 0] = torch.tensor(init['bvx'], dtype=torch.float64)
sim.boid_vel[0, :nlive, 1] = torch.tensor(init['bvy'], dtype=torch.float64)
sim.pred_pos[0, 0] = init['px']; sim.pred_pos[0, 1] = init['py']
sim.pred_vel[0, 0] = init['pvx']; sim.pred_vel[0, 1] = init['pvy']
sim.pred_size[0] = init['psize']; sim.pred_last_feed_ms[0] = init['lastFeed']
sim._frame_ms = torch.tensor(float(init['nowMs']), dtype=torch.float64, device=dev)
sim.frame = J['warm']

K, D, Hs, roll_M = 16, 8, 60, 120
B = 1; rows = torch.arange(B)
cand = pp._candidate_targets(sim, K)
base = pp._save_state(sim)
roll = Sim(seeds=list(range(K)), weights=pp.WEIGHTS, device=dev,
           auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=True)
pp._load_state(roll, pp._tile_state(base, K))
roll_tgt = cand.reshape(B * K, 2).contiguous()
frozen = ~roll.boid_alive
c0 = roll.catches.clone()
for _ in range(Hs):
    sp = roll.boid_pos.clone(); sv = roll.boid_vel.clone()
    roll._step_boids()
    roll.boid_pos[frozen] = sp[frozen]; roll.boid_vel[frozen] = sv[frozen]
    pp._analytic_steer(roll, roll_tgt)
    roll._check_catches(); roll._decay_size(); roll.frame += 1; roll._frame_ms += st.FRAME_MS
c_near = (roll.catches - c0).reshape(B, K).float()
tcand = pp._candidate_targets(roll, K)
tfeat, tctx = fp.candidate_features(roll, tcand)
with torch.no_grad():
    tv = model(tfeat, tctx)
roll_score = c_near + tv.max(dim=1).values.reshape(B, K)
f0, x0 = fp.candidate_features(sim, cand)
with torch.no_grad():
    vprior = model(f0, x0)
pscore = (f0[:, :, 18] - f0[:, :, 16])
top1 = int(pscore.argmax(dim=1)[0])
thr = torch.topk(pscore, 1, dim=1).values[:, -1:]
is_top = pscore >= thr
score = torch.where(is_top, roll_score, vprior)
pick = int(score.argmax(dim=1)[0])

cand_np = cand[0].tolist()
print(f"=== state: nlive={nlive} pred=({init['px']:.2f},{init['py']:.2f}) ===")
# candidate position diff
maxcd = max(abs(cand_np[k][0]-dbg['cands'][k][0])+abs(cand_np[k][1]-dbg['cands'][k][1]) for k in range(K))
print(f"max candidate-position |GPU-JS| = {maxcd:.4e}")
print(f"ballistic top1: GPU={top1} JS={dbg['top1']}   pick: GPU={pick} JS={dbg['pick']}")
print(f"rollout catches top1: GPU c_near[{top1}]={float(c_near[0,top1]):.0f}  JS rr.catches={dbg['rrCatches']}")
print(f"boot(terminal max V): GPU={float(tv.max()):.4f}  JS={dbg['boot']:.4f}")
# pscore + vprior + score per candidate
print(" k | GPUpsc  JSpsc | GPUvpr  JSvpr | GPUscore JSscore")
for k in range(K):
    print(f"{k:2d} | {float(pscore[0,k]):+.3f} {dbg['pscore'][k]:+.3f} | "
          f"{float(vprior[0,k]):+.3f} {dbg['vprior'][k]:+.3f} | "
          f"{float(score[0,k]):+.3f} {dbg['score'][k]:+.3f}")
print(f"feat0 GPU={[round(float(x),3) for x in f0[0,0,:8]]}")
print(f"feat0 JS ={[round(x,3) for x in dbg['feat0'][:8]]}")

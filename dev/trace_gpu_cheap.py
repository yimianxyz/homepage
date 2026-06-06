"""Per-frame trace of the GPU cheap policy (seed 200000) to diff vs the JS browser.
Logs frame, predator (x,y), cumulative catches, and the chosen target each decision.
Mirrors run_value_lookahead_cheap (K_roll=1, prune=ball, Hs=60, roll_M=120)."""
import sys, json
import torch
import planner_probe as pp
import sim_torch as st
from sim_torch import Sim
import feat_planner as fp
from eval_value import Deploy

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 160
K, D, Hs, roll_M = 16, 8, 60, 120
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
pp.WEIGHTS = st.load_weights('../js/predator_weights.json', device=dev)
pp.TWO_PASS = True
st.FAST_TWO_PASS = False
import os
_np = next(p for p in ['/tmp/net_strict.pt', os.path.expanduser('~/net_strict.pt')] if os.path.exists(p))
model = Deploy(torch.load(_np, map_location='cpu'), dev)

sim = Sim(seeds=[SEED], weights=pp.WEIGHTS, device=dev, auto_target='evolved',
          auto_target_opts=dict(pp.E3D), two_pass=pp.TWO_PASS)
roll = Sim(seeds=list(range(K)), weights=pp.WEIGHTS, device=dev, auto_target='evolved',
           auto_target_opts=dict(pp.E3D), two_pass=pp.TWO_PASS)
B = 1
rows = torch.arange(B)
held = None
out = []
for f in range(FRAMES):
    if f % D == 0:
        cand = pp._candidate_targets(sim, K)
        base = pp._save_state(sim)
        pp._load_state(roll, pp._tile_state(base, K))
        roll_tgt = cand.reshape(B * K, 2).contiguous()
        frozen = ~roll.boid_alive  # roll_M=120 => all active; frozen=dead only
        c0 = roll.catches.clone()
        for _ in range(Hs):
            sp = roll.boid_pos.clone(); sv = roll.boid_vel.clone()
            roll._step_boids()
            roll.boid_pos[frozen] = sp[frozen]; roll.boid_vel[frozen] = sv[frozen]
            pp._analytic_steer(roll, roll_tgt)
            roll._check_catches()
            roll._decay_size(); roll.frame += 1; roll._frame_ms += st.FRAME_MS
        c_near = (roll.catches - c0).reshape(B, K).float()
        f0, x0 = fp.candidate_features(sim, cand)
        with torch.no_grad():
            vprior = model(f0, x0)
        tcand = pp._candidate_targets(roll, K)
        tfeat, tctx = fp.candidate_features(roll, tcand)
        with torch.no_grad():
            tv = model(tfeat, tctx)
        roll_score = c_near + tv.max(dim=1).values.reshape(B, K)
        pscore = (f0[:, :, 18] - f0[:, :, 16])
        thr = torch.topk(pscore, 1, dim=1).values[:, -1:]
        is_top = pscore >= thr
        score = torch.where(is_top, roll_score, vprior)
        pick = int(score.argmax(dim=1)[0])
        held = cand[rows, score.argmax(dim=1)]
        dec = dict(pick=pick, tgt=[round(float(held[0, 0]), 1), round(float(held[0, 1]), 1)])
    else:
        dec = None
    pp._step_with_target(sim, held)
    out.append(dict(f=f, px=round(float(sim.pred_pos[0, 0]), 4), py=round(float(sim.pred_pos[0, 1]), 4),
                    pvx=round(float(sim.pred_vel[0, 0]), 4), pvy=round(float(sim.pred_vel[0, 1]), 4),
                    b0x=round(float(sim.boid_pos[0, 0, 0]), 4), b0y=round(float(sim.boid_pos[0, 0, 1]), 4),
                    sz=round(float(sim.pred_size[0]), 2), catches=int(sim.catches[0])))
for r in out:
    print(json.dumps(r))

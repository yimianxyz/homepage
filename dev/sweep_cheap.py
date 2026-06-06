"""Sweet-spot search for the HONEST cheap policy (tie-bug fixed in
feat_planner.run_value_lookahead_cheap). Sweeps prune_by x K_roll x Hs and prints
the mean catches @ given frames/seeds on the faithful GPU proxy. Cost per decision
~ K_roll x Hs rollout-frames (the browser budget knob).
  python3 sweep_cheap.py --prune ball,mindist,ballmin,v,vmin --kroll 1,2,3,4 \
      --Hs 60 --seed0 200000 --nseed 64 --frames 1500
"""
import sys, os, argparse, time
import torch
import sim_torch as st
import planner_probe as pp
import feat_planner as fp
from eval_value import Deploy

st.FAST_TWO_PASS = False
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
_wp = '../js/predator_weights.json' if os.path.exists('../js/predator_weights.json') else os.path.expanduser('~/js_eval/js/predator_weights.json')
pp.WEIGHTS = st.load_weights(_wp, device=dev)
_np = next(p for p in ['/tmp/net_strict.pt', os.path.expanduser('~/net_strict.pt')] if os.path.exists(p))
model = Deploy(torch.load(_np, map_location='cpu'), dev)

ap = argparse.ArgumentParser()
ap.add_argument('--prune', default='ball')
ap.add_argument('--kroll', default='1')
ap.add_argument('--Hs', default='60')
ap.add_argument('--seed0', type=int, default=200000)
ap.add_argument('--nseed', type=int, default=64)
ap.add_argument('--frames', type=int, default=1500)
ap.add_argument('--D', default='8')
ap.add_argument('--no_value', action='store_true')
a = ap.parse_args()
seeds = list(range(a.seed0, a.seed0 + a.nseed))
prunes = a.prune.split(',')
krolls = [int(x) for x in a.kroll.split(',')]
Hss = [int(x) for x in a.Hs.split(',')]
Ds = [int(x) for x in a.D.split(',')]
PLANNER = 18.75  # GPU strict ceiling (n=64) — for %-of-planner

print(f"# dev={dev} seeds={a.seed0}..{a.seed0+a.nseed-1} frames={a.frames} no_value={a.no_value}")
print(f"{'prune':>8} {'Kroll':>5} {'Hs':>4} {'D':>3} {'cost/f':>7} {'mean':>7} {'SE':>5} {'%plan':>6}")
for prune in prunes:
    for K_roll in krolls:
        for Hs in Hss:
            for D in Ds:
                t0 = time.time()
                c = fp.run_value_lookahead_cheap(seeds, a.frames, dev, model, K=16, D=D,
                        Hs=Hs, roll_M=120, K_roll=K_roll, prune_by=prune, no_value=a.no_value)
                m = float(c.mean()); se = float(c.std() / (len(c) ** 0.5))
                costf = K_roll * Hs / D   # rollout-frames per game-frame (browser cost)
                print(f"{prune:>8} {K_roll:>5} {Hs:>4} {D:>3} {costf:>7.1f} {m:>7.2f} {se:>5.2f} {100*m/PLANNER:>5.0f}%  ({time.time()-t0:.0f}s)", flush=True)

"""Head-to-head: run the SAME cheap policy (K16/D8/Hs60/roll_M120/K_roll1/ball)
in sim_torch on the SAME seeds the JS production harness uses, so we can compare
GPU-cheap vs boid.js-cheap per seed. If they match, the historical 14.64-vs-7.0
gap was an apples-to-oranges artifact; if GPU catches ~2x more on identical
seeds, there is a real execution divergence to localize."""
import sys, torch
import sim_torch as st
import planner_probe as pp
import feat_planner as fp
from eval_value import Deploy

import os
st.FAST_TWO_PASS = False
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
_wp = '../js/predator_weights.json' if os.path.exists('../js/predator_weights.json') else os.path.expanduser('~/js_eval/js/predator_weights.json')
pp.WEIGHTS = st.load_weights(_wp, device=dev)
seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 4
frames = int(sys.argv[3]) if len(sys.argv) > 3 else 600
seeds = list(range(seed0, seed0 + nseed))
_np = next(p for p in ['/tmp/net_strict.pt', os.path.expanduser('~/net_strict.pt')] if os.path.exists(p))
model = Deploy(torch.load(_np, map_location='cpu'), dev)
Kroll = int(sys.argv[4]) if len(sys.argv) > 4 else 1
prune = sys.argv[5] if len(sys.argv) > 5 else 'ball'
c = fp.run_value_lookahead_cheap(seeds, frames, dev, model, K=16, D=8, Hs=60,
                                 roll_M=120, K_roll=Kroll, prune_by=prune, no_value=False)
print(f'GPU-cheap K_roll={Kroll} prune={prune} per-seed:', [int(x) for x in c], 'mean=', round(float(c.mean()), 2))

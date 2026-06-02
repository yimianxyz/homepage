"""sim_torch side of the boid-trajectory faithfulness check (see twopass_check.js).

Builds one seed's boids, pins the predator at a FIXED position, runs F frames of
boid-only dynamics in 'single' (sequential, = JS render-only) or 'two' (= JS
tick+render) mode, dumps init+final positions. Compared against the JS dump.

  python3 twopass_check.py --seed 200000 --frames 300 --mode two \
      --px 840 --py 840 --weights ../js/predator_weights.json --out /tmp/sim_two.json
"""
import argparse, json
import torch
import sim_torch as st
from sim_torch import Sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=200000)
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--mode', choices=['single', 'two'], default='two')
    ap.add_argument('--px', type=float, default=840.0)
    ap.add_argument('--py', type=float, default=840.0)
    ap.add_argument('--weights', default='../js/predator_weights.json')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    W = st.load_weights(args.weights, device='cpu')
    sim = Sim(seeds=[args.seed], weights=W, device='cpu', two_pass=True)
    # Pin predator (boid avoidance reads sim.pred_pos); never stepped here.
    sim.pred_pos[:, 0] = args.px
    sim.pred_pos[:, 1] = args.py

    init = sim.boid_pos[0].clone().cpu().tolist()
    step = sim._step_boids_twopass if args.mode == 'two' else sim._step_boids_sequential
    for _ in range(args.frames):
        step()
        sim.pred_pos[:, 0] = args.px      # keep predator pinned
        sim.pred_pos[:, 1] = args.py
    final = sim.boid_pos[0].cpu().tolist()
    res = dict(mode=args.mode, seed=args.seed, frames=args.frames,
               px=args.px, py=args.py, numBoids=sim.N, init=init, final=final)
    s = json.dumps(res)
    if args.out:
        open(args.out, 'w').write(s)
    else:
        print(s)


if __name__ == '__main__':
    main()

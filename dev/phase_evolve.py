"""Phase-conditional evolved-patrol search (situation decomposition by live-boid
count). The game flows through phases — 120-boid herding, small-N chase, 1-boid
endgame — and production uses ONE evolved-patrol param set for all of them. This
CMA-ES searches a SEPARATE param set per phase (selected per-frame by the live
count), a strict superset of the single global set, to measure how much headroom
phase-specialization buys over the global E3D baseline (8.34 catches/1500fr).

The whole population is one batched sim (pop*S envs); each generation restores
the cached init state, writes per-env phase-param tensors, and replays.

  python3 phase_evolve.py --gens 60 --pop 24 --seeds 192 --seedStart 30000 \
      --edges 2,8,25,70 --sigma0 0.15 --seed 1 --out ~/situ/ckpt/ph1
"""
import argparse, json, time, os, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_torch import Sim, load_weights

WEIGHTS = 'js/predator_weights.json'

# name, lo, hi, E3D-production-init
PARAMS = [
    ('cluster_r',    40.0,  400.0,  178.09),
    ('dens_pow',      0.0,    4.0,    2.373),
    ('reach_scale', 150.0, 6000.0, 1515.0),
    ('sharp',         0.5,   25.0,    9.25),
    ('lead_scale',    0.0,    1.5,    0.454),
    ('lead_max',      0.0,  300.0,  230.6),
    ('nbhd',          0.0,    1.0,    0.461),
]
NAMES = [p[0] for p in PARAMS]
K = len(PARAMS)
STATE = ['boid_pos', 'boid_vel', 'boid_alive', 'pred_pos', 'pred_vel',
         'pred_size', 'pred_auto', 'pred_last_feed_ms', 'catches', '_frame_ms']


def main():
    import cma
    ap = argparse.ArgumentParser()
    ap.add_argument('--gens', type=int, default=60)
    ap.add_argument('--pop', type=int, default=24)
    ap.add_argument('--seeds', type=int, default=192)
    ap.add_argument('--frames', type=int, default=1500)
    ap.add_argument('--seedStart', type=int, default=30000)
    ap.add_argument('--edges', default='2,8,25,70')  # P-1 ascending live-count thresholds
    ap.add_argument('--sigma0', type=float, default=0.15)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'log.jsonl', 'a')
    def log(o): print(json.dumps(o), flush=True); logf.write(json.dumps(o) + '\n'); logf.flush()

    dev = a.device
    pop, S = a.pop, a.seeds
    B = pop * S
    edges = [float(x) for x in a.edges.split(',')]
    P = len(edges) + 1
    dim = P * K

    seeds = [a.seedStart + s for c in range(pop) for s in range(S)]
    w = load_weights(WEIGHTS, device=dev)
    sim = Sim(seeds=seeds, weights=w, device=dev, sequential=False,
              auto_target='evolved', auto_target_opts={})
    # per-env phase-param tensor, written in place each gen
    pp = torch.zeros((B, P, K), dtype=torch.float64, device=dev)
    sim.auto_target_opts = {
        '_phase_params': pp,
        '_phase_edges': edges,
        '_phase_names': NAMES,
    }
    init_state = {k: getattr(sim, k).clone() for k in STATE}

    lo = np.array([p[1] for p in PARAMS]); hi = np.array([p[2] for p in PARAMS])
    init = np.array([p[3] for p in PARAMS])
    x0 = np.tile((init - lo) / (hi - lo), P)  # all phases start at E3D

    def decode(xrow):
        # xrow: (dim,) in [0,1] -> (P,K) physical params
        g = np.clip(xrow, 0, 1).reshape(P, K)
        return lo + g * (hi - lo)

    def set_params(X):
        # X: (pop, dim). Build (B,P,K) by repeating each candidate over S seeds.
        phys = np.stack([decode(X[c]) for c in range(pop)], axis=0)  # (pop,P,K)
        t = torch.tensor(np.repeat(phys, S, axis=0), dtype=torch.float64, device=dev)
        pp.copy_(t)

    def restore():
        for k in STATE:
            getattr(sim, k).copy_(init_state[k])
        sim.frame = 0

    def eval_pop(X):
        set_params(X); restore()
        for _ in range(a.frames):
            sim.step()
        return sim.catches.float().view(pop, S).mean(dim=1).cpu().numpy()

    es = cma.CMAEvolutionStrategy(list(x0), a.sigma0, {
        'bounds': [0.0, 1.0], 'popsize': pop, 'seed': a.seed, 'maxiter': a.gens,
        'verbose': -9})
    log({'phase': 'start', 'P': P, 'edges': edges, 'names': NAMES, 'dim': dim,
         'pop': pop, 'seeds': S, 'frames': a.frames, 'seedStart': a.seedStart, 'B': B})
    best_f = -1.0; best = None
    for gen in range(a.gens):
        t0 = time.time()
        Xl = es.ask(); X = np.array(Xl)
        catch = eval_pop(X)
        es.tell(Xl, [-c for c in catch])
        gi = int(np.argmax(catch))
        if catch[gi] > best_f:
            best_f = float(catch[gi])
            best = {NAMES[j]: [float(decode(X[gi])[p, j]) for p in range(P)] for j in range(K)}
            json.dump({'mean_catches': best_f, 'edges': edges, 'P': P,
                       'phase_params': best, 'gen': gen, 'seeds': S, 'frames': a.frames},
                      open(out / 'best.json', 'w'), indent=2)
        log({'gen': gen, 'gen_best': float(catch[gi]), 'gen_mean': float(np.mean(catch)),
             'best_so_far': best_f, 'gen_s': round(time.time() - t0, 1)})
    log({'phase': 'done', 'best': best_f})


if __name__ == '__main__':
    main()

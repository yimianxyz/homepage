"""Batched CMA-ES over the 'evolved' patrol params. The whole population is
evaluated in ONE sim (pop*S envs) per generation via per-env param tensors, so
the per-frame kernel-launch overhead is amortized across all candidates — the
sim is overhead-bound at small batch, so this is ~10-20x faster than evaluating
candidates one at a time.

The sim is built once (one RNG init); each generation restores the cached
initial state, writes the population's params into the per-env tensors, and
replays. AlphaEvolve flavor: structural mutations to the 'evolved' branch happen
between runs; this drives the continuous parameter search.

  python3 dev/evolve_patrol_batched.py --gens 200 --pop 32 --seeds 192 \
      --frames 1500 --seedStart 6000 --device cuda --seed 1 --out ~/ckpt/ev1
"""
import argparse, json, time, os, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_torch import Sim, load_weights

WEIGHTS = 'js/predator_weights.json'
PARAMS = [
    ('cluster_r',   40.0,  400.0, 150.0),
    ('dens_pow',     0.0,    4.0,   1.0),
    ('reach_scale',150.0, 6000.0,2000.0),
    ('sharp',        0.5,   25.0,   6.0),
    ('lead_scale',   0.0,    1.5,   0.4),
    ('lead_max',     0.0,  300.0, 120.0),
]
# state tensors that step() mutates — cached after init, restored each gen
STATE = ['boid_pos', 'boid_vel', 'boid_alive', 'pred_pos', 'pred_vel',
         'pred_size', 'pred_auto', 'pred_last_feed_ms', 'catches', '_frame_ms']


def x0_norm():
    return [(init - lo) / (hi - lo) for (_, lo, hi, init) in PARAMS]


def main():
    import cma
    p = argparse.ArgumentParser()
    p.add_argument('--gens', type=int, default=200)
    p.add_argument('--pop', type=int, default=32)
    p.add_argument('--seeds', type=int, default=192)
    p.add_argument('--frames', type=int, default=1500)
    p.add_argument('--seedStart', type=int, default=6000)
    p.add_argument('--sigma0', type=float, default=0.25)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'evolve_log.jsonl', 'a')
    def log(o): print(json.dumps(o), flush=True); logf.write(json.dumps(o) + '\n'); logf.flush()
    dev = a.device
    pop, S = a.pop, a.seeds
    B = pop * S
    # seed layout: env c*S+s uses unique seed (seedStart+s); each candidate sees
    # the SAME S seeds -> fair comparison.
    seeds = [a.seedStart + s for c in range(pop) for s in range(S)]
    w = load_weights(WEIGHTS, device=dev)
    sim = Sim(seeds=seeds, weights=w, device=dev, sequential=False,
              auto_target='evolved', auto_target_opts={})
    # per-env param tensors (written in place each gen)
    ptens = {name: torch.zeros(B, dtype=torch.float64, device=dev) for name, *_ in PARAMS}
    sim.auto_target_opts = ptens
    # cache initial state
    init_state = {k: getattr(sim, k).clone() for k in STATE}

    def set_params(X):
        # X: (pop, 6) in [0,1]; write per-env param tensors
        for j, (name, lo, hi, _) in enumerate(PARAMS):
            vals = lo + np.clip(X[:, j], 0, 1) * (hi - lo)   # (pop,)
            t = torch.tensor(np.repeat(vals, S), dtype=torch.float64, device=dev)
            ptens[name].copy_(t)

    def restore():
        for k in STATE:
            getattr(sim, k).copy_(init_state[k])
        sim.frame = 0

    def eval_pop(X):
        set_params(X)
        restore()
        for _ in range(a.frames):
            sim.step()
        c = sim.catches.float().view(pop, S).mean(dim=1).cpu().numpy()
        return c

    es = cma.CMAEvolutionStrategy(x0_norm(), a.sigma0, {
        'bounds': [0.0, 1.0], 'popsize': pop, 'seed': a.seed, 'maxiter': a.gens,
        'verbose': -9})
    log({'phase': 'start', 'params': [n for n, *_ in PARAMS], 'pop': pop,
         'seeds': S, 'frames': a.frames, 'seedStart': a.seedStart, 'B': B})
    best_f = -1.0; best_opts = None
    for gen in range(a.gens):
        t0 = time.time()
        Xl = es.ask()
        X = np.array(Xl)
        catch = eval_pop(X)
        es.tell(Xl, [-c for c in catch])
        gi = int(np.argmax(catch))
        if catch[gi] > best_f:
            best_f = float(catch[gi])
            best_opts = {name: float(lo + np.clip(X[gi, j], 0, 1) * (hi - lo))
                         for j, (name, lo, hi, _) in enumerate(PARAMS)}
            json.dump({'mean_catches': best_f, 'opts': best_opts, 'gen': gen,
                       'seeds': S, 'frames': a.frames}, open(out / 'best.json', 'w'), indent=2)
        log({'gen': gen, 'gen_best': float(catch[gi]), 'gen_mean': float(np.mean(catch)),
             'best_so_far': best_f, 'best_opts': best_opts, 'gen_s': round(time.time() - t0, 1)})
    log({'phase': 'done', 'best': best_f, 'best_opts': best_opts})


if __name__ == '__main__':
    main()

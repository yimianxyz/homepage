"""CMA-ES search over the 'evolved' patrol-target params, GPU-evaluated through
the deployed NN chase. The patrol target drove every historical +30-41% win;
this searches a richer family than nearest_cluster (adds reachability + a
sharpness knob). AlphaEvolve flavor: when params converge, new mechanisms are
added to the sim_torch 'evolved' branch and the search restarts wider.

  python3 dev/evolve_patrol.py --gens 120 --pop 24 --seeds 256 --frames 1500 \
      --seedStart 6000 --device cuda --seed 1 --out ~/ckpt/ev1
"""
import argparse, json, time, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sim import evaluate

WEIGHTS = 'js/predator_weights.json'

# param name -> (low, high, init)
PARAMS = [
    ('cluster_r',   40.0,  400.0, 150.0),
    ('dens_pow',     0.0,    4.0,   1.0),
    ('reach_scale',150.0, 6000.0,2000.0),
    ('sharp',        0.5,   25.0,   6.0),
    ('lead_scale',   0.0,    1.5,   0.4),
    ('lead_max',     0.0,  300.0, 120.0),
]


def vec_to_opts(x):
    """x in [0,1]^6 -> param dict (clipped to bounds)."""
    o = {}
    for xi, (name, lo, hi, _) in zip(x, PARAMS):
        o[name] = float(lo + min(max(xi, 0.0), 1.0) * (hi - lo))
    return o


def x0_norm():
    return [(init - lo) / (hi - lo) for (_, lo, hi, init) in PARAMS]


def fitness(x, seeds, frames, device, use_graph):
    opts = vec_to_opts(x)
    r = evaluate(WEIGHTS, seeds=seeds, frames=frames, device=device,
                 use_graph=use_graph, sequential=False,
                 auto_target='evolved', auto_target_opts=opts)
    return r['mean_catches'], opts


def main():
    import cma
    p = argparse.ArgumentParser()
    p.add_argument('--gens', type=int, default=120)
    p.add_argument('--pop', type=int, default=24)
    p.add_argument('--seeds', type=int, default=256)
    p.add_argument('--frames', type=int, default=1500)
    p.add_argument('--seedStart', type=int, default=6000)
    p.add_argument('--sigma0', type=float, default=0.25)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--no_graph', action='store_true')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'evolve_log.jsonl', 'a')
    def log(o): print(json.dumps(o), flush=True); logf.write(json.dumps(o) + '\n'); logf.flush()
    seeds = list(range(a.seedStart, a.seedStart + a.seeds))
    use_graph = (not a.no_graph) and a.device == 'cuda'

    es = cma.CMAEvolutionStrategy(x0_norm(), a.sigma0, {
        'bounds': [0.0, 1.0], 'popsize': a.pop, 'seed': a.seed, 'maxiter': a.gens,
        'verbose': -9})
    log({'phase': 'start', 'params': [n for n, *_ in PARAMS], 'pop': a.pop,
         'seeds': a.seeds, 'frames': a.frames, 'seedStart': a.seedStart})
    best_f = -1.0; best_opts = None
    for gen in range(a.gens):
        t0 = time.time()
        X = es.ask()
        fits = []; catch = []
        for x in X:
            c, opts = fitness(x, seeds, a.frames, a.device, use_graph)
            fits.append(-c); catch.append(c)
        es.tell(X, fits)
        gi = int(np.argmax(catch))
        if catch[gi] > best_f:
            best_f = catch[gi]; best_opts = vec_to_opts(X[gi])
            json.dump({'mean_catches': best_f, 'opts': best_opts, 'gen': gen,
                       'seeds': a.seeds, 'frames': a.frames}, open(out / 'best.json', 'w'), indent=2)
        log({'gen': gen, 'gen_best': catch[gi], 'gen_mean': float(np.mean(catch)),
             'best_so_far': best_f, 'best_opts': best_opts, 'gen_s': time.time() - t0})
    log({'phase': 'done', 'best': best_f, 'best_opts': best_opts})


if __name__ == '__main__':
    main()

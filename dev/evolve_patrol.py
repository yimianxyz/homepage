"""Evolutionary search over the 'evolved' patrol-target params, GPU-evaluated
through the deployed NN chase. The patrol target drove every historical +30-41%
win; this searches a richer family than nearest_cluster (adds reachability + a
sharpness knob). Self-contained CEM optimizer (no external deps) — robust to
the sim's seed noise via elite ranking. AlphaEvolve flavor: new mechanisms are
added to the sim_torch 'evolved' branch between runs; this drives the params.

  python3 dev/evolve_patrol.py --gens 150 --pop 16 --elite 5 --seeds 256 \
      --frames 1500 --seedStart 6000 --device cuda --seed 1 --out ~/ckpt/ev1
"""
import argparse, json, time, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sim import evaluate

WEIGHTS = 'js/predator_weights.json'
# name, low, high, init
PARAMS = [
    ('cluster_r',   40.0,  400.0, 150.0),
    ('dens_pow',     0.0,    4.0,   1.0),
    ('reach_scale',150.0, 6000.0,2000.0),
    ('sharp',        0.5,   25.0,   6.0),
    ('lead_scale',   0.0,    1.5,   0.4),
    ('lead_max',     0.0,  300.0, 120.0),
]
LO = np.array([p[1] for p in PARAMS])
HI = np.array([p[2] for p in PARAMS])
INIT = np.array([p[3] for p in PARAMS])
NAMES = [p[0] for p in PARAMS]


def to_opts(xn):
    """xn in [0,1]^6 -> param dict."""
    v = LO + np.clip(xn, 0, 1) * (HI - LO)
    return {NAMES[i]: float(v[i]) for i in range(len(NAMES))}


def fit(xn, seeds, frames, device, use_graph):
    opts = to_opts(xn)
    r = evaluate(WEIGHTS, seeds=seeds, frames=frames, device=device,
                 use_graph=use_graph, sequential=False,
                 auto_target='evolved', auto_target_opts=opts)
    return r['mean_catches']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gens', type=int, default=150)
    p.add_argument('--pop', type=int, default=16)
    p.add_argument('--elite', type=int, default=5)
    p.add_argument('--seeds', type=int, default=256)
    p.add_argument('--frames', type=int, default=1500)
    p.add_argument('--seedStart', type=int, default=6000)
    p.add_argument('--sigma0', type=float, default=0.30)
    p.add_argument('--sigma_floor', type=float, default=0.02)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--no_graph', action='store_true')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'evolve_log.jsonl', 'a')
    def log(o): print(json.dumps(o), flush=True); logf.write(json.dumps(o) + '\n'); logf.flush()
    rng = np.random.default_rng(a.seed)
    seeds = list(range(a.seedStart, a.seedStart + a.seeds))
    use_graph = (not a.no_graph) and a.device == 'cuda'
    D = len(PARAMS)

    mu = (INIT - LO) / (HI - LO)            # init mean in [0,1]
    sig = np.full(D, a.sigma0)
    best_f = -1.0; best_opts = None
    log({'phase': 'start', 'params': NAMES, 'pop': a.pop, 'elite': a.elite,
         'seeds': a.seeds, 'frames': a.frames, 'seedStart': a.seedStart})
    # always include the current mean as a candidate (elitism on the mean)
    for gen in range(a.gens):
        t0 = time.time()
        X = np.clip(mu + sig * rng.standard_normal((a.pop, D)), 0, 1)
        X[0] = np.clip(mu, 0, 1)            # evaluate the mean itself
        fits = np.array([fit(x, seeds, a.frames, a.device, use_graph) for x in X])
        order = np.argsort(-fits)
        elite = X[order[:a.elite]]
        mu = elite.mean(axis=0)
        sig = np.maximum(elite.std(axis=0), a.sigma_floor)
        gi = int(order[0])
        if fits[gi] > best_f:
            best_f = float(fits[gi]); best_opts = to_opts(X[gi])
            json.dump({'mean_catches': best_f, 'opts': best_opts, 'gen': gen,
                       'seeds': a.seeds, 'frames': a.frames}, open(out / 'best.json', 'w'), indent=2)
        log({'gen': gen, 'gen_best': float(fits[gi]), 'gen_mean': float(fits.mean()),
             'best_so_far': best_f, 'best_opts': best_opts,
             'mu_opts': to_opts(mu), 'sig': sig.tolist(), 'gen_s': round(time.time() - t0, 1)})
    log({'phase': 'done', 'best': best_f, 'best_opts': best_opts})


if __name__ == '__main__':
    main()

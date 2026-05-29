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
    ('nbhd',         0.0,    1.0,   0.5),
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
    p.add_argument('--pop', type=int, default=18)
    p.add_argument('--elite', type=int, default=5)
    p.add_argument('--seeds', type=int, default=256, help='seeds per generation (resampled each gen)')
    p.add_argument('--frames', type=int, default=1500)
    p.add_argument('--poolStart', type=int, default=10000)
    p.add_argument('--poolSize', type=int, default=20000, help='training seed pool (resampled per gen, disjoint from val/gate)')
    p.add_argument('--valStart', type=int, default=5000)
    p.add_argument('--valSeeds', type=int, default=512)
    p.add_argument('--valEvery', type=int, default=4)
    p.add_argument('--sigma0', type=float, default=0.30)
    p.add_argument('--sigma_floor', type=float, default=0.05)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--no_graph', action='store_true')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'evolve_log.jsonl', 'a')
    def log(o): print(json.dumps(o), flush=True); logf.write(json.dumps(o) + '\n'); logf.flush()
    rng = np.random.default_rng(a.seed)
    use_graph = (not a.no_graph) and a.device == 'cuda'
    D = len(PARAMS)
    val_seeds = list(range(a.valStart, a.valStart + a.valSeeds))

    mu = (INIT - LO) / (HI - LO)            # init mean in [0,1]
    sig = np.full(D, a.sigma0)
    best_val = -1.0; best_opts = None
    log({'phase': 'start', 'params': NAMES, 'pop': a.pop, 'elite': a.elite,
         'seeds': a.seeds, 'frames': a.frames, 'pool': [a.poolStart, a.poolStart + a.poolSize],
         'valStart': a.valStart, 'valSeeds': a.valSeeds, 'resample': True})
    for gen in range(a.gens):
        t0 = time.time()
        # fresh seed block each gen -> CEM optimizes EXPECTED catches, not a
        # fixed noisy block (kills the seed-overfitting we saw at gen 0).
        base = int(rng.integers(a.poolStart, a.poolStart + a.poolSize - a.seeds))
        seeds = list(range(base, base + a.seeds))
        X = np.clip(mu + sig * rng.standard_normal((a.pop, D)), 0, 1)
        X[0] = np.clip(mu, 0, 1)            # carry the mean as a candidate
        fits = np.array([fit(x, seeds, a.frames, a.device, use_graph) for x in X])
        order = np.argsort(-fits)
        elite = X[order[:a.elite]]
        mu = elite.mean(axis=0)
        sig = np.maximum(elite.std(axis=0), a.sigma_floor)
        rec = {'gen': gen, 'gen_best': float(fits[order[0]]), 'gen_mean': float(fits.mean()),
               'mu_opts': to_opts(mu), 'sig': [round(s, 3) for s in sig.tolist()],
               'seed_base': base, 'gen_s': round(time.time() - t0, 1)}
        # validate the robust center (mu) on the FIXED held-out block
        if gen % a.valEvery == 0 or gen == a.gens - 1:
            vmu = fit(mu, val_seeds, a.frames, a.device, use_graph)
            rec['val_mu'] = float(vmu)
            if vmu > best_val:
                best_val = float(vmu); best_opts = to_opts(mu)
                json.dump({'val_mu': best_val, 'opts': best_opts, 'gen': gen,
                           'valSeeds': a.valSeeds, 'frames': a.frames}, open(out / 'best.json', 'w'), indent=2)
            rec['best_val'] = best_val
        log(rec)
    log({'phase': 'done', 'best_val': best_val, 'best_opts': best_opts})


if __name__ == '__main__':
    main()

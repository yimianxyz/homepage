"""Held-out re-score of a phase-conditional evolved-patrol winner.

The phase_evolve CMA-ES selects on its own 160-seed block, so its `best.json`
mean is optimistic (selection bias). This re-scores the winning phase params on
a FRESH held-out seed block and, on the SAME seeds, re-scores the global-E3D
baseline — a paired comparison that isolates the phase-specialization gain from
seed-block luck.

Uses sequential=False + eager (use_graph=False) to match the search config
exactly (the phase hook is not CUDA-graph-safe; eager == graph numerically).

  python3 phase_rescore.py ~/situ/ckpt/ph2/best.json --seedStart 200000 --n 2048
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sim import evaluate

WEIGHTS = 'js/predator_weights.json'
NAMES = ['cluster_r', 'dens_pow', 'reach_scale', 'sharp', 'lead_scale', 'lead_max', 'nbhd']
E3D = {"cluster_r": 178.09, "dens_pow": 2.373, "reach_scale": 1515.0, "sharp": 9.25,
       "lead_scale": 0.454, "lead_max": 230.6, "nbhd": 0.461}


def score(seeds, frames, device, opts):
    r = evaluate(WEIGHTS, seeds=seeds, frames=frames, device=device,
                 use_graph=False, sequential=False,
                 auto_target='evolved', auto_target_opts=opts)
    ps = np.asarray(r['per_seed_catches'], dtype=float)
    return ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('best')                       # path to phase_evolve best.json
    ap.add_argument('--seedStart', type=int, default=200000)
    ap.add_argument('--n', type=int, default=2048)
    ap.add_argument('--frames', type=int, default=1500)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    d = json.load(open(a.best))
    edges = d['edges']; P = d['P']
    pp_dict = d['phase_params']                   # {name: [P floats]}
    seeds = list(range(a.seedStart, a.seedStart + a.n))
    S = len(seeds)

    # phase config: (S, P, K) per-env tensor, same candidate on every seed
    K = len(NAMES)
    phys = np.zeros((P, K), dtype=np.float64)
    for j, nm in enumerate(NAMES):
        phys[:, j] = pp_dict[nm]
    pp = torch.tensor(np.repeat(phys[None], S, axis=0), dtype=torch.float64,
                      device=a.device)
    phase_opts = {'_phase_params': pp, '_phase_edges': edges, '_phase_names': NAMES}

    t0 = time.time()
    base_ps = score(seeds, a.frames, a.device, dict(E3D))      # global E3D
    phase_ps = score(seeds, a.frames, a.device, phase_opts)    # phase-conditional
    el = time.time() - t0

    def stat(ps):
        return float(ps.mean()), float(ps.std(ddof=1) / np.sqrt(len(ps)))
    bm, bse = stat(base_ps)
    pm, pse = stat(phase_ps)
    diff = phase_ps - base_ps                                  # paired
    dm = float(diff.mean()); dse = float(diff.std(ddof=1) / np.sqrt(len(diff)))

    out = {
        'best': a.best, 'P': P, 'edges': edges,
        'seedStart': a.seedStart, 'n': a.n, 'frames': a.frames,
        'search_best_mean': d.get('mean_catches'),
        'heldout_E3D_mean': round(bm, 4), 'heldout_E3D_se': round(bse, 4),
        'heldout_phase_mean': round(pm, 4), 'heldout_phase_se': round(pse, 4),
        'paired_gain': round(dm, 4), 'paired_gain_se': round(dse, 4),
        'gain_sigma': round(dm / dse, 2) if dse > 0 else None,
        'pct_better_vs_8.3447': round(100 * (pm - 8.3447) / 8.3447, 2),
        's': round(el, 1),
    }
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()

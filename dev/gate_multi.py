"""Gate several 'evolved' patrol configs across multiple large held-out seed
blocks, so we pick the config that GENERALIZES (not one that overfit its own
CEM validation block). Prints one JSON line per (config, block)."""
import json, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sim import evaluate

WEIGHTS = 'js/predator_weights.json'
SEEDS = 2048
FRAMES = 1500
BLOCKS = [50000, 80000]

CONFIGS = {
    'E3D':     {"cluster_r":178.09,"dens_pow":2.373,"reach_scale":1515,"sharp":9.25,"lead_scale":0.454,"lead_max":230.6,"nbhd":0.461},
    'ev3d_g4': {"cluster_r":179.38,"dens_pow":2.460,"reach_scale":1782.15,"sharp":8.620,"lead_scale":0.4498,"lead_max":233.93,"nbhd":0.5314},
    'ev2c_g12':{"cluster_r":176.50,"dens_pow":1.142,"reach_scale":1225.39,"sharp":9.858,"lead_scale":0.4728,"lead_max":168.38,"momentum":0.0756},
    'NC':      None,  # deployed nearest_cluster baseline
}
NC = {"cluster_r":150.0,"lead_scale":0.4,"lead_max":120.0}

def run(name, opts, target, block):
    t0 = time.time()
    seeds = list(range(block, block + SEEDS))
    r = evaluate(WEIGHTS, seeds=seeds, frames=FRAMES, device='cuda',
                 use_graph=True, sequential=False,
                 auto_target=target, auto_target_opts=opts)
    ps = np.asarray(r['per_seed_catches'], dtype=float)
    se = float(ps.std(ddof=1) / np.sqrt(len(ps)))
    print(json.dumps({'cfg': name, 'block': block, 'mean': r['mean_catches'],
                      'se': round(se, 4), 's': round(time.time()-t0, 1)}), flush=True)

if __name__ == '__main__':
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else list(CONFIGS)
    for block in BLOCKS:
        # rebind block via env consumed by eval_sim? evaluate uses seedStart kw
        for name in only:
            if name == 'NC':
                run(name, NC, 'nearest_cluster', block)
            else:
                run(name, CONFIGS[name], 'evolved', block)

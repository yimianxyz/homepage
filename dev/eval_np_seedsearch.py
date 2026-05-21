"""Run the 10 seed-search NNs through the NumPy sim, plus the shipped NN
for a baseline. Compare rank order with the JS 8-seed eval to see if
NumPy is a reliable proxy for ranking candidate policies.

Outputs JSON to dev/reports/np_seedsearch_compare.json.
"""
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, 'dev')
from sim_np import Sim, load_weights


def eval_one(weights_path, seeds, max_frames=5000):
    w = load_weights(weights_path)
    sim = Sim(seeds=seeds, weights=w, auto_target='flock_centroid')
    t0 = time.time()
    out = sim.run(max_frames)
    out['elapsed'] = time.time() - t0
    out['weights'] = weights_path
    return out


def main():
    SEEDS = list(range(100, 116))  # same 16 seeds as JS eval baseline
    MAX_FRAMES = 5000
    results = []

    # First the shipped baseline (matches dev/reports/autotarget_flock_centroid.json)
    print(f"=== shipped ===")
    r = eval_one('js/predator_weights.json', SEEDS, MAX_FRAMES)
    print(f"  mean={r['mean_catches']:.2f} elapsed={r['elapsed']:.1f}s per_seed={r['per_seed_catches']}")
    results.append({'name': 'shipped', **r})

    for s in range(1, 11):
        wp = f'dev/weights/seed_search/w_seed{s}.json'
        if not Path(wp).exists():
            print(f"=== seed={s}: missing ===")
            continue
        print(f"=== seed={s} ===")
        r = eval_one(wp, SEEDS, MAX_FRAMES)
        print(f"  mean={r['mean_catches']:.2f} elapsed={r['elapsed']:.1f}s per_seed={r['per_seed_catches']}")
        results.append({'name': f'seed{s}', **r})

    out_path = 'dev/reports/np_seedsearch_compare.json'
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n=== SUMMARY ===")
    # Sort by mean catches
    results.sort(key=lambda r: -r['mean_catches'])
    for r in results:
        print(f"  {r['name']:>10s}  mean={r['mean_catches']:6.2f}")


if __name__ == '__main__':
    main()

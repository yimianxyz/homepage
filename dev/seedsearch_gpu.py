"""Big seed search on GPU. Evaluate many candidate NN weight files in
sequence using sim_torch.Sim on CUDA. Each candidate gets B seeds.

Usage:
    python3 dev/seedsearch_gpu.py --weights "dev/weights/seed_search/w_seed*.json" \
        --baseline js/predator_weights.json \
        --seeds 64 --frames 5000 --device cuda \
        --out dev/reports/gpu_seedsearch.json
"""
import argparse
import json
import time
import glob
import sys
from pathlib import Path

sys.path.insert(0, 'dev')
from sim_torch import Sim, load_weights


def eval_one(weights_path, seeds, max_frames, device):
    w = load_weights(weights_path, device=device)
    sim = Sim(seeds=seeds, weights=w, device=device)
    t0 = time.time()
    r = sim.run(max_frames)
    r['elapsed'] = time.time() - t0
    r['weights_path'] = str(weights_path)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', nargs='+', required=True,
                   help='Glob patterns or paths to candidate weight files.')
    p.add_argument('--baseline', default='js/predator_weights.json')
    p.add_argument('--seeds', type=int, default=64)
    p.add_argument('--seed_start', type=int, default=100)
    p.add_argument('--frames', type=int, default=5000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default='dev/reports/gpu_seedsearch.json')
    args = p.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    paths = []
    for pat in args.weights:
        if '*' in pat or '?' in pat:
            paths.extend(sorted(glob.glob(pat)))
        else:
            paths.append(pat)

    results = []
    print(f"=== Baseline ({args.baseline}) ===")
    r = eval_one(args.baseline, seeds, args.frames, args.device)
    print(f"  mean={r['mean_catches']:.2f}  elapsed={r['elapsed']:.1f}s")
    results.append({'name': 'baseline', **r})

    for path in paths:
        print(f"=== {Path(path).stem} ===")
        r = eval_one(path, seeds, args.frames, args.device)
        print(f"  mean={r['mean_catches']:.2f}  elapsed={r['elapsed']:.1f}s")
        results.append({'name': Path(path).stem, **r})

    # Save and summary
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")
    results.sort(key=lambda r: -r['mean_catches'])
    print("\n=== SORTED BY mean_catches (desc) ===")
    for r in results:
        print(f"  {r['name']:>20s}  {r['mean_catches']:6.2f}")


if __name__ == '__main__':
    main()

"""Load-bearing-horizon retention curve from the H-sweep.

Reads planner_probe --out JSONs (sweep_single_H*.json), each with mean catches at
a given rollout horizon H (K, D fixed). Prints catches(H), retention =
catches(H)/catches(Hmax), and flags the smallest H still >= a target retention.
Retention is computed PAIRED per-seed when per_seed is present (chaos-robust).

  python3 analyze_sweep.py sweep_single_H*.json --target 0.999
"""
import argparse, glob, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--target', type=float, default=0.999)
    args = ap.parse_args()
    paths = []
    for f in args.files:
        paths += glob.glob(f)
    rows = []
    for p in sorted(set(paths)):
        d = json.load(open(p))
        rows.append(dict(H=d.get('H'), mean=d.get('mean'), n=d.get('n'),
                         per=np.asarray(d['per_seed'], float) if d.get('per_seed') else None,
                         seedStart=d.get('seedStart'), path=p))
    rows.sort(key=lambda r: -r['H'])
    if not rows:
        print('no files'); return
    base = rows[0]                      # largest H = reference planner
    print(f"reference H={base['H']} mean={base['mean']:.3f} n={base['n']}")
    print(f"{'H':>5} {'mean':>8} {'retention':>10} {'paired_dCI95':>22}")
    for r in rows:
        ret = r['mean'] / base['mean'] if base['mean'] else float('nan')
        ci = ''
        if r['per'] is not None and base['per'] is not None and \
           r['seedStart'] == base['seedStart'] and len(r['per']) == len(base['per']):
            d = r['per'] - base['per']
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(d), size=(20000, len(d)))
            bm = d[idx].mean(1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            ci = f"[{lo:+.2f},{hi:+.2f}]"
        flag = ' <-- >=target' if ret >= args.target else ''
        print(f"{r['H']:>5} {r['mean']:>8.3f} {ret:>10.4f} {ci:>22}{flag}")
    ok = [r for r in rows if (r['mean'] / base['mean']) >= args.target]
    if ok:
        sm = min(ok, key=lambda r: r['H'])
        print(f"\nsmallest H >= {args.target} retention: H={sm['H']} "
              f"(mean {sm['mean']:.3f}, {sm['mean']/base['mean']:.4f})")
    else:
        print(f"\nNO H meets {args.target}; best retention "
              f"{max(r['mean']/base['mean'] for r in rows):.4f}")


if __name__ == '__main__':
    main()

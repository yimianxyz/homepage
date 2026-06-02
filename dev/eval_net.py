"""eval_net — paired held-out re-scoring of a distilled scorer net vs the planner.

Loads a net saved by distill_v5 (--save_net: {state_dict, arch, ...}), runs it
closed-loop on a fresh held-out seed set, and on the SAME seeds runs the planner
(K/H/D, same --dense) and the production E3D baseline. Reports per-seed paired
means with standard errors AND a bootstrap 95% CI on the net/planner ratio — the
honest "fraction of the planner we recover" number, never an on-search mean.

The 99% goal is stated against the ORIGINAL integer-gain planner (21.40 catches
@ 1500 frames), so the headline pass/fail is: net_mean >= 21.19 (absolute).

Usage:
  python3 eval_net.py --net net_deepsets_d08.pt --n 1024 --frames 1500 \
      --K 16 --H 120 --D 8 --dense 0.8 --hold_D 8 --weights predator_weights.json
"""
import argparse, json, time
import numpy as np
import torch

import sim_torch as st
import planner_probe as pp
from distill_v4 import SetScorer, eval_closed_loop

BASE = 8.3447          # production E3D @ 1500 frames (sim_torch)
ORIG_PLANNER = 21.40   # integer-gain K16/H120/D8 planner; 99% bar = 21.186


def e3d_baseline(seeds, frames, device):
    sim = st.Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
                 auto_target='evolved', auto_target_opts=dict(pp.E3D))
    for _ in range(frames):
        sim.step()
    return sim.catches.cpu().numpy().astype(np.float64)


def boot_ci(a, b, reps=10000, seed=0):
    """95% bootstrap CI on mean(a)/mean(b) over paired resamples."""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(reps, n))
    ratios = a[idx].mean(1) / b[idx].mean(1)
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def boot_ci_mean(a, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(reps, n))
    m = a[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True, help='saved net .pt (state_dict+arch)')
    ap.add_argument('--n', type=int, default=1024)
    ap.add_argument('--seedStart', type=int, default=300000,
                    help='held-out start; keep disjoint from gen (4e5/6e5/8e5) and eval (2e5)')
    ap.add_argument('--frames', type=int, default=1500)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--H', type=int, default=120)
    ap.add_argument('--D', type=int, default=8)
    ap.add_argument('--hold_D', type=int, default=8)
    ap.add_argument('--dense', type=float, default=0.0)
    ap.add_argument('--weights', default='predator_weights.json')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--skip_planner', action='store_true',
                    help='skip the (expensive) planner re-score; report net + baseline only')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    pp.WEIGHTS = st.load_weights(args.weights, device=device)
    pp.DENSE_LAMBDA = args.dense
    seeds = list(range(args.seedStart, args.seedStart + args.n))
    t0 = time.time()

    blob = torch.load(args.net, map_location=device)
    a = blob['arch']
    net = SetScorer(a['enc'], a['d'], a['layers'], heads=a.get('heads', 4),
                    hid=a['hid']).to(device)
    net.load_state_dict(blob['state_dict'])
    net.eval()
    print(f"[eval_net] {args.net} enc={a['enc']} d={a['d']} L={a['layers']} "
          f"hid={a['hid']} | n={args.n} frames={args.frames} K={args.K} "
          f"H={args.H} D={args.D} dense={args.dense} hold_D={args.hold_D}", flush=True)

    net_c, pick0 = eval_closed_loop(net, seeds, args.frames, device, args.K, args.hold_D)
    net_c = net_c.astype(np.float64)
    print(f"[eval_net] net done {time.time()-t0:.0f}s pick0={pick0:.3f}", flush=True)

    base_c = e3d_baseline(seeds, args.frames, device).astype(np.float64)
    print(f"[eval_net] baseline done {time.time()-t0:.0f}s", flush=True)

    res = dict(net=args.net, arch=a, n=args.n, frames=args.frames,
               K=args.K, H=args.H, D=args.D, dense=args.dense, hold_D=args.hold_D,
               pick0=pick0)

    def stat(x):
        return dict(mean=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))))

    res['net_catches'] = stat(net_c)
    res['base_catches'] = stat(base_c)
    nlo, nhi = boot_ci_mean(net_c)
    res['net_mean_ci95'] = [nlo, nhi]

    if not args.skip_planner:
        plan_c = pp.run_planner(seeds, args.frames, device, args.K, args.H, args.D)[0].astype(np.float64)
        print(f"[eval_net] planner done {time.time()-t0:.0f}s", flush=True)
        res['planner_catches'] = stat(plan_c)
        delta = net_c - plan_c
        res['paired_delta_net_minus_planner'] = stat(delta)
        rlo, rhi = boot_ci(net_c, plan_c)
        res['ratio_net_over_planner'] = dict(point=float(net_c.mean() / plan_c.mean()),
                                             ci95=[rlo, rhi])

    # ---- report ----
    print("\n==================== PAIRED HELD-OUT RE-SCORING ====================")
    print(f"  E3D baseline   : {res['base_catches']['mean']:.3f} ± {res['base_catches']['se']:.3f}")
    print(f"  net (closed)   : {res['net_catches']['mean']:.3f} ± {res['net_catches']['se']:.3f}  "
          f"(95% CI [{nlo:.2f}, {nhi:.2f}])  pick0={pick0:.3f}")
    if not args.skip_planner:
        pc = res['planner_catches']; r = res['ratio_net_over_planner']
        d = res['paired_delta_net_minus_planner']
        print(f"  planner (dense={args.dense}): {pc['mean']:.3f} ± {pc['se']:.3f}")
        print(f"  net - planner  : {d['mean']:+.3f} ± {d['se']:.3f} (paired)")
        print(f"  net / planner  : {r['point']*100:.1f}%  (95% CI [{r['ci95'][0]*100:.1f}%, {r['ci95'][1]*100:.1f}%])")
    bar = 0.99 * ORIG_PLANNER
    nm = res['net_catches']['mean']
    verdict = "PASS" if nm >= bar else "below"
    print(f"  --- GOAL: net_mean >= {bar:.2f} (99% of {ORIG_PLANNER}) : "
          f"{nm:.3f} -> {verdict} ({100*nm/ORIG_PLANNER:.1f}% of orig planner)")
    print("====================================================================\n")

    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(res, fh, indent=2)
        print(f"[eval_net] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()

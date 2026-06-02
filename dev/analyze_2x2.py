"""Paired analysis of the Phase-0 2x2 premise table {E3D,planner} x {single,two-pass}.

Consumes planner_probe.py --out JSONs (each carries per_seed catches on
seedStart=200000). Computes per-cell mean+SE, and for any two arms run on
IDENTICAL seeds, a PAIRED bootstrap 95% CI on the per-seed difference
(the decision-critical statistic — boid flocking is chaotic so per-seed
catches are noisy, but the per-seed DELTA on shared seeds cancels the
shared scene-difficulty and is far tighter than the unpaired SE suggests).

Also reports the selection-on-chaos null: two-pass planner (default tiebreak,
ties -> cand0=E3D) vs two-pass planner (--randtie). If the planner's edge over
two-pass E3D survives random tiebreaking it's genuine lookahead; if the edge
collapses to the randtie arm it was an artifact of how near-tied sub-catch
rollouts resolve.

  python3 analyze_2x2.py \
      --label "single E3D"      single_e3d.json \
      --label "single planner"  single_planner.json \
      --label "two-pass E3D"    e3d_twopass.json \
      --label "two-pass planner" planner_twopass.json \
      --label "two-pass planner randtie" planner_twopass_randtie.json
"""
import argparse, json
import numpy as np


def load(path):
    d = json.load(open(path))
    ps = d.get('per_seed')
    return d, (np.asarray(ps, dtype=np.float64) if ps is not None else None)


def boot_paired(a, b, iters=100000, seed=0):
    """Paired bootstrap 95% CI + p(Δ>0) on a-b over shared seeds."""
    rng = np.random.default_rng(seed)
    d = a - b
    n = len(d)
    idx = rng.integers(0, n, size=(iters, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), float((means > 0).mean())


def mde(a, b):
    """Minimum detectable effect (paired, 80% power, two-sided 0.05) and the
    n needed to detect a +0.5/episode edge. Addresses the red-team's power
    worry: a 'n.s.' CI is only informative if the test COULD have seen a real
    sub-catch edge. sd_delta is the paired per-seed Δ SD (scene difficulty
    cancels), which drives power far more than the raw catch SD."""
    d = a - b
    n = len(d)
    sd = float(d.std(ddof=1))
    # 80% power two-sided 0.05: MDE = (z_a/2 + z_b) * sd/sqrt(n) = 2.80 * SE
    se = sd / np.sqrt(n)
    mde80 = 2.80 * se
    n_for_half = (2.80 * sd / 0.5) ** 2 if sd > 0 else 0.0
    return sd, se, mde80, n_for_half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', action='append', nargs=2, metavar=('NAME', 'PATH'),
                    required=True, help='repeatable: --label NAME path.json')
    ap.add_argument('--planner-key', default='two-pass planner',
                    help='label of the main planner arm for paired Δ')
    ap.add_argument('--e3d-key', default='two-pass E3D',
                    help='label of the E3D arm to pair the planner against')
    ap.add_argument('--randtie-key', default='two-pass planner randtie')
    args = ap.parse_args()

    cells, per = {}, {}
    print('=== cells ===')
    for name, path in args.label:
        d, ps = load(path)
        cells[name] = d
        per[name] = ps
        n = d.get('n')
        mean = d.get('mean')
        se = d.get('se')
        fne = (d.get('stats') or {}).get('frac_non_e3d')
        seeds_tag = f"seedStart={d.get('seedStart')}"
        extra = f" frac_non_e3d={fne}" if fne is not None else ""
        print(f"  {name:30s} mean={mean:7.3f}  se={se:6.3f}  n={n}  {seeds_tag}{extra}")

    def seeds_of(name):
        d = cells[name]
        s0 = d.get('seedStart')
        n = d.get('n')
        return None if s0 is None else (s0, n)

    def paired(a_name, b_name, tag):
        if a_name not in per or b_name not in per:
            print(f"  [{tag}] missing arm(s): {a_name} / {b_name}")
            return
        if per[a_name] is None or per[b_name] is None:
            print(f"  [{tag}] no per_seed in one arm -> cannot pair")
            return
        if seeds_of(a_name) != seeds_of(b_name):
            print(f"  [{tag}] seed blocks differ {seeds_of(a_name)} vs {seeds_of(b_name)} -> NOT paired")
            return
        a, b = per[a_name], per[b_name]
        if len(a) != len(b):
            print(f"  [{tag}] length mismatch {len(a)} vs {len(b)}")
            return
        m, lo, hi, pgt = boot_paired(a, b)
        sd, se, mde80, n_half = mde(a, b)
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s. (CI spans 0)"
        print(f"  [{tag}] Δ(mean per-seed) = {m:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]  "
              f"P(Δ>0)={pgt:.3f}  {sig}")
        print(f"        paired SD(Δ)={sd:.3f}  SE={se:.3f}  MDE@80%pow={mde80:.3f}/episode"
              f"  n_for_0.5edge≈{n_half:.0f}")
        if not (lo > 0 or hi < 0):
            print(f"        ^ n.s. is only conclusive if MDE < the edge you'd act on; "
                  f"a real edge < {mde80:.2f}/episode would be invisible at this n.")

    print('\n=== paired bootstrap (shared seeds) ===')
    paired(args.planner_key, args.e3d_key, f"{args.planner_key} - {args.e3d_key}")
    paired(args.planner_key, args.randtie_key,
           f"{args.planner_key} - {args.randtie_key} (default vs random tiebreak)")
    paired(args.randtie_key, args.e3d_key, f"{args.randtie_key} - {args.e3d_key} (null edge)")


if __name__ == '__main__':
    main()

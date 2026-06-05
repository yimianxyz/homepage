"""Rigorous check: is the fast (predictor-corrector) two-pass statistically
equal to the faithful sequential two-pass? Compares catch rates at large n for
E3D (cheap, big n -> tight test of boid+predator dynamics) and the planner (the
teacher whose gains we'd train on). If fast/seq ~= 1.0 the fast sim is unbiased
and safe to generate training data with.
  python3 validate_equiv.py <n_e3d> <n_planner> <frames>
"""
import sys, time
import numpy as np
import planner_probe as pp
import sim_torch as st

N_E3D = int(sys.argv[1]) if len(sys.argv) > 1 else 128
N_PLAN = int(sys.argv[2]) if len(sys.argv) > 2 else 24
FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
dev = 'cuda'
pp.WEIGHTS = st.load_weights('../js/predator_weights.json', device=dev)
pp.TWO_PASS = True


def run(policy, n, fast):
    st.FAST_TWO_PASS = fast
    seeds = list(range(200000, 200000 + n))
    if policy == 'e3d':
        return np.asarray(pp.run_e3d(seeds, FRAMES, dev))
    return np.asarray(pp.run_planner(seeds, FRAMES, dev, 16, 120, 8))


for pol, n in [('e3d', N_E3D), ('planner', N_PLAN)]:
    t = time.time(); seq = run(pol, n, False); ts = time.time() - t
    t = time.time(); fast = run(pol, n, True); tf = time.time() - t
    sm, fm = float(seq.mean()), float(fast.mean())
    # paired SE of the difference (same seeds)
    diff = fast - seq
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    print(f'{pol} n={n}: seq={sm:.3f} fast={fm:.3f} ratio={fm/max(sm,1e-9):.3f} '
          f'diff={fm-sm:+.3f}±{se:.3f}SE  (seq {ts:.0f}s, fast {tf:.0f}s, {ts/max(tf,1e-9):.1f}x)',
          flush=True)

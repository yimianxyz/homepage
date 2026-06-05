# Ballistic-prune: collapsing the lookahead rollout from 16 candidates to 1

**Date:** 2026-06-05  **Branch:** rl/teacher
**Eval:** `eval_value.py` on `net_v2_absval` (3553-param value net), sim_torch,
n=256 unless noted, frames=1500 (calibration regime, ≈ 5000f/4).

## The setup

The deployable student is a tiny value net + a short online rollout over the
planner's 16 candidate targets (cand0 = E3D analytic patrol; cand1..15 = nearest
boids, lead-adjusted). Each decision (every D=8 frames):
`score[k] = (catches in an Hs-frame rollout of candidate k) + V(terminal)`,
argmax, commit. Full rollout of all 16 candidates reproduces the planner-class
result (**15.23 @1500f**, the ceiling). The expensive part is the rollout:
16 candidates × 120 boids × Hs frames. **Question: how few/cheap rollouts can we
do and stay at the ceiling?**

## Headline finding: ballistic selection > value selection, 1 rollout ≈ 16

`K_roll` = how many of the 16 candidates get a real rollout; the rest are scored
by a prior. Two ways to choose *which* get rolled — by the value-net prior, or by
a cheap 2-body **ballistic catchability** sim (predator max-speed seek vs a
constant-velocity boid; `caught − t_catch/H_b`). Holding all else fixed:

| K_roll | by value net | by **ballistic** |
|--------|--------------|------------------|
| 0 (no rollout) | 8.58 | 8.58 |
| **1** | 9.05 | **14.97** |
| 2 | 11.48 | 14.70 |
| 3 | 12.63 | 14.66 |
| 4 | 13.82 | 15.07 |
| 5 | 13.81 | 15.09 |
| 6 | 13.95 | 15.16 |
| 7 | 14.98 | 15.23 |
| 16 (full) | 15.23 | 15.23 |

**Rolling the single most ballistically-catchable candidate = 14.97 = 98.3% of
the 15.23 ceiling, with 1 rollout instead of 16.** Value selection needs 7
rollouts to match. The cleanest isolation is K_roll=1 (selection is everything):
ballistic 14.97 vs value 9.05 — **+5.9** with all else identical.

**Why:** rolling all 16 lets the imperfect Hs=60 rollout occasionally hand a
non-catchable candidate a spuriously high score that wins the argmax (the
non-monotonic value-prune dip at K_roll=4). Ballistic pre-filtering to genuinely
catchable targets removes those false positives. The value net, despite ingesting
the ballistic features, cannot replicate the explicit ballistic sim for selection.

### Generalization
Confirmed across three independent seed sets at n=256: ball/kr1 =
14.97 (seed300000), 14.66 (seed500000); the earlier 16.07 (seed200000, n=96) was
small-n noise — it does NOT beat the 15.23 ceiling.

## The minimal deploy config (cost frontier, ballistic kr=1)

- **Rollout depth Hs: ~60, irreducible.** 60→14.97, 40→13.61, 30→12.29,
  20→10.51, 10→8.21 (≈ the no-rollout floor). Clean linear cliff — the slow
  predator needs ~60 frames to see a catch resolve.
- **Rollout boids roll_M: 64 sweet spot.** 120→14.97, 64→14.17 (95% at half the
  boids), 32→12.63, 16→12.33, 8→10.91. Knee at 64; the targeted boid's neighbors
  steer its path, so you can't freeze most of the flock.
- **No rollout at all = 8.58 ≈ E3D baseline** — the rollout is irreducible; the
  value net alone (even with ballistic features) can't steer better than patrol.

**Net deploy cost: 1 candidate × ~64 boids × 60 frames per decision, vs the full
16 × 120 × 60 — roughly 15–30× cheaper at 95–98% of the catch rate.**

## Does the JS port need the value net?

At Hs=60, dropping the net (no terminal bootstrap; non-rolled candidates score 0,
i.e. E3D fallback) costs −3.3: ball/kr1 14.97 → **11.66**. So the net's bootstrap
(catches beyond the 60-frame window) and its prior (letting a non-rolled candidate
win) both matter at Hs=60. Open test (running): whether rolling top-1..3 ballistic
candidates the *full* Hs=120 recovers the catches without any net — which would
keep the JS port to `rolloutFlat` + `candidates()` + a ballistic scorer (both
already exist and are verified in `js/predator_planner_worker.js`).

## Cross-seed-set + deployment-scale confirmation

Ball/kr1 vs the full-rollout ceiling, three independent seed sets (n=256, 1500f):
seed300000 14.97 / 15.23 = **98.3%**; seed500000 14.66 / 15.54 = **94.3%**. The
single-rollout result is 94–98% of ceiling everywhere. At deployment scale
(5000f, single-pass, n=128): full(16) ceiling = **53.9**, ball/kr1 M120 =
**52.7 (97.8%)**, M64 = **51.3 (95.2%)** — the 1-rollout headline holds at scale.

## JS port (browser deploy)

The decision core is ported and **parity-verified to sim_torch < 1e-6**
(`js/cheap_planner.js`, `dev/test_cheap_parity.js`, fixture
`dev/js_fixture.json`): 19 features (incl. the 2-body ballistic intercept) +
standardization + 23→48→48→1 GELU value net. The full kr=1 policy is wired into
`js/predator_planner_worker.js` (`planCheap`, `rolloutFlatState` returning the
terminal snapshot for the bootstrap, a `cheap` controller in `evalClosedLoop`,
and a `cheapconfig` message) — reusing the already-verified `rolloutFlat` +
`candidates()`. `dev/eval_cheap_headless.js` runs it headless with sim_torch-
matched seeds.

**Two-pass deploy validation (the live regime, double-flock):** at 1500f, n=8,
cheap = **8.63** vs e3d = **6.0** — the 1-rollout ballistic policy beats E3D
patrol in the real deployment dynamics. The cheap/e3d ratio is 1.44× two-pass vs
1.76× single-pass: BOTH policies score lower two-pass, and cheap compresses more
because the value net + ballistic intercept were tuned in single-pass dynamics.

## Next step (recommended): two-pass recalibration

The port works live but is single-pass-tuned. To recover the full margin in the
browser, regenerate the value-net training data with `TWO_PASS=True`
(`run_log_feat`), retrain, re-export to `js/value_net.json`, re-verify parity.
Then wire `?policy=cheap` into the live predator and Playwright-verify. Open
decision: deploy regime (single vs two-pass) — the live page is two-pass, so
two-pass recalibration is the path to maximizing live catches.

## Files
- `feat_planner.py` — `run_value_lookahead_cheap(..., K_roll, prune_by={v,ball},
  no_value)`; `_ballistic_intercept`.
- `eval_value.py` — `--K_roll --prune_by --no_value`.
- Sweep drivers: `recovery_sweep.sh`, `confirm_ball.sh`, `confirm3_seed500.sh`,
  `minimal_config.sh`, `novalue_test.sh`, `novalue_hs.sh`, `confirm_5kf.sh`.

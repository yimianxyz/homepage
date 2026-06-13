# Cheap-surrogate-rollout probe — can a cheaper rollout predict `boot`/`catches`?

**side-a EXACT-NN.** Question: prod's `planCheap` rolls the top-4 ballistic
candidates Hs=90 frames through a *faithful* flock sim and scores each
`catches + boot` (boot = max terminal value-net value). ~69% of plan decisions
turn on the **boot** difference between two equal-catch rolled candidates
(median decisive margin ~0.019). **Can a cheaper rollout — boids without the
expensive flocking — predict the true `boot` (and `catches`) well enough to use
as an NN feature?**

All numbers below are from a standalone reproduction of `rolloutFlatState` +
`candidates` + the boot computation, loading prod constants /
`computeEvolvedTarget` / `cp_features` / `cp_value` via `dev/fasteval.js`
`buildHarness` (so the math is the shipped code). Scratch scripts are in this
directory; raw results in `*_result.json`.

---

## 0. Validation gate (full flocking ON) — PASSED, exactly

With `flockMode='full'`, reproduce the logged `r.rolled[rk] = [ci, catches, boot]`.

| sample | pairs | catches match | boot match (<1e-7) | max boot error |
|--------|-------|---------------|--------------------|----------------|
| 48 shards (stride 8, 15/shard) | **2800** | **2800/2800 = 100%** | **2800/2800 = 100%** | **0.0 (bit-exact)** |

Boot error is not merely <1e-7 — it is **identically 0**. The reproduction is
byte-faithful to the oracle. The full-flock reproduction also gives **S_dec =
1.000** and catch agreement = 1.000 on 500 held-out records (sanity), confirming
the surrogate-as-policy pipeline matches the oracle when the rollout is faithful.
Only then do we trust the cheap variants.

Cheap variants tested:
- **V_const** — boids integrate at constant velocity (skip `accumulateFlock`
  entirely; torus wrap; predator step + catch unchanged).
- **V_avoid** — boids get ONLY the predator-avoidance `qx/qy` force (no
  cohesion / separation / alignment); predator step + catch unchanged.

Measurement sample: **3,500 plans across 350 shards** (1 per shard × 10),
14,000 rolled-candidate pairs per variant. Correlation/residual analysis on a
separate 2,500-plan / 10,000-pair dump.

---

## 1. Catch agreement — P(catches_cheap == catches_true)

| variant | overall | rk0 | rk1 | rk2 | rk3 |
|---------|---------|-----|-----|-----|-----|
| **V_avoid** | **0.679** | 0.661 | 0.686 | 0.683 | 0.686 |
| **V_const** | **0.614** | 0.595 | 0.618 | 0.620 | 0.623 |
| full (sanity) | 1.000 | — | — | — | — |

Both variants get the catch count *wrong on ~32–39% of rolled candidates*.
Signed-error structure differs: **V_avoid mostly UNDER-counts** (boids it
should have caught escape via flock dispersal it no longer models — the
predator-avoid force pushes them away), while **V_const both over- and
under-counts** (constant-velocity boids fly straight into / away from the
predator unrealistically). Catches alone (an integer that directly enters the
score) are already corrupted before boot even matters.

## 2. Boot-error CDF — |boot_cheap − boot_true|

| variant | median | frac<0.001 | frac<0.01 | frac<0.05 | frac<0.1 | p90 |
|---------|--------|-----------|----------|----------|---------|-----|
| **V_avoid** | **0.249** | 0.6% | **3.7%** | 14.5% | 25.7% | 0.890 |
| **V_const** | **0.279** | 0.5% | **3.3%** | 13.3% | 23.4% | 1.020 |
| full (sanity) | 0.000 | 100% | 100% | 100% | 100% | 0.000 |

The target precision is **~0.01** (the decision margin is ~0.019). Only
**~3–4% of cheap-boot estimates land within 0.01** of truth; the **median error
(~0.25–0.28) is ~13–15× the decisive margin.** `boot_true` itself has std 0.67,
so this is a large fraction of the full signal range.

## 3. Surrogate-as-policy S_dec (committed coord vs `lab.tx/ty`)

score' = vprior with the 4 rolled candidates overridden by
`catches_cheap + boot_cheap`; deduped argmax; coord compared to the oracle label.

| variant | S_dec | S_dec on catch-decided | S_dec on boot-decided |
|---------|-------|------------------------|------------------------|
| **V_avoid** | **0.416** | 0.450 | 0.366 |
| **V_const** | **0.406** | 0.371 | 0.415 |
| full (sanity) | **1.000** | 1.000 | 1.000 |

A cheap surrogate plugged directly in as the rollout reproduces the prod
decision **~41% of the time** — barely above the ~37% NN-alone-no-rollout
baseline that side-a already measured, and nowhere near the L1h target. It fails
on *both* boot-decided and catch-decided plans, because both `catches_cheap` and
`boot_cheap` are corrupted.

## 4. The residual-learning question (the real test)

Could an NN *correct* the cheap boot (residual learning), rather than use it raw?
The ceiling for that is how much true-boot variance the cheap boot explains.
On 10,000 finite pairs:

| variant | Pearson r | Spearman ρ | r² (var explained) | resid frac<0.01 after best linear fit | resid frac<0.01 + cheap-catches |
|---------|-----------|-----------|--------------------|---------------------------------------|---------------------------------|
| **V_avoid** | **0.657** | 0.681 | **0.43** | **1.6%** | 1.6% |
| **V_const** | **0.619** | 0.645 | **0.38** | **1.3%** | 1.9% |

`boot_true` std = 0.67. After the *best possible* global linear correction
`boot_true ≈ a·boot_cheap + b`, the residual median is still ~0.29–0.31 and only
**~1.3–1.6% of residuals fall below 0.01**. Adding `catches_cheap` as a second
regressor barely helps (≤1.9%). So a feature with r≈0.62–0.66 leaves an
irreducible residual std of ~0.50 — **~26× the 0.019 decision margin.** A
residual NN cannot manufacture signal the cheap rollout discarded; it can only
de-bias what is there, and what is there is too coarse.

## 5. Cost proxy (wall time, isolated warmed passes)

| variant | µs / rollout (sim only) | µs / rollout (incl. boot recompute) | speedup vs full (sim only) |
|---------|------------------------:|------------------------------------:|---------------------------:|
| **V_const** | 432 | ~1,520 | **23.8×** |
| **V_avoid** | 776 | ~2,640 | **13.2×** |
| full | 10,283 | ~10,600 | 1× |

The flocking neighbor-scan dominates the sim; dropping it is genuinely cheap
(13–24× faster). But the speed buys nothing here: the cheap rollout's output is
not a usable proxy. (Note: a real NN feature would also pay the shared boot
recompute — `candidates`+`cp_features`+`cp_value` on the terminal — which is
~1.1 ms and unavoidable in all variants.)

---

## Verdict

**No.** A cheap surrogate rollout — whether constant-velocity (V_const) or
predator-avoid-only (V_avoid) — does **not** plausibly let an NN predict `boot`
to ~0.01. Three independent measurements agree:

1. **Boot precision:** only ~3–4% of cheap-boot estimates are within the 0.01
   target; the median error (~0.25–0.28) is ~13–15× the median decisive margin
   (0.019).
2. **Residual ceiling:** the cheap boot correlates with the true boot at only
   r≈0.62–0.66 (r²≈0.4), so even an oracle residual-learner — best-fit linear,
   *or* the same plus cheap-catches — leaves ~1.3–1.9% of residuals under 0.01
   and a residual std (~0.50) ~26× the decision margin. The signal a residual NN
   would need is destroyed by removing flocking, not merely biased.
3. **End-to-end:** dropped straight into the policy, the surrogate reproduces
   the prod decision only ~41% of the time — essentially the NN-alone level —
   versus 100% for the faithful rollout.

The reason is structural: the decisive boot differences come from where the
*flock* ends up after 90 frames of the predator perturbing it (cohesion pulling
boids back, separation spreading them, alignment turning the herd). V_const
ignores the predator entirely (boids fly off on stale velocities); V_avoid keeps
only the predator repulsion but loses the flock's internal restoring dynamics
that actually shape the terminal configuration the value net reads. The catch
count — an integer term in the score — is itself wrong ~32–39% of the time, so
the surrogate fails even before boot precision is in play.

**Caveats.** (a) These two variants are the cheap-end bookends the brief
specified; an *intermediate* surrogate (e.g. cohesion+separation but coarser
grid, or fewer frames) might land between full and these — not tested here, but
the cost win shrinks as you re-add flocking, and the r≈0.65 ceiling suggests the
flock dynamics that matter are exactly the expensive part. (b) The linear-fit
residual is an *upper bound* on a 1–2 feature residual NN; a deep NN could in
principle exploit nonlinear structure, but with r²≈0.4 and ~98% of residuals
above 0.01 there is little headroom — boot is dominated by terminal flock
geometry that the cheap sim does not produce. (c) S_dec here overrides only the
4 rolled slots (as prod does); a different NN architecture that re-derives all
16 scores from features is a separate question (the existing NN-alone ~37%
baseline), but feeding *this* cheap rollout as the rollout feature does not
move the needle.

**Bottom line for the program:** the expensive flocking in the rollout is
load-bearing, not incidental. A cheap surrogate rollout is not a viable feature
for closing the boot gap; the path to high S_dec remains the faithful rollout
(L1h), not a cheaper approximation of it.

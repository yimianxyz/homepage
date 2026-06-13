# L0 — unified-exact policy: verification report (#6)

**Status: FINAL.** 84-cell matrix complete — 12,894,083 frames, **0 mismatches**.

L0 is the program's **T1 floor** (SPEC §4): one policy module, one entry point
covering all N, output **bitwise-identical to frozen prod** (`main@6dce76f`) by
construction. This report is the evidence: how L0 is built so the claim is
*provable*, the lockstep differential harness that is the program's instrument,
and the verification matrix (zero mismatches) + honest per-regime throughput.

## 1. What L0 is, and why it is T1 (exact by construction)

L0 (`dev/exact_nn/policy_unified.js`) is **not a rewrite** of prod's arithmetic
— a rewrite would only be T2 (exact-on-corpus). It is a **mechanical verbatim
build**: `dev/exact_nn/build_l0.js` embeds the three prod policy sources
**byte-for-byte** inside a shadow-scope wrapper and adds zero arithmetic.

- Embedded byte ranges (sha256-attested in the artifact header, re-checkable
  with `node build_l0.js --check`):
  - `js/cheap_planner.js` — `cp_erf/cp_gelu/cp_ballistic/cp_features/cp_value`
  - `js/predator.js` — `EVOLVED_PATROL` + `computeEvolvedTarget`
  - `js/predator_cheap.js` — flat rollout sim, grid, `candidates`, `planCheap`,
    `steer`, `intercept`, the `force()` dispatcher + persistent state
- The single entry point is **prod's own `force(pred, boids)`**, exposed
  directly — it already dispatches `N==0 → Vector(0,0)`, `N≤5 → intercept`,
  `N>5 → plan/steer`. No re-dispatch, no indirection.
- The value net loads through the embedded fetch chain (browser: real `fetch`;
  harness: the stub) → the identical `JSON.parse` of the identical bytes.

Because every output-influencing operation runs **in prod's own source bytes in
prod's own order — including the same `Math.*` calls in the same sequence** —
T1 is **engine-invariant**: L0 is bit-exact in every JS engine prod runs in,
including ones never tested (SPEC §2 T1). The load-bearing quirks the SPEC §7 /
§4 review flagged are preserved automatically because the bytes are copied, not
re-expressed:

- two-pass `accumulateFlock` (frozen pass 1, sequential in-place pass 2);
- the **no-tie-break** `candidates()` distance sort (`predator_cheap.js:241`,
  relies on V8 stable sort) vs the **tie-broken** pidx sort at `:272` (`||(a-b)`);
- the `NaN → −Infinity` bootstrap-max path;
- LIFO linked-list grid insertion order;
- `frame===0 || frame>=cfg.D` gate timing and the strict-`>` argmax;
- `egBoid` object-identity commit across regime flips (re-checked via `indexOf`).

The wrapper's only deviations are **provably output-irrelevant** (equivalence is
defined on the returned force, SPEC §2): no activation-viz side effects (prod's
post-argmax `cp_value_viz` writes only viz arrays), and no `window.__predatorReady`
ownership (boot gate stays with prod). Both documented in `build_l0.js`.

## 2. The lockstep differential harness (program instrument, SPEC §2a)

`dev/exact_nn/diff_harness.js` + the shared `dev/exact_nn/stepper.js` (adopted
from side-a's `ae48889`, side-b extensions layered on as opt-in params).

- **One sim, both policies every frame, every regime** (N>5, N≤5, N==0).
  Prod's force is applied (lockstep); the candidate runs on the same state.
- **Force compared as raw u64 bit patterns** via a shared `Float64Array`/
  `Uint32Array` view — `−0 ≠ +0`, NaN payloads compared exactly.
- **Decision-level metric is primary** (SPEC §2a/§3): committed target
  coordinates compared bitwise with **dedup-by-coordinates** (padded duplicate
  candidates are the same decision); `egBoid` by live-object identity. Per-frame
  force equality is the secondary, stricter check.
- **Resync** after any disagreement (copy prod's `{target,frame,egBoid}` into
  the candidate) → per-decision counts, no cascade inflation.
- **State injection** get/set hook (`__cheapDebug`, an anchored in-memory
  transform proven digest-inert) for synthetic states.
- **Post-extinction frames** keep stepping after N==0 so the zero path is
  exercised (the live page keeps ticking).
- **Modes:** lockstep (default) and trajectory-fork (apply candidate force,
  report first-divergence frame).

**The harness is proven on itself** (`--selftest`, 49/49 across fullgame /
endgame / gate / spawn cells): determinism, `__cheapDebug` inertness, fastRender
purity, candidate-eval non-disturbance, identity candidate → 0 mismatches, and a
deliberately **1-ulp-broken candidate caught at the exact perturbed frame** in
both modes. Notable: a 1-ulp force fault is invisible to trajectory-level checks
(it rounds away below velocity ulp, ~4.4e-16) yet lockstep catches it — the
empirical justification for the lockstep-bitwise doctrine.

## 3. Verification matrix

`dev/exact_nn/shard_runner.js` — 84 cells = 7 device configs × {full game,
endgame N=1..5, gate-crossing N=6..8, spawn-schedule A/B incl. same-coordinate
double-taps, pristine-committed-artifact slice}, disjoint held-out seeds
≥272000, fanned across local cores. The committed L0 artifact is the candidate.

Device configs: `390×844` (mobile UA→N=60), `820×1180` (desktop N=120 + a real
**iPad-UA** cell → N=60), `1024×768`, `1512×982`, `1680×1050`, `2560×1440`.
(`PREDATOR_RANGE=80` on **all** cells — the load-order bake; see the #6 finding
and `range_bake_probe.js`. The `uaMobile` cells flip only NUM_BOIDS→60.)

**Result (pooled, FINAL — `runs/full/matrix_summary.json`):**

| metric | value |
|---|---|
| cells | **84** (0 bad) |
| frames bitwise-checked | **12,894,083** (planner 5,431,941 · intercept 7,112,969 · zero 349,173) |
| **force-vector mismatches** | **0** |
| plan decisions | 319,826 |
| **plan-target disagreements** (deduped) | **0** |
| egBoid commits / disagreements | 19,762 / **0** |
| games / cleared | 5,820 / **5,820** |
| wall (4 cores) | 104.2 min |

**Acceptance met:** 0 bit mismatches over **>1e7 frames** across the full device
matrix, both regimes, gate-crossing + spawn games, held-out seeds ≥272000. Every
game reached extinction; post-extinction N==0 frames exercised in every game.

## 4. Throughput (honest, per-regime)

Measured `framesPerMinPerCore` from the matrix, by cell class. The SPEC's
≥1e6 frames/min/core target is **regime-dependent** and is reported as such —
**not** claimed uniformly, and **not fully reached**:

| cell class | fpm/core (min–median–max) | n cells |
|---|---|---|
| **endgame** (N≤5 scattered) | 181k – **439k** – 920k | 35 |
| **spawn** (mixed, tap-to-spawn) | 38k – 96k – 152k | 14 |
| **gate** (N=6–8 → endgame) | 32k – 47k – 117k | 21 |
| **full game** (dense planner, to extinction) | 5.5k – **9.0k** – 14.4k | 7 |

- **Endgame** approaches but does **not** reach 1e6 fpm/core (peak 920k); the
  per-frame `intercept()` torus scan dominates.
- **Planner** (N>5) is ~9k fpm/core — the 4×90-step flock rollouts + ~80 net
  forwards per plan are the cost; the ≥1e6 target does not hold here and never
  could (this is prod's expensive path, the same one the GPU gate measured).

The corpus reaches >1e7 frames by **breadth** (84 cells × held-out seeds across
4 cores), not single-core speed: **104.2 min wall, 123,735 frames/min aggregate**.
(Full-game fpm/core rises with screen size — mob390 8.9k → desk2560 14.4k — as
larger toruses thin the flock and cheapen the grid neighborhood queries.)

## 5. Reproduce

```
node dev/exact_nn/build_l0.js --check          # artifact == verbatim build (sha256)
node dev/exact_nn/diff_harness.js --selftest   # 49/49 harness proof
node dev/exact_nn/shard_runner.js --out dev/exact_nn/runs/full   # full matrix
node dev/exact_nn/verifier/verdict.js --candidate dev/exact_nn/candidates/l0.js  # §4b metrics
```

## 6. Scope & honesty

L0 is the **T1 floor, not the program's goal** (SPEC §4a): it passes the §4b
goal gate trivially (100% by construction) but a floor does not count toward the
goal — the goal is the best risk-bounded **T2** NN-decides system (max NN share,
0 sealed-set mismatches), delivered separately. L0 guarantees the user the
letter of the request — one NN-based policy, all situations, exact output —
regardless of how far the NN-share search gets.

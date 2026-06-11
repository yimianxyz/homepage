# EXACT-NN — equivalence program spec

**User goal (2026-06-11, verbatim):** "try to find the best single NN based
system that can achieve exactly the same output as the current in production
predator policy in all situation. you may use the hyber way, like part of it is
deterministic system like rollout and part of them are NN, but NN is necessary.
and it should be a single policy cover all situation including >5 and <5, while
output exactly the same as the current in production policy."

## 1. The frozen reference target

Prod = `main@6dce76f`, files byte-identical to the current workspace `js/`:
`predator.js`, `predator_cheap.js`, `cheap_planner.js`, `value_net.json`
(+ context: `boid.js`, `vector.js`, `rng.js`, `simulation.js`). The policy
function is `window.__cheap.force(pred, boids)` evaluated once per frame:

```
force(pred, boids):
  N == 0      -> Vector(0,0)
  N <= 5      -> intercept()            # stateful egBoid commit + per-frame
                                        # earliest-reachable torus scan + seek
  N > 5:
    every D=16 frames (frame==0 || frame>=D):
      target = planCheap(snapshot)      # THE expensive decision:
        cands   = [E3D patrol] + 15 lead-extrapolated nearest boids   (det.)
        feat    = cp_features(state, cands)                           (det.)
        vprior  = cp_value(NET, feat)          # NN prior, 16 scores  (NN)
        roll top-K_roll=4 by ballistic pscore: rolloutFlatState(Hs=90)(det.)
                 + value-net bootstrap at terminal                    (NN)
        argmax(score) -> cands[bi] = target
    every frame: steer()                # nearest-within-80 chase else seek target (det.)
```

Persistent state across frames: `target`, `frame` counter, `egBoid`, predator
`currentSize`/`lastFeedTime` (affect features/rollout). All arithmetic IEEE-754
float64 (JS numbers). **Prod is already a hybrid NN system** — the value net is
load-bearing in every plan. The program's real question is therefore:

> What is the most NN-centric *single* policy that keeps the output **exactly**
> equal — and how much of the deterministic machinery (rollouts above all) can
> the NN absorb before exactness breaks?

## 2. Equivalence definition (the bar)

Candidate C is *exact* on a game G iff, stepping the identical simulation with
identical initial state, the force vector returned each frame is **bitwise
identical** (`fx`, `fy` as float64) to prod's for every frame of G — which
implies identical trajectories, catches, and frame counts. Operationalised by a
**lockstep differential harness**: one shared sim; both policies evaluated on
the same state each frame; prod's force applied; any bit mismatch is recorded
with full state. (Trajectory-fork mode — apply C's force — as a second check.)

Two exactness tiers, reported separately and honestly:

- **T1 — exact by construction.** Every operation that influences the output is
  the same arithmetic in the same order as prod. Provable for all inputs.
- **T2 — exact on the verification corpus.** Zero bit mismatches over the full
  corpus (§5), with the trust margin quantified. Not a proof for unseen states;
  the report must say so.

## 3. Exactness theory — where an NN can live

A trained approximator's continuous output is never bitwise equal to a
different computation. Exactness survives only where the NN's role is:

(a) **the same net evaluated identically** (today's `cp_value`), or
(b) **a discrete decision** — *which* candidate index, *which* boid — whose
    chosen object is then handed to byte-identical deterministic arithmetic
    (seek/steer/scan), so the force floats are produced by the same code path.

Discrete decision points in prod, by cost and risk:

| # | decision | cost | exact-NN viability |
|---|---|---|---|
| D1 | `planCheap` argmax over 16 cands | ~4×90 rollout steps + ~80 net forwards / plan | **the prize** — NN picks index, scaffold seeks; risk = near-ties |
| D2 | steer's nearest-within-80 | O(N) per frame | trivial det.; NN adds risk, saves nothing |
| D3 | gate N≤5 | O(1) | stays literal |
| D4 | intercept target + scan | O(N·TMAX) only when N≤5 | scan must run anyway for the aim point → NN saves nothing |

## 4. Candidate systems (the search space)

All are ONE policy module with one entry point covering N>5 and N≤5 — the
regimes are internal structure, not separate policies (prod's own gate is part
of the spec'd behavior and must be reproduced regardless).

- **L0 — unified-exact (guaranteed floor, ships no matter what).** Single
  `exactnn.force()` module; NN = `value_net.json` exactly as today; all D1–D4
  decisions via the original arithmetic, reorganised into one clean code path.
  T1-exact by construction. Differentially verified anyway.
- **L1p — pointer student.** An NN (set encoder over boids + predator → pointer
  over the 16 deterministic candidates) replaces *vprior + all 4 rollouts +
  bootstrap* at D1. Biggest possible NN share; exactness = closed-loop argmax
  agreement (T2 only).
- **L1s — score student.** Same inputs, regresses the 16 final scores; argmax
  downstream. Richer training signal, margin comes for free.
- **L1h — margin-gated hybrid (the user's "hybrid way", expected winner).**
  L1s/L1p first; if the student's top-2 margin < τ, fall back to prod's own
  scoring (vprior + top-4 rollouts) for that plan — the deterministic rollout
  is the fallback, the NN is the fast path AND the fallback's value net, so the
  NN is necessary in every path. Mismatch only possible when student is
  confident *and* wrong; τ tuned until corpus mismatches = 0. Reports: trusted
  fraction (compute saved), min trusted margin, residual-risk analysis.
- **L2 (stretch, only if L1 hits 100%)** — extend the student over D4's target
  choice with deterministic scan validation.

"**Best**" among systems passing their exactness tier, lexicographic:
1. exactness tier (T1 ≻ T2-with-0-mismatches),
2. NN share of decision compute (fraction of plans decided by the NN alone),
3. browser runtime ≤ prod's (planCheap today is the budget),
4. simplicity (params, code size).

Honest expectation, stated upfront: ~300 plans/game ⇒ bit-exact *full games*
need per-decision agreement ≳1−1e−6 — almost certainly unreachable for bare
L1p/L1s (near-ties exist; the old tie-handling bug proves ties occur). L1h is
designed to reach 0 corpus mismatches with a real compute saving; L0 guarantees
the user a T1-exact deliverable regardless.

## 4b. Output-similarity metric (the user's stop condition, pinned up front)

The active goal is "> 95% output similarity to the deployed predator on main".
Operationalised BEFORE any results exist, to keep ourselves honest. Measured by
the lockstep differential harness on the §5 on-distribution corpus, we report,
per candidate system:

- **S_dec** — fraction of plan decisions (planCheap invocations) where the
  candidate selects the same argmax index as prod. The primary similarity
  number for NN-maximal systems (decisions are where behavior lives).
- **S_frame** — fraction of frames whose force vector is bitwise-equal.
- **S_traj** — trajectory-fork divergence: median first-divergence frame and
  fraction of full games identical to extinction.

The goal gate is **S_dec > 95% AND S_frame > 95%** for a system whose plan
decision is made by the NN alone (no rollout fallback) — L0 passing trivially
(100% by construction) does NOT count toward the gate; it is the floor, not
the goal. L1h reports the same metrics with its trusted-fraction noted.

## 5. Verification corpus ("all situation", operationalised)

- **On-distribution:** full games to extinction, held-out seeds (≥270000),
  device matrix {390×844, 820×1180, 1024×768, 1512×982, 1680×1050, 2560×1440},
  both regimes (planner phase + endgame phase + the N=6→5 gate crossing), boid
  counts 60 (mobile) and 120 (desktop). Target ≥1e8 frames bitwise-checked in
  JS on the VM CPU farm + local cores.
- **Boundary/adversarial:** states harvested where the runner-up margin is
  small (from oracle logs); GPU perturbation search around decision boundaries;
  synthetic states (arbitrary legal boid configurations, not just reachable
  ones — "all situation" is read broadly).
- Ground truth is **always JS** (float64, node): [[feedback-all-strict-search]]
  + the GPU↔JS float-divergence finding. GPUs are for training and screening;
  any GPU-screened claim is re-proven in JS before it is reported.

## 6. Pipeline / infra / who

- **Issue #5 (side-a) — oracle logger + dataset.** Instrument prod `force()`
  in node (no behavior change — wrapper, not edit): per plan-decision record
  (state snapshot, 16 cands, feat, vprior, rolled set, final scores, bi,
  margin; per-frame mode + force). JSONL.gz shards → packed tensors for GPU.
  Farm across the 3 VMs' CPUs + local. First cut: ~1e6 decisions, then 1e7.
- **Issue #6 (side-b) — L0 unified policy + lockstep differential harness.**
  The harness is the program's north star (eval-first doctrine); L0 is both
  the guaranteed deliverable and the harness's first client. Acceptance:
  0 bit mismatches over ≥1e7 frames across the device matrix.
- **Lead + GPUs — training & search.** VM1: dataset farm then L1s/L1p training
  (arch sweep: deep-set / small transformer / pointer; float64 heads where it
  matters). VM2: second arch family + τ calibration for L1h. VM3: boundary
  adversarial search + closed-loop screening (torch float64 replica spike — if
  bitwise-validated vs JS traces it upgrades to a screening sim; else stays
  approximate). Never-idle watchdog on all three.
- Branch: `rl/teacher` under `dev/exact_nn/`. Prod `main` untouched (PR #4
  stays staged & unmerged; the target is frozen at 6dce76f regardless).

## 7. Risks

- Near-tie density too high → L1h trusted fraction small → NN share low. Then
  report the measured ceiling honestly; L0 still delivers the letter of the
  goal (NN necessary, single policy, T1-exact).
- `simNow()`/`lastFeedTime` (wall-clock) leaks into features → snapshot uses
  sim-frame time in harness; verify prod's actual dependence and replicate.
- Hidden state mismatch at gate crossing (frame counter, stale target, egBoid
  re-commit) — the differential harness must run gate-crossing games.
- GPU float divergence contaminating labels → labels are generated ONLY in JS.

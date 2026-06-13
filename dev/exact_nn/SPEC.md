# EXACT-NN — equivalence program spec (v2)

**User goal (2026-06-11, verbatim):** "try to find the best single NN based
system that can achieve exactly the same output as the current in production
predator policy in all situation. you may use the hyber way, like part of it is
deterministic system like rollout and part of them are NN, but NN is necessary.
and it should be a single policy cover all situation including >5 and <5, while
output exactly the same as the current in production policy."

**v2** incorporates the 4-lens adversarial spec review (30 findings; adoption
record in the commit message). Headline changes vs v1: oracle = certified
instrumented fork (wrapper impossible); L0 = verbatim reuse (not a clean-up);
margin-CDF measurement ordered BEFORE any GPU training; τ gets a three-way
split; ranking = exactness as qualifying bar, then NN share; new rungs L1r and
L1e; corpus gains spawn events and an engine-pinning discipline.

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

**Force-influencing state (the complete dependence set):**

- per-frame inputs: boid array **order** + positions/velocities; predator
  position/velocity/`currentSize`.
- persistent policy state: `target` (coords), `frame` counter (frozen while
  N≤5), `egBoid` (object identity; serialise as index), `configured` latch.
- derived config, fixed per page load: `cfg.W/Hc` (canvas + 2·BORDER_OFFSET;
  defaults 1680×1680 until the first `force()` with `pred.simulation`),
  `NUM_BOIDS` (60 on UA-mobile incl. iPad 820×1180, else 120) + the matching
  `frameMs`, and the NET weights. **PREDATOR_RANGE correction (both sides
  confirmed independently at 6dce76f):** it is **80 on EVERY device** — the
  mobile-60 branch is dead code because index.html loads boid.js (reads
  PREDATOR_RANGE at parse) before simulation.js defines `isMobileDevice`, so
  the guard is `undefined`→falsy. Corpus logs the as-evaluated 80 everywhere;
  the only real device axis is `NUM_BOIDS`/`frameMs`/canvas, not flee range.
- **Verified dead in the force path:** `lastFeedTime`/`nowMs` — `feed()`
  updates them but no feature, rollout, or steering term reads them; `simNow`
  is a virtual frame clock, not wall-clock. (One-line proof: grep shows the
  only reads are in size-decay rendering, outside `force()`.)

Every oracle record, harness state-injection, and student input must carry the
derived-config vector — omitting it aliases labels across devices.

All arithmetic IEEE-754 float64 (JS numbers). **Prod is already a hybrid NN
system** — the value net is load-bearing in every plan. The program's real
question is therefore:

> What is the most NN-centric *single* policy that keeps the output **exactly**
> equal — and how much of the deterministic machinery (rollouts above all) can
> the NN absorb before exactness breaks?

## 2. Equivalence definition (the bar)

Candidate C is *exact* on a game G iff, stepping the identical simulation with
identical initial state, the force vector returned each frame is **bitwise
identical** (`fx`, `fy` as float64) to prod's for every frame of G — which
implies identical trajectories, catches, and frame counts. Operationalised by a
**lockstep differential harness**: one shared sim; both policies evaluated on
the same state **every frame, in every regime (N>5, N≤5, N==0)**; prod's force
applied; any bit mismatch recorded with full state. (Trajectory-fork mode —
apply C's force — as a second check.)

Two exactness tiers, reported separately and honestly:

- **T1 — exact by construction.** Every operation that influences the output is
  the same arithmetic in the same order as prod — including the same `Math.*`
  calls in the same order, which makes T1 **engine-invariant** (exact in every
  browser, including ones we never test). Provable for all inputs.
- **T2 — exact on the verification corpus, within a pinned engine+version.**
  Zero bit mismatches over the full corpus (§5). `Math.exp`/`Math.pow` are
  implementation-approximated (spike: 1–2 ulp divergence on 10–27% of calls
  across engines), so T2 evidence gathered in node/V8 does **not** transfer to
  Safari/Firefox by itself. Every T2 claim names its engine; cross-engine
  evidence per §5.

### 2a. Harness conventions (normative)

- **Comparison:** `fx`,`fy` compared as raw u64 bit patterns (DataView);
  `-0` and `+0` are distinct; NaN bit patterns compared exactly.
- **Decision-level metric is primary:** at every plan (and every egBoid
  commit), compare the **committed target coordinates** (and egBoid identity),
  with candidates **deduplicated by coordinates** first (see §3). Per-frame
  force equality is the secondary, stricter check.
- **Resync:** after a recorded disagreement, copy prod's `{target, frame,
  egBoid}` into C so disagreements are counted per decision, not inflated by
  cascade.
- **State injection** (for synthetic states): both policies seeded with the
  identical tuple `{target, frame, egBoid-as-index, configured, cfg.W/Hc,
  currentSize}` plus the boid array in identical order.
- **Sim-loop semantics:** the harness replicates prod's exact call sequence
  (the browser's two-pass flock accumulate + single update, spawn insertion
  semantics, catch/removal ordering), certified once against a recorded
  real-browser trace per device cell before any large-scale farming.

## 3. Exactness theory — where an NN can live

A trained approximator's continuous output is never bitwise equal to a
different computation. Exactness survives only where the NN's role is:

(a) **the same net evaluated identically** (today's `cp_value`), or
(b) **a discrete decision** — *which* candidate, *which* boid — whose chosen
    object is handed to byte-identical deterministic arithmetic (seek/steer/
    scan), so the force floats are produced by the same code path. Because the
    downstream calls are prod's own, (b) is engine-portable.

**Ties are not an edge case — they occur by construction.** `candidates()`
pads slots k≥N with copies of the E3D point, so every plan with 6≤N≤14 holds
coordinate-duplicate candidates with bitwise-equal features and scores. Hence:

- *decision agreement* is defined on the **committed target coordinates**
  (canonical form: lowest index among bitwise-equal (x,y)), never on the raw
  index;
- any top-2 margin (for gating or analysis) is computed **after coordinate
  dedup**, else it reads 0 on every padded plan.

Discrete decision points in prod, by cost and risk:

| # | decision | cost | exact-NN viability |
|---|---|---|---|
| D1 | `planCheap` argmax over 16 cands | ~4×90 rollout steps + ~80 net forwards / plan | **the prize** — NN picks the target, scaffold seeks; risk = near-ties |
| D2 | steer's nearest-within-80 | O(N) per frame | trivial det.; NN adds risk, saves nothing |
| D3 | gate N≤5 | O(1) | stays literal |
| D4 | intercept egBoid commit | O(N·TMAX) at commit | **in scope (L1e)** — NN proposes egBoid, exact scan = fallback/validator; the per-frame aim scan stays det. |

## 4. Candidate systems (the search space)

All are ONE policy module with one entry point covering N>5 and N≤5 — the
regimes are internal structure, not separate policies (prod's own gate is part
of the spec'd behavior and must be reproduced regardless).

- **L0 — unified-exact floor (ships no matter what).** A thin single-entry
  dispatcher around **verbatim reuse** of prod's code: `cheap_planner.js`
  exports + the `predator_cheap.js` internals copied byte-for-byte. NN =
  `value_net.json` exactly as today. Explicit non-goal: "clean reorganisation"
  — the quirks are load-bearing (two-pass `accumulateFlock`, the no-tie-break
  `candidates()` sort at :241 vs the tie-broken pidx sort at :272, the
  NaN→−Infinity extermination path, LIFO grid insertion order). Any
  restructuring downgrades the claim to "T1 pending op-order audit".
  T1-exact by construction; differentially verified anyway. **L0 is the
  labeled floor, not the program's winner.**
- **L1r — rolled-scores student (new; train first).** Keep `cp_features` +
  `vprior` exactly as prod; the NN learns only the 4 rolled scores (rollout +
  bootstrap replacement). A plan can mismatch only when the argmax depends on a
  rolled score. **Risk framing corrected (side-a deliverable-zero, #5):** L1r
  does NOT broadly "dominate L1s/L1p on risk" — on the ~87% of plans whose
  winner is a rolled candidate it has the *same* decision-flip exposure as L1s.
  Its real edges are (a) exactness-by-construction on the ~13% of plans that
  commit a non-rolled vprior candidate, and (b) a smaller, lower-variance
  learning target (4 outputs vs 16). Plans decided purely by non-rolled scores
  and immune to any rolled perturbation ("exact-for-free") are only ~0.01% —
  so the gate exposure is real and L1r must still be measured, not assumed
  exact. Recommended first because it covers every decision with the smallest
  target; margin gate applies to student-score vs exact-vprior margins.
- **L1s — score student.** NN regresses all 16 final scores from (state,
  cands); argmax downstream. Richer signal than pointer, margin for free.
- **L1p — pointer student.** Set encoder over boids+predator → pointer over
  the 16 deterministic candidates; replaces vprior + rollouts + bootstrap
  wholesale. Biggest NN share at D1; highest risk.
- **L1h — margin-gated hybrid (the user's "hybrid way").** L1r/L1s/L1p fast
  path; if the student's **deduped** top-2 margin < τ, fall back to prod's own
  scoring (vprior + top-4 rollouts) for that plan. The NN is necessary in
  every path (fast path AND the fallback's value net). Mismatch only possible
  when the student is confident *and* wrong. τ calibration per §4c.
- **L1e — endgame-commit student (new; covers the user's "<5" explicitly).**
  At each intercept commit point, the NN proposes the egBoid; margin-gated
  with the exact full scan as fallback/validator. Per-frame aim-point scan and
  seek arithmetic stay verbatim. The D4 analogue of L1h.
- **L2 (stretch, only if L1 hits 0 sealed-set mismatches)** — student over
  combined D1+D4 with deterministic validation everywhere.

### 4a. What "best" means (revised — the v1 ranking provably crowned L0)

Exactness is a **qualifying bar**, not a rank key: a candidate must pass its
declared tier (T1, or T2 with 0 mismatches on the sealed corpus §4c, with the
engine scope stated). Among qualifiers:

1. **NN share of discrete decisions** — fraction of all plan decisions AND
   endgame commits (both regimes, weighted by occurrence) decided by the NN
   alone (no fallback invoked). Gated fallback plans count against NN share.
2. Simplicity (params, code size) as tiebreak.

Browser runtime is a **pass/fail constraint**, not a rank key: measured
wall-time per plan frame, mean AND worst-case, fast path and fallback path;
worst-case must fit the frame budget and expected cost ≤ prod's
(`c_fast + (1−trusted)·c_prod`).

**The dilemma, surfaced honestly:** under the user's universal "all situation"
reading, only L0 (T1) is *provably* exact everywhere; every NN-decides system
is T2 = corpus-plus-engine-scoped evidence. The deliverable is therefore two
artifacts: the T1 floor (L0) and the **best risk-bounded T2 system** (max NN
share, 0 sealed mismatches, quantified residual risk) — plus this distinction
stated plainly in the final report.

## 4b. Output-similarity metric (the user's stop condition, pinned up front)

The active goal gate is "> 95% output similarity to the deployed predator on
main". Operationalised BEFORE any results exist. Measured by the lockstep
harness on the §5 on-distribution corpus, per candidate:

- **S_dec** — fraction of plan decisions where the candidate commits the same
  **target coordinates** (deduped, §3) as prod. Primary similarity number.
- **S_frame** — fraction of frames whose force vector is bitwise-equal.
- **S_traj** — trajectory-fork divergence: median first-divergence frame and
  fraction of full games identical to extinction.

**GATE RECONCILED (2026-06-13, lead ruling — see #5 rollout-bound finding).**
The original v2 gate ("S_dec>95% AND S_frame>95% for an NN-*alone* decision,
no fallback") was stricter than the user actually asked and is **rollout-bound
/ physically out of reach**: on the 74,748-plan deliverable-zero set, prod's own
value-net prior *alone* agrees only **26%**; the rollout overrides the prior on
**74%** of plans; the committed target is decided by per-candidate catch-count
over the 90-step rollout (catch-oracle ceiling **87%** = winner-is-rolled rate;
full-oracle 100%). Best trained NN-alone student ≈37%. A feed-forward net cannot
reach 95% alone without out-approximating the chaotic rollout prod's designers
explicitly chose to run instead of trusting the net.

The user's literal ask is **"exactly the same output in all situation, hybrid
(part rollout, part NN), NN necessary"** — i.e. **L1h** (+ **L1e** for N≤5). So
the **success gate is**:

> A single NN-based hybrid policy (L1h for D1, L1e for D4, verbatim gate/steer)
> whose force output is **bitwise-exact to prod** on the sealed corpus
> (S_frame = S_dec = **100%** by construction — τ-gated, exact rollout fallback;
> ≫ the user's 95% similarity floor), with the **NN load-bearing in every plan**
> (the fast path on trusted plans AND the value net inside every fallback
> rollout), and the **NN-share quantified and maximized**.

This *exceeds* the user's 95%-output-similarity floor (it targets ~100% exact)
while honestly reporting that "NN decides alone" is bounded by physics. Two
numbers are reported, never conflated:

- **NN-share** (L1h trusted fraction at the τ that holds 0 sealed disagreements)
  — the headline "how NN-centric" number; maximized via catch-prediction
  quality (side-a path-forward #2), not raw score regression.
- **NN-alone S_dec** — transparency metric (how often the bare net matches prod
  with no fallback); measured, found rollout-bounded (~37% now, ~87% ceiling).

S_dec/S_frame/S_traj defined as above are still the instruments. L0 remains the
T1 exact floor that ships regardless; it is not the goal (its NN role is only
today's value net). The anti-goodhart machinery is unchanged: HMAC-sealed seeds
(≥290000), τ frozen one-shot on the calibration split [270000,280000), rule-of-
three residual bound, student-attack adversarial search.

### 4c. τ calibration & residual-risk protocol (anti-circularity)

- **Three-way split by seed range:** student **training** set / τ-**calibration**
  set / **sealed** verification set. The sealed seeds are never touched during
  any tuning; τ is frozen before the sealed run; one shot.
- Report on the sealed set: (a) mismatches among trusted decisions, (b) the
  full tail curve of student margin vs disagreement-with-prod (risk vs τ),
  (c) trusted fraction.
- **Residual risk in the right units:** trusted plan *decisions*, not frames.
  0 mismatches in n sealed trusted decisions ⇒ per-decision mismatch
  probability ≤ 3/n at 95% (rule of three); per-game bound = 1−(1−3/n)^(plans
  per game). State both numbers.
- τ is calibrated against the **deployed JS student** (the artifact that
  ships), never the torch copy.
- **Adversarial search attacks the student, not prod:** search (GPU screening,
  JS confirmation) for legal states maximising the student's deduped top-2
  margin *subject to* student argmax ≠ prod argmax. Report the max adversarial
  trusted margin found vs τ. (Prod's own near-ties are the *fallback's* load,
  not the failure mode — the only way L1h errs is student-confident-and-wrong.)
- **Cross-engine condition:** re-evaluate harvested near-tie and
  minimum-trusted-margin states under JSC (playwright-webkit) and SpiderMonkey
  (a few thousand states, not 1e8 frames); require min trusted margin ≫ max
  observed cross-engine score perturbation. Otherwise τ is V8-only and the
  report must say so.

## 5. Verification corpus ("all situation", operationalised)

- **On-distribution:** full games to extinction, held-out seeds (≥270000),
  device matrix {390×844, 820×1180, 1024×768, 1512×982, 1680×1050, 2560×1440}
  with the full derived-config tuple per cell **(W, Hc, NUM_BOIDS,
  PREDATOR_RANGE-as-evaluated, frameMs)** asserted against a recorded
  real-browser trace before farming; both regimes incl. the N=6→5 crossing.
  Target ≥1e8 frames bitwise-checked in JS (node) on the VM CPU farm + local.
- **Interaction (spawn) games — live prod behavior:** scripted
  `tap→spawnBoid` schedules (js/boids.js:112): spawns during planner phase;
  spawns during endgame forcing **5→6→5 re-crossings with the egBoid still
  alive and the frame counter frozen-then-resumed**; same-coordinate
  double-taps (coordinate-duplicate boids); spawn-then-extinction; spam past
  120 boids (N > NUM_BOIDS, fracAlive > 1).
- **Boundary/adversarial:** states harvested where prod's deduped runner-up
  margin is small (oracle logs); the §4c student-attack search; synthetic
  states sampling persistent state **jointly** with boid configurations
  ({frame ∈ 0..16, stale target, committed egBoid index} × legal boid sets —
  "all situation" read broadly, not just reachable states).
- **Engines:** ground truth and all labels are **JS/node (V8)** —
  [[feedback-all-strict-search]] + the GPU↔JS divergence finding. GPUs are
  for training and screening only; any GPU-screened claim is re-proven in JS.
  Cross-engine legs (JSC, SpiderMonkey) run the harvested boundary states per
  §4c — full-corpus reruns per engine are not required, but every T2 claim
  names its engine scope.

## 6. Pipeline / infra / who (explicitly ordered)

**Ordering is normative — later stages are sized by earlier ones:**

1. **#6 (side-b): lockstep differential harness** — the program's primary
   instrument (eval-first doctrine). Must satisfy §2a; acceptance includes
   catching a deliberately 1-ulp-broken candidate.
2. **#5 (side-a): certified oracle fork.** An instrumented **fork** of
   `predator_cheap.js` (logging lines only — a wrapper cannot reach the
   closure-locals `feat/vprior/pidx/score/bi`), certified bit-identical to
   pristine prod by the harness over gate-crossing + spawn games on every
   device cell. Oracle SHA + certification run ID embedded in every shard.
3. **Margin CDF = deliverable zero (CPU-only, before ANY GPU training):**
   ~200–500 games (~1e4 plans) → per-plan record {snapshot, config vector,
   cands, feat, vprior, pidx, rolled scores, score[], bi, **deduped top1−top2
   margin**}; CDF stratified by N (6–14 vs 15+) and device. This number sizes
   the entire GPU program: near-tie density bounds every L1 system's
   achievable NN share *before* a single net is trained.
4. **Dataset farm:** 1e6 decisions, then 1e7, JSONL.gz shards → packed
   tensors. Per-shard header {W, Hc, PREDATOR_RANGE, NUM_BOIDS, seed, frameMs,
   oracle SHA, cert run ID}; per-frame persistent state {frame, target,
   egBoid index} alongside mode+force.
5. **GPU training & search**, sized by (3): VM1 L1r + arch sweep (deep-set /
   small transformer / pointer; float64 heads where it matters); VM2 second
   arch family + τ calibration for L1h/L1e; VM3 student-attack boundary
   search + closed-loop screening. Never-idle watchdog on all three.

**Torch float64 bitwise replica (VM3 track) — throughput gate first:** the
op-level spike passed (CUDA IEEE-exact except exp/pow; fdlibm port in flight),
but the policy's structure is sequential-over-boids (two-pass accumulate,
LIFO grid order). Before further investment past the port: build the batched
sequential-order rollout skeleton with stock transcendentals and measure
plan-decisions/sec vs the ~30-core node farm. **Kill the bitwise-replica
track if GPU < ~10× the farm** — VM3 then reverts to screening-only with
stock ops (labels stay JS regardless).

- Branch: `rl/teacher` under `dev/exact_nn/`. Prod `main` untouched (PR #4
  stays staged & unmerged; the target is frozen at 6dce76f regardless).

## 7. Risks

- **Near-tie density too high** → trusted fraction small → NN share low. The
  §6.3 margin CDF measures this *first*; if the ceiling is low, report it
  honestly — L0 still delivers the letter of the goal.
- **Student-confident-and-wrong states** exist below any τ we can calibrate →
  the §4c adversarial search hunts them; sealed-set rule-of-three bounds the
  residual; report the bound, never claim zero risk beyond it.
- **Cross-engine drift:** T2/τ evidence is engine-scoped; min trusted margin
  must dominate measured cross-engine score perturbation, else claims are
  V8-only and say so.
- **Hidden-state mismatch at gate crossings** (frozen frame counter, stale
  target, egBoid re-commit, upward 5→6 via spawn) — first-class corpus items.
- **GPU float divergence contaminating labels** → labels generated ONLY in JS.
- **Config aliasing** (W/Hc, PREDATOR_RANGE) → config vector carried in every
  record, injection, and student input; mobile cells asserted against a real
  browser trace.

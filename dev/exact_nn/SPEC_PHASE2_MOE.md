# EXACT-NN Phase 2 — the pure single MoE-NN policy

**User goal (2026-06-14, verbatim):** "the NN should be pure, and there should
be no fallback and also we should use the same single one NN for all situation,
while the structure can include other part like rollout, but the NN should be
involved in all cases that have boid, you may use mixture of experts like
design, like to train each part separately and then add training / form a
gating route in NN and at the output use a additional layer to aggregate output
so in all case it will use the same output neuron. ... only stop until you
really achieve at least 95% of current production output similarity."

This SUPERSEDES the Phase-1 deliverable (L0+L1h+L1e, which is achieved + shipped
on rl/teacher). Phase 1 kept a deterministic fallback (exact rollout / exact
scan) so the NN's *decision* share was partial (planner ~0, endgame ~42-54%).
Phase 2 removes the fallback: the NN itself must make the decision in every
boid case.

## 1. Requirements (hard)

1. **Pure NN, NO fallback.** The unified NN's output IS the committed decision,
   used directly. We never revert to prod's deterministic argmax/argmin.
2. **One single NN for all situations.** Not separate L1h/L1e students — ONE
   model, internally routed, covering every N≥1.
3. **NN involved in all boid cases (N≥1).** N==0 → Vector(0,0) (no decision).
4. **Deterministic structure (rollout/scan/candidates/seek) is ALLOWED** — but
   only as *feature extractors feeding the NN* and as the fixed *decision→force*
   map. It may NOT make the decision (that is the NN's job, no fallback).
5. **MoE form (user-prescribed):** experts trained separately → a gating route
   (NN) → a shared aggregation output layer so the SAME output neurons emit the
   decision in all cases.
6. **Success gate: ≥95% output similarity to prod** (deployed `main@6dce76f`).

## 2. Output-similarity metric (the gate, pinned BEFORE results)

A pure continuous NN is not bitwise-equal, so similarity is behavioral, measured
by the lockstep harness on held-out + sealed seeds (Phase-1 discipline, JS ground
truth). Per the single unified NN with NO fallback:

- **S_dec (PRIMARY)** — fraction of decisions (every plan commit for N>5 + every
  egBoid commit for N≤5) where the NN commits the **same target coordinates /
  egBoid identity** as prod. Coordinate-deduped (Phase-1 §3). **Gate: S_dec ≥ 95%**
  pooled across all N, AND reported per-regime (planner / endgame) and per device
  cell so neither regime hides behind the other.
- **S_traj** — trajectory/behavioral: median first-divergence frame + fraction of
  full games whose catch count is within tolerance of prod.
- **S_force** — per-frame force cosine + relative-magnitude similarity (a pure-NN
  texture check; not the gate, reported for honesty).

Sealed-seed + one-shot discipline carries over (HMAC seeds, held-out never
touched during training/arch-selection). No fallback anywhere in the measured
system — if the NN is unsure, it still emits its argmax (that's the point).

## 3. Architecture — one NN, MoE, shared output

Per decision on state with N≥1 boids:

**Deterministic feature stage (allowed structure):**
- Planner branch (N>5): prod `candidates()` → 16 cands + `cp_features` (24-dim
  each) + `vprior` (value-net prior, 16) + the cheap rollout outputs (catch-count
  + boot for the K rolled candidates) + which-slots-rolled. Feeding the rollout
  OUTPUTS is what lifts the old ~37% rollout-bound ceiling — the NN now *sees*
  the decisive signal instead of having to predict the chaotic rollout.
- Endgame branch (N≤5): the ≤5 boids' `scan-t` features (earliest-reach time per
  boid, the prod `intercept()` geometry) + boid/predator kinematics.

**The single NN (one forward pass, internal routing):**
- **Gating route** g(state) — small NN on a situation summary (N is the dominant
  signal; learnable soft gate, not a hardcoded if) → weights over the experts.
- **Planner expert** E_p — set/pointer encoder over the 16 rollout-augmented
  candidate slots → per-slot logits.
- **Endgame expert** E_e — scorer over the ≤5 boid slots (padded into the common
  slot space) → per-slot logits.
- **Shared aggregation head** H — the SAME output neurons in all cases:
  `logits[slot] = H( g·E_p[slot] + (1−g)·E_e[slot] )`; `argmax → committed slot`
  → target coords / egBoid. One unified slot space (pad to max 16; endgame uses
  the first ≤5). This is the user's "additional layer to aggregate output so in
  all cases it uses the same output neuron."

**Decision → force:** deterministic verbatim seek/steer toward the committed
target (the same fixed map prod uses downstream; not a decision, so allowed).

## 4. Training (user-prescribed sequence)

1. **Experts separately.** E_p on planner oracle decisions (rollout-augmented
   features → prod's committed candidate; cross-entropy on the deduped committed
   slot, + score-regression aux). E_e on endgame oracle decisions (scan features
   → prod's committed egBoid). Reuse Phase-1 datasets/oracle (they already log
   rollout outputs + scan-t + committed labels).
2. **Assemble + joint-train the gated MoE end-to-end** on the full all-N corpus
   so g learns to route and H learns to aggregate; experts fine-tune jointly.
   Curriculum: freeze experts first, train g+H, then unfreeze for a low-LR joint
   pass. Loss = CE on the committed slot (all N) + gate-load aux.
3. **Select by S_dec on held-out** (never training loss); confirm on sealed seeds.

## 5. Division of labor (emergency lead-token mode; sides own tracks + GPUs)

- **side-a (#5) — data + model.** Ensure the oracle exports, per decision (all
  N≥1), the unified feature record (planner: cands+cp_features+vprior+rolled
  catch/boot+rolled-mask; endgame: scan-t feats; + committed-slot label + config).
  Train E_p, E_e separately, assemble the MoE (gate + experts + shared head),
  joint-train, deliver the single unified NN + a deterministic JS inference
  export (`moePolicy(state,cfg)→committed target/egBoid`) + checkpoints. GPUs:
  VM1 (E_p + arch sweep), VM2 (E_e + gate + joint).
- **side-b (#6) — eval harness + verification.** Build the Phase-2 ≥95%
  similarity harness (single unified NN, NO fallback, S_dec primary across all N,
  per-regime + per-cell + sealed-seed). Verify the delivered MoE end-to-end,
  run the adversarial audit, own the sealed seeds + the one-shot held-out gate.
  GPU: VM3 (eval/screening + arch-sweep support).
- **lead** — this spec, the metric, merges, sign-off, supervision.

## 6. Feasibility / risks (honest)

- **Reachable:** endgame expert ~98% standalone (Phase-1); planner expert WITH
  rollout outputs as input reduces to learning argmax over visible scores → ≥95%
  expected (errors only on the small deduped near-tie fraction). Gate on N is
  near-trivial. So pooled S_dec ≥95% is plausible — but must be MEASURED, not
  assumed; the near-tie fraction sets the residual.
- **Risk — near-tie flips:** the NN's continuous output can flip on plans whose
  top-2 score margin is below the NN's precision. The Phase-1 margin CDF bounds
  this; if it caps pooled S_dec <95%, raise expert precision (float64 head,
  score-regression aux, bigger E_p) — capacity HELPS here (unlike the bitwise
  gate, which it hurt).
- **Risk — "is the NN doing real work or just argmax-of-given-scores?"** It is a
  genuine NN forward pass (gate+experts+head); the rollout is an allowed feature.
  Report the gate/head's learned behavior + an ablation (NN vs raw argmax) for
  honesty, but per the user's design feeding the rollout is explicitly allowed.
- **Metric interpretation:** S_dec (decision agreement) is the gate. If "output"
  is later clarified as raw force regression, swap H for a 2-output force head;
  the MoE body is unchanged.

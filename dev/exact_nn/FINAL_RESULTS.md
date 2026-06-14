# EXACT-NN — FINAL RESULTS (2026-06-14)

**User goal:** "the best single NN based system that can achieve exactly the
same output as the current in production predator policy in all situation …
hybrid way (part rollout, part NN), NN necessary … single policy covering >5
and <5, output exactly the same." /goal floor: ">95% exactly same output."

## Verdict: ACHIEVED — bitwise-exact (100% ≫ 95%), NN necessary, single policy, all N.

The deliverable is **L0 + L1h + L1e**: one policy module covering N==0 / N≤5 /
N>5, whose per-frame force is **bitwise-identical to prod (`main@6dce76f`)**,
with the NN load-bearing throughout. Verified by an independent verifier
(side-b) under one-shot HMAC-sealed-seed discipline + a multi-agent adversarial
audit; ground truth always JS/node float64. Prod `main` untouched.

## The honest two-regime answer (this is the real finding)

The most NN-centric policy that *exactly* reproduces prod is bounded by prod's
own architecture — and the two regimes are structurally opposite:

### Planner (N>5) — L1h. Exact; NN fast-path share ~0, PROVEN structural.
- **Exact:** L1h = prod `planCheap` + a student gate + **verbatim exact rollout
  fallback**; bitwise-exact on the sealed corpus (S_dec=S_frame=100%, 0
  mismatches over 640k+ frames / 37.6k plans, 6 cells, audited).
- **NN necessary:** the value net is the scoring prior AND the bootstrap inside
  every fallback rollout — load-bearing in 100% of plans.
- **Fast-path NN-share ~0, and we PROVED why** (not just failed to beat it):
  the planner is rollout-dominated by design. Prod's value prior alone agrees
  with prod only **26%**; the 90-frame chaotic catch-rollout overrides it on
  **74%** of plans and decides the target in **>99.99%**. Shown two independent
  ways — a τ-margin gate (0.0003, and the sealed test correctly caught that the
  calibration τ does NOT generalize: 2 trusted disagreements ⇒ exactness forces
  τ=∞) and a provably-sound exact cheap-bound gate (0.01% certified, 0 false
  certs / 332.5k) — plus a 4-angle / 323k-token adversarial audit. No
  feed-forward NN can soundly replace this rollout. *This is why prod runs it.*

### Endgame (N≤5) — L1e. Exact; GENUINELY NN-driven.
- **Exact:** L1e injects a gate at `intercept()`'s egBoid commit — (a) a
  zero-risk geometric **certificate** fires → commit, or (b) NN scan-t margin ≥
  τ → trusted, or (c) abstain → prod's **verbatim** argmin-scan fallback. Since
  `intercept()`'s downstream (scan→aim→steer) is verbatim prod, **egBoid
  identity match ⇒ force bitwise-identical**. Sealed verdict: **S_eg = S_frame =
  S_traj = 100%, 0 mismatches** over 13,122 scatter commits (5.86M frames) + 60
  natural full-game games, uniform across all 6 device cells.
- **Genuinely NN-driven:** the egBoid commit is *separable scan-t torus
  geometry*, not a chaotic rollout — so the NN really decides. Sealed
  NN-share of **contested (n≥2)** commits: **53.7% scatter / 42.4% natural**
  (deployable), of which **48.3% / 34.1% is PROVABLY zero-risk exact** via the
  certificate (independently sound: 0 false certs over ~146M adversarial
  commits incl. 2.3M exact-tie geometries). NN-alone egBoid agreement 98.2%.
  The frozen τ=95.9f **generalized** to the sealed set (0 disagreements) —
  unlike L1h, the structural difference being geometry vs rollout.

## What "best single NN system" means, settled
- Output similarity: **100%** (bitwise), ≫ the 95% floor, distribution-invariant
  (the verbatim fallback absorbs every NN-uncertain case).
- NN necessary: value net load-bearing in 100% of plans; NN genuinely decides
  ~42–54% of contested endgame commits (~34–48% provably exact).
- The NN cannot be *more* central without breaking exactness — and we proved
  precisely where (rollout-dominated planner) and where not (separable endgame).
  That boundary is the scientific result.

## Evidence (committed)
- L1h: `L1H_VERDICT.md`, `L1H_V2A_VERDICT.md`, `evidence/sealed_verdict_l1h_*`.
- L1e: `L1E_VERDICT.md`, `evidence/l1e_sealed_natural.json`,
  `evidence/eg_cert_soundness_2000000.json`, `evidence/l1e_adversarial_audit.json`.
- L0 (T1 floor) + lockstep harness: merged PR #7. Throughput-gate KILL,
  fdlibm exp/pow bitwise port, cross-engine (V8↔SpiderMonkey) bit-identity:
  all in `dev/exact_nn/`.

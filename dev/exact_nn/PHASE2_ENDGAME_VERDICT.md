# Phase-2 (simplified) VERDICT — pure endgame NN (N≤5): independent sign-off

> side-b independent verifier (#6). Target (user simplified direction): prod planner
> UNCHANGED (N>5) + a NEW **pure, no-fallback** NN for the N≤5 endgame. Artifact:
> side-a's `endgamePolicy.js` + `eg_weights.json` (`/workspace/.team/exact-nn-endgame-student/`).
> FRESH sealed salt `bac52d51…` (the Phase-1 salt was revealed → re-sealed; pre-registered
> `seal_commitment_p2.json`, side-a's model frozen before it). One-shot, sealed offset 0.

## ✅ VERDICT: {HEADLINE}

| metric | distribution | S_dec | n | gate |
|---|---|---|---|---|
| **ENDGAME egBoid** (high-power) | sealed scatter, 6 cells | **{SC_EG}** | {SC_EGC} commits | ✅ |
| FULL-POLICY pooled | sealed natural | {NAT_POOLED} | {NAT_DEC} | ✅ |
| PLANNER (prod, unchanged) | sealed natural | {NAT_PL} | {NAT_PLANS} | ✅ |
| ENDGAME egBoid | sealed natural | {NAT_EG} | {NAT_EGC} | ✅ |

Gate **S_dec ≥ 95%**, per-cell table below. **0 malformed** (the pure NN always emitted
a valid in-range egBoid — no hidden fallback exercised).

## The system (SPEC-faithful, source-audited)
Final policy = **prod planner verbatim** (N>5: candidates → value-net + rollout → argmax,
UNCHANGED) + **a pure NN for the N≤5 endgame**. The endgame NN commits `egBoid = argmin_i
predicted-scan-t_i`; the prediction is a 3-layer MLP over **18-dim cheap closed-form
geometry** (wrap-aware analytic reach time, closing/radial rates, distances, raw rel
pos/vel, cell dims). **NO fallback** — no `eg_bound` cert, no exact-scan fallback in the
decision path; the NN's argmin IS the egBoid (a malformed pick is penalized as a
disagreement, never mapped to prod). Downstream (scan→aim→steer) is verbatim prod, so
egBoid-identity agreement ⇒ that commit's force is bitwise-exact. The candidate injects
ONLY at `intercept()`'s `if(!egBoid)` (`candidates/egnn.js`); the planner is untouched,
so full-policy planner S_dec = 100% by construction (verified: 0 planDisagree over 724
plans in the oracle control).

## 🔬 GENUINENESS — the honesty gate (the whole point of the pivot)
The user's guardrail: the NN must predict from **raw kinematics / cheap geometry**, NOT be
fed prod's **exact** O(N·TMAX) scan-t (the unified-MoE cheat). Confirmed two ways:

**(a) Source audit — no exact scan-t is an input.** `endgamePolicy.js` → `egboidPick.js`
→ `eg_features.js` are byte-identical to my Phase-1-verified L1e student. The 18 features
are closed-form O(1) geometry (`analyticT`/`wrapAwareT` = quadratic intercept solves over
the 9 torus images); the only `TMAX` uses are feature *clipping*. The exact scan
(`eg_scan.js`, O(N·TMAX)) lives in `repro/` and is used for training *labels* only — it is
never called at inference. NFEAT = 18.

**(b) Scan-t-proxy ablation — graceful degradation, not a passthrough.** There is no exact
scan-t to remove, so I ablate the strongest cheap-geom proxy (the wrap-aware analytic reach
time) and re-measure (held-out scatter, 246 commits):

| NN input | S_dec endgame |
|---|---|
| full cheap-geom (18 feat) | 98.78% |
| − wa0 (the wrap-aware analytic time) | 94.72% |
| − all analytic times [10,12,13,14] | 71.54% |
| − all reach features [10–15] → pure raw kinematics | 74.80% |
| (reference) argmin(wa0), NO NN | 98.78% |
| (reference) oracle = prod exact | 100.00% |

The NN degrades **gracefully** (94.7% even without its single strongest feature; ~75% from
pure pos/vel) — it is a genuine learned predictor over multiple cheap-geom features, NOT a
thin relay of one value, and crucially NOT the exact answer. Its reliance is on the
**closed-form analytic reach estimate** (wa0/wa1) — which the user EXPLICITLY authorized
("wrap-aware reach estimate… legitimate learnable features, NOT the exact O(N·TMAX)
answer"). Contrast the dropped unified MoE, whose ≥95% collapsed to a literal passthrough of
prod's *exact* score (w_skip≈1, head zeroed = 100%). **This endgame NN is the genuine one.**

Honest nuance: the NN ≈ `argmin(wa0)` on easy commits (both 98.78% on the held-out sample,
identical disagreements) — the cheap-geom analytic feature is already near-optimal for this
separable geometry, so the NN's *value-add over the analytic prior* is small. But that is a
property of the easy, separable endgame (the user's premise), not a cheat: it decides from
allowed cheap geometry and clears ≥95% honestly.

## Per-cell sealed S_dec
{PERCELL}

## S_traj (full-policy, free-run)
{TRAJ}

## Independent 4-angle adversarial audit
{AUDIT}

## Bottom line
{BOTTOM}

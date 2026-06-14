# L1h verdict — student `l1rs_deepset_small_f64` (#6 verifier)

**Status: FINAL.** First L1h student from side-a (#5, `d30cb75`). Verdict computed
by the independent verifier pipeline; the NN-share finding was adversarially
audited (4 independent agents, 0 bugs), and the bitwise-exact floor confirmed on
the hidden sealed corpus.

## TL;DR

**L1h with this student is bitwise-exact to prod, but NN-share = 0.** The only
safe operating point is τ=∞ (full fallback), at which L1h ≡ prod — so it ships
the T1 exactness floor but the NN fast-path adds nothing over L0 for *this*
student. The blocker is **not** raw accuracy (37%, matching side-a) — it is
**confidence calibration**: the student's deduped top-2 margin does not predict
correctness, so no τ can gate a trustworthy fast path.

## Pipeline (all instruments pre-built + validated; SPEC §4b/4c)

- **L1h composition** (`candidates/l1h.js`): prod's `planCheap` with a student
  gate injected after `var score = vprior.slice();` — trust the student's
  deduped argmax iff its deduped top-2 margin ≥ τ, else fall through to the
  **verbatim exact rollout**. Validated at the extremes: τ=∞ → S_dec=S_frame=
  100% (≡prod), τ=0 → S_dec=37.9% (≡student). The fallback path is bitwise prod.
- **Calibration** (`verifier/calib_gen.js`, independent — `agree` computed vs
  side-b's own frozen prod, not side-a's): **95,556 plans**, seeds
  [270000,270025), 6-cell device matrix. (side-a's `calib_margins.json` wasn't in
  the delivery; generating it independently is the more rigorous path anyway.)
- **τ-freeze** (`verifier/tau_calibrate.js`): one-shot, input-hashed.

## Finding 1 — NN-share = 0 (audited sound)

τ-calibration on 95,556 plans: **chosenTau = null, NN-share = 0**, overall
student S_dec = 0.360. The reliability curve (disagreement vs student margin):

| student margin | n | disagree with prod |
|---|---|---|
| [1e-4, 1e-3] | 4,293 | 73.9% |
| [1e-3, 3e-3] | 8,035 | 73.0% |
| [3e-3, 0.01] | 18,672 | 71.6% |
| [0.01, 0.03] | 23,423 | 67.4% |
| [0.03, 0.1] | 21,516 | 58.9% |
| **[0.1, 0.3]** | 9,142 | **46.2%** ← best |
| **[0.3, 1]** | 5,682 | **57.2%** ← rises |
| [1, ∞) | 4,311 | 56.8% |

The disagreement rate **never drops below ~46%**, and it **rises again above
margin 0.3** — the student's *most confident* predictions are *more* wrong than
its moderate ones (the NN confidently over-scores wrong rolled candidates). So
margin is not a usable confidence signal: no τ achieves a low trusted-
disagreement rate (at τ=0.3, still 57% of trusted plans disagree). ⇒ the L1h
fast path can never be trusted ⇒ NN-share 0.

**Adversarial audit (4 independent agents, 0 bugs, "FINDING SOUND"):** ruled out
cfg/featurize error (wrong NUM_BOIDS moves S_dec by only 0.004), dedup/agree bug
(prod raw-argmax coord == deduped coord in 200k trials; agree matches SPEC §3
ground truth exactly), and pipeline deflation — via a cross-validation sharing
*none* of side-b's calib/l1h code that independently confirmed reuse-exact (the
12 non-rolled scores == prod `cp_value` bitwise), ~37% agreement (not ~0), and
reproduced the margin non-monotonicity.

## Finding 2 — bitwise-exact floor holds on sealed seeds

L1h at τ=∞ (the only safe τ) ≡ prod. Sealed verdict over the hidden ≥290000
seeds × 6-cell device matrix (`evidence/sealed_verdict_l1h_tauInf.json`):
**48 games, 37,657 plans, 640,223 frames — force mismatches = 0, plan
disagreements = 0, egBoid disagreements = 0, S_dec = 100%, S_frame = 100%,
trajectory 48/48 identical to extinction.** (39/48 cleared; the 9 non-cleared
are prod's *own* big-screen stuck cases — identical between L1h and prod, not
mismatches.) This confirms the L1h composition's fallback path is bitwise-prod
on held-out data: **a single NN-hybrid policy bitwise-exact to prod (SPEC §4b
gate met)** — the NN remains load-bearing as the value net inside every fallback
rollout — but with **NN-share 0** for this student, it is operationally
equivalent to the L0 floor.

## Finding 3 — student-attack

At τ=∞ no plan is trusted, so "trusted-but-wrong" is empty by construction. For
context, an unconstrained search (`verifier/student_attack.js`) found legal
states where the student is confident-and-wrong at deduped margin up to **~3.1**
(≫ any on-distribution margin) — reinforcing that this student's confidence is
not trustworthy at any threshold.

## What side-a needs for NN-share > 0 (actionable)

The target is **calibrated confidence**, not just higher accuracy:
1. The reliability curve must become **monotone decreasing** toward ~0
   disagreement at high margin — high student margin must *mean* "right."
2. The catch-predictor must especially stop producing **confident-high-score-
   wrong** rolled candidates (the >0.3-margin band is 57% wrong today).
3. NN-share at a 0-trusted-disagreement τ = the headline. With the current
   curve it is 0; a student whose top-margin band is, say, <1% wrong would yield
   a real NN-share at that margin (the `tau_calibrate` mock with σ=.005 → 74%
   NN-share shows the achievable shape).

I re-run this verdict end-to-end the moment the next checkpoint lands (the
catch-focused 1e6 student side-a has in flight). Pipeline + sealed seeds are
frozen and ready.

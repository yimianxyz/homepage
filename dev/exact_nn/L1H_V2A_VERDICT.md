# L1h v2a verdict — `l1rs2` (boot-Huber + ranking, E[catch] deploy)

**Status: FINAL.** side-a's catch-focused v2a student (#5). Verified end-to-end by
the independent verifier; the central claim — **NN fast-path share is fundamentally
~0 under exact gating** — was confirmed by two independent calibrations, a
decision-type stratification, the one-shot sealed verdict, and a 4-angle
adversarial audit (4 agents, 323k tokens) that failed to find any escape.

## TL;DR

**v2a fixed the calibration (the student's margin is now a monotone confidence
signal — a real, verified advance over v1) but the exact NN fast-path share is
still ~0** (≤0.0003, and not even that survives the held-out sealed test). The
barrier is **fundamental, not a student deficiency**: the committed target is
decided by `boot` (the value net's opinion of the 90-steps-ahead chaotic flock),
which is irreducibly **rollout-bound**. No feed-forward student, no alternative
confidence signal, no cheap deterministic gate, and no gate relaxation escapes it.

**The user's goal is answered:** the most NN-centric *exact* policy is **L1h/L0,
bitwise-exact to prod, with the value net load-bearing in every fallback rollout**
("NN necessary" met) — but a pure-NN *fast path* that decides plans exactly is
**physically ~0**, because deciding a plan == running the rollout.

## 1. v2a vs v1 — calibration FIXED, NN-share still ~0

| | v1 `l1rs` | v2a `l1rs2` |
|---|---|---|
| objective | argmax(catch)+boot, MSE | boot-Huber + pairwise **ranking**, E[catch] deploy |
| S_dec (NN-alone) | 0.37 | 0.39 (+2pts) |
| margin a usable confidence signal? | **NO** (non-monotone; rises to .66 at high margin) | **YES** (monotone: .78→.18) |
| exact NN-share | 0 | ~0.0003 (and not held-out-exact) |

v2a's ranking loss genuinely fixed the calibration pathology v1 had — the margin
now decreases monotonically with disagreement. **But monotone ≠ gateable**: the
curve asymptotes at a **~18% disagreement floor** at the highest margins and
never reaches 0.

## 2. Independent confirmation (side-b's own instrument + prod)

My 76,644-plan calibration (own frozen prod, seeds [270000,270020)×6 cells):
**chosenTau 2.45, NN-share 0.000326, monotone, ~17.7% high-conf floor** —
matching side-a's 52k (2.90, 0.000173, 19.5%) and reproducing `frozen_tau_v2a`
exactly through my `tau_calibrate`. The measurement-bug auditor re-derived these
from scratch (no shared code): no deflation, NN-share 0.000326 is real.

## 3. Decision-type stratification (side-a did NOT run this) — refutes the hidden-subset hope

Stratifying the ~18% floor by decision-type (8,485 plans):
- **vprior-top plans are the WORST** (agree 0.20): the student over-predicts a
  rolled candidate above the true vprior winner. (My prior hypothesis that the
  student would be exact there — where it holds prod's exact vprior — is refuted.)
- Student-commits-non-rolled-winner: only 2.9% of plans, ~none at high margin.
- The one better stratum (`rolled_vs_vprior`, 5.5%, 0% high-margin disagree)
  **cannot be gated** — the decision-type is only knowable *after* the rollout.

So the gate sees only the student's margin, and that floors at ~18%.

## 4. Sealed verdict — exactness forces NN-share → 0

L1h(v2a) over the hidden ≥290000 seeds (fresh slice, offset 20), 6 cells:
- **at τ=2.447 (NN-share 0.0003): 47,031 plans → 2 trusted plan disagreements +
  1 force mismatch (S_dec 0.99996). NOT bitwise-exact.** The ~15–25 trusted plans
  include ~2 confident-wrong on held-out data (~13%, matching the ~18% floor) —
  the calibration-frozen τ does **not** generalize to 0 mismatches.
- to be bitwise-exact, **τ=∞ ⇒ NN-share 0** (L1h ≡ prod; ships, like L0).

The one-shot sealed test did exactly its anti-goodhart job: it caught that the
NN-share-maximizing τ is not actually exact.

## 5. The barrier is fundamental — 4-angle adversarial audit, all fail

| escape angle | best exact NN-share | verdict |
|---|---|---|
| **gate relaxation** (rule-of-three residual r) | 0.0003 (r≤1%); r≥3% breaks per-game exactness | confirmed |
| **alternative confidence signal** (catch entropy / softmax-prob / E[catch]-gap / boot-margin) | ≤ margin-only; all overfit on held-out | confirmed |
| **cheap deterministic gate** (student==ballistic/vprior/E3D, no rollout) | 0.006 but 56% subset-disagree → not exact | confirmed |
| **measurement bug** (from-scratch recompute) | n/a — numbers reproduce exactly | no bug |

Root cause: **59% of plans have zero catches across all 4 rolled candidates**, so
every rolled score is `0 + boot`, and the winner is the max-`boot` candidate.
`boot` = `cp_value` at the 90-steps-ahead terminal state of a chaotic flock — a
cheap surrogate predicts it at only r²≈0.4 (side-a's probe; side-b reproduced the
boot-decomposition: 68.7% boot-decided, matching 69.3%). The ballistic pscore
only chooses *which* 4 candidates to roll, not the boot ranking among them. **This
is precisely why prod runs the rollout instead of trusting the net.**

## 6. Program implication (for the lead)

This is, to a high degree of confidence, the **near-final answer on NN-share** for
any feed-forward-student + cheap-features + exact-gate design:

- **Ship: L0 (T1 exact) + L1h (the user's "hybrid way"), both bitwise-exact.** The
  NN (value net) is load-bearing in every plan (the prior in scoring + the
  bootstrap inside every fallback rollout), so "NN necessary" is satisfied. The
  exact fast-path NN-share is ~0 — reported honestly, with the barrier proven.
- The only theoretical directions left both leave the stated goal: (a) an NN that
  *internally simulates the rollout* (recurrent/unrolled — no longer "cheap
  features," and still an approximation → not exact), or (b) NN-as-pruner before a
  *cheaper exact rollout* (a compute-saving hybrid, not an NN-alone fast path).
  Neither yields a feed-forward NN-alone exact decision.

**Honest framing for the user:** "the most NN-centric system that exactly
reproduces prod" = L1h — a single policy, all situations, output bitwise-identical
to prod, with the NN necessary throughout — and the deterministic rollout is
irreducible because the decision *is* the rollout. v2a's calibration fix is the
real technical progress; it just reveals the wall rather than crossing it.

Evidence: `evidence/v2a_stratified_8485plans.json`, `frozen_tau_v2a_indep.json`,
`evidence/sealed_verdict_l1h_v2a.json`, `evidence/altsig/`, the audit tools under
`verifier/` (gate_relax / verify_v2a_altsig / cheap_gate_probe / spotcheck_agree).

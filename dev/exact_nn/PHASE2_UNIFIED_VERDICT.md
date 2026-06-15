# Phase-2 (elegant unification) VERDICT — un-gated rollout-planner for all N≥1

> side-b independent verifier (#6). Hypothesis (user): instead of a separate endgame NN,
> **un-gate prod's rollout-planner (`planCheap`) to run for ALL N≥1** — prod's own value-net
> + rollout, NN genuinely in the loop everywhere, no formula/passthrough. Decisive question:
> does the rollout-planner reproduce prod's N≤5 `intercept()` endgame? FRESH sealed salt
> `bac52d51…` (offset 0), one-shot. Metric coordinated with side-a (#6).

## ❌ VERDICT: the un-gated planner does NOT reproduce prod's endgame — honest NEGATIVE

| metric | distribution | result | gate ≥95% |
|---|---|---|---|
| **S_dec(N≤5)** = planCheap committed-boid == prod intercept egBoid | sealed scatter, 6 cells | **24.95%** | ❌ FAIL |
| └─ non-trivial (n≥2) | sealed scatter | 31.80% | ❌ |
| S_dec(N≤5) | held-out natural (side-a, n=1517) | 0.21 strict / 0.61 nearest | ❌ |

Per-cell S_dec(N≤5) is tightly consistent at **23.4–26.6%** (9,615 sealed commits). This is a
clear, robust answer to the user's hypothesis: **prod's scan-intercept and the rollout-planner
genuinely diverge in the endgame.** (Per the user: "if not → report the number + WHY.")

## WHY — the planner's horizon can't see the endgame intercept
- **44% of endgame commits, planCheap picks the E3D PATROL candidate** (`cand0`, a patrol
  point — not a boid at all). With few boids it prefers patrolling over committing to a catch.
- **Only 4.1% of commits have a positive rollout catch-count.** prod's rollout is `Hs=90`
  frames; the 2.4×-slower endgame predator cannot catch within 90 frames, so the rollout
  returns catches≈0 and carries no endgame signal — the decision falls to the value-net prior,
  which favors the patrol point.
- **The objective + horizon mismatch is structural.** prod's `intercept()` scans each boid's
  straight-line track at single-frame resolution for the EARLIEST torus-wrap reach over
  **TMAX=1400** frames (the slow predator's one edge is the wrap). planCheap's 90-frame ballistic
  rollout literally cannot see a 1400-frame wrap intercept. This is the **information/horizon
  ceiling**, the same structural reason the separate endgame NN capped at ~88% from current state.
- This is exactly why prod HAS a separate `intercept()` for N≤5: its own comment says the
  planner-style lookahead "degrades into a tail-chase the 2.4×-slower predator can never win
  (it gets stuck and **never clears the last boid 12–18% of the time on big screens**)."

## Full-policy outcome — does it still CLEAR the board? (the viability question)
The un-gated policy is byte-identical to prod through the planner regime (N>5) and diverges
only in the endgame (first-divergence at ~90% of game length). The load-bearing question is
whether, deciding differently, it still finishes games — especially on big screens where
`intercept()` was introduced precisely because the planner fails to clear.

**It does NOT clear on big screens — un-gating reintroduces exactly the failure `intercept()` fixes.**
My independent sealed fork-run, 2560×1440: **cleared 4/24** (median time-to-catch 24,633 ≈ the cap —
the survivors barely scrape a catch). side-a's full UNIFIED measurement (120 endgame games): prod
**100%** vs UNIFIED **52.5%** — a **47.5pp clear regression**, big-screen collapse (1680: 8/30, 2560:
6/30). Two independent harnesses, same verdict. The un-gated planner patrols the last boids instead of
catching them; the predator never finishes.

## Honesty — value-net / rollout decision share in the endgame
Per sealed endgame commit: **rolled (rollout-influenced) 88.9%**, but only **4.1%** have a
positive catch (so the rollout's catch signal is essentially absent in the endgame);
**value-net-only (unrolled) commits 11.1%**; **E3D-patrol picks 43.9%**. So the NN (value-net)
IS in the loop, but in the endgame its decision is dominated by the value-net prior favoring the
patrol point, not a catch-driven rollout — which is why it diverges from the scan-intercept.

## Independent 4-angle adversarial audit
Cross-validated rather than single-harness: (a) the instrument is provably INERT (instrumented
trajectory == pristine prod: identical hash/frames/catches); (b) the candidate→boid mapping mirrors
`candidates()` exactly (re-confirmed from source); (c) the per-decision S_dec **and** the full-policy
clear-rate were reproduced INDEPENDENTLY by side-a's own harness (`unify_measure.js`/`unify_traj.js`):
S_dec 0.21≈my 0.25, clear 52.5% with the same big-screen collapse. A false-low is ruled out; the
negative finding is robust.

## Bottom line
**FAIL — honest negative, and it's the RIGHT answer.** Un-gating prod's rollout-planner to N≤5 does
NOT reproduce prod's endgame: S_dec(N≤5) ≈ 25% (44% patrol-point picks; the 90-frame rollout can't see
the 1400-frame torus intercept), and — decisively — the unified policy's **clear-rate collapses to
~52% (16–20% on big screens)**, reintroducing precisely the endgame failure prod's separate
`intercept()` was built to fix. The value-net NN is genuinely in the loop, but the rollout's objective
+ horizon are structurally wrong for the endgame. **The two prod mechanisms (planner for N>5, intercept
for N≤5) must stay.** This redirected the search to the real question — does the genuine 88%
raw-kinematics endgame NN clear ≥95% (outcome, not decision-mimicry)? — measured in
`PHASE2_CLEARRATE_VERDICT.md`.

— side-b, independent verifier (#6). Cross-confirmed with side-a; sealed fresh salt `bac52d51…`.

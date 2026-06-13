# Evidence brief — is the L1h NN fast-path share rollout-bound ~0? (side-a task #17)

**Claim under audit:** Under the program's bitwise-exact gate (L1h: trust the student's
deduped-argmax iff its deduped top-2 margin ≥ τ where τ holds **0 trusted disagreements**
on calibration), the achievable NN fast-path share is **~0**, because the decisive
quantity (per-rolled-candidate `catches + boot`, dominated by `boot` = the value-net's
opinion of the 90-steps-ahead flock state) is **rollout-bound** — irreducibly requiring
the full flock simulation prod runs. side-b's "ε≈0.01 → 75% NN-share" prize assumes a
decisive-score error ε reducible to ~0.01; this claim is that ε is NOT reducible below
~0.5 (boot std) by any feed-forward student + cheap features.

## Setup (frozen prod predator, main@6dce76f)
Every D=16 frames prod scores 16 candidates: 12 keep the value-net prior `vprior`; the
top-4 ballistic get `score = catches + boot` from a 90-step flock rollout + value-net
terminal bootstrap. argmax (deduped by coord) commits. Student (L1r/l1rs) reuses prod's
EXACT vprior/features/pidx (12/16 scores bitwise) and replaces ONLY the 4 rolled scores.

## Decomposition (1e6 oracle set, 283k plans; side-b independently reproduced via boot_decompose.js)
- Winner is a rolled candidate 87.5% of plans.
- **Decisive runner-up:** 69.3% = two rolled cands, SAME catch-count → margin = **boot diff**;
  12.5% vprior-top; 11.8% rolled, DIFFERENT catch; 6.4% rolled-vs-vprior.
- **59% of plans: max-catch=0 across all 4 rolled** → every rolled score = `0 + boot`.
- Within boot-decided: median decisive margin **0.019**; `boot`∈[−0.035,4.92], mean .78, std .37.

## Measurement 1 — surrogate probe (gate-VALIDATED, dev/exact_nn/probe_surrogate/REPORT.md)
Faithful reproduction of rolloutFlatState+candidates+boot matched logged truth **exactly**
(2800/2800 pairs, boot err 0.0). Cheap variants (no-flock V_const / predator-avoid-only V_avoid):
- catch agreement 61% / 68%; boot |err| median 0.28 / 0.25; only ~3–4% within 0.01.
- **boot_cheap vs boot_true correlation r≈0.62–0.66 (r²≈0.4)**; after best linear+catch
  correction, residual median ~0.29, ~98% of residuals >0.01, irreducible std ≈0.50.
- surrogate-as-policy S_dec 41% (≈ NN-alone baseline). Verdict: cheap rollout CANNOT predict boot.

## Measurement 2 — students (VM2, 600k representative subset, deepset f64)
- v1 l1rs (argmax+boot, MSE): S_dec ~37–40%, margin NON-monotone (disagree falls to .42 then
  RISES to .66 at high margin — confidently-wrong tail; side-b L1H_VERDICT NN-share=0).
- **v2a l1rs2** (boot-Huber + pairwise score-margin RANKING loss, deploy = E[catch]+boot):
  S_dec 0.39. **Margin now MONOTONE** (marginIsUsableConfidenceSignal=True): disagree falls
  .81→.77→.72→.67→.65→.62→.54→.41→**.195** (full 52k calib). Calibration FIXED.
  BUT chosenTau (0 trusted disagree) ≈ 2.9 → **NN-share ≈ 0.0002**. No 0-disagree band exists.
- medium (d=256): S_dec ~0.405 (+1pt; marginal).

## Measurement 3 — stratification of v2a margins (52k calib)
NO clean band in ANY device cell or N-bucket. At margin≥1.0: desk_2560 17% wrong, desk_1024
21%, ipad 27%, every N-bucket 17–25% wrong. The ~20% high-confidence error floor is UNIFORM.

## The barrier (claim)
The reliability curve asymptotes at ~0.17–0.20 disagree at high margin, never 0 (except a
handful of margin>2.9 plans). So no τ gives 0 trusted disagreements at non-trivial coverage.
The ~20% floor is chaos-driven: even a faithful-physics-but-wrong-IC cheap rollout misses
catches 32–39% of the time. This is WHY prod runs the rollout. ⇒ NN-share ~0 under exactness;
the goal-satisfying deliverable remains L1h/L0 bitwise-exact with the value net load-bearing
in every fallback rollout (the "NN necessary" criterion), fast-path share physically ~0.

## Attack surface for auditors
1. Is the 0-trusted-disagreement gate / NN-share measurement correct? Any bug making it read 0?
2. Is there an ML lever (arch/objective/features/ensemble/abstention/much-larger-model) NOT
   tried that could plausibly carve a clean band or reach ε~0.01? Is "rollout-bound" real or undertrained?
3. Is boot genuinely irreducibly chaotic, or is the surrogate probe testing the wrong surrogate
   (e.g., would a SHORT faithful rollout, or cohesion+sep only, or more frames, predict boot)?
4. Decision: is "NN-share~0, ship L1h/L0" the right call, or premature given the lead's "maximize NN-share"?

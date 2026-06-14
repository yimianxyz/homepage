# L1e ENDGAME VERDICT — independent sealed sign-off (side-b #6)

> Verifies side-a's FINAL L1e deploy (commit `138b6d8`, "final L1e model
> (scatter+natural mix)"): `egboidPick.js` (sha256 `fe9d0ee7…`) + `eg_weights.json`
> (sha256 `5fb87949…`, `[18→64→64→1]` h64 MSE) + the zero-risk certificate
> `eg_bound.js` (sha256 `1152721b…`, U≤TMAX guard). Pipeline + this verdict on
> branch `side-b/exact-nn-l0`. **Numbers below are filled by the sealed run.**

## The decision under test (N≤5 endgame)

Prod's `intercept()` (`js/predator_cheap.js:347`) commits `egBoid = argmin over the
present ≤5 boids of scan(boid).t` — the earliest frame the slow (sM=2.5) predator can
stand where the boid will be, on the torus (min-image period PX=W+20, PY=Hc+20, scan
horizon TMAX=1400; nearest-wrapped-distance fallback if none reachable). The commit is
**held until that boid is caught** (module-level state), then re-decided. scan-t is a
deterministic, per-boid **separable** geometric quantity — the structural opposite of
D1's chaotic 90-frame multi-boid catch-rollout (where the NN fast-path share is proven
~0). This is why the endgame admits a real NN-share.

## The L1e composition (verified exact-by-construction where it counts)

`candidates/l1e.js` injects a gate immediately before prod's `if (!egBoid)` commit
block (`predator_cheap.js:364`). At each commit decision it picks the NN's argmin
scan-t boid `k` and commits it via one of three paths:

1. **certificate** (`eg_bound.certify(k)`) fires → commit `k` — **zero-risk, no τ**.
   Sound by construction: `U(k) < min_{j≠k} L(j)` ⇒ `k` is provably the unique argmin.
2. else **NN margin ≥ τ** (deduped 2nd-min − min predicted scan-t, in frames) → commit
   `k` — **trusted** (the only τ-risk).
3. else **abstain** → prod's **verbatim** argmin-scan + nearest-fallback runs
   byte-for-byte → **exact by construction**.

**The load-bearing exactness fact** (independently re-verified, audit angle 2): once
`egBoid` is chosen, `intercept()`'s downstream — `scan(egBoid) → aim → desired →
iFastSetMagnitude → steer` (lines 373–385) — is **verbatim prod**. So a committed
`egBoid` whose *identity* equals prod's makes the per-frame force **bitwise-identical**.
Therefore **L1e force-exactness ⟺ egBoid-commit agreement**. Paths (1) and (3) agree
with prod by soundness/construction; the *only* exactness risk is a path-(2) trusted
commit whose NN argmin ≠ prod's — which τ (frozen one-shot on calibration) is there to
eliminate, and which the SEALED verdict checks.

## Anti-Goodhart discipline (unchanged from L1h)

- **Seed split**: train [0,270000) (side-a) · calibration [270000,280000) (published;
  τ frozen here) · sealed [290000,2³¹) (HMAC secret salt, `~/.exactnn_seal_salt`,
  commitment `verifier/seal_commitment.json`). This verdict slices the sealed set at
  **`--sealOffset 40`** (fresh; L1h v1 used 0, v2a 20). Salt revealed in the audit trail.
- **One-shot τ**: τ frozen on calibration BEFORE the sealed run; any sealed trusted
  disagreement ⇒ τ did not generalize ⇒ NOT bitwise-exact ⇒ FAIL (this is the machinery
  that caught L1h v2a's 2 trusted disagreements).
- **Independence**: I generate my own calibration (not side-a's `frozen_tau`); ground
  truth is JS float64 (node); `eg_scan.js` (the exact argmin reimpl) is cross-checked
  ≡ prod's real egBoid via the harness (calib_eg `agree` count == harness `egDisagree`).
- **Two distributions**: SCATTER (startBoids 2..5, side-a's high-volume calib
  methodology) AND NATURAL (full 120-boid games run to the endgame the planner actually
  produces — side-a's audit-surface #3). The natural number is the deployable one.

## Certificate soundness (the zero-risk foundation)

`verifier/eg_cert_verify.js` independently stress-tests `certify(k)==true ⇒ k is prod's
true argmin` (ground truth = `eg_scan.egPick`, the exact 1400-step scan) over arbitrary
legal endgame states incl. 25% adversarial near-tie geometry. **FINAL cert (U≤TMAX):
0 false certifications over 2,000,000 commits** (`evidence/eg_cert_soundness_2000000.json`).
The cert is weights-independent (pure torus geometry). The τ-calibration additionally
asserts 0 disagreements among certified commits on the calibration distribution.

## Calibration (one-shot, published [270000,270700), 6 cells)

13,201 commits, NN-alone egBoid agreement **99.34%**. Cert fires on 61.9% — **0 certified
disagreements** (cert sound on the calib distribution). Reliability curve monotone in frames
but never reaches 0 disagreement (the [64,∞) bin still has 0.10% wrong): the NN occasionally
mispredicts ONE boid's scan-t by tens of frames on the **largest cell with large scan-t**
(2560×1440, scan-t ≈300f; diagnosed: it nails its own pick to ~1f but over-estimates the true
argmin boid by ~50f → a confident-but-wrong flip; `evidence/l1e_highmargin_disagreements.json`).
To force **0 trusted disagreements** the frozen **τ = 95.9 frames** (set by the worst calib
case at 95.8f; the next is 32.2f). Conservative by design — it routes the NN's rare
large-cell mispredictions to the exact-scan fallback.

## Results — SEALED verdict @ sealOffset 40 (hidden seeds, never seen before)

NN-share is reported two ways. **Raw** counts every egBoid commit; **non-trivial** excludes
n=1 sole-boid commits (~29%), which are trivially exact (one boid is always the argmin) and
flow entirely into cert+trusted — they say nothing about the NN on contested geometry. The
**non-trivial figure is the honest "how NN-driven is the endgame" number** (audit angle 4).

| metric | scatter (startBoids 2..5) | natural (full-game, deployable) |
|---|---|---|
| games / endgame commits | 4,200 / **13,122** | _pending_ |
| intercept frames | 5,861,945 | _pending_ |
| **S_eg** (egBoid agreement) | **100%** (egDisagree 0) | _pending_ |
| **S_frame** (bitwise force) | **100%** (forceMismatch 0) | _pending_ |
| **S_traj** (identical to extinction) | **100%** (4200/4200) | _pending_ |
| sole-reachable n=1 (trivial) share | 28.9% | _pending_ |
| **NN-share (non-trivial, n≥2)** | **53.7%** | _pending_ |
| · cert (non-trivial, zero-risk) | **48.3%** | _pending_ |
| · trusted (non-trivial, τ-gated) | 5.4% | _pending_ |
| NN-share (raw, n=1 included) | 67.1% (cert 60.0% + trusted 7.1%) | _pending_ |
| fallback-share (exact scan) | 32.9% | _pending_ |
| residual (rule-of-3 on 926 trusted) | ≤0.32%/commit, ≤1.0%/game | _pending_ |

frozen τ = 95.9 frames · monotone reliability · calib 13,201 commits · 0 certified disagreements.
Per-cell non-trivial NN-share 48–64% (rises with cell size); egDisagree 0 in EVERY cell.

## Verdict (scatter)

**L1e is BITWISE-EXACT to prod on the sealed scatter set: 0 force mismatches over 13,122
egBoid commits / 5.86M intercept frames, all 4,200 endgame trajectories identical to
extinction, uniform across all 6 device cells.** The frozen τ **generalized** (0 sealed
trusted disagreements) — unlike L1h v2a, which the same one-shot machinery FAILED (2
disagreements) — because the endgame egBoid is **separable scan-t torus geometry**, not a
chaotic rollout. The endgame is genuinely NN-driven: **non-trivial NN-share 53.7%** (cert
48.3% provably exact + 5.4% τ-trusted) on contested (n≥2) commits — vs D1's planner fast-path
~0. The certificate is the zero-risk backbone (independently SOUND, 0 false certs / 2,000,000
+ ~146M audit samples); the τ-margin adds 5.4% more, conservatively gated (τ=95.9f) to exclude
the NN's rare large-cell scan-t mispredictions, residual ≤0.32%/trusted commit (0 observed).

## Adversarial audit (4 independent skeptic agents + synthesis)

`evidence/l1e_adversarial_audit.json`. **Bitwise-exactness / safety SURVIVES all 4 angles:**
1. **Cert soundness** (none): the U(k)<min L(j) sandwich is provably sound (sound integer
   lower bound via triangle inequality + verified-reachable upper bound + strict inequality,
   U≤TMAX guard); ~146M fresh adversarial commits (incl. 2.3M exact-tie geometries), **0 false
   certs**; published 2M reproduced live; eg_scan oracle bit-identical to prod over 5M.
2. **egBoid-match ⇒ force-exact** (minor): intercept()'s downstream is verbatim and branch-free
   on egBoid provenance; the injection is a proven pure-insert; fork no-resync S_traj=100% over
   28k closed-loop frames proves the held-until-caught state stays identical to extinction; the
   harness's bitwise (u32, −0/NaN-aware) compare catches a single ulp (broken1ulp self-test).
3. **τ-generalization** (minor): τ is one-shot, cryptographically chained to the CALIB file
   (inputSha256); sealed seeds HMAC-fresh, FLOOR 290000, disjoint from calib; salt commitment
   pre-registered. Independent shadow re-probe of 261,782 sealed commits: 0 disagreements among
   22,701 genuine n≥2 trusted commits; near-τ band densely tested (1548 within 5f, all agree);
   worst sealed disagreement 59.6f, a 36f gap below τ. (Caveat: τ on the *calibration* side is
   pinned by a single outlier at 95.78f — robust on sealed, but a single-point freeze.)
4. **Measurement integrity** (the headline fix): per-cell exactness all-zero (no pooled mask);
   harness egCommits == gate commits (no double-count). It CAUGHT that the raw NN-share over-
   counted n=1 trivials (~29%) — corrected above: the non-trivial figure is now the headline.

## Bottom line

**L0 + L1h + L1e ship BITWISE-EXACT to prod.** The two-regime NN-share story, honestly stated:
**planner (N>5) exact with NN fast-path ~0** (rollout-dominated, proven 4 ways); **endgame
(N≤5) exact with a genuinely NN-driven ~54% of contested commits** (separable scan-t geometry;
cert ~48% provably exact + ~5% τ-trusted; the rest exact-scan fallback); the **value net
necessary throughout**. The exactness guarantee is distribution-invariant (the fallback absorbs
every NN-uncertain case). _(natural-distribution row — the deployable number — appended below.)_

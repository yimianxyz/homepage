# Fast (vectorized) two-pass is biased high — large-n equivalence test (2026-06-05)

## Context
The faithful two-pass (`_step_boids_twopass`, sequential per-boid pass-2) is
validated <1e-7 vs the live browser (`twopass_validation.md`), but its pass-2 is
a Python per-boid loop → ~11× slower than single-pass, the data-gen bottleneck.
`FAST_TWO_PASS` (commit 3a08d04) added a vectorized predictor-corrector
approximation: pass-1 flock(frame-start); predict moved positions
`pos+limit(vel+accel1)`; pass-2 flock(predicted); one update. It passed an early
E3D n=128 check (ratio 1.043, within noise) and was used to generate all the
two-pass training data + `net_tp.pt`.

## Finding — the fast approximation over-catches ~11% (significant at n=512)
`validate_equiv.py` (paired, same seeds, sequential vs fast):

| policy | n | seq | fast | ratio | paired diff |
|--------|--:|----:|-----:|------:|------------:|
| E3D | 512 | 7.154 | 7.924 | **1.108** | **+0.770 ± 0.250 SE (~3.1σ)** |
| E3D | 128 | — | — | 1.043 | (within noise — why it slipped through) |

The fast sim makes the predator **~11% more effective** than the faithful
sequential sim. The n=128 pre-check was underpowered; n=512 exposes it.

## The bias is the boid-boid pass-2, NOT the predator term (diagnostic)
Hypothesis: the fast path recomputes predator-avoidance at the boid's already-
fled predicted position → weaker pass-2 avoidance. Tested a fix that computes
predator-avoidance **once at frame-start and applies it 2×** — which is exactly
what the true sequential does (in the sequential loop boid *i* evaluates its own
pass-2 flock BEFORE it moves, so it sees the frame-start predator distance both
passes). Result: **fixed fast E3D n=512 = 8.484** — *worse* (ratio 1.186).

So the predator term was not the culprit (the fix matched the true predator
handling and the bias grew). The remaining difference is the **boid-boid**
forces: the true sequential pass-2 moves only boids 0..i-1 (in order) when
boid *i* flocks; any vectorized version moves **all** boids at once. That
in-order partial update is inherently serial — **no cheap vectorized
approximation reproduces it.** The fix was reverted; `sim_torch.py` is back to
the exact version (md5 62f62ba5) that generated the data + net.

## Implications
1. **Fast two-pass is not a faithful drop-in for the sequential (live-browser)
   regime.** Training labels/gains generated with it are inflated.
2. Faithful two-pass gen requires the **slow sequential** sim (~11× cost) — but
   it is correct and was already done once at small scale.
3. The bottom-line question is whether the fast-trained `net_tp.pt`, **deployed
   in the true sequential regime**, still beats E3D-seq (7.15) and approaches the
   true-sequential planner. If yes, the gen bias was benign for the *policy* and
   we ship into today's (sequential) production. If no, options are (a) regen
   faithfully in slow sequential + retrain, or (b) redefine production itself as
   the fast sim (zero train/deploy gap by construction, at a ~11%-easier-catch
   visual cost + a JS predictor-corrector reimplementation).

## Repro
`python3 validate_equiv.py 512 2 1500` (E3D arm) — VM2, us-central1-a.
`python3 validate_equiv.py 8 64 1500` (planner arm) — VM3.

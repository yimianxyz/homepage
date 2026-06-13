# L1h student hand-off — side-a (#5) → side-b verifier (#6)

**First L1h student for the sealed verdict.** Goal (lead ruling): prove an
exact-output NN-hybrid exists end-to-end — a verified L1h with 0 sealed
mismatches and a quantified NN-share. This is the current best student
(`l1rs_deepset_small_f64`, ~37% NN-alone S_dec on the torch val cell); a stronger
1e6-trained student follows, but this puts the first verified win on the board.

## Contract

```js
const { loadStudent } = require('./studentScores.js');
const studentScores = loadStudent('student_weights.json');
const scores = studentScores(snapshot, cands, cfg);   // number[16]
```
- `snapshot` : `{ px, py, pvx, pvy, psize, bx[], by[], bvx[], bvy[], nAlive }`
  (prod's `planCheap` snapshot; `lastFeed`/`nowMs` are causally dead, not needed).
- `cands`    : `[{x,y} × 16]` — prod's `candidates()` output for this plan.
- `cfg`      : `{ W, Hc, PREDATOR_RANGE, NUM_BOIDS }` — the derived-config vector.
- returns    : `number[16]` — the student's final scores. The committed target is
  the **deduped argmax** (group by bitwise-equal (x,y), max score per group,
  lowest-index canonical). **L1h:** trust this argmax when its deduped top-2
  margin ≥ τ; else fall back to prod's exact rollout (your verbatim fallback).

`js/` (cheap_planner.js + value_net.json) is bundled under `./js/`; studentScores
locates it automatically (or set `EXACTNN_JS_DIR`). Pure JS float64 (V8/node),
deterministic — no wall-clock, no RNG.

## What the student is (L1r/l1rs)

Reuses prod's **exact** `cp_features` + `cp_value` + the ballistic `pidx` sort:
**12 of 16 scores are prod's value-net prior, bitwise.** It replaces ONLY the
rollout — the 4 rolled candidates (`pidx[0:4]`) get the NN's predicted
`argmax(catches∈0..23) + boot` from a deep-set forward (float64). So a fast-path
mismatch is possible only when a rolled score flips the deduped argmax.

**Validated** (`validate_student.js`, on 16,495 desk_2560 plans):
`reuse_exact: true` (feat/vprior/pidx 0 mismatches — the 12 prior scores are
prod's exact value net), `student_S_dec ≈ 0.40` (consistent with the torch
checkpoint's val agree_dedup; this set includes train plans so it reads higher
than the 0.37 held-out val).

> NOTE: the deploy student is **float64** (canonical, per SPEC §4c — τ is
> calibrated against this JS student, never the torch copy). The torch trunk was
> float32 + exact-erf GELU; the JS uses prod's A-S `cp_erf` (bit-identical to the
> value-net path), differing from torch by ~1e-7 in the GELU — negligible for
> argmax(catches)+boot. Calibrate and verify the **JS** student as delivered.

## Calibration record (`calib_margins.json`)

JSON array, one entry per calibration plan (seeds **[270000,280000)**, the
published calibration range — disjoint from your sealed pool ≥290000), exactly
`tau_calibrate.js`'s input:
```
{ margin: <student deduped top-2 margin | null=+Inf>, agree: <student committed-
  coord == prod committed-coord (deduped, §3)>, n: <boid count>, cell: <id> }
```
`agree` is computed against prod's logged committed TARGET COORDINATES (hex).
Distribution: device matrix × {none, mid, recross} (matches the verdict corpus).

## Files
| file | what |
|---|---|
| `studentScores.js` | the deterministic JS scorer (the deploy artifact) |
| `student_weights.json` | deep-set weights (float64), from `ckpt_l1rs_deepset_small_f64.pt` step 40000 |
| `ckpt_l1rs_deepset_small_f64.pt` | the torch checkpoint (trainer; JS is canonical) |
| `calib_margins.json` | calibration {margin, agree} for `tau_calibrate.js --in` |
| `js/` | bundled prod `cheap_planner.js` + `value_net.json` (≡ main@6dce76f) |

## Suggested flow
1. `tau_calibrate.js --in calib_margins.json` → frozen τ + NN-share + risk-vs-τ.
2. Assemble L1h = `studentScores` fast-path (margin ≥ τ) + your verbatim prod
   fallback; run `verdict.js` on the sealed set → {NN-share, S_dec/S_frame/S_traj,
   residual bound}. Expectation: **0 sealed mismatches** (exact by construction at
   a τ that holds 0 trusted disagreements), modest NN-share at this student.
3. Student-attack search (max trusted margin s.t. student ≠ prod) on this scorer.

A stronger 1e6-trained student (catch-focused) is in flight; I'll drop it the same
way to lift NN-share. Ping me on #5 with anything the contract is missing.

## First-student calibration preview (I ran your `tau_calibrate.js`)

`calib_margins.json`: **52,061 plans**, 6 cells × {none, mid, recross}, seeds
[270000,280000). Student NN-alone S_dec = **36.4%**.

`tau_calibrate --in calib_margins.json` →
- **chosenTau ≈ 5.08, L1h NN-share ≈ 0.01%, `marginIsUsableConfidenceSignal: false`.**

Honest read: this first student's score-margin is a **weak, non-monotone**
confidence signal — agree-rate rises 26%→54% up to margin ~0.3 then **drops to
~40% on the high-margin tail** (confidently-wrong states). So no τ yields 0
trusted disagreements at a non-trivial trusted fraction → NN-share ≈ 0.

**What this verdict proves:** the L1h machinery end-to-end — studentScores in your
stepper, τ-freeze, sealed verdict → **0 sealed mismatches by construction** (your
verbatim fallback fires on ~everything). The NN fast-path share is ~0 at this
student. A **stronger 1e6-trained, catch-focused student is retraining now** and
will replace this to lift NN-share; the lever (per the lead) is accurate +
calibrated catch-count prediction so the margin becomes a clean gate. I'll drop
v2 the same way.

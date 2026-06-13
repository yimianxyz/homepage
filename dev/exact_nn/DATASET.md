# EXACT-NN oracle dataset (#5)

Offline plan-decision records of the **frozen prod predator** (`js/` ≡
`main@6dce76f`), logged by a **certified instrumented fork** of
`predator_cheap.js`, for training the L1 students (SPEC §4). Ground truth is
JS/node (V8) float64 — never GPU (SPEC §6 "labels generated ONLY in JS").

> Status: **WIP**. This file is filled in as the farm runs. Numbers below marked
> `(TBD)` are stamped from `manifest.json` + `analyze_margin.js` after each farm.

## How it's produced (regeneration)

```
# 0. (re)generate + certify the fork  (gate: refuses to farm uncertified)
node dev/exact_nn/gen_oracle_policy.js            # oracle_policy.js from js/predator_cheap.js
node dev/exact_nn/gen_oracle_policy.js --verify   # strips to prod source byte-for-byte
node dev/exact_nn/certify_oracle.js               # -> CERT.json (0 mismatches, all cells)

# 1. farm shards (local cores and/or side-a VMs)
node dev/exact_nn/shard_runner.js --quota <decisions/cell> --spawnFrac 0.15 \
     --concurrency 4 --outDir dev/exact_nn/data            # --mode vm for ml-forecast-1/2

# 2. margin CDF (deliverable zero) — sizes the GPU program
node dev/exact_nn/analyze_margin.js --data dev/exact_nn/data --md DZERO.md --out dzero.json

# 3. pack -> tensors for training
python3 dev/exact_nn/train_shakedown/pack_oracle.py --data dev/exact_nn/data --out packed.npz
```

Determinism: each (cell, seed, spawn-profile) game is byte-reproducible
(rng.js virtual clock; verified by `oracle_logger --selftest` check 2). Re-running
`shard_runner` is idempotent — it scans `*.meta.json` and continues seed
allocation past whatever exists.

## Certification provenance

Every shard's `.meta.json` carries `oracleSha` + `certRunId`; `oracle_logger`
refuses to farm unless `CERT.json.ok` and its `oracleSha` matches the live
`oracle_policy.js`. Current: `oracleSha=f2b99385…`, `certRunId=6f946c4619e1ef56`.

## Schema

**`<shard>.meta.json`** (the shard header, SPEC §6.4): `{cell, W, H, numBoids,
predRange, frameMs, spawn, seedStart, seeds, maxFrames, oracleSha, certRunId,
node, nDecisions, nNegZero, nNegInf, games[]}`.

**`<shard>.decisions.jsonl.gz`** — one plan decision per line:

| field | shape | meaning |
|---|---|---|
| `seed`,`cell`,`W`,`H` | | game id + device |
| `cfg` | `{W,Hc,PREDATOR_RANGE,NUM_BOIDS}` | derived-config vector, as the policy evaluated it (SPEC §1) |
| `f`,`N` | | 1-based sim frame; live boid count (≥6, planner regime) |
| `s` | `{px,py,pvx,pvy,psize,lastFeed,nowMs, bx,by,bvx,bvy}` | full input snapshot (`lastFeed`/`nowMs` causally dead) |
| `cands` | `[16][x,y]` | candidate targets (slot 0 = E3D patrol; 1..15 nearest boids lead-adjusted; k≥N = E3D copies) |
| `feat` | `[16][19]` | `cp_features` per-candidate |
| `ctx` | `[4]` | `cp_features` shared context |
| `vprior` | `[16]` | value-net prior |
| `pidx` | `[4]` | roll order (ballistic pscore desc, tie→lowest index) |
| `rolled` | `[4][ci,catches,boot]` | per rolled candidate; aligned to `pidx`; `boot` JSON null = −Infinity (extermination) |
| `score` | `[16]` | final scores (`vprior` with rolled overrides); null = −Infinity |
| `bi` | | prod argmax slot |
| `lab` | `{ti,tx,ty}` | **THE LABEL**: committed target COORDINATES, slot canonicalized to lowest bitwise-equal (x,y) |
| `margin` | | slot-level runner-up margin |
| `dmargin` | | top1−top2 over coordinate-DEDUPED groups; JSON null = +Infinity (coordinate forced, no competitor) |
| `nDistinct` | | distinct coordinate groups among the 16 slots |

Every record is replay-verified at log time (`recheckPlan`: recomputes
pidx/score/bi/margin/label/dedup and throws on any bit of disagreement).

**`<shard>.frames.jsonl.gz`** — one columnar line per game: `mode` (0 zero / 1
intercept / 2 plan / 3 steer), `N`, force `fx`/`fy` (hex u64), `pf` (policy
frame counter), `tx`/`ty` (committed target hex), `eg` (egBoid index).

## Non-finite encoding

`score[k]` and `rolled` `boot` may be **−Infinity** (prod's NaN→−Infinity
extermination path when a rolled candidate's 90-step rollout clears every boid;
~0.5% of plans). JSON serializes these as `null`; the packer maps `null → −inf`.
These are always argmax-losers (the winner keeps a finite vprior), so they are
masked in regression and never become the label. `dmargin` may be **+Infinity**
(single coordinate group) → `null` → decoded to `+inf` (maximally safe, never a
near-tie). NaN and +Inf anywhere else are hard errors.

## Seed discipline (SPEC §4c)

- Train/calibration seeds **< 270000**; verification (sealed) seeds **≥ 270000**,
  reserved for side-b — I never farm or train on them. `oracle_logger` and
  `shard_runner` refuse to cross 270000.
- Per-cell seed blocks are disjoint (`device_matrix.seedBase`: 100000, 110000,
  …, 150000); spawn profiles occupy disjoint sub-blocks (offset 0/2000/4000/6000).

## Device matrix & corpus

Cells (SPEC §5): `iphone_390x844`, `ipad_820x1180` (mobile-by-UA → 60 boids,
frameMs 18), `desk_1024x768`, `desk_1512x982`, `desk_1680x1050`,
`desk_2560x1440`. PREDATOR_RANGE = **80 on every cell** (load-order bake at
6dce76f; the 60 branch is dead code — see `device_matrix.js`).

Spawn corpus (SPEC §5 interaction games), `--spawnFrac`: `mid` (planner-phase
taps incl. same-coord double-tap), `recross` (reactive 5→6→5 gate re-crossings
with egBoid + frozen frame counter alive across), `spam` (48 taps past the cap).

## Contents (stamped after farm)

**Deliverable-zero set** (`data/`, none-profile, seeds 100000–160000): 74,748
decisions, 116 games, 6 cells. Margin CDF: see `DZERO.md`. Used for the first
L1h student.

**1e6 set** (`data_1e6/`, spawnFrac 0.15, seeds 100000–160000+): **1,132,267
decisions, 1,864 games, 386 shards, 0 farm failures.** −inf scores: **0** (the
slow predator never exterminates all boids in a 90-step rollout, even in spam).

| split | decisions | games | per-cell (k) | by spawn profile |
|---|---|---|---|---|
| deliverable-zero | 74,748 | 116 | ~9–16 each | none only |
| 1e6 | 1,132,267 | 1,864 | 178–202 each | none 890k / mid 74k / recross 78k / spam 90k |

**Calibration set** (`calib_data/`, seeds [270000,280000), the published
calibration range): 52,061 decisions, 6 cells × {none, mid, recross} — for
side-b's τ-freeze (`student/calib_margins.json`). Sealed seeds (≥290000) are
side-b's, never farmed here.

Certified oracle: `oracleSha=f2b99385…`, `certRunId=6f946c4619e1ef56` (510,991
frames lockstep, 0 mismatches; `CERT.md`).

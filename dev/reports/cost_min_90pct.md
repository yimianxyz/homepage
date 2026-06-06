# Cheap-policy cost-minimization under a %-of-planner constraint (2026-06-06)

Goal: smallest browser rollout cost that reaches a target % of the planner
(18.75 strict GPU ceiling), with the deployed `predator_cheap.js` policy.

## Faithfulness first — the GPU proxy was fixed, then characterized
Two real bugs in the GPU eval (`feat_planner.run_value_lookahead_cheap`) were
found and fixed; one scare was a false alarm:
1. **Tie-selection bug** (`is_top = pscore >= topk`): the ballistic prune ties at
   -1 for a slow predator (no ballistic catch), so `>=` selected ALL candidates →
   the "cheap K_roll=1" eval was secretly the full 16-roll planner. The historic
   "GPU 14.64 vs JS 7.0 2x gap" was this. Fixed: deterministic stable top-K_roll.
2. **E3D freeze-during-chase**: `_update_auto_target` froze cand0 while chasing;
   the browser recomputes E3D fresh every decision. Fixed: `always_recompute_target`.
3. **(false alarm)** a "candidate mismatch" in diff_decision was a harness bug
   (compared pre- vs post-step states). E3D matches JS exactly (1939.6562,1587.6002).

Residual: even fully fixed, GPU `ball@1`=10.42 vs JS=6.92 on identical seeds
(200000-11), per-seed uncorrelated. Cause: the re-planning decisions diverge at
float precision (float32 torch net + erf-approx GELU + `.float()` in
`_analytic_steer` vs JS float64); the chaotic 1500-frame closed loop amplifies it.
**GPU = reliable for SHAPE; JS = ground truth for ABSOLUTE.** % of planner is
~engine-invariant (the ~1.5x inflation roughly cancels), so search on GPU, JS-verify.

## Faithful sweet-spot map (GPU, ball prune, n=64, 1500f; planner=18.75)
K_roll @ Hs=60: 8.97(48%) / 12.25(65%) / 14.08(75%) / 15.83(84%)  [K=1/2/3/4]
Depth @ K=2:   Hs60 12.25 / Hs90 15.08 / Hs120 14.92 / Hs150 13.52  (knee Hs~90)
Depth @ K=1:   Hs60 8.97 / Hs90 9.94 / Hs120 10.28 / Hs200 10.92    (caps ~11)
- `ball` (roll E3D + nearest boids) beats every learned prune at every K_roll.
- Catches track total rollout WORK (K_roll x Hs); depth beats breadth at equal work
  (K4/Hs90=92% > K6/Hs60=85%, both work 360).

## Cost = K_roll x Hs / D  (rollout-frames per game-frame; net is ~free, 3553 params)
≥90% (>=16.875) configs, cheapest first:
- **K4 / Hs90 / D16  = 16.88 (90%)  cost 22.5  <- min-cost @ current net (~3x shipped 7.5)**
- K4 / Hs90 / D8  = 17.22 (92%)  cost 45
- K3 / Hs120 / D8 = 17.25 (92%)  cost 45
- K4 / Hs75 / D8  = 17.19 (92%)  cost 37.5  (Hs75 fails at D>=12 -> 85%)
- D=24 fails everywhere (K4/Hs90/D24 = 79%, too stale)
Key lever: **D=16 halves per-frame cost vs D8 with no catch loss** — a DEEP rollout
(Hs90) yields a target that stays good ~16 frames; shallow Hs can't use large D.

## Does the net matter? (no_value = pure rollout, no terminal bootstrap)
- K4/Hs90 no_value = 15.67 (84%)  vs with-net 17.22 (92%)  -> net = +1.55 (+10%), LOAD-BEARING (it is what crosses 90%).
- K6/Hs90 no_value = 17.73 (95%): rolling MORE candidates removes net dependence,
  but at cost 67.5 (3x pricier than net-assisted K4/Hs90).
=> the net SUBSTITUTES for rollout breadth -> a better net is a direct cost-saver.

## Retrain decision: YES (DAgger), AFTER fixing the rollout shape, then iterate
The net is planner-trained (wrong distribution: cheap policy visits different
states and catches less, so the bootstrap over-estimates). DAgger-relabel on the
chosen policy's own visited states (incl. Hs-terminals) should let a CHEAPER
rollout clear 90% (target: K4/Hs60/D16, cost ~15 ~= 2x shipped, currently 84%).
Co-design loop: fix rollout -> DAgger-retrain net -> re-minimize cost -> iterate.

## JS-validated deployable absolutes (seeds 200000-11, real predator_cheap.js path)
- K1/Hs60 (shipped) = 6.92
- K2/Hs60 = 8.75 (+26%)
- K2/Hs90 = 9.42 (+36%)
- (K4/Hs90/D16 JS-validation pending)

## Recommendation
1. Ship now (no retrain): **K4 / Hs90 / D16, ball** — 90% of planner at ~3x shipped
   rollout cost. JS-validate first.
2. To reach ~2x shipped: DAgger-retrain the net on K4-policy states, re-test
   K4/Hs60/D16.

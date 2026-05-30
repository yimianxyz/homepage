# End-to-end predator NN — distillation search log

Goal (user): replace the production predator policy (E3D evolved patrol + 35-feature
builder + 35->4->2 chase NN) with ONE simple raw-observation end-to-end NN that
reproduces production behaviour. Stop only when both 100% match and minimality hold.

## M0 — feasibility (see M0_findings.md)
Exact per-seed catch identity is **physically impossible** (chaos: a 1e-7 steering
perturbation already decorrelates per-seed catches to chance; ε* < 1e-7). Goal reframed
to **behavioral equivalence**: match mean/distribution of catches (~8.1) and hunting.

## M1 — raw-obs encoding + on-policy dataset
`raw_obs.py`: predator-centric, torus-wrapped. pred_vel (2) + K-nearest boids rel
pos/vel (4K) + GxG soft density grid + GxG*2 momentum field. `gen_dataset_e2e.py`
captures (obs, production steering force, d1, pred_auto) on-policy from the shipped E3D
policy. Train 512–1024 seeds, val held-out.

## M2 — architecture / objective search (GPU, L4)
Behavioural eval = `eval_e2e.py` (E2ESim plugs net in place of the whole pipeline);
512 held-out seeds 70000+; production baseline 8.19.

| step | finding | catches |
|---|---|---|
| force-head MLP, density grid | learns chase (~20° ang) but FAILS patrol (~66°, capacity-independent) | 4.3 |
| + velocity/momentum field | did NOT fix patrol; HURT chase (signal dilution) | — |
| diag: why? | production force is Reynolds (desired_vel − cur_vel): 56°/105° off dir-to-target; force is an ill-conditioned regression target. dir-to-target itself is learnable at ~24°. | — |
| **Reynolds output head** (net predicts desired velocity, fixed `clip(desired−vel,0.05)` head) | unlocks patrol | **5.7–5.9** |
| capacity sweep (1.9k→44k params) | catches flat 5.5→5.7 → not capacity-bound | ~5.7 |
| direct dir-label training | worse (crude chase label, no lead) | 4.2 |
| grid G9 vs G13 (+2× data) | dir-to-target 23°→21° — resolution barely helps | — |
| **regime decomposition** (hybrid_diag.py) | **chase_e2e 7.86 (96% of prod), patrol_e2e 6.24 (76%)** → CHASE is ~solved, PATROL is the whole gap | — |
| **scale-up** G13 grid + 170k params (256,128) + 1024 seeds + 250 ep | full **6.48**, patrol_e2e **6.77**, chase_e2e **8.49** (>prod) | 6.48 |
| **scale-up** G17 grid + 256k params + 1280 seeds + 300 ep | full **6.99**, patrol_e2e **7.08**, chase_e2e **8.44** | 6.99 |

Grid scaling curve on patrol_e2e: G9 6.24 → G13 6.77 → G17 7.08 → **G21 7.18** (gains
+0.53, +0.31, +0.10 — PLATEAUING). Full: 5.9 → 6.48 → 6.99 → **7.16**. The grid/raw_obs
MLP saturates near ~7.2 and needs a 430k-param, 1357-dim net at G21 — accurate to 88% of
prod but NOT minimal and NOT converging to equivalence. **Grid path ruled out as the final
answer.** Parallel
test: a set/attention encoder (see set_*.py) that aims for the same patrol with far fewer
params by reading exact boid positions instead of a histogram.

### Set-encoder path (set_obs/set_net/set_e2e) — boid SET vs histogram
| net | params | full | patrol_e2e | chase_e2e | val ang pat/chs |
|---|---|---|---|---|---|
| DeepSets (phi→mean→rho) | 3.6k | 4.27 | 5.80 | 6.57 | 45/72 |
| 1× self-attn → mean pool | 7.9k | 5.38 | 5.65 | **8.44** | 40/**11** |

Key structural finding: **chase and patrol want opposite encodings.**
- CHASE wants the single nearest boid. Self-attention nails it (11° angle, chase_e2e 8.44
  = matches the KNN-grid, exceeds prod). DeepSets mean-pool dilutes it (chase 6.57).
- PATROL wants a density-weighted centroid. A *single* attention layer is WORSE than the
  histogram (40° vs grid ~21°): per-boid local density is a PAIRWISE quantity, and one
  attn layer (per-boid keys) can't encode it, whereas a GxG histogram materializes density
  directly as cell counts. So the grid's "lossy" binning is actually the right structure
  for density — the set net needs ≥2 attention layers (layer 1 computes per-boid density,
  a learned attn-POOL then forms the weighted centroid) to match production's algorithm.
2-block self-attn + cross-attention pool set-transformer (set_net AttnPool, ~15k params)
RESULT: full 6.10 / patrol_e2e **6.15** / chase_e2e 8.60 (val ang pat/chs = 36.8/4.1). Chase
is superb (4.1°, exceeds prod); patrol 6.15 is still WORSE than the grid (7.18). **Set/attention
path RULED OUT for patrol.** Root cause confirmed: a softmax-attention pool *normalises* the
per-boid neighbour count into an average — it cannot COUNT density, which is exactly the
quantity production weights by. The histogram counts (scatter-add) but quantises.

### Density-feature DeepSets path (set_obs `--density-radii`) — RUNNING
The decisive idea: stop asking the net to *infer* per-boid local density (a pairwise count
that attention averages away and the grid quantises) and instead **materialise it exactly as
an input feature**. `set_obs(..., density_radii=[r1,r2,...])` appends, per boid, the exact
#alive-neighbours within each radius (torus, O(N²) in the encoder). Given density as input, a
plain DeepSets `phi→masked-mean→rho` suffices: the masked mean is an *un-normalised* density-
weighted sum Σφ/N, and since N is a per-frame scalar, that sum has the **same DIRECTION** as
production's normalised centroid Σwₚ/Σw. So the minimal phi→mean→rho net can represent patrol
direction with NO histogram quantization and NO attention-averaging. Three VMs in parallel:
- VM1 CONTROL: identical pipeline, NO density (feats=5) — isolates the density-feature effect.
- VM2 densA: radii [80,178,320] (range/cluster_r/2×), deepsets d32/48/64, mean pool.
- VM3 densB: radii [60,120,240,480] (4-scale), deepsets-mean vs deepsets-attnpool vs 1×attn.
Decision metric: patrol_e2e and val patrol angle vs grid 7.18 / 10.8°. If density-deepsets
beats the grid at far fewer params, it is the elegant minimal answer.

RESULTS (512-seed decompose, 70000+):
| config | params | patrol_e2e | chase_e2e | full | val ang pat |
|---|---|---|---|---|---|
| control deepsets-MEAN, NO density | 6.0k | 5.996 | 8.217 | 5.49 | 43.4 |
| density deepsets-MEAN, 3 radii | 6.2k | 5.715 | 7.94 | 5.35 | 40.5 |
| density deepsets-MEAN, 4 radii | 6.2k | 5.711 | 8.01 | 5.04 | 40.7 |
| density deepsets-MEAN, d64 rho128,64 | 21.7k | 5.912 | 7.89 | 5.57 | 34.1 |
| density + attn-POOL, 4 radii (lucky draw) | 13.4k | 7.123 | 8.305 | 6.906 | 32.6 |
| density + attn-POOL, 4 radii (repro ×4) | 13.4k | ~4.9 | 8.33 | ~4.9 | **36–38** |

Two clean conclusions:
1. **Density + mean-pool does NOT help** (5.7 ≈ 6.0 control). The earlier theory ("mean-pool =
   un-normalised weighted sum = same direction as centroid") was WRONG: production patrol is a
   cluster **SELECTION** (dens_pow=2.37 picks the single densest cluster, ignoring sparse boids),
   which is an argmax-like op a smooth pooled MEAN cannot represent — it averages across all
   clusters. Same failure as the soft-argmax-centroid head (54°). Scaling capacity (21.7k) or
   radii (4 vs 3) barely moved it. So the bottleneck was never density materialization alone — it
   is SELECTION.
2. **CORRECTION — the "7.12 winner" was chaotic-metric NOISE, not a real result.** The 7.123 came
   from a single training draw that happened to land at 32.6° val patrol angle. Re-running the
   EXACT same config (4-radii [60,120,240,480], d48, heads2, attn-pool) four independent times —
   VM1 m4r_d48, VM2 iso3r_h2a/b, VM3 iso_h2a/b — gives patrol_e2e **4.76–5.14** at val angle
   **36–38°**, NOT 7.12. The catch-count (`patrol_e2e`) is a CHAOTIC metric: a ~5° angle change
   amplifies into a ~2-catch swing through the boid sim. **The reliable, deterministic signal is
   the val patrol ANGLE.** By that metric the density+attn-pool set encoder sits at ~36–38°, far
   worse than the grid's **10.8°**. Lesson recorded: never rank architectures by single-run
   decompose catch-counts; rank by val angle, confirm with multi-seed.

**Set / attention path is RULED OUT for patrol** by the reliable angle metric (all set variants
36–43° vs grid 10.8°). Mean-pool can't select; cross-attn pool selects but on a 36° error budget —
the softmax over per-boid density is too soft to match the dens_pow=2.37 hard argmax that prod
uses. Chase IS solved by the set path (attn-pool ~5° angle, 8.3 catches). The grid histogram
remains the best patrol encoder found (10.8° @ G21, plateaus ~7.18 at 430k params).

### GatePool (density-gated self-weighted pool) — the FINAL pooling test, also ~36°
Hypothesis: the attn-pool failed because it queried from pred_vel, but production patrol
selection is predator-INDEPENDENT (weight each boid by its OWN density^2.37). GatePool
implements exactly that: per-boid scalar score → softmax(score/tau) with a LEARNABLE
temperature → weighted sum of values = Σ(dens^p·pos)/Σ dens^p. Result: **val patrol angle
still 36–40°** (3-radii d48: 36.1/37.2; 4-radii: 37.0/37.6; d32: 39.3; d24: 39.4). Predator-
independent gating did NOT sharpen patrol. **DECISIVE variance evidence** — the SAME config
(gate_d48, 3-radii) two seeds gave full-e2e **7.047 vs 4.945** at near-identical angle
(36.1° vs 37.2°): a 2.1-catch swing from sim chaos at the same direction accuracy. This is the
SAME trap as the "7.12 winner" — catch-count is noise; angle is the signal. Confirmed twice now.

**FINAL VERDICT on the set path:** every permutation-invariant pooling mechanism (mean,
cross-attention, density-gated self-weighted) floors at ~36–40° patrol angle. The bottleneck is
NOT the pool — it is the set bottleneck itself: compressing all boids into one D-dim pooled
vector destroys the absolute spatial structure needed to SELECT one cluster and emit a precise
direction. The Cartesian grid keeps that structure (each cell = a fixed direction) and the MLP
reads off the densest cell → 10.8°. Set path closed for patrol; grid is the encoder of record.
Note minimality upside not realized: gate-pool at 8.6k params is 50× smaller than grid-G21
(430k) but its angle (36°) is too poor — so "minimal but wrong" loses to "big but right".

### CLEAN matched grid baseline (variance pairs) — corrects the bogus "10.8°"
Earlier notes cited grid patrol angle "10.8°"; that was an unreproducible artifact (likely
measured on train data or a different angle definition). A clean matched sweep — identical recipe
(hidden 256,128, reynolds head, force target, NO dirw, 1280 seeds, 300 ep), TWO training seeds
each, measured by measure_patrol_angle.py on held-out val — gives:

| G | catches seed a | catches seed b | mean | patrol angle (val, reliable) |
|---|---|---|---|---|
| G17 | 6.79 | 7.05 | 6.92 | — |
| G21 | 6.95 | 7.04 | **6.99** | **21.3 / 21.2** (both seeds!) |
| G25 | 6.44 | 7.06 | 6.75 | — |

Two decisive conclusions:
1. **The grid plateaus at ~7.0 catches = 85% of prod 8.19, and resolution beyond G17 does NOT
   help** (G17 6.92 ≈ G21 6.99 > G25 6.75; G25 even adds variance). The plateau is real and
   resolution-saturated — NOT a quantization limit you can resolve away.
2. **Reproducible patrol angle is ~21°**, not 10.8° — both G21 seeds land at 21.2–21.3°. So the
   true grid patrol direction error is ~21°, and that 21° → ~7.0 catches. The whole residual gap
   (7.0 vs 8.19) is patrol cluster-selection the histogram MLP cannot sharpen past ~21°.

Catch-count variance across seeds is ~0.1–0.6 (G25 worst at 0.6) — modest, so the means above are
defensible. NEXT: is 21° capacity-bound or representational? Test a much bigger/deeper head at
fixed G21 (reuse data). If big head still ~21° → representational ceiling for raw-obs grid encoders.

### BREAKTHROUGH: LOG-POLAR encoder beats the Cartesian plateau
The Cartesian grid is resolution+capacity saturated: bigger head (512,256,128 / 1024,512,256,
400-500 ep) on G21 drops patrol angle 21°→17.3° but catches stay ~7.0 (cap_512 7.03, cap_1024
6.92). So Cartesian tops out ~7.0 catches / 17° angle. A predator-centric LOG-POLAR histogram
(nr log-radial × nt angular bins, angular bins WRAP and align with the output direction) breaks
it — matched recipe (hidden 256,128, reynolds force, 1280 seeds, 300 ep), variance pairs:

| encoder | params-class | catches a / b | mean | dist_gap |
|---|---|---|---|---|
| Cartesian G21 (256,128) | ~430k | 6.95 / 7.04 | 6.99 | 1.21 |
| Cartesian G21 (1024,512,256) | ~2M | 6.92 / — | ~7.0 | 1.31 |
| **polar 8×48** | ~370k | 7.115 / 7.121 | **7.12** | 1.09 |
| **polar 8×64** | ~430k | 7.227 / 7.455 | **7.34** | **0.74–1.02** |

Finer angular resolution keeps helping (48→64: +0.2 catches, dist_gap 1.09→0.74). The 8×64 seed-b
reached **7.46 = 91% of prod 8.19** at dist_gap 0.74. The earlier "info-limit" worry (count
histograms can't see cluster tightness) was empirically WRONG: aligning bins with the output
DIRECTION matters more — "densest angular sector = patrol heading" is a far easier readout than
reducing a Cartesian field to a direction. NEXT: push angular (8×96, 8×128) + radial (12×64) to
find where polar saturates, and re-measure patrol angle (measure_patrol_angle.py was missing on
the VMs, so polar angle numbers are still pending).

### Current read (post scale-up)
The "~6.0 raw-obs ceiling" from prior PPO is **NOT** a hard wall. Jointly raising grid
resolution (G9→G13), capacity (44k→170k), data (512→1024 seeds) and epochs (→250) moved
**both** regimes up: chase_e2e 7.86→8.49 (now exceeds prod — chase is saturated/solved),
patrol_e2e 6.24→6.77. The full e2e net is 6.48 (errors in both regimes compound). The whole
remaining gap is still PATROL (6.77 vs prod 8.19 = 83%), but it is responding to scale —
so we are capacity/resolution/data-bound, not at a representational ceiling.

The patrol computation is a density-weighted cluster-SELECTION centroid. A predator-centric
density histogram approximates it and finer resolution helps (G13>G9). Open: which of the
four levers drove the +0.53 patrol gain, and where does the grid curve plateau. Next:
isolate by pushing grid resolution + data further (one decisive higher-res run), and in
parallel test a set/attention encoder that sees exact boid positions (the architecture that
can in principle compute the density-weighted centroid without histogram quantization).

# Evidence brief — L1e endgame egBoid-commit student (side-a task #18)

**Decision (N≤5):** prod's `intercept()` commits `egBoid = argmin over the present boids
of scan(boid).t` (earliest frame the predator can stand where the boid will be; torus
min-image; nearest-wrapped-distance fallback if none reachable within TMAX=1400). Held
until caught. scan-t depends ONLY on a boid's (relpos, vel) + torus dims (PX=W+20,
PY=Hc+20) + sM=2.5 — **per-boid SEPARABLE** (no inter-boid interaction), the structural
opposite of D1's chaotic 90-frame multi-boid rollout.

**L1e composition (D4 twin of L1h):** NN predicts per-boid scan-t → argmin = egBoid,
margin = 2nd-min − min predicted scan-t. Commit the NN's egBoid iff margin ≥ τ (or the
sound certificate fires), else prod's EXACT `intercept()` scan fallback → bitwise-exact
by construction (the chosen egBoid is handed to prod's verbatim aim/seek; only egBoid
IDENTITY can differ on a trusted commit).

## Pipeline + cross-checks (all in dev/exact_nn/endgame/)
- `eg_logger.js`: digest-inert anchored transform on intercept() logs each commit's
  per-boid exact scan-t + prod egIdx + state. Train: 38,877 commits (seeds 100000–101499,
  6 cells, startBoids=5+scatter). Calib: 13,025 commits (seeds [270000,270499)).
- `eg_scan.js`: independent EXACT scan()/egPick reimpl. CROSS-CHECK: 0 scan-t mismatches
  + 0 egIdx mismatches vs prod logs; eg_pack egDerivedMismatch=0 over all 38,877 commits.
- `eg_features.js`: per-boid features (key = WRAP-AWARE analytic intercept time = min over
  9 torus images; argmin over it alone = 99.1% egBoid agreement vs 89.3% no-wrap). Shared
  by packer + deploy → bit-identical features.
- `eg_train.py`: per-boid scan-t MLP (18 feat, h64, MSE). val egBoid agree 99.1%, RMSE 4.2
  frames. (argmin-CE + h128 variant OVERFIT: train 99.4/val 94.6 — rejected; MSE-only kept.)
- `egboidPick.js`: deploy (float64, A-S erf GELU). JS parity: 95.2%→(wrap-aware)99.0% on
  train, matching torch.
- `eg_bound.js`: SOUND zero-risk certificate — U(k)=verified-reachable integer near
  analytic-t (a few scan-evals, not the 1400 scan) < L(j)=ceil(dist0(j)/(sM+bspeed_j)) ∀j≠k
  ⇒ k provably the unique argmin = prod egBoid. 0 bound violations, **0 false certs** on
  the data (adversarial all-index test).

## Headline numbers (held-out calib [270000,280000), 13,025 commits)
- **NN-alone egBoid agreement: 99.0%** (the user's "<5" is genuinely NN-driven).
- **margin-gated bitwise-exact NN-share: 72.3%** (τ=24.5 frames, 0 trusted disagreements,
  `marginIsUsableConfidenceSignal: True`; reliability margin>1f disagree 0.56%).
- **sound-certified zero-risk share: 54.0%** (provable, no τ).
- **COMBINED (cert ∪ margin≥τ) bitwise-exact NN-share: 77.6%.**
- 20% of commits are sole-reachable (n=1 boid or only 1 reachable) → trivially correct.
vs D1's NN fast-path ~0: the endgame admits a real NN-share, as the lead predicted.

## Attack surface for auditors
1. SOUNDNESS: is eg_bound's certificate provably sound for ALL legal inputs (not just the
   tested data)? Scrutinize the U(k)<L(j) argument (integer scan-t, torus, fast boids,
   nearest-fallback, unreachable competitors). Any input where a certified k ≠ true argmin?
2. MEASUREMENT: is the 72.3%/77.6% NN-share computed correctly (tau_calibrate 0-disagree,
   sole-reachable handling, margin in frames)? Is egBoid-agreement the right exactness
   proxy (does matching egBoid identity ⇒ bitwise-exact force, given prod's verbatim
   downstream + the stateful hold-until-caught)?
3. DISTRIBUTION REALISM: train/calib use startBoids=5+scatter (side-b's endgame_margin
   methodology). Does the NN-share hold on NATURAL full-game endgames (boids that survived
   the planner) — the distribution side-b's SEALED verdict (≥290000) will use? (A natural
   full-game calib is being farmed; reason about/measure the gap.)
4. UNDER/OVER-CLAIM: is 77.6% honest? Could it be materially higher (more data — the MSE
   model is data-limited not overfit; better re-wrap features) or is something inflating it?

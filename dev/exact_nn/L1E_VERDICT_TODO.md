# L1e verdict — cold-resume handoff (side-b #6)

**This file is the resume contract after `/clear`.** Read it + the #6 thread, then
execute the L1e sealed verdict when side-a hands off `egboidPick`. Everything
below is already committed on branch `side-b/exact-nn-l0`.

## Where the program stands (2026-06-14)

Lead ship-call (#6) ACCEPTED. **Final deliverable shape = L0 + L1h + L1e.**
- **D1 (planner, N>5): DONE, SHIPPED.** L0 (T1 provably-exact) + L1h (hybrid,
  bitwise-exact, value net load-bearing). NN fast-path share ~0 — proven
  structural (rollout-bound boot), confirmed 4 ways (independent calib,
  decision-type stratification, sealed test catching τ-non-generalization,
  4-angle adversarial audit). **Do NOT iterate the planner student.** v2a
  (`l1rs2`, monotone margin) is the planner student of record. Verdicts:
  `L1H_VERDICT.md` (v1), `L1H_V2A_VERDICT.md` (v2a).
- **D4 (endgame, N≤5): L1e — the genuine NN-share win, IN PROGRESS (this task).**
  Decided by scan-t torus geometry (separable, not chaos). side-a's zero-risk
  certificate `endgame/eg_bound.js` is **independently verified SOUND** by me
  (`verifier/eg_cert_verify.js`, 0/800k false certs; evidence
  `evidence/eg_cert_soundness_500k.json`), certifying ~27% (broad-random) /
  ~51% (real game commits) exactly, NN-free. side-a's scan-t NN: 99.1% standalone
  egBoid agreement. THIS is where 'NN-centric exact' delivers + covers '<5'.

## The L1e verdict to run (when side-a hands off `egboidPick`)

side-a will deliver (watch `/workspace/.team/`, side-a branch `exact-nn-oracle`,
and #5/#6): a deterministic JS `egboidPick(state) → egIdx` (the scan-t NN +
margin gate + certificate), the NN weights (`endgame/eg_weights.json` exists),
and the deploy features (`endgame/eg_features.js`, `eg_scan.js`, `eg_bound.js`
already pulled into my tree). Then:

1. **Independent egBoid agreement**: run the L1e egboidPick vs prod's exact
   `egPick` (`endgame/eg_scan.js`, the bit-identical reimpl of intercept()'s
   scan + argmin) on the calibration range [270000,280000), 6-cell device
   matrix, N≤5 endgame games (startBoids 1-5 scatter). Confirm ~99% agreement.
2. **τ-freeze on the scan-t margin** (the NN path; the certificate path is
   zero-risk, no τ): `tau_calibrate.js` analogue — but the "agree" is egBoid
   match, "margin" is the scan-t-margin the NN/gate exposes. The L1e gate:
   commit egBoid via (a) certificate if it fires [exact, no τ], else (b) NN if
   its margin ≥ τ [trusted], else (c) the exact full scan [fallback].
3. **Build the L1e composition** (the N≤5 analogue of `candidates/l1h.js`):
   inject the egBoid gate into `intercept()` (anchor: the `if (!egBoid)` commit
   block in `js/predator_cheap.js:364`) — replace the argmin-scan egBoid choice
   with: certificate→NN-if-margin≥τ→exact-scan-fallback. CRITICAL EXACTNESS
   FACT: if the chosen egBoid == prod's egBoid, the downstream per-frame aim
   scan + seek is verbatim ⇒ **force bitwise-exact**. So L1e force-exactness ==
   egBoid-commit agreement. The per-frame aim scan() stays prod's (verbatim).
4. **Sealed verdict on N≤5 force output** (FRESH sealed slice — v1 used offset
   0, v2a offset 20; use **offset 40**): `verdict.js --candidate <l1e> --seeds N
   --sealOffset 40 --cells <matrix>` over endgame games; the harness already
   handles N≤5 (intercept regime) — confirm 0 force mismatches on certified+
   trusted commits, measure NN-share (= cert-frac + NN-trusted-frac) + the
   rule-of-three residual on the NN-trusted (non-certified) commits.
5. **Adversarial audit** (Workflow, like the v2a barrier audit but PRO — does the
   L1e NN-share survive: certificate soundness at scale [done], egBoid-match⇒
   force-exact [verify], the scan-t margin gate generalizes to sealed [the
   one-shot test], no measurement bug).
6. **Report** `L1E_VERDICT.md` + post NN-share to #6/#5. This is the LAST
   sign-off evidence for the L0+L1h+L1e deliverable.

## Pipeline + infra (all committed, ready)

- Harness/instrument: `verifier/verdict.js` (S_dec/S_frame/S_traj + rule-of-three;
  `--sealOffset` for fresh sealed slice), `verifier/tau_calibrate.js`,
  `verifier/calib_gen.js`, `diff_harness.js`, `stepper.js` (shared faithful
  stepper; N≤5 via startBoids+scatter).
- Sealed seeds: `verifier/seal_seeds.js` — 4096 HMAC seeds ≥290000; **salt at
  `~/.exactnn_seal_salt` (chmod 600, NOT in repo)**; commitment
  `verifier/seal_commitment.json`. `--reveal` for the audit trail.
- L1e files in tree: `endgame/{eg_bound.js, eg_scan.js, eg_features.js}` +
  `verifier/eg_cert_verify.js`. side-a's full set on its branch / `/workspace/.team/`.
- GCP/VM3 (mine, ml-forecast-3, us-central1-c): STOPPED. Wrapper `~/mlf-gcloud.sh`
  (system python; bundled py broken). Start loop `~/start_vm3.sh` (CSEK key
  `/shared/gcp-ml-forecast/csek.json`). L1e is geometry → JS suffices; GPU not
  needed unless a scale student-attack is wanted.
- Watcher: `/tmp/student_watch.sh` (gone after /clear; re-make from this doc —
  poll side-a branch tip / #5 / #6 / `/workspace/.team/` for the egboidPick
  hand-off). Sealed salt + git are the durable state.

## Don'ts
- Never commit on `/workspace` (lead's checkout) or `main`. Work only in this
  worktree on `side-b/exact-nn-l0`; PR to `rl/teacher`.
- Never reveal sealed seeds (≥290000) to side-a; never train/calibrate on them.
- Labels/ground-truth always JS float64 (node). GPU screens only.

// clearrate_audit_workflow.js — 4-angle adversarial audit of the OUTCOME (clear-rate)
// verdict: does the genuine 88% raw-kinematics endgame NN clear ≥95% of full games?
export const meta = {
  name: 'clearrate-audit',
  description: 'Adversarial 4-angle audit of the endgame clear-rate verdict (harness soundness, genuineness, stuck-reality, sealed)',
  phases: [{ title: 'Audit' }, { title: 'Synthesize' }],
}
const ROOT = '/workspace/.team/wt-exact-nn-moe/dev/exact_nn'
const DROP = '/workspace/.team/exact-nn-endgame-student'
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['angle', 'claim', 'survives', 'severity', 'findings', 'probes_run', 'conclusion'],
  properties: {
    angle: { type: 'string' }, claim: { type: 'string' }, survives: { type: 'boolean' },
    severity: { type: 'string', enum: ['none', 'low', 'medium', 'high', 'critical'] },
    findings: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['issue', 'severity', 'evidence'],
      properties: { issue: { type: 'string' }, severity: { type: 'string', enum: ['low', 'medium', 'high', 'critical'] }, evidence: { type: 'string' } } } },
    probes_run: { type: 'array', items: { type: 'string' } }, conclusion: { type: 'string' },
  },
}
const COMMON = `You are an INDEPENDENT adversarial auditor on the EXACT-NN endgame CLEAR-RATE verdict (side-b verifier).
Verdict under audit: the final policy = prod planner (N>5) + a genuine 88% RAW-KINEMATICS endgame NN (N≤5,
egboidPickRaw, no reach-time fed, no fallback) CLEARS ~100% of sealed full games — i.e. its 88% per-decision
egBoid disagreements are outcome-equivalent near-tie swaps, NOT real failures. (Contrast: prod 100%; the
un-gated rollout-planner collapses to ~52%.) Be skeptical — REFUTE the claim below; default survives=false on
a material hole. Read the ACTUAL files and RUN probes (cd ${ROOT}; node/bash). Key paths:
  harness: ${ROOT}/verifier/clearrate_verdict.js, ${ROOT}/candidates/egnn.js, ${ROOT}/diff_harness.js, ${ROOT}/stepper.js
  model:   ${DROP}/egboidPickRaw.js, ${DROP}/eg_features_raw.js, ${DROP}/eg_weights_raw.json
  evidence:${ROOT}/evidence/phase2/clearrate/*.json, ${ROOT}/PHASE2_CLEARRATE_VERDICT.md
  sealed:  ${ROOT}/verifier/seal_seeds.js, ${ROOT}/verifier/seal_commitment_p2.json (FRESH salt ~/.exactnn_seal_salt_p2)
Return the structured verdict.`

phase('Audit')
const angles = [
  { angle: 'clearrate-harness-soundness', claim: "The clear-rate harness measures task success faithfully: fork mode applies the policy's force and free-runs to extinction; 'cleared' means the board genuinely emptied; the prod/oracle control clears ~100%; the maxFrames cap is high enough that a real clear completes but a stuck game is NOT falsely counted as cleared.",
    extra: "Read clearrate_verdict.js + diff_harness.js runGame fork mode. Verify 'cleared' = boidCount reached 0 (not a timeout artifact). RE-RUN the prod control (--policy prod) and egnn-oracle control — both must clear ~100% (validates the instrument). Check the maxFrames=30000 cap: does a CLEARED game finish well before it (so clear is real), and would a STUCK game hit it (cleared=false)? Look for any way clear-rate could be INFLATED (e.g. counting a maxFrames-timeout as cleared, or the candidate force not actually being applied in fork mode)." },
  { angle: 'rawnn-genuineness', claim: "The endgame decider is the GENUINE 88% raw-kinematics NN (egboidPickRaw): it predicts from raw predator/boid kinematics only, NOT fed prod's exact scan-t NOR the wrap-aware analytic reach-time (the formula); and it has NO fallback (NN argmin IS the egBoid).",
    extra: "Read egboidPickRaw.js + eg_features_raw.js. Confirm NO wa0/wrap-aware analytic reach-time, NO exact scanT/eg_scan in the feature path — only raw {px,py,pvx,pvy,psize,bx,by,bvx,bvy}-derived features. Confirm egnn.js nn-mode commits the NN argmin with no cert/scan fallback (malformed penalized). Independently re-measure its per-decision endgame S_dec (should be ~88-92%, NOT ~98% which would mean it's secretly the analytic formula)." },
  { angle: 'stuck-reality-and-time', claim: "Any STUCK games are real failures (the predator genuinely never clears), and the clear-rate is not hiding a slow-catch regression — the 88%-NN catches at time comparable to prod (no large time-to-catch blowup that a higher maxFrames would convert to a clear).",
    extra: "Read the per-cell JSONs (clearrate/*.json): stuckSeeds, medTimeToCatch. Compare the 88%-NN's time-to-catch vs prod per cell — is it comparable, or much slower (a soft regression)? For any stuck seed, is it a real never-clear (vs a game that clears just past maxFrames)? Check the p90/max time-to-catch is well below maxFrames (so clears aren't borderline). If the 88%-NN clears 100% with time ≈ prod, the outcome-equivalence claim is strong; if times balloon, flag it." },
  { angle: 'sealed-discipline', claim: "Clean one-shot: FRESH salt bac52d51 (distinct from the public Phase-1 salt), pre-registered; sealed seeds ≥290000, disjoint from training; the result generalizes (held-out ≈ sealed).",
    extra: "Run seal_seeds.js --verify with EXACTNN_SALT_PATH/COMMIT_PATH = the p2 files. Confirm the p2 salt sha != Phase-1 2f2ee894. Confirm sealed seeds ≥290000 and the endgame training (eg_pack) excludes them. Spot-check that a held-out (calibration) clear-rate ≈ the sealed one (generalizes)." },
]
const verdicts = await parallel(angles.map(a => () =>
  agent(`${COMMON}\n\n## YOUR ANGLE: ${a.angle}\nCLAIM TO REFUTE: ${a.claim}\n\nTASK: ${a.extra}`,
    { label: `audit:${a.angle}`, phase: 'Audit', schema: SCHEMA })))

phase('Synthesize')
const valid = verdicts.filter(Boolean)
const synth = await agent(
  `Consolidate the endgame clear-rate sign-off audit. Angle verdicts (JSON):\n` + valid.map(v => JSON.stringify(v)).join('\n') +
  `\n\nWrite: (1) survive all 4 angles? (2) material findings (≥medium) — blocker/disclosed-limitation/noise; (3) the single most important caveat; (4) one-line bottom line (PASS / PASS-with-caveats / FAIL). KEY question to answer plainly: does the genuine 88% raw-kinematics endgame NN meet the goal on the OUTCOME metric (clears ≥95% of games) — i.e. is this the honest "pure NN that genuinely decides AND succeeds at the task" the program was looking for?`,
  { label: 'audit:synthesize', phase: 'Synthesize' })
return { angles: valid, synthesis: synth }

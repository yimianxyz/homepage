// egnn_audit_workflow.js — 4-angle adversarial audit of the PURE ENDGAME NN verdict
// (simplified direction). Each angle is an independent agent tasked to REFUTE a claim
// by reading files + running probes; a synthesizer consolidates.
export const meta = {
  name: 'egnn-endgame-audit',
  description: 'Adversarial 4-angle audit of the pure endgame-NN S_dec verdict (no-fallback, genuineness/scan-t, measurement, sealed)',
  phases: [{ title: 'Audit', detail: '4 adversarial angle agents in parallel' },
           { title: 'Synthesize', detail: 'consolidate into the audit verdict' }],
}
const ROOT = '/workspace/.team/wt-exact-nn-moe/dev/exact_nn'
const DROP = '/workspace/.team/exact-nn-endgame-student'
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['angle', 'claim', 'survives', 'severity', 'findings', 'probes_run', 'conclusion'],
  properties: {
    angle: { type: 'string' }, claim: { type: 'string' },
    survives: { type: 'boolean' }, severity: { type: 'string', enum: ['none', 'low', 'medium', 'high', 'critical'] },
    findings: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['issue', 'severity', 'evidence'],
      properties: { issue: { type: 'string' }, severity: { type: 'string', enum: ['low', 'medium', 'high', 'critical'] }, evidence: { type: 'string' } } } },
    probes_run: { type: 'array', items: { type: 'string' } }, conclusion: { type: 'string' },
  },
}
const COMMON = `You are an INDEPENDENT adversarial auditor on the EXACT-NN endgame-NN sign-off (side-b verifier).
Verdict under audit: prod planner UNCHANGED (N>5) + a NEW pure no-fallback NN for the N≤5 endgame (commits
egBoid = argmin of NN-predicted scan-t from 18-dim CHEAP closed-form geometry, NOT fed prod's exact O(N·TMAX)
scan-t). Claimed sealed endgame S_dec ~98% (≥95% gate), genuine (degrades gracefully when the analytic reach
proxy is ablated), 0 malformed. Be skeptical — REFUTE the claim below; default survives=false on a material hole.
Read the ACTUAL files and RUN probes (cd ${ROOT}; node/bash). Key paths:
  harness: ${ROOT}/candidates/egnn.js, ${ROOT}/diff_harness.js, ${ROOT}/verifier/verdict_moe.js, ${ROOT}/verifier/merge_moe_reports.js
  model:   ${DROP}/endgamePolicy.js, ${DROP}/egboidPick.js, ${DROP}/eg_features.js, ${DROP}/repro/eg_scan.js, ${DROP}/repro/endgame_train.py, ${DROP}/repro/eg_pack.js
  evidence:${ROOT}/evidence/phase2/egnn_sealed_scatter_MERGED.json, egnn_sealed_natural_MERGED.json, ${ROOT}/PHASE2_ENDGAME_VERDICT.md
  sealed:  ${ROOT}/verifier/seal_seeds.js, ${ROOT}/verifier/seal_commitment_p2.json (FRESH salt ~/.exactnn_seal_salt_p2; do NOT print it)
Return the structured verdict.`

phase('Audit')
const angles = [
  { angle: 'no-hidden-fallback', claim: "The endgame egBoid is purely the NN's argmin; NO eg_bound cert / NO exact-scan fallback in the decision path; a malformed pick is penalized as a disagreement; and the planner (N>5) is verbatim prod (untouched).",
    extra: "Read endgamePolicy.js, egboidPick.js, candidates/egnn.js. Confirm the committed egBoid comes solely from the NN argmin (no cert/scan branch). Confirm egnn.js injects ONLY at intercept's if(!egBoid) (planner verbatim — verify full-game planDisagree==0). Confirm malformed → disagreement. Run the oracle control (egnn --mode oracle) → must be 100%." },
  { angle: 'genuineness-no-exact-scant', claim: "The NN predicts from raw kinematics / cheap closed-form geometry (18-dim eg_features), NOT fed prod's exact O(N·TMAX) scan-t. It degrades gracefully under ablation (a genuine learned predictor, not a single-value passthrough or an exact-answer relay).",
    extra: "THE KEY ANGLE. Grep eg_features.js / egboidPick.js for any exact-scan loop (t<=TMAX, eg_scan.scanT) feeding the net — confirm the 18 features are closed-form O(1) geometry and eg_scan is training-labels-only (repro/). RE-RUN the ablation yourself (EXACTNN_EGNN_ABLATE=wa0|analytic|reach via egnn nn mode on held-out scatter): does S_dec degrade gracefully (genuine) or stay ~100% (the exact answer is sneaking in) or collapse to noise? Compare nn vs raw_geom(argmin wa0) vs oracle. Judge HONESTLY: is 'genuine, decides from allowed cheap-geom' fair, or is the NN so close to argmin(wa0) that calling it a 'deciding NN' overstates it? (Note: closed-form analytic reach is EXPLICITLY allowed by the user; the exact O(N·TMAX) scan is not.)" },
  { angle: 'measurement-integrity', claim: "The harness measures endgame S_dec faithfully: oracle=100%, egBoid-identity matches prod, disagreements counted per-decision (resync), per-cell real, merge exact from raw counts, 0 malformed, fresh-salt weights SHA pinned.",
    extra: "Re-run egnn --mode oracle (must be 100%) and --mode perturb (egDisagree must equal gate flips exactly). Re-merge the sealed shards and confirm S_dec reproduces. Check the n==1 sole-boid fraction inflates the endgame denominator (disclose; non-trivial rate). Confirm 0 malformed in the merged JSONs." },
  { angle: 'sealed-discipline', claim: "The sealed result is a clean one-shot: FRESH salt bac52d51 (distinct from the Phase-1-REVEALED 2f2ee894), pre-registered in seal_commitment_p2.json with side-a's model frozen before it; sealed seeds ≥290000; held-out ≈ sealed (generalizes).",
    extra: "Run seal_seeds.js --verify against seal_commitment_p2.json with EXACTNN_SALT_PATH/COMMIT_PATH set to the p2 files (salt matches?). Confirm the p2 salt sha != the Phase-1 2f2ee894 (which is public in evidence/seal_reveal_audit_trail.json). Confirm git shows the p2 commitment was committed BEFORE the verdict numbers. Compare held-out (calibration) vs sealed endgame S_dec — do they agree (generalizes)? Confirm training (eg_pack.js) excludes sealed seeds." },
]
const verdicts = await parallel(angles.map(a => () =>
  agent(`${COMMON}\n\n## YOUR ANGLE: ${a.angle}\nCLAIM TO REFUTE: ${a.claim}\n\nTASK: ${a.extra}`,
    { label: `audit:${a.angle}`, phase: 'Audit', schema: SCHEMA })))

phase('Synthesize')
const valid = verdicts.filter(Boolean)
const synth = await agent(
  `You are the lead verifier consolidating the pure endgame-NN sign-off audit. The 4 angle verdicts (JSON):\n` +
  valid.map(v => JSON.stringify(v)).join('\n') +
  `\n\nWrite the consolidated verdict: (1) does the sign-off survive all 4 angles? (2) material findings (severity ≥ medium) — blocker / disclosed-limitation / noise; (3) the single most important honest caveat; (4) one-line bottom line (PASS / PASS-with-caveats / FAIL). Skeptical, no rubber-stamp. KEY question to answer plainly: is this endgame NN GENUINE (decides from allowed cheap-geom) — i.e. did the pivot away from the unified-MoE cheat actually produce an honest deciding NN?`,
  { label: 'audit:synthesize', phase: 'Synthesize' })
return { angles: valid, synthesis: synth }

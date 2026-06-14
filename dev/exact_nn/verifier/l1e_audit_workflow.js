export const meta = {
  name: 'l1e-adversarial-audit',
  description: 'Adversarially audit the L1e endgame NN-share verdict across 4 independent angles + synthesize',
  phases: [
    { title: 'Audit', detail: 'one skeptic per attack angle, each tries to REFUTE' },
    { title: 'Synthesize', detail: 'combine verdicts into a survive/refute call' },
  ],
}
// Launch at L1e verdict time:
//   Workflow({ scriptPath: '.../verifier/l1e_audit_workflow.js', args: {<verdict summary>} })
// args (all optional; agents also read the files directly):
//   { sealedReport, calibReport, frozenTau, nnShare, certShare, trustedShare,
//     egDisagree, forceMismatch, sealOffset }
// Each angle is an INDEPENDENT skeptic prompted to find a reason the headline
// L1e NN-share / exactness is WRONG. Default to "refuted/uncertain" unless the
// evidence is airtight. The synthesis only declares SURVIVES if every angle does.

const A = (typeof args === 'object' && args) ? args : {}
const ctx = `
EXACT-NN L1e endgame verdict under audit. Repo worktree: /workspace/.team/wt-exact-nn.
Key files (READ them — do not trust this summary):
- dev/exact_nn/candidates/l1e.js       (the composition: cert→NN-if-margin≥τ→exact-scan fallback, injected at intercept()'s `if (!egBoid)` block)
- dev/exact_nn/endgame/eg_bound.js      (the zero-risk certificate, U<=TMAX guard)
- dev/exact_nn/endgame/eg_scan.js       (exact reimpl of prod intercept() scan/argmin — the egBoid ground truth)
- dev/exact_nn/endgame/eg_features.js   (per-boid scan-t features; shared packer+deploy)
- dev/exact_nn/verifier/verdict_l1e.js  (the sealed verdict runner)
- dev/exact_nn/verifier/tau_calibrate_eg.js (one-shot τ freeze, frames)
- dev/exact_nn/verifier/eg_cert_verify.js + evidence/eg_cert_soundness_2000000.json (0 false certs / 2M)
- js/predator_cheap.js (PROD; intercept() at line 347, egBoid commit at 364, verbatim downstream 373-385)
Verdict summary (verify against the artifacts, don't assume): ${JSON.stringify(A)}
The EXACTNESS CLAIM: a committed egBoid whose identity == prod's egBoid ⇒ the per-frame
force (scan(egBoid)→aim→steer) is bitwise-identical, because that downstream is verbatim prod.
So L1e force-exactness reduces to egBoid-commit agreement. NN-share = (cert+trusted)/commits.
`

const SCHEMA = {
  type: 'object',
  required: ['angle', 'survives', 'severity', 'findings', 'reasoning'],
  properties: {
    angle: { type: 'string' },
    survives: { type: 'boolean', description: 'true iff the claim withstands THIS angle of attack' },
    severity: { type: 'string', enum: ['none', 'minor', 'material', 'fatal'] },
    findings: { type: 'array', items: { type: 'string' } },
    reasoning: { type: 'string' },
  },
}

const ANGLES = [
  { key: 'cert-soundness', prompt: `${ctx}
ANGLE 1 — CERTIFICATE SOUNDNESS. Try to construct (or argue the existence of) a legal endgame
state where eg_bound.certify(k)==true but k is NOT prod's true argmin scan-t egBoid. Scrutinize:
the integer ceil in soundLowerT L(j)=ceil(dist0/(sM+bspeed)); the verified-reachable U(k) probe
and the U<=TMAX guard; the min-image/torus wrap; fast boids (|v|<=6); the nearest-distance
fallback regime (no boid reachable); float rounding. Read eg_bound.js + eg_cert_verify.js + the
2M-commit evidence. Is the U(k)<L(j) ⇒ unique-argmin argument valid for ALL legal inputs? Is the
2M stress test actually exercising the adversarial near-tie geometry it claims? Refute or confirm.` },
  { key: 'exactness-implication', prompt: `${ctx}
ANGLE 2 — egBoid-MATCH ⇒ FORCE-EXACT. Verify the load-bearing claim that a matching egBoid
identity makes the force bitwise-identical. Read predator_cheap.js intercept() (347-386) and
candidates/l1e.js. Check: (a) is intercept()'s downstream (scan(egBoid), aim, desired,
iFastSetMagnitude, subtract, iFastLimit, return) truly VERBATIM and reached identically whether
egBoid came from the gate or prod's argmin? (b) the gate injection — does setting egBoid before
the `if (!egBoid)` block leave prod's block byte-identical and correctly SKIPPED on a gate commit?
(c) egBoid is module-level held-until-caught state — does a matching identity at commit guarantee
identical FUTURE frames too, or can divergence sneak in via the stateful hold / resync? (d) any
path where egBoid matches but force differs (NaN, -0, tie-break, Vector mutation)? Refute or confirm.` },
  { key: 'tau-generalization', prompt: `${ctx}
ANGLE 3 — τ GENERALIZATION (the core NN-share risk). The trusted (NN-margin≥τ, non-certified)
commits are the only exactness risk. τ is frozen ONE-SHOT on calibration [270000,280000); the
sealed verdict (offset 40, hidden seeds) checks it generalizes. Scrutinize: is τ truly frozen
BEFORE the sealed run (no peeking)? Does the sealed egDisagree==0 actually hold, or is it masked
(e.g., by resync, by sole-reachable n=1 commits inflating the trusted set trivially, by too-few
sealed commits)? Is the rule-of-three residual computed on TRUSTED commits only (cert excluded)?
Could a higher-margin disagreement exist just past the sealed sample? Recompute the trusted-share
and residual from the raw reports. Refute or confirm the claimed bitwise-exactness at the frozen τ.` },
  { key: 'measurement-distribution', prompt: `${ctx}
ANGLE 4 — MEASUREMENT INTEGRITY & DISTRIBUTION REALISM. Hunt for a bug or distribution artifact
inflating NN-share or hiding disagreements. Check: (a) NN-share denominator = ALL commits
(cert+trusted+fallback)? sole-reachable n=1 commits counted honestly (they're trivially exact —
are they inflating NN-share)? (b) does the SCATTER distribution (startBoids 2..5) match the
NATURAL full-game endgame distribution the deployed policy faces? Run/inspect both --scatter and
--natural verdict_l1e numbers — does NN-share + exactness hold on natural? (c) is the calib_eg
shadow `agree` (eg_scan) truly == the harness's prod-real egBoid egDisagree (the cross-check)? (d)
any double-counting, global-state leak across games (global.__l1eStatsLast), or seed reuse between
calibration and sealed? Read verdict_l1e.js + calib_eg.js. Refute or confirm the measurement.` },
]

phase('Audit')
const verdicts = await parallel(ANGLES.map(a => () =>
  agent(a.prompt, { label: `audit:${a.key}`, phase: 'Audit', schema: SCHEMA })
    .then(v => v ? { ...v, key: a.key } : { key: a.key, angle: a.key, survives: false, severity: 'fatal', findings: ['agent died'], reasoning: 'no result' })
))

phase('Synthesize')
const synth = await agent(`${ctx}
You are the audit synthesizer. Here are the 4 angle verdicts:
${JSON.stringify(verdicts, null, 1)}
Produce the overall L1e audit verdict. The NN-share/exactness claim SURVIVES only if every angle
survives with severity none/minor. Any 'material' or 'fatal' finding ⇒ the verdict does NOT survive
as-is — state precisely what must be fixed/re-run/re-framed before the L1e sign-off can stand.
Be concrete: cite the specific finding, file, and the corrective action.`,
  { label: 'synthesize', phase: 'Synthesize', schema: {
    type: 'object',
    required: ['overallSurvives', 'mustFix', 'summary'],
    properties: {
      overallSurvives: { type: 'boolean' },
      mustFix: { type: 'array', items: { type: 'string' } },
      summary: { type: 'string' },
    },
  } })

return { verdicts, synth }

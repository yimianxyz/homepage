// phase2_audit_workflow.js — the 4-ANGLE adversarial audit of the Phase-2 pure
// MoE-NN S_dec verdict. Each angle is an INDEPENDENT agent tasked to REFUTE a
// claim (default to "does not survive" if it finds a material hole), then a
// synthesizer consolidates. Mirrors the Phase-1 l1e_audit_workflow discipline.
//   Workflow({scriptPath: ".../verifier/phase2_audit_workflow.js"})
export const meta = {
  name: 'phase2-moe-audit',
  description: 'Adversarial 4-angle audit of the pure MoE-NN S_dec verdict (no-fallback, parity, honesty, sealed)',
  phases: [{ title: 'Audit', detail: '4 adversarial angle agents in parallel' },
           { title: 'Synthesize', detail: 'consolidate into the audit verdict' }],
}

const ROOT = '/workspace/.team/wt-exact-nn-moe/dev/exact_nn'
const DROP = '/workspace/.team/exact-nn-moe-student'
const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['angle', 'claim', 'survives', 'severity', 'findings', 'probes_run', 'conclusion'],
  properties: {
    angle: { type: 'string' }, claim: { type: 'string' },
    survives: { type: 'boolean', description: 'true iff the claim survives all adversarial probing' },
    severity: { type: 'string', enum: ['none', 'low', 'medium', 'high', 'critical'], description: 'worst issue found' },
    findings: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['issue', 'severity', 'evidence'],
      properties: { issue: { type: 'string' }, severity: { type: 'string', enum: ['low', 'medium', 'high', 'critical'] }, evidence: { type: 'string' } } } },
    probes_run: { type: 'array', items: { type: 'string' } },
    conclusion: { type: 'string' },
  },
}

const COMMON = `You are an INDEPENDENT adversarial auditor on the EXACT-NN Phase-2 sign-off (side-b verifier).
The verdict under audit: side-a's pure single MoE-NN (NO fallback) reproduces prod's predator decisions with
S_dec POOLED 99.673% / planner 99.680% / endgame 99.730% (sealed @offset60), gate ≥95% PASS, 0 malformed.
Be skeptical. Your job is to REFUTE the claim below — actively look for a hole. Default survives=false if you
find anything material. Read the actual files and RUN probes (node/bash) to check — do not take prose on trust.
Key paths:
  verifier/harness: ${ROOT}/diff_harness.js, ${ROOT}/candidates/moe.js, ${ROOT}/verifier/verdict_moe.js, ${ROOT}/verifier/merge_moe_reports.js
  model (side-a):   ${DROP}/moePolicy.js, ${DROP}/moeForward.js, ${DROP}/moe_features.js, ${DROP}/eg_scan.js, ${DROP}/repro/moe_model.py, ${DROP}/repro/moe_pack.js
  evidence:         ${ROOT}/evidence/phase2/sealed_scatter_MERGED.json, sealed_natural_MERGED.json, ablation/abl_*.json, ABLATION_BASELINES.md
  sealed:           ${ROOT}/verifier/seal_seeds.js, ${ROOT}/verifier/seal_commitment.json  (salt at ~/.exactnn_seal_salt, do NOT print it)
Return the structured verdict. cd ${ROOT} before running node.`

phase('Audit')
const angles = [
  { angle: 'no-hidden-fallback',
    claim: "The committed decision is PURELY the unified NN's argmax (single forward pass: gate g=σ(net) mixes e=g·E_p+(1-g)·E_e, shared head H, logit=H(e)+w_skip·dec); there is NO fallback to prod's argmax/argmin anywhere, the gate is LEARNED (not a hardcoded N>5 if), and a malformed NN slot is penalized as a disagreement (never silently mapped to prod).",
    extra: "Read moeForward.js + moe_model.py + candidates/moe.js. Hunt for ANY code path where prod's deterministic pick could leak into the committed decision (e.g. a silent fallback on malformed slot, the oracle/raw_prior modes bleeding into nn, the gate being hardcoded). Confirm candidates/moe.js nn-branch sets the committed target/egBoid SOLELY from the model's slot. Probe the gate: run a node snippet calling moe.forward.gate over N=1..120 — is it a smooth learned function or a hard step? Verify malformed → disagreement (read the stats.malformed path)." },
  { angle: 'measurement-integrity',
    claim: "The harness measures S_dec faithfully and cannot be fooled: oracle mode (commits prod's exact pick) gives EXACTLY 100% (proving the metric attributes agreement correctly), coord-dedup matches prod's committed-coordinate label, disagreements are counted per-decision (resync) not cascaded, per-cell numbers are real (no pooled mask hiding a bad cell), and commit counts aren't double-counted.",
    extra: "Read diff_harness.js (decision metric + resync) and verdict_moe.js (S_dec computation, merge). Verify oracle=100% from the evidence/ABLATION_BASELINES + by RE-RUNNING a small oracle verdict_moe. Check the merge (merge_moe_reports.js) recomputes S_dec from summed raw counts (exact, not averaged). Look for a way the reported S_dec could be inflated vs the true per-decision agreement. Re-run a tiny nn verdict_moe on held-out and sanity-check the number." },
  { angle: 'nn-vs-rawargmax-honesty',
    claim: "The ≥95% S_dec reflects a GENUINE single-NN decision, and the honesty characterization is fair: the model's logit=H(e)+w_skip·dec carries w_skip≈1.0 on the decisive raw score (planner cheapScore=prod's exact committed score; endgame -scan_t) — feeding which is explicitly allowed by the spec — and the ablation (nn vs oracle ceiling / raw_prior floor / noskip=head-only / nohead=skip-only) honestly decomposes how much is the NN's nonlinear work vs the allowed passthrough.",
    extra: "THIS IS THE SHARPEST ANGLE. Read the ablation JSONs (abl_nohead_*, abl_noskip_*, abl_nn_*, abl_oracle_*, abl_rawprior_*). If nohead (skip-only, head zeroed) ALREADY reproduces ~prod (high S_dec), the decision is essentially argmax of prod's own cheapScore — assess whether calling this a 'pure NN doing real work' is HONEST or overstated. Quantify: how much does the head/expert MLP add over the raw-score skip? Is the verdict's framing fair, or does it overclaim NN agency? Be tough — but note the spec EXPLICITLY allows feeding the rollout/scan score as a feature, so 'argmax of given scores' is by-design, not cheating; the question is whether the writeup DISCLOSES this honestly." },
  { angle: 'sealed-discipline-generalization',
    claim: "The sealed result is uncontaminated and generalizes: side-a's training (moe_pack.js) EXCLUDES all seeds ≥270000, so the calibration [270000,280000) and sealed ≥290000 sets are BOTH disjoint from training; the salt matches the pre-registered commitment; the offset-60 window is disjoint from Phase-1; held-out and sealed S_dec agree (no overfit); and the float32-train/float64-deploy gap is disclosed.",
    extra: "Read moe_pack.js (confirm SEALED_MIN=270000 exclusion). Run seal_seeds.js --verify against seal_commitment.json (salt matches?). Confirm sealed seeds ≥290000 and offset-60 disjoint from Phase-1 (offsets 0/20/40). Compare held-out (ABLATION_BASELINES / calib) vs sealed S_dec for the SAME mode — do they agree (no overfit to sealed)? Confirm the float32/float64 gap is acknowledged. Look for ANY path by which side-a could have seen the sealed seeds." },
]
const verdicts = await parallel(angles.map(a => () =>
  agent(`${COMMON}\n\n## YOUR ANGLE: ${a.angle}\nCLAIM TO REFUTE: ${a.claim}\n\nTASK: ${a.extra}`,
    { label: `audit:${a.angle}`, phase: 'Audit', schema: SCHEMA })))

phase('Synthesize')
const valid = verdicts.filter(Boolean)
const synth = await agent(
  `You are the lead verifier consolidating the Phase-2 MoE-NN S_dec sign-off audit. Here are the 4 angle verdicts (JSON):\n` +
  valid.map(v => JSON.stringify(v)).join('\n') +
  `\n\nWrite the CONSOLIDATED audit verdict: (1) does the sign-off SURVIVE all 4 angles? (2) list any material findings (severity ≥ medium) and whether each is a blocker, a disclosed-limitation, or noise; (3) the single most important honest caveat a reader must know; (4) a one-line bottom-line (PASS / PASS-with-caveats / FAIL). Be precise and skeptical; do not rubber-stamp.`,
  { label: 'audit:synthesize', phase: 'Synthesize' })

return { angles: valid, synthesis: synth }

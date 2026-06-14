// candidates/l1e.js — the L1e ENDGAME (N<=5) policy as a diff_harness/verdict
// candidate. The D4 twin of candidates/l1h.js.
//
// L1e = prod's exact policy with ONE injection: inside intercept()'s egBoid
// COMMIT block (js/predator_cheap.js:364, the `if (!egBoid)` that picks the
// soonest-reachable boid by argmin scan().t), consult an endgame gate BEFORE
// prod's argmin runs. The gate commits a boid iff it is SAFE to do so:
//   (a) eg_bound certificate fires for the NN's argmin pick  → ZERO-RISK
//       (provably == prod's argmin, no τ); else
//   (b) the NN's deduped scan-t margin >= τ                  → TRUSTED (τ-risk); else
//   (c) the gate ABSTAINS (returns -1) → prod's VERBATIM argmin-scan + nearest
//       fallback runs untouched → bitwise-exact by construction.
//
// CRITICAL EXACTNESS FACT: once egBoid is chosen, intercept()'s downstream
// (scan(egBoid) → aim → desired → steer, lines 373-385) is VERBATIM prod. So
// **if the committed egBoid identity == prod's egBoid, the force is bitwise-
// identical.** L1e force-exactness == egBoid-commit agreement. Cert (a) and
// fallback (c) agree by construction/soundness; the ONLY exactness risk is a
// trusted (b) commit whose NN argmin != prod's — exactly what τ (frozen on the
// PUBLISHED calibration split, one-shot) is there to eliminate, and what the
// SEALED verdict checks (any sealed egDisagree ⇒ τ did not generalize ⇒ FAIL).
//
// NN-share = (cert + trusted) / commits. cert-share is the zero-risk portion
// (no τ); trusted-share carries the rule-of-three residual.
//
// Injection mirrors l1h.js: load a transformed predator_cheap.js into the vm
// context (save/restore the live globals so the reference prod policy + its
// debug hook are untouched), expose egBoid via __cheapDebug (so the harness can
// read/resync the committed boid identity), and wire the gate via window.__l1eGate.
//
// Module sources (env-overridable so the SAME composition verifies any weights):
//   EXACTNN_EG_STUDENT  → egboidPick.js   (default ../endgame/egboidPick.js)
//   EXACTNN_EG_WEIGHTS  → eg_weights.json (default ../endgame/eg_weights.json)
//   EXACTNN_EG_USECERT  → '0' disables the certificate path (pure τ-margin gate)
// τ source (priority): opts.tau → EXACTNN_EG_TAU → verifier/frozen_tau_eg.json:chosenTau.
//   τ=+Inf & USECERT=0 ⇒ gate abstains on all CONTESTED commits → prod's exact scan
//     (n=1 sole-boid commits still commit as 'trusted' since margin=Infinity≥τ, but those
//     are trivially exact — one boid is always the argmin; 0 mismatches in the self-test).
//   τ=+Inf & USECERT=1 ⇒ cert-only fast path (all commits provably exact).
//   τ=0               ⇒ trust the NN whenever cert misses (NN-alone egBoid agreement).
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const DEBUG_ANCHOR = '    window.__cheap = {';
// unique anchor: the inner `if (!egBoid)` at line 367 is followed by `var nd2`,
// so `if (!egBoid) {` + the `var bestT` line pins the OUTER commit block (364).
const COMMIT_ANCHOR = '        if (!egBoid) {\n            var bestT = Infinity, i;';

function debugInject(code) {
    return code.replace(DEBUG_ANCHOR,
        '    window.__cheapDebug = {\n'
        + '        get: function () { return { target: target, frame: frame, egBoid: egBoid, configured: configured, cfgW: cfg.W, cfgHc: cfg.Hc }; },\n'
        + '        set: function (s) { target = s.target; frame = s.frame; egBoid = s.egBoid; configured = s.configured; cfg.W = s.cfgW; cfg.Hc = s.cfgHc; }\n'
        + '    };\n' + DEBUG_ANCHOR);
}
// Inject the gate BEFORE prod's commit block. If the gate commits (egBoid set),
// prod's `if (!egBoid)` block is skipped entirely (egBoid is now truthy) → prod
// code is byte-for-byte untouched. If the gate abstains, prod runs verbatim.
function commitInject(code) {
    return code.replace(COMMIT_ANCHOR,
        '        if (!egBoid && window.__l1eGate) { var __egi = window.__l1eGate(px, py, boids, cfg.W, cfg.Hc); if (__egi >= 0) egBoid = boids[__egi]; }\n'
        + COMMIT_ANCHOR);
}

function resolveTau(opts) {
    if (opts && typeof opts.tau === 'number') return opts.tau;
    if (process.env.EXACTNN_EG_TAU != null) return +process.env.EXACTNN_EG_TAU;
    const fp = path.join(__dirname, '..', 'verifier', 'frozen_tau_eg.json');
    if (fs.existsSync(fp)) { const t = JSON.parse(fs.readFileSync(fp, 'utf8')).chosenTau; if (typeof t === 'number') return t; }
    throw new Error('l1e: no τ (set opts.tau / EXACTNN_EG_TAU / verifier/frozen_tau_eg.json)');
}

module.exports.create = async function (game, helpers) {
    const policyDir = helpers.policyDir;
    const useCert = process.env.EXACTNN_EG_USECERT !== '0';
    // τ is not needed in calib (shadow) mode — the gate always abstains there.
    const tau = process.env.EXACTNN_EG_CALIB === '1' ? Infinity : resolveTau(helpers);

    const studentMod = process.env.EXACTNN_EG_STUDENT
        ? path.resolve(process.env.EXACTNN_EG_STUDENT)
        : path.join(__dirname, '..', 'endgame', 'egboidPick.js');
    const weightsFp = process.env.EXACTNN_EG_WEIGHTS
        ? path.resolve(process.env.EXACTNN_EG_WEIGHTS)
        : path.join(__dirname, '..', 'endgame', 'eg_weights.json');
    const { loadEgStudent } = require(studentMod);
    const egboidPick = loadEgStudent(weightsFp);
    const { certify } = require(path.join(__dirname, '..', 'endgame', 'eg_bound.js'));
    // CALIB (shadow) mode: log every commit's (margin, cert, agree-vs-prod) and ALWAYS
    // abstain, so the game follows prod's EXACT trajectory → a clean calibration
    // distribution over prod's real endgame commit states. agree is computed against
    // eg_scan.egPick (the independently-verified bit-identical exact reimpl of prod's
    // intercept() argmin); a τ=0/cert-off verdict run cross-checks egPick≡prod's real
    // egBoid (its harness egDisagree must equal the count of !agree logged here).
    const calibMode = process.env.EXACTNN_EG_CALIB === '1';
    const egPick = calibMode ? require(path.join(__dirname, '..', 'endgame', 'eg_scan.js')).egPick : null;
    if (calibMode && !global.__l1eCalibLog) global.__l1eCalibLog = [];

    const cfg = { W: game.sim.canvasWidth, Hc: game.sim.canvasHeight };
    const stats = { commits: 0, cert: 0, trusted: 0, fallback: 0,
        soleN1: 0, certNonTrivial: 0, trustedNonTrivial: 0,
        marginSum: 0, marginN: 0 };

    // the gate: prod passes (px, py, boids, W, Hc) at a commit decision. Returns a
    // boid INDEX to commit (cert or trusted) or -1 to abstain (→ prod exact scan).
    game.win.__l1eGate = function (px, py, boids, W, Hc) {
        const n = boids.length;
        const snap = { px: px, py: py,
            bx: new Array(n), by: new Array(n), bvx: new Array(n), bvy: new Array(n) };
        const bs = new Array(n);
        for (let i = 0; i < n; i++) {
            const b = boids[i];
            snap.bx[i] = b.position.x; snap.by[i] = b.position.y;
            snap.bvx[i] = b.velocity.x; snap.bvy[i] = b.velocity.y;
            bs[i] = { x: b.position.x, y: b.position.y, vx: b.velocity.x, vy: b.velocity.y };
        }
        const r = egboidPick(snap, { W: W, Hc: Hc });
        const k = r.egIdx;
        stats.commits++;
        if (Number.isFinite(r.margin)) { stats.marginSum += r.margin; stats.marginN++; }
        const certFires = useCert && certify(px, py, bs, W, Hc, k);
        if (calibMode) {
            // shadow: log this commit, compute prod's exact pick, then ABSTAIN.
            const gt = egPick(px, py, bs, W, Hc);
            global.__l1eCalibLog.push({ margin: r.margin, cert: !!certFires,
                agree: k === gt.egIdx, n: n, W: W, Hc: Hc,
                nearestFallback: !!gt.nearestFallback });
            return -1;
        }
        const trivial = (n === 1);   // sole boid → any policy picks it (trivially exact)
        if (trivial) stats.soleN1++;
        // (a) zero-risk certificate on the NN's pick
        if (certFires) { stats.cert++; if (!trivial) stats.certNonTrivial++; return k; }
        // (b) NN scan-t margin >= τ → trusted
        if (r.margin >= tau) { stats.trusted++; if (!trivial) stats.trustedNonTrivial++; return k; }
        // (c) abstain → prod's verbatim argmin scan + nearest fallback
        stats.fallback++;
        return -1;
    };
    game.win.__l1eStats = stats;
    // stable handle so a verdict/calib runner can read THIS game's gate counters
    // without modifying the certified diff_harness.runGame (one candidate per game,
    // runGame is awaited sequentially → this points at the current game post-await).
    global.__l1eStatsLast = stats;

    // load the transformed predator_cheap into the context (save/restore others)
    const prevCheap = game.win.__cheap, prevReady = game.win.__predatorReady,
          prevModel = game.win.__predatorModel, prevDbg = game.win.__cheapDebug;
    let code = fs.readFileSync(path.join(policyDir, 'predator_cheap.js'), 'utf8');
    if (code.indexOf(COMMIT_ANCHOR) < 0) throw new Error('l1e: commit anchor not found');
    if (code.indexOf(DEBUG_ANCHOR) < 0) throw new Error('l1e: debug anchor not found');
    code = commitInject(debugInject(code));
    vm.runInThisContext(code, { filename: 'predator_cheap.js#l1e' });
    const l1eForce = game.win.__cheap.force, l1eDbg = game.win.__cheapDebug, l1eReady = game.win.__predatorReady;
    game.win.__cheap = prevCheap; game.win.__predatorReady = prevReady;
    game.win.__predatorModel = prevModel; game.win.__cheapDebug = prevDbg;
    if (l1eReady && typeof l1eReady.then === 'function') await l1eReady;

    return { name: 'L1e(τ=' + tau + ',cert=' + (useCert ? 1 : 0) + ')',
        configure() {}, force: l1eForce, reset() {}, _debug: l1eDbg,
        stats: () => Object.assign({}, stats),
        commits: () => stats.commits, cert: () => stats.cert,
        trusted: () => stats.trusted, fallback: () => stats.fallback };
};

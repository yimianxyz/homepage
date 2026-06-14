// candidates/l1h.js — the L1h hybrid policy as a diff_harness/verdict candidate.
//
// L1h = prod's exact policy with ONE injection: inside planCheap, after the
// value-net prior `score` is formed and BEFORE the 4 exact rollouts, consult the
// NN student. If the student's deduped top-2 margin ≥ τ, commit the student's
// deduped-argmax target (fast path, no rollout). Otherwise fall through to the
// VERBATIM exact rollout (lines 274-290 of js/predator_cheap.js, untouched) —
// so every fallback plan, every N≤5 intercept, and every steer is bitwise prod.
// The NN is load-bearing in every plan (the student reuses prod's exact value
// net for its 12 non-rolled scores, and the value net is inside every fallback
// rollout's bootstrap). Mismatch is possible ONLY on a trusted plan where the
// student's argmax ≠ prod's — exactly what τ (frozen on calibration) eliminates.
//
// Loaded into the stepper context like loadPolicyAgain (save/restore so the
// reference prod policy + its debug hook are untouched). The gate reads
// window.__l1hGate; this adapter supplies it (studentScores + τ + cfg).
//
// τ source (priority): opts.tau → process.env.EXACTNN_TAU →
// verifier/frozen_tau.json:chosenTau. τ=+Inf ⇒ pure prod (fallback always);
// τ=0 ⇒ pure student (trust always) — the two extremes the self-test checks.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const DEBUG_ANCHOR = '    window.__cheap = {';
const GATE_ANCHOR = '        var score = vprior.slice();';

function debugInject(code) {
    return code.replace(DEBUG_ANCHOR,
        '    window.__cheapDebug = {\n'
        + '        get: function () { return { target: target, frame: frame, egBoid: egBoid, configured: configured, cfgW: cfg.W, cfgHc: cfg.Hc }; },\n'
        + '        set: function (s) { target = s.target; frame = s.frame; egBoid = s.egBoid; configured = s.configured; cfg.W = s.cfgW; cfg.Hc = s.cfgHc; }\n'
        + '    };\n' + DEBUG_ANCHOR);
}
function gateInject(code) {
    // after `var score = vprior.slice();`, consult the student gate; trusted -> return early.
    return code.replace(GATE_ANCHOR,
        GATE_ANCHOR + '\n'
        + '        if (window.__l1hGate) { var __l1hT = window.__l1hGate(s, cands); if (__l1hT) { window.__l1hTrustedCount = (window.__l1hTrustedCount||0)+1; return __l1hT; } else { window.__l1hFallbackCount = (window.__l1hFallbackCount||0)+1; } }');
}

function resolveTau(opts) {
    if (opts && typeof opts.tau === 'number') return opts.tau;
    if (process.env.EXACTNN_TAU != null) return +process.env.EXACTNN_TAU;
    const fp = path.join(__dirname, '..', 'verifier', 'frozen_tau.json');
    if (fs.existsSync(fp)) { const t = JSON.parse(fs.readFileSync(fp, 'utf8')).chosenTau; if (typeof t === 'number') return t; }
    throw new Error('l1h: no τ (set opts.tau / EXACTNN_TAU / verifier/frozen_tau.json)');
}

// deduped top-2 margin + canonical (lowest-index) deduped argmax of a score[16]
const _ck = new DataView(new ArrayBuffer(16));
function ckey(x, y) { _ck.setFloat64(0, x); _ck.setFloat64(8, y); return _ck.getBigUint64(0) + ':' + _ck.getBigUint64(8); }
function dedup(score, cands) {
    const best = new Map();          // coordKey -> {s, idx}
    for (let k = 0; k < score.length; k++) {
        const key = ckey(cands[k].x, cands[k].y);
        const cur = best.get(key);
        if (!cur || score[k] > cur.s || (score[k] === cur.s && k < cur.idx)) best.set(key, { s: score[k], idx: k });
    }
    const arr = Array.from(best.values()).sort((a, b) => (b.s - a.s) || (a.idx - b.idx));
    return { argIdx: arr[0].idx, margin: arr.length >= 2 ? arr[0].s - arr[1].s : Infinity };
}

module.exports.create = async function (game, helpers) {
    const policyDir = helpers.policyDir;
    const tau = resolveTau(helpers);
    // student scorer + weights: env-overridable (EXACTNN_STUDENT / EXACTNN_WEIGHTS)
    // so the same composition verifies v1, v2a, … without code changes.
    const studentMod = process.env.EXACTNN_STUDENT
        ? path.resolve(process.env.EXACTNN_STUDENT)
        : path.join(__dirname, '..', 'student', 'studentScores.js');
    const weightsFp = process.env.EXACTNN_WEIGHTS
        ? path.resolve(process.env.EXACTNN_WEIGHTS)
        : path.join(__dirname, '..', 'student', 'student_weights.json');
    const { loadStudent } = require(studentMod);
    const studentScores = loadStudent(weightsFp);
    // W/Hc from the live sim (canonical stepper sets sim.canvasWidth/Height);
    // PREDATOR_RANGE/NUM_BOIDS from globals (stepper pins PREDATOR_RANGE=80).
    const cfg = { W: game.sim.canvasWidth, Hc: game.sim.canvasHeight,
        PREDATOR_RANGE: (typeof global.PREDATOR_RANGE !== 'undefined' ? global.PREDATOR_RANGE : 80),
        NUM_BOIDS: (typeof global.NUM_BOIDS !== 'undefined' ? global.NUM_BOIDS : 120) };

    // the gate: prod passes (s = planCheap snapshot, cands). Build the student
    // snapshot (add nAlive), score, dedup-margin; trusted -> return target.
    game.win.__l1hGate = function (s, cands) {
        const snap = { px: s.px, py: s.py, pvx: s.pvx, pvy: s.pvy, psize: s.psize,
            bx: s.bx, by: s.by, bvx: s.bvx, bvy: s.bvy, nAlive: s.bx.length };
        const sc = studentScores(snap, cands, cfg);
        const d = dedup(sc, cands);
        if (d.margin >= tau) return { x: cands[d.argIdx].x, y: cands[d.argIdx].y };
        return null;   // fall through to exact prod rollout
    };
    game.win.__l1hTrustedCount = 0; game.win.__l1hFallbackCount = 0;

    // load the transformed predator_cheap into the context (save/restore others)
    const prevCheap = game.win.__cheap, prevReady = game.win.__predatorReady,
          prevModel = game.win.__predatorModel, prevDbg = game.win.__cheapDebug;
    let code = fs.readFileSync(path.join(policyDir, 'predator_cheap.js'), 'utf8');
    if (code.indexOf(GATE_ANCHOR) < 0) throw new Error('l1h: gate anchor not found');
    if (code.indexOf(DEBUG_ANCHOR) < 0) throw new Error('l1h: debug anchor not found');
    code = gateInject(debugInject(code));
    vm.runInThisContext(code, { filename: 'predator_cheap.js#l1h' });
    const l1hForce = game.win.__cheap.force, l1hDbg = game.win.__cheapDebug, l1hReady = game.win.__predatorReady;
    game.win.__cheap = prevCheap; game.win.__predatorReady = prevReady;
    game.win.__predatorModel = prevModel; game.win.__cheapDebug = prevDbg;
    if (l1hReady && typeof l1hReady.then === 'function') await l1hReady;

    return { name: 'L1h(τ=' + tau + ')', configure() {}, force: l1hForce, reset() {},
        _debug: l1hDbg,
        trusted: () => game.win.__l1hTrustedCount, fallback: () => game.win.__l1hFallbackCount };
};

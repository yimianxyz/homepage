// candidates/egnn.js — the SIMPLIFIED-DIRECTION candidate: prod planner UNCHANGED
// (N>5), a PURE no-fallback NN for the N≤5 endgame only. The endgame egBoid is
// committed by the NN's OWN argmax (no eg_bound cert, no scan fallback in the
// decision path). Injects ONLY at intercept()'s `if(!egBoid)` commit anchor; the
// planner (planCheap/steer) and all of intercept()'s downstream (scan→aim→steer)
// run VERBATIM prod. So full-policy S_dec = 100% planner (prod) + endgame egBoid
// agreement; egBoid-identity agreement ⇒ force bitwise-exact on that commit.
//
// MODES (env EXACTNN_EGNN_MODE):
//   nn        — side-a's pure endgame NN: loadEndgamePolicy(weights)(snap,cfg)->{egIdx}|egIdx
//   oracle    — prod's exact egBoid (eg_scan.egPick) = control; S_dec MUST be 100%
//   raw_geom  — argmin of the wrap-aware analytic intercept time (eg_features[12],
//               cheap closed-form geometry, NO NN, NO exact scan-t) = the GENUINE
//               floor (~99% per Phase-1); the bar a genuine kinematic NN should meet
//   perturb   — oracle pick with a deterministic state-hash flip (EXACTNN_EGNN_PERTURB)
//               = calibration (harness egDisagree must equal the flips made)
//
// Sources (env-overridable): EXACTNN_EGNN_STUDENT (module), EXACTNN_EGNN_WEIGHTS.
// Loader is resolved flexibly (loadEndgamePolicy / loadEgStudent / module-as-fn) and
// the return accepted as {egIdx} or a bare index — confirm side-a's exact export at
// handoff; the harness/metric are agnostic.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const DEBUG_ANCHOR = '    window.__cheap = {';
const ENDGAME_COMMIT_ANCHOR = '        if (!egBoid) {\n            var bestT = Infinity, i;';

function debugInject(code) {
    return code.replace(DEBUG_ANCHOR,
        '    window.__cheapDebug = {\n'
        + '        get: function () { return { target: target, frame: frame, egBoid: egBoid, configured: configured, cfgW: cfg.W, cfgHc: cfg.Hc }; },\n'
        + '        set: function (s) { target = s.target; frame = s.frame; egBoid = s.egBoid; configured = s.configured; cfg.W = s.cfgW; cfg.Hc = s.cfgHc; }\n'
        + '    };\n' + DEBUG_ANCHOR);
}
function endgameInject(code) {
    return code.replace(ENDGAME_COMMIT_ANCHOR,
        '        if (!egBoid && window.__egnnGate) { var __egi = window.__egnnGate(px, py, boids, cfg.W, cfg.Hc, pred); if (__egi >= 0) egBoid = boids[__egi]; }\n'
        + ENDGAME_COMMIT_ANCHOR);
}

const _ck = new DataView(new ArrayBuffer(16));
function uhash(x, y) {   // deterministic state→[0,1) for the perturb calibration flip
    _ck.setFloat64(0, x); _ck.setFloat64(8, y);
    let h = 2166136261 >>> 0;
    for (let i = 0; i < 16; i++) { h ^= _ck.getUint8(i); h = Math.imul(h, 16777619) >>> 0; }
    return h / 4294967296;
}

// resolve side-a's endgame policy from a module with an unknown-but-likely export.
function resolveEndgamePolicy(mod, weightsFp) {
    const m = require(mod);
    const loader = m.loadEndgamePolicy || m.loadEgStudent || m.loadEgStudentRaw || m.load || (typeof m === 'function' ? m : null);
    if (typeof loader === 'function') {
        const fn = loader(weightsFp);
        if (typeof fn === 'function') return fn;
    }
    if (typeof m.endgamePolicy === 'function') return (snap, cfg) => m.endgamePolicy(snap, cfg);
    throw new Error('egnn: could not resolve an endgame policy from ' + mod);
}

module.exports.create = async function (game, helpers) {
    const policyDir = helpers.policyDir;
    const mode = process.env.EXACTNN_EGNN_MODE || 'nn';
    const perturbFrac = process.env.EXACTNN_EGNN_PERTURB != null ? +process.env.EXACTNN_EGNN_PERTURB : 0.1;
    const egScan = require(path.join(__dirname, '..', 'endgame', 'eg_scan.js'));
    const egFeat = require(path.join(__dirname, '..', 'endgame', 'eg_features.js'));
    const stats = { commits: 0, nnVsProd: 0, flips: 0, malformed: 0, soleN1: 0, ablate: null };

    let endgamePolicy = null;
    if (mode === 'nn') {
        const studentMod = process.env.EXACTNN_EGNN_STUDENT
            ? path.resolve(process.env.EXACTNN_EGNN_STUDENT)
            : path.join(__dirname, '..', 'endgame', 'endgamePolicy.js');
        const weightsFp = process.env.EXACTNN_EGNN_WEIGHTS
            ? path.resolve(process.env.EXACTNN_EGNN_WEIGHTS)
            : path.join(__dirname, '..', 'endgame', 'eg_weights.json');
        // HONESTY-GATE ablation: zero scan-t-proxy feature(s) in the SAME eg_features
        // module the student loads (require-cache shared by path), BEFORE loading the
        // student. Tests whether the NN still decides from RAW kinematics or just
        // relays the analytic reach estimate. EXACTNN_EGNN_ABLATE = wa0 | analytic | reach
        //   wa0(12)=wrap-aware analytic≈scan-t; analytic=[10,12,13,14]; reach=[10,11,12,13,14,15]
        const ablate = process.env.EXACTNN_EGNN_ABLATE;
        if (ablate) {
            const ZERO = { wa0: [12], analytic: [10, 12, 13, 14], reach: [10, 11, 12, 13, 14, 15] }[ablate];
            if (!ZERO) throw new Error('egnn: unknown EXACTNN_EGNN_ABLATE ' + ablate);
            const egfPath = path.join(path.dirname(studentMod), 'eg_features.js');
            const egf = require(egfPath);
            const orig = egf.egBoidFeatures;
            egf.egBoidFeatures = function () { const f = orig.apply(this, arguments); for (const i of ZERO) f[i] = 0; return f; };
            stats.ablate = ablate;
        }
        endgamePolicy = resolveEndgamePolicy(studentMod, weightsFp);
    }

    // the gate: prod passes (px,py,boids,W,Hc,pred) at the endgame commit. Returns a
    // boid index (NO fallback — the NN's argmax IS the decision) or -1 (test escape).
    game.win.__egnnGate = function (px, py, boids, W, Hc, pred) {
        const n = boids.length;
        stats.commits++; if (n === 1) stats.soleN1++;
        const bs = new Array(n);
        for (let i = 0; i < n; i++) bs[i] = { x: boids[i].position.x, y: boids[i].position.y,
            vx: boids[i].velocity.x, vy: boids[i].velocity.y };
        const prodIdx = egScan.egPick(px, py, bs, W, Hc).egIdx;
        let idx;
        if (mode === 'nn') {
            const snap = { px, py, pvx: pred ? pred.velocity.x : 0, pvy: pred ? pred.velocity.y : 0,
                psize: pred ? pred.currentSize : 0,
                bx: bs.map(b => b.x), by: bs.map(b => b.y), bvx: bs.map(b => b.vx), bvy: bs.map(b => b.vy) };
            const r = endgamePolicy(snap, { W, Hc });
            idx = (r && typeof r.egIdx === 'number') ? r.egIdx : (typeof r === 'number' ? r : -1);
            // malformed pick PENALIZED as a disagreement (never silently → prod; no hidden fallback)
            if (!(idx >= 0 && idx < n)) { stats.malformed++; idx = n >= 2 ? (prodIdx + 1) % n : prodIdx; }
        } else if (mode === 'raw_geom') {
            idx = argminWrapAware(px, py, bs, W, Hc, egFeat);   // cheap-geom argmin, NO NN, NO exact scan-t
        } else { // oracle / perturb
            idx = prodIdx;
            if (mode === 'perturb' && n >= 2 && uhash(bs[prodIdx].x, bs[prodIdx].y) < perturbFrac) { idx = (prodIdx + 1) % n; stats.flips++; }
        }
        if (idx !== prodIdx) stats.nnVsProd++;
        return idx;
    };
    game.win.__egnnStats = stats;
    global.__egnnStatsLast = stats;

    // load transformed predator_cheap (endgame anchor only; planner verbatim)
    const prevCheap = game.win.__cheap, prevReady = game.win.__predatorReady,
          prevModel = game.win.__predatorModel, prevDbg = game.win.__cheapDebug;
    let code = fs.readFileSync(path.join(policyDir, 'predator_cheap.js'), 'utf8');
    if (code.indexOf(ENDGAME_COMMIT_ANCHOR) < 0) throw new Error('egnn: endgame anchor not found');
    if (code.indexOf(DEBUG_ANCHOR) < 0) throw new Error('egnn: debug anchor not found');
    code = endgameInject(debugInject(code));
    vm.runInThisContext(code, { filename: 'predator_cheap.js#egnn' });
    const egForce = game.win.__cheap.force, egDbg = game.win.__cheapDebug, egReady = game.win.__predatorReady;
    game.win.__cheap = prevCheap; game.win.__predatorReady = prevReady;
    game.win.__predatorModel = prevModel; game.win.__cheapDebug = prevDbg;
    if (egReady && typeof egReady.then === 'function') await egReady;

    return { name: 'egnn(' + mode + (mode === 'perturb' ? ',p=' + perturbFrac : '') + ')',
        configure() {}, force: egForce, reset() {}, _debug: egDbg,
        stats: () => Object.assign({}, stats) };
};

// wrap-aware analytic intercept-time argmin (eg_features[12]=wa0/100): cheap closed-
// form geometry from kinematics, NO NN, NO exact O(N·TMAX) scan. The genuine floor.
function argminWrapAware(px, py, bs, W, Hc, egFeat) {
    let idx = 0, best = Infinity;
    for (let i = 0; i < bs.length; i++) {
        const t = egFeat.egBoidFeatures(px, py, bs[i].x, bs[i].y, bs[i].vx, bs[i].vy, W, Hc)[12];
        if (t < best) { best = t; idx = i; }
    }
    return idx;
}

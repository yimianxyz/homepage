// candidates/split.js — the EXACT deployed policy (prod planner N>5 + prod endgame
// intercept) with a CONFIGURABLE planner/endgame SPLIT RULE + hysteresis, for the
// optimal-split (T*) search. The ONLY change vs prod: the `boids.length <= 5` gate in
// __cheap.force becomes a configurable predicate. Endgame = verbatim intercept(),
// planner = verbatim planCheap+steer; only WHEN we switch differs.
//
// Hysteresis (anti-flap): once in the endgame we stay until the EXIT condition (a gap
// above the ENTER condition), mirroring prod's commit-and-hold spirit.
//
// THREE RULES (env EXACTNN_SPLIT_RULE), authoritative JS defs (side-a's GPU surface
// should match these for the cross-check):
//   count    — endgame when N ≤ T            (enter); planner when N ≥ T+2 (exit).  T=5 is PROD.
//              env EXACTNN_SPLIT_T (default 5).
//   density  — count-T that SCALES with screen area: Td = round(T_ref · A/A_ref), where
//              A = torus area (W+20)(Hc+20), A_ref = (1024+20)(768+20). enter N≤Td / exit N≥Td+2.
//              env EXACTNN_SPLIT_TREF (default 5). Tests "T* rises with screen size".
//   horizon  — endgame when the planner CAN'T reach the soonest boid within its ~90-frame
//              rollout horizon: min_boid(wrap-aware analytic reach-time wa0) > H.
//              enter minWa0 > H / exit minWa0 < 0.9·H. Auto-adapts to screen size.
//              env EXACTNN_SPLIT_H (default 90 = the planner's Hs).
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const egFeat = require(path.join(__dirname, '..', 'endgame', 'eg_features.js'));

const DEBUG_ANCHOR = '    window.__cheap = {';
const GATE_LINE = '            if (boids.length <= 5) return intercept(pred, boids);   // ENDGAME: torus head-on intercept';
const AREF = (1024 + 20) * (768 + 20);

function debugInject(code) {
    return code.replace(DEBUG_ANCHOR,
        '    window.__cheapDebug = {\n'
        + '        get: function () { return { target: target, frame: frame, egBoid: egBoid, configured: configured, cfgW: cfg.W, cfgHc: cfg.Hc }; },\n'
        + '        set: function (s) { target = s.target; frame = s.frame; egBoid = s.egBoid; configured = s.configured; cfg.W = s.cfgW; cfg.Hc = s.cfgHc; }\n'
        + '    };\n' + DEBUG_ANCHOR);
}
function gateInject(code) {
    if (code.indexOf(GATE_LINE) < 0) throw new Error('split: gate line not found');
    return code.replace(GATE_LINE,
        '            if (window.__splitGate(pred, boids, cfg)) return intercept(pred, boids);   // CONFIGURABLE SPLIT');
}

module.exports.create = async function (game, helpers) {
    const policyDir = helpers.policyDir;
    const rule = process.env.EXACTNN_SPLIT_RULE || 'count';
    const T = process.env.EXACTNN_SPLIT_T != null ? +process.env.EXACTNN_SPLIT_T : 5;
    const Tref = process.env.EXACTNN_SPLIT_TREF != null ? +process.env.EXACTNN_SPLIT_TREF : 5;
    const P = process.env.EXACTNN_SPLIT_P != null ? +process.env.EXACTNN_SPLIT_P : 1.0;   // density exponent (T*~A^p, p~0.3 sub-linear)
    const H = process.env.EXACTNN_SPLIT_H != null ? +process.env.EXACTNN_SPLIT_H : 90;

    let inEndgame = false;   // hysteresis state (per game; reset() clears)
    const stats = { switches: 0, framesEndgame: 0, framesPlanner: 0 };

    function minWa0(pred, boids, cfg) {
        const px = pred.position.x, py = pred.position.y;
        let m = Infinity;
        for (let i = 0; i < boids.length; i++) {
            const b = boids[i];
            const t = egFeat.egBoidFeatures(px, py, b.position.x, b.position.y, b.velocity.x, b.velocity.y, cfg.W, cfg.Hc)[12] * 100;
            if (t < m) m = t;
        }
        return m;
    }

    game.win.__splitGate = function (pred, boids, cfg) {
        const N = boids.length;
        let enter, exit;
        if (rule === 'count') { enter = N <= T; exit = N >= T + 2; }
        else if (rule === 'density') {
            const A = (cfg.W + 20) * (cfg.Hc + 20);
            const Td = Math.max(1, Math.round(Tref * Math.pow(A / AREF, P)));   // sub-linear area scaling
            enter = N <= Td; exit = N >= Td + 2;
        } else if (rule === 'horizon') {
            const m = minWa0(pred, boids, cfg);
            enter = m > H; exit = m < 0.9 * H;
        } else throw new Error('split: unknown rule ' + rule);
        const was = inEndgame;
        if (!inEndgame && enter) inEndgame = true;
        else if (inEndgame && exit) inEndgame = false;
        if (inEndgame !== was) stats.switches++;
        if (inEndgame) stats.framesEndgame++; else stats.framesPlanner++;
        return inEndgame;
    };
    game.win.__splitStats = stats;
    global.__splitStatsLast = stats;

    const prevCheap = game.win.__cheap, prevReady = game.win.__predatorReady,
          prevModel = game.win.__predatorModel, prevDbg = game.win.__cheapDebug;
    let code = fs.readFileSync(path.join(policyDir, 'predator_cheap.js'), 'utf8');
    code = gateInject(debugInject(code));
    vm.runInThisContext(code, { filename: 'predator_cheap.js#split' });
    const force = game.win.__cheap.force, dbg = game.win.__cheapDebug, ready = game.win.__predatorReady;
    game.win.__cheap = prevCheap; game.win.__predatorReady = prevReady;
    game.win.__predatorModel = prevModel; game.win.__cheapDebug = prevDbg;
    if (ready && typeof ready.then === 'function') await ready;
    return { name: 'split(' + rule + ',T=' + T + ',Tref=' + Tref + ',H=' + H + ')',
        configure() {}, force, reset() { inEndgame = false; }, _debug: dbg, stats: () => Object.assign({}, stats) };
};

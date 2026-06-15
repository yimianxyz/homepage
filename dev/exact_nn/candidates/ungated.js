// candidates/ungated.js — the ELEGANT-UNIFICATION full policy: prod's rollout-planner
// (planCheap) UN-GATED to run for ALL N≥1 (the N≤5 intercept() endgame gate removed).
// Prod's own value-net + rollout decide every N; no separate endgame NN, no formula.
//
// The single change vs prod: the `if (boids.length <= 5) return intercept(...)` gate in
// __cheap.force is removed, so planCheap+steer runs for all N≥1. planCheap keeps its
// natural every-D-frame re-plan cadence (prod's intercept commits-and-holds an egBoid;
// the un-gated planner re-plans — a documented behavioral difference, NOT a per-decision
// equivalence). Used in diff_harness FORK mode to ask the load-bearing question: does the
// un-gated policy still CLEAR the board (catch the last ≤5 boids), or patrol forever?
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const DEBUG_ANCHOR = '    window.__cheap = {';
const GATE_LINE = '            if (boids.length <= 5) return intercept(pred, boids);   // ENDGAME: torus head-on intercept';

function debugInject(code) {
    return code.replace(DEBUG_ANCHOR,
        '    window.__cheapDebug = {\n'
        + '        get: function () { return { target: target, frame: frame, egBoid: egBoid, configured: configured, cfgW: cfg.W, cfgHc: cfg.Hc }; },\n'
        + '        set: function (s) { target = s.target; frame = s.frame; egBoid = s.egBoid; configured = s.configured; cfg.W = s.cfgW; cfg.Hc = s.cfgHc; }\n'
        + '    };\n' + DEBUG_ANCHOR);
}
function ungateInject(code) {
    if (code.indexOf(GATE_LINE) < 0) throw new Error('ungated: N≤5 gate line not found');
    return code.replace(GATE_LINE, '            // [UN-GATED] N≤5 endgame gate removed → planCheap runs for ALL N≥1');
}

module.exports.create = async function (game, helpers) {
    const policyDir = helpers.policyDir;
    const prevCheap = game.win.__cheap, prevReady = game.win.__predatorReady,
          prevModel = game.win.__predatorModel, prevDbg = game.win.__cheapDebug;
    let code = fs.readFileSync(path.join(policyDir, 'predator_cheap.js'), 'utf8');
    code = ungateInject(debugInject(code));
    vm.runInThisContext(code, { filename: 'predator_cheap.js#ungated' });
    const force = game.win.__cheap.force, dbg = game.win.__cheapDebug, ready = game.win.__predatorReady;
    game.win.__cheap = prevCheap; game.win.__predatorReady = prevReady;
    game.win.__predatorModel = prevModel; game.win.__cheapDebug = prevDbg;
    if (ready && typeof ready.then === 'function') await ready;
    return { name: 'ungated(planCheap-all-N)', configure() {}, force, reset() {}, _debug: dbg };
};

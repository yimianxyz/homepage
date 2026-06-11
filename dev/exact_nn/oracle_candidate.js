// oracle_candidate.js — diff_harness candidate that loads the INSTRUMENTED
// FORK (oracle_policy.js) with its logging hooks ACTIVE (counting sink).
//
// This is the farm-gate certification subject: diff_harness lockstep-compares
// the fork's force against pristine prod bitwise every frame, with the hooks
// firing exactly as they will during dataset farming. Zero mismatches across
// the device matrix (incl. gate-crossing and spawn games) == certificate.
//
//   node dev/exact_nn/diff_harness.js --candidate dev/exact_nn/oracle_candidate.js ...
//
// Hook-coverage counters land on global.__oracleCertCounts after each game so
// certify_oracle.js can assert the hooks actually ran (a cert that never
// exercised the logging paths would be vacuous).
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const FORK = path.join(__dirname, 'oracle_policy.js');

module.exports.create = async function (game) {
    const win = game.win;
    // Re-eval the fork in the live context (same pattern as stepper's
    // loadPolicyAgain, different file): an independent policy closure with its
    // own target/frame/egBoid/NET state; the reference closure and the running
    // sim stay untouched.
    const prevCheap = win.__cheap, prevReady = win.__predatorReady,
          prevModel = win.__predatorModel;
    vm.runInThisContext(fs.readFileSync(FORK, 'utf8'), { filename: 'oracle_policy.js#cert' });
    const forkCheap = win.__cheap, forkReady = win.__predatorReady;
    win.__cheap = prevCheap; win.__predatorReady = prevReady;
    win.__predatorModel = prevModel;
    if (forkReady && typeof forkReady.then === 'function') await forkReady;

    // Counting sink: hooks active (farm configuration), output discarded.
    const counts = { planStart: 0, roll: 0, planEnd: 0, frameEnd: 0 };
    global.__oracleCertCounts = counts;
    const sink = {
        planStart() { counts.planStart++; },
        roll() { counts.roll++; },
        planEnd() { counts.planEnd++; },
        frameEnd() { counts.frameEnd++; },
    };

    return {
        name: 'oracle-fork(hooks-on)',
        configure() {},
        reset() { counts.planStart = counts.roll = counts.planEnd = counts.frameEnd = 0; },
        force(pred, boids) {
            // window.__oracle is process-global: scope it to exactly this call
            // so the pristine reference (and any other closure) never sees it.
            win.__oracle = sink;
            try { return forkCheap.force(pred, boids); }
            finally { win.__oracle = null; }
        },
    };
};

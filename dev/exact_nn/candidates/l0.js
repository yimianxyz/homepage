// candidates/l0.js — diff_harness adapter for the L0 unified policy.
//
// Loads dev/exact_nn/policy_unified.js (the MECHANICAL BUILD — see
// build_l0.js) into the harness context the way a browser <script> would
// (vm.runInThisContext, window path). The artifact's embedded fetch chain
// resolves value_net.json through the harness stub — the identical bytes and
// JSON.parse prod uses — and `ready` gates on it.
//
// configure()/reset() are deliberate noops: L0's force() IS prod's closure
// function, which configures lazily on first call (byte-identical timing),
// and per-game fresh-load (the harness contract) replaces reset().
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

module.exports.create = async function (game, helpers) {
    const src = fs.readFileSync(path.join(__dirname, '..', 'policy_unified.js'), 'utf8');
    const prev = game.win.__exactnn;                  // (defensive; normally undefined)
    vm.runInThisContext(src, { filename: 'policy_unified.js' });
    const h = game.win.__exactnn;
    game.win.__exactnn = prev;
    if (h.ready && typeof h.ready.then === 'function') await h.ready;
    return { name: 'L0-unified@' + 'mechanical-build', configure() {}, force: h.force, reset() {}, _inner: h._inner };
};

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
// When the harness requests decision-level metrics (helpers.transform), the
// artifact is REBUILT IN-MEMORY from the same pristine sources with the same
// anchored __cheapDebug insert the reference gets (build_l0.js buildSource) —
// the committed file stays pristine; the selftest separately proves the hook
// is digest-inert. Without transform, the committed artifact itself is loaded.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { buildSource } = require('../build_l0.js');

module.exports.create = async function (game, helpers) {
    let src, label;
    if (helpers && helpers.transform) {
        src = buildSource(helpers.policyDir, helpers.transform).source;
        label = 'L0-unified@inmem-debug-build';
    } else {
        src = fs.readFileSync(path.join(__dirname, '..', 'policy_unified.js'), 'utf8');
        label = 'L0-unified@committed-artifact';
    }
    const prev = game.win.__exactnn;                  // (defensive; normally undefined)
    vm.runInThisContext(src, { filename: 'policy_unified.js' });
    const h = game.win.__exactnn;
    game.win.__exactnn = prev;
    if (h.ready && typeof h.ready.then === 'function') await h.ready;
    return { name: label, configure() {}, force: h.force, reset() {},
             _inner: h._inner, _debug: h._inner.__cheapDebug || null };
};

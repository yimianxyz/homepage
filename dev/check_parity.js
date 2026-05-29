// JS<->sim_torch parity check for the evolved patrol target.
// Loads dev/parity_state.json (state + sim_torch's target per patrol-mode env),
// recomputes the target with predator.js's computeEvolvedTarget, and asserts a
// tight match. Run: node dev/check_parity.js
'use strict';
const fs = require('fs');
const path = require('path');

// minimal Vector shim: computeEvolvedTarget only reads .x/.y and uses
// predPos.x/.y (no Vector methods), so plain objects suffice.
global.window = undefined;
const { computeEvolvedTarget } = require(path.resolve(__dirname, '../js/predator.js'));

const data = JSON.parse(fs.readFileSync(path.resolve(__dirname, 'parity_state.json'), 'utf8'));
const opt = data.opts;
let maxErr = 0, worst = null;
for (const c of data.cases) {
    const t = computeEvolvedTarget(c.pred, c.boids, opt, c.pred);
    const ex = Math.abs(t.x - c.target.x);
    const ey = Math.abs(t.y - c.target.y);
    const err = Math.max(ex, ey);
    if (err > maxErr) { maxErr = err; worst = { seed: c.seed, js: t, torch: c.target, n: c.n_alive }; }
}
console.log(JSON.stringify({ cases: data.cases.length, max_abs_err: maxErr, worst }, null, 2));
const TOL = 1e-6;
if (maxErr > TOL) { console.error(`FAIL: max abs err ${maxErr} > ${TOL}`); process.exit(1); }
console.log(`OK: JS patrol target matches sim_torch within ${TOL} over ${data.cases.length} cases`);

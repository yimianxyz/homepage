// moe_parity.js — verify moeForward.js (JS deploy) reproduces the trained torch
// MoE on dumped samples (moe_parity.py). Reports max |logit_js - logit_torch| and
// argmax/argmin agreement (the only thing that affects S_dec).
//   node moe_parity.js moe_weights.json parity.json
'use strict';
const fs = require('fs');
const { loadMoE } = require('./moeForward.js');

const moe = loadMoE(process.argv[2] || 'moe_weights.json');
const samples = JSON.parse(fs.readFileSync(process.argv[3] || 'parity.json', 'utf8'));

let maxAbs = 0, maxG = 0, argOk = 0, n = 0;
const worst = [];
for (const s of samples) {
    const { logit, g } = moe.forward(s.planner_block, s.endgame_block, s.gate_feat,
        s.slot_valid.map(Boolean), s.p_valid.map(Boolean), s.e_valid.map(Boolean));
    let mx = 0;
    for (let k = 0; k < 16; k++) {
        if (!s.slot_valid[k]) continue;
        const d = Math.abs(logit[k] - s.torch_logit[k]);
        if (d > mx) mx = d;
    }
    maxAbs = Math.max(maxAbs, mx);
    maxG = Math.max(maxG, Math.abs(g - s.torch_g));
    // argmax over valid slots, both sides
    const am = (arr) => { let bi = -1, bs = -Infinity; for (let k = 0; k < 16; k++) if (s.slot_valid[k] && arr[k] > bs) { bs = arr[k]; bi = k; } return bi; };
    if (am(logit) === am(s.torch_logit)) argOk++;
    else worst.push({ regime: s.regime, js: am(logit), torch: am(s.torch_logit), maxAbs: mx });
    n++;
}
console.log(`parity: n=${n} maxAbs(logit)=${maxAbs.toExponential(3)} maxAbs(gate)=${maxG.toExponential(3)} argmaxAgree=${(argOk / n).toFixed(4)}`);
if (worst.length) console.log('argmax disagreements:', JSON.stringify(worst.slice(0, 10)));

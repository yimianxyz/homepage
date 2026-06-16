// moe_ablate_weights.js — produce ABLATED MoE weight files for the honesty
// decomposition (4-angle audit: "is the NN doing real work or argmax-of-scores?").
// The logit is `head(e) + w_skip·dec` where dec = the gated decisive RAW score
// (planner cheapScore=prod's exact score; endgame −scan_t). Two ablations isolate
// the contributions, both run through side-a's UNCHANGED moeForward/moePolicy:
//   noskip: w_skip=0          -> logit = head(e) only (does the expert+head ALONE decide?)
//   nohead: H.out.weight/bias=0 -> logit = w_skip·dec only (does the raw-score skip alone decide?)
// S_dec(nn) vs S_dec(noskip) vs S_dec(nohead) decomposes the decision.
//   node moe_ablate_weights.js <in_weights.json> <outdir>
'use strict';
const fs = require('fs');
const path = require('path');
const inFp = process.argv[2], outDir = process.argv[3] || path.dirname(inFp);
const J = JSON.parse(fs.readFileSync(inFp, 'utf8'));
function clone(o) { return JSON.parse(JSON.stringify(o)); }

const noskip = clone(J); noskip.weights['w_skip'] = [0.0];
fs.writeFileSync(path.join(outDir, 'moe_weights_noskip.json'), JSON.stringify(noskip));

const nohead = clone(J);
const ho = nohead.weights['H.out.weight'];
for (let r = 0; r < ho.length; r++) for (let c = 0; c < ho[r].length; c++) ho[r][c] = 0.0;
nohead.weights['H.out.bias'] = nohead.weights['H.out.bias'].map(() => 0.0);
fs.writeFileSync(path.join(outDir, 'moe_weights_nohead.json'), JSON.stringify(nohead));

console.log('wrote moe_weights_noskip.json (w_skip=0, head-only) + moe_weights_nohead.json (H.out=0, skip-only) to ' + outDir);
console.log('orig w_skip =', J.weights['w_skip'][0]);

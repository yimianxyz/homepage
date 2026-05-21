#!/bin/bash
# Verify an RL candidate against the baseline with a larger seed pool to
# rule out false-positive accepts from the hill climb's 1·SE threshold.
#
#   ./dev/rl_verify.sh <candidate_weights.json>
#
# Reports per-seed catches for baseline and candidate side-by-side, plus
# the paired delta mean ± SE.

set -e
CAND="${1:-dev/weights/rl_sigma005.json}"
[ ! -f "$CAND" ] && { echo "no such file: $CAND"; exit 1; }

BASE_REPORT="dev/reports/verify_baseline.json"
CAND_REPORT="dev/reports/verify_$(basename "$CAND" .json).json"
SEEDS=16
SEED_START=100
MAX_FRAMES=5000

echo "=== Baseline (js/predator_weights.json) ==="
node dev/eval_tte.js js/predator_weights.json \
    --seeds $SEEDS --seedStart $SEED_START \
    --maxFrames $MAX_FRAMES --workers 4 \
    --report "$BASE_REPORT" 2>&1 | tail -15

echo
echo "=== Candidate ($CAND) ==="
node dev/eval_tte.js "$CAND" \
    --seeds $SEEDS --seedStart $SEED_START \
    --maxFrames $MAX_FRAMES --workers 4 \
    --report "$CAND_REPORT" 2>&1 | tail -15

echo
echo "=== Paired comparison ==="
node -e "
const b = require('./$BASE_REPORT');
const c = require('./$CAND_REPORT');
const bMap = new Map(b.perSeed.map(p => [p.seed, p.catches]));
const cMap = new Map(c.perSeed.map(p => [p.seed, p.catches]));
const seeds = Array.from(bMap.keys()).sort((a,b)=>a-b);
const deltas = [];
console.log('seed   baseline   candidate   delta');
for (const s of seeds) {
    const bc = bMap.get(s), cc = cMap.get(s);
    const d = cc - bc;
    deltas.push(d);
    console.log(s + '  ' + String(bc).padStart(8) + '  ' + String(cc).padStart(10) + '  ' + (d > 0 ? '+' : '') + d);
}
const n = deltas.length;
const mean = deltas.reduce((a,b)=>a+b,0)/n;
const variance = deltas.reduce((a,b)=>a+(b-mean)**2,0)/n;
const se = Math.sqrt(variance / n);
const z = mean / se;
console.log();
console.log('paired delta:  mean=' + mean.toFixed(2) + '  SE=' + se.toFixed(2) + '  z=' + z.toFixed(2));
console.log('baseline mean catches:  ' + b.meanCatches.toFixed(2));
console.log('candidate mean catches: ' + c.meanCatches.toFixed(2));
console.log('improvement: ' + ((c.meanCatches - b.meanCatches) / b.meanCatches * 100).toFixed(1) + '%');
if (z > 2) console.log('VERDICT: candidate statistically beats baseline (z > 2 ≈ 97.5% one-sided)');
else if (z > 1) console.log('VERDICT: weakly better (z > 1 ≈ 84% one-sided), inconclusive');
else if (z > 0) console.log('VERDICT: marginal — noise');
else console.log('VERDICT: candidate WORSE than baseline');
"

#!/bin/bash
# Promote a trained weights file to js/predator_weights.json after verifying
# all gates pass.
#
# Usage: ./dev/ship.sh dev/weights/c3v3_H4.json
set -e
W="$1"
[ -z "$W" ] && { echo "usage: $0 <weights.json>"; exit 1; }
[ ! -f "$W" ] && { echo "no such file: $W"; exit 1; }

ID=$(basename "$W" .json)
REPORT="dev/reports/${ID}.final_eval.json"

echo "[1/3] Full eval ($ID) ..."
node dev/eval.js \
  --weights "$W" \
  --report "$REPORT" \
  --frames 5000 --testStates 50000 --divSeeds 8 --behaviorSeeds 4

echo "[2/3] Gate summary:"
node -e "
const r = require('./${REPORT}');
console.log('  gates.allPassed:', r.gates.allPassed);
for (const [k,v] of Object.entries(r.gates)) {
    if (k === 'allPassed') continue;
    console.log('   ', k.padEnd(22), v ? 'PASS' : 'FAIL');
}
console.log('  params:', r.model.totalParams, 'featureDim:', r.model.featureDim);
console.log('  mse:', r.metrics.regression.mse, 'maxAbs:', r.metrics.regression.maxAbs);
console.log('  decisionAgr:', r.metrics.decision.agreement, 'rEdgeAgr:', r.metrics.decision.rEdgeAgreement);
console.log('  catchLCPs:', r.metrics.divergence.map(d => d.seed+':'+d.catchLCP+'/'+d.ruleCatchCount).join(' '));
"

ALL_PASS=$(node -e "console.log(JSON.parse(require('fs').readFileSync('$REPORT')).gates.allPassed)")
if [ "$ALL_PASS" != "true" ]; then
    echo
    echo "Gates did NOT all pass. Not shipping."
    exit 2
fi

echo "[3/3] All gates passed. Copying $W -> js/predator_weights.json"
cp "$W" js/predator_weights.json
echo "Done."

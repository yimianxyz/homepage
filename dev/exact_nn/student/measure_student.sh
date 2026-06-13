#!/usr/bin/env bash
# measure_student.sh — side-a's local eps→NN-share preview for an L1h student.
# Runs the SAME instruments side-b uses: validate_student.js (S_dec + reuse_exact
# + calib_margins on the calibration range [270000,280000)) then verifier/
# tau_calibrate.js (chosenTau, L1h NN-share, monotonicity, reliability curve).
#
#   ./measure_student.sh <weights.json> <label> [max_plans]
# writes calib_margins to /tmp/calib_<label>.json and frozen_tau to /tmp/frozen_<label>.json
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
W="$1"; LABEL="${2:-student}"; MAX="${3:-40000}"
CALIB="/tmp/calib_${LABEL}.json"; FROZEN="/tmp/frozen_${LABEL}.json"
TAU="/workspace/.team/wt-exact-nn/dev/exact_nn/verifier/tau_calibrate.js"

echo "=== validate_student ($LABEL, $MAX plans, calib_data [270000,280000)) ==="
WRITE_CALIB="$CALIB" node "$HERE/validate_student.js" --weights "$W" --data "$HERE/../calib_data" --max "$MAX" | tail -8

echo "=== tau_calibrate ($LABEL) ==="
node "$TAU" --in "$CALIB" --out "$FROZEN" 2>/dev/null >/dev/null
python3 - "$FROZEN" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print('  calibPlans                :', d['calibPlans'])
print('  overallStudentAgree (S_dec):', d['overallStudentAgree'])
print('  chosenTau (0-disagree)    :', d['chosenTau'])
print('  L1h NN-share estimate     :', d['L1h_NNshare_estimate'])
print('  marginIsUsableConfidence  :', d['marginIsUsableConfidenceSignal'])
print('  reliability (disagree-rate should FALL with margin → monotone):')
for b in d['reliability']:
    print('    margin %-16s n=%-6d disagree=%.3f' % (str(b['marginRange']), b['n'], b['disagreeRate']))
print('  riskVsTau (trustedFrac @ 0 trusted-disagreements is the NN-share):')
for c in d['riskVsTau']:
    print('    tau=%-8s trustedFrac=%.4f trustedDisagreements=%d' % (c['tau'], c['trustedFrac'], c['trustedDisagreements']))
PY

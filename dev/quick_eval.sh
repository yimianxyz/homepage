#!/bin/bash
# Quick triage eval. Usage: ./dev/quick_eval.sh dev/weights/c3v3_H8.json
# Writes report to dev/reports/$(basename weights .json).eval.json
set -e
W="$1"
[ -z "$W" ] && { echo "usage: $0 <weights.json>"; exit 1; }
ID=$(basename "$W" .json)
REPORT="dev/reports/${ID}.eval.json"
node dev/eval.js \
  --weights "$W" \
  --report "$REPORT" \
  --frames 1500 --testStates 8000 --divSeeds 3 --behaviorSeeds 2
echo "Report: $REPORT"
node dev/compare.js "$REPORT"

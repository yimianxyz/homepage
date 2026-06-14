#!/bin/bash
# handoff_watch_moe.sh — Phase-2 standby. Exits (→ side-b is notified) the moment
# side-a drops the unified MoE model, so the one-shot sealed verdict can run.
# Run via Bash run_in_background; the completion notification re-engages side-b.
#
#   bash verifier/handoff_watch_moe.sh
DROP="${1:-/workspace/.team/exact-nn-moe-student}"
echo "[handoff_watch_moe] watching $DROP for moePolicy.js + moe_weights.json ($(date -u +%H:%M:%S))"
while true; do
  # accept any of the likely export names side-a may use
  POL=""; WTS=""
  for p in moePolicy.js moe_policy.js moePolicy.mjs; do [ -f "$DROP/$p" ] && POL="$DROP/$p"; done
  for w in moe_weights.json moe_weights.bin weights.json; do [ -f "$DROP/$w" ] && WTS="$DROP/$w"; done
  if [ -n "$POL" ] && [ -n "$WTS" ]; then
    echo "[handoff_watch_moe] HANDOFF DETECTED ($(date -u +%H:%M:%S)): $POL + $WTS"
    ls -la "$DROP"
    exit 0
  fi
  sleep 60
done

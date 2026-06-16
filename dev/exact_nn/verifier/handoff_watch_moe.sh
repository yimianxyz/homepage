#!/bin/bash
# handoff_watch_moe.sh — Phase-2 standby. Exits (→ side-b is notified) the moment
# side-a drops the unified MoE model anywhere under /workspace/.team, so the
# one-shot sealed verdict can run. Launch via Bash run_in_background; the
# completion notification re-engages side-b.
#
#   bash verifier/handoff_watch_moe.sh [SEARCH_ROOT]
ROOT="${1:-/workspace/.team}"
echo "[handoff_watch_moe] watching $ROOT for a unified MoE (moePolicy + weights) ($(date -u +%H:%M:%S))"
while true; do
  # any *moe*Policy / moe_policy under .team, plus a sibling moe*weights json
  POL=$(find "$ROOT" -maxdepth 3 -type f \( -iname 'moepolicy.js' -o -iname 'moe_policy.js' -o -iname 'moepolicy.mjs' \) 2>/dev/null | head -1)
  WTS=$(find "$ROOT" -maxdepth 3 -type f \( -iname 'moe_weights.json' -o -iname 'moe_weights.bin' -o -iname 'weights.json' \) 2>/dev/null | head -1)
  if [ -n "$POL" ] && [ -n "$WTS" ]; then
    echo "[handoff_watch_moe] HANDOFF DETECTED ($(date -u +%H:%M:%S)):"
    echo "  policy : $POL"
    echo "  weights: $WTS"
    ls -la "$(dirname "$POL")"
    exit 0
  fi
  sleep 60
done

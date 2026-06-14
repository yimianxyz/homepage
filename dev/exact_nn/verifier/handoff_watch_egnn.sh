#!/bin/bash
# handoff_watch_egnn.sh — standby for side-a's PURE ENDGAME NN drop (simplified
# direction). Exits (→ side-b notified) when an endgame-policy module + weights land
# under /workspace/.team. Distinct from the unified-MoE moePolicy.js (now dropped).
ROOT="${1:-/workspace/.team}"
echo "[handoff_watch_egnn] watching $ROOT for a pure endgame NN (endgamePolicy + weights) ($(date -u +%H:%M:%S))"
while true; do
  # an endgame-policy module side-a is asked to deliver (endgamePolicy.js / endgame_policy.js / eg_nn*.js)
  POL=$(find "$ROOT" -maxdepth 3 -type f \( -iname 'endgamepolicy.js' -o -iname 'endgame_policy.js' -o -iname 'eg_nn*.js' -o -iname 'egpolicy.js' \) 2>/dev/null | head -1)
  if [ -n "$POL" ]; then
    DIR=$(dirname "$POL")
    WTS=$(find "$DIR" -maxdepth 1 -type f -iname '*weights*.json' 2>/dev/null | head -1)
    echo "[handoff_watch_egnn] ENDGAME NN DETECTED ($(date -u +%H:%M:%S)):"
    echo "  policy : $POL"
    echo "  weights: ${WTS:-<none found in dir>}"
    ls -la "$DIR"
    exit 0
  fi
  sleep 60
done

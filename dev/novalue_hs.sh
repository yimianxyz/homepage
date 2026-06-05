#!/bin/bash
# Can a LONGER / WIDER single-rollout remove the value net? no_value means no
# bootstrap and E3D fallback. Since we only roll a few candidates, roll them the
# full Hs=120 so post-window catches are seen directly. If kr<=3 @ Hs120 ~= the
# with-net 14.97, the JS port needs NO net (rolloutFlat already does any horizon).
# n=128 for speed (exploratory). seed300000 to compare with the with-net curve.
cd ~/js_eval/dev || exit 1
NET=/home/sa_105440658512316303279/net_v2_absval.pt
OUT=~/novalue_hs.jsonl
: > "$OUT"
echo "=== novalue_hs n128 seed300000 $(date -u +%Y%m%dT%H%M%SZ) ===" >> "$OUT"
run() {  # kr Hs
  sm=$(python3 eval_value.py --net "$NET" --n 128 --seedStart 300000 --frames 1500 \
       --Hs "$2" --lookahead --roll_M 120 --K_roll "$1" --prune_by ball --no_value \
       --device cuda --weights ../js/predator_weights.json --skip-planner 2>/dev/null \
       | grep -oE '"student_mean": [0-9.]+' | grep -oE '[0-9.]+$')
  echo "{\"no_value\": true, \"K_roll\": $1, \"roll_M\": 120, \"Hs\": $2, \"n\": 128, \"student_mean\": $sm}" >> "$OUT"
  echo "novalue kr=$1 Hs=$2 -> $sm"
}
run 1 120
run 2 120
run 3 120
run 3 90
run 3 60
run 1 90
echo "NOVALUE_HS_DONE" >> "$OUT"
setsid bash ~/watchdog.sh > ~/watchdog.log 2>&1 < /dev/null &
echo "watchdog resumed pid $!" >> "$OUT"

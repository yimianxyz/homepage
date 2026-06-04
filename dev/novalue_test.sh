#!/bin/bash
# Does kr=1 need the value net? no_value = roll top-1 ballistic candidate, score 0
# for the rest (E3D fallback when the single rollout yields no catch). If this ~=
# the with-net 14.97, the JS port needs NO net (huge simplification: rolloutFlat +
# candidates + a ballistic scorer already exist in predator_planner_worker.js).
cd ~/js_eval/dev || exit 1
NET=/home/sa_105440658512316303279/net_v2_absval.pt
OUT=~/novalue_test.jsonl
: > "$OUT"
echo "=== novalue test n256 seed300000 $(date -u +%Y%m%dT%H%M%SZ) ===" >> "$OUT"
run() {  # kr M extraflag tag
  sm=$(python3 eval_value.py --net "$NET" --n 256 --seedStart 300000 --frames 1500 \
       --Hs 60 --lookahead --roll_M "$2" --K_roll "$1" --prune_by ball $3 \
       --device cuda --weights ../js/predator_weights.json --skip-planner 2>/dev/null \
       | grep -oE '"student_mean": [0-9.]+' | grep -oE '[0-9.]+$')
  echo "{\"tag\": \"$4\", \"K_roll\": $1, \"roll_M\": $2, \"Hs\": 60, \"n\": 256, \"student_mean\": $sm}" >> "$OUT"
  echo "$4 kr=$1 M=$2 -> $sm"
}
run 1 120 "--no_value" novalue_kr1_M120
run 1 120 ""           withval_kr1_M120_ref
run 2 120 "--no_value" novalue_kr2_M120
run 3 120 "--no_value" novalue_kr3_M120
run 1 64  "--no_value" novalue_kr1_M64
echo "NOVALUE_DONE" >> "$OUT"
setsid bash ~/watchdog.sh > ~/watchdog.log 2>&1 < /dev/null &
echo "watchdog resumed pid $!" >> "$OUT"

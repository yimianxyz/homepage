#!/bin/bash
# Prune-by-ballistic vs prune-by-value frontier on net_v2_absval.
# Apples-to-apples with the V-prune sweep: n=96, frames=1500, M=120, Hs=60.
cd ~/js_eval/dev || exit 1
NET=/home/sa_105440658512316303279/net_v2_absval.pt
OUT=~/ballistic_prune.jsonl
: > "$OUT"
echo "=== ballistic-prune experiment $(date -u +%Y%m%dT%H%M%SZ) ===" >> "$OUT"
for pb in v ball; do
  for kr in 2 3 4 5; do
    sm=$(python3 eval_value.py --net "$NET" --n 96 --seedStart 200000 --frames 1500 \
         --Hs 60 --lookahead --roll_M 120 --K_roll "$kr" --prune_by "$pb" \
         --bias0 0 --device cuda --weights ../js/predator_weights.json --skip-planner 2>/dev/null \
         | grep -oE '"student_mean": [0-9.]+' | grep -oE '[0-9.]+$')
    echo "{\"prune_by\": \"$pb\", \"K_roll\": $kr, \"roll_M\": 120, \"Hs\": 60, \"student_mean\": $sm}" >> "$OUT"
    echo "pb=$pb kr=$kr -> $sm"
  done
done
echo "BALLISTIC_PRUNE_DONE" >> "$OUT"
# resume the never-idle watchdog
setsid bash ~/watchdog.sh > ~/watchdog.log 2>&1 < /dev/null &
echo "watchdog resumed pid $!" >> "$OUT"

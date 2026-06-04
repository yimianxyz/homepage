#!/bin/bash
# Second independent seed set (400000) for the recovery curve's key points, so
# the v-vs-ballistic shape is confirmed across two fresh seed sets at n=256.
cd ~/js_eval/dev || exit 1
NET=/home/sa_105440658512316303279/net_v2_absval.pt
OUT=~/recov2_seed400.jsonl
: > "$OUT"
echo "=== recov2 n256 seed400000 $(date -u +%Y%m%dT%H%M%SZ) ===" >> "$OUT"
run() {  # pb kr
  sm=$(python3 eval_value.py --net "$NET" --n 256 --seedStart 400000 --frames 1500 \
       --Hs 60 --lookahead --roll_M 120 --K_roll "$2" --prune_by "$1" \
       --bias0 0 --device cuda --weights ../js/predator_weights.json --skip-planner 2>/dev/null \
       | grep -oE '"student_mean": [0-9.]+' | grep -oE '[0-9.]+$')
  echo "{\"prune_by\": \"$1\", \"K_roll\": $2, \"roll_M\": 120, \"Hs\": 60, \"n\": 256, \"seedStart\": 400000, \"student_mean\": $sm}" >> "$OUT"
  echo "pb=$1 kr=$2 -> $sm"
}
run ball 2
run ball 3
run ball 6
run v 2
run v 3
run v 6
run v 16
echo "RECOV2_DONE" >> "$OUT"
setsid bash ~/watchdog.sh > ~/watchdog.log 2>&1 < /dev/null &
echo "watchdog resumed pid $!" >> "$OUT"

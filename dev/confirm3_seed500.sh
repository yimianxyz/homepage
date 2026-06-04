#!/bin/bash
# Third independent seed set (500000) to lock the headline: ball/kr1 ~= ceiling,
# and the minimal config (M64). n=256, Hs=60.
cd ~/js_eval/dev || exit 1
NET=/home/sa_105440658512316303279/net_v2_absval.pt
OUT=~/confirm3_seed500.jsonl
: > "$OUT"
echo "=== confirm3 n256 seed500000 $(date -u +%Y%m%dT%H%M%SZ) ===" >> "$OUT"
runM() {  # pb kr M
  sm=$(python3 eval_value.py --net "$NET" --n 256 --seedStart 500000 --frames 1500 \
       --Hs 60 --lookahead --roll_M "$3" --K_roll "$2" --prune_by "$1" \
       --device cuda --weights ../js/predator_weights.json --skip-planner 2>/dev/null \
       | grep -oE '"student_mean": [0-9.]+' | grep -oE '[0-9.]+$')
  echo "{\"prune_by\": \"$1\", \"K_roll\": $2, \"roll_M\": $3, \"Hs\": 60, \"n\": 256, \"seedStart\": 500000, \"student_mean\": $sm}" >> "$OUT"
  echo "pb=$1 kr=$2 M=$3 -> $sm"
}
runM ball 1 120    # the winner
runM ball 1 64     # the cheap minimal config
runM v 1 120       # contrast (value picks wrong candidate)
runM v 16 120      # ceiling
echo "CONFIRM3_DONE" >> "$OUT"
setsid bash ~/watchdog.sh > ~/watchdog.log 2>&1 < /dev/null &
echo "watchdog resumed pid $!" >> "$OUT"

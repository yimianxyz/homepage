#!/bin/bash
# Real deployment-scale (5000f) numbers for the winning config. All prior sweeps
# were @1500f (calibration regime, ~/4 of 5000f). Confirms the headline at the
# scale the browser actually runs. n=128 seed300000.
cd ~/js_eval/dev || exit 1
NET=/home/sa_105440658512316303279/net_v2_absval.pt
OUT=~/confirm_5kf.jsonl
: > "$OUT"
echo "=== confirm 5000f n128 seed300000 $(date -u +%Y%m%dT%H%M%SZ) ===" >> "$OUT"
run() {  # kr M tag
  sm=$(python3 eval_value.py --net "$NET" --n 128 --seedStart 300000 --frames 5000 \
       --Hs 60 --lookahead --roll_M "$2" --K_roll "$1" --prune_by ball \
       --device cuda --weights ../js/predator_weights.json --skip-planner 2>/dev/null \
       | grep -oE '"student_mean": [0-9.]+' | grep -oE '[0-9.]+$')
  echo "{\"tag\": \"$3\", \"K_roll\": $1, \"roll_M\": $2, \"Hs\": 60, \"frames\": 5000, \"n\": 128, \"student_mean\": $sm}" >> "$OUT"
  echo "$3 kr=$1 M=$2 -> $sm"
}
run 1 120 ball_kr1_M120_5kf
run 1 64  ball_kr1_M64_5kf
run 16 120 full_ceiling_5kf
echo "CONFIRM_5KF_DONE" >> "$OUT"
setsid bash ~/watchdog.sh > ~/watchdog.log 2>&1 < /dev/null &
echo "watchdog resumed pid $!" >> "$OUT"

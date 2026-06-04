#!/bin/bash
# How much of the full-rollout ceiling (15.1 @M120/Hs60) can we recover by
# rolling K_roll candidates, and does ballistic ranking recover it faster than V?
# Independent seed set (300000) + n=256 to test whether the K_roll=4 V-dip
# generalizes beyond seeds 200000-95. Front-load top6/top5/top4 (the user's ask).
cd ~/js_eval/dev || exit 1
NET=/home/sa_105440658512316303279/net_v2_absval.pt
OUT=~/recovery_sweep.jsonl
: > "$OUT"
echo "=== recovery sweep n256 seed300000 $(date -u +%Y%m%dT%H%M%SZ) ===" >> "$OUT"
for kr in 6 5 4 7 8 3; do
  for pb in v ball; do
    sm=$(python3 eval_value.py --net "$NET" --n 256 --seedStart 300000 --frames 1500 \
         --Hs 60 --lookahead --roll_M 120 --K_roll "$kr" --prune_by "$pb" \
         --bias0 0 --device cuda --weights ../js/predator_weights.json --skip-planner 2>/dev/null \
         | grep -oE '"student_mean": [0-9.]+' | grep -oE '[0-9.]+$')
    echo "{\"prune_by\": \"$pb\", \"K_roll\": $kr, \"roll_M\": 120, \"Hs\": 60, \"n\": 256, \"seedStart\": 300000, \"student_mean\": $sm}" >> "$OUT"
    echo "pb=$pb kr=$kr -> $sm"
  done
done
# full-rollout ceiling reference on the same fresh seeds
smf=$(python3 eval_value.py --net "$NET" --n 256 --seedStart 300000 --frames 1500 \
      --Hs 60 --lookahead --roll_M 120 --K_roll 16 --prune_by v \
      --bias0 0 --device cuda --weights ../js/predator_weights.json --skip-planner 2>/dev/null \
      | grep -oE '"student_mean": [0-9.]+' | grep -oE '[0-9.]+$')
echo "{\"prune_by\": \"full\", \"K_roll\": 16, \"roll_M\": 120, \"Hs\": 60, \"n\": 256, \"seedStart\": 300000, \"student_mean\": $smf}" >> "$OUT"
echo "RECOVERY_SWEEP_DONE" >> "$OUT"
setsid bash ~/watchdog.sh > ~/watchdog.log 2>&1 < /dev/null &
echo "watchdog resumed pid $!" >> "$OUT"

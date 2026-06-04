#!/bin/bash
# Find the minimal browser config. A: does NO rollout (pure value-net argmax over
# candidates, every D=8) hold up, given the net already ingests ballistic features?
# B: with ballistic K_roll=1, how cheap can the OTHER knobs go (Hs depth, roll_M boids)?
# n=256, seeds 300000 to match the main fresh-seed curve. Reference: full=15.1, ball/kr1=14.97.
cd ~/js_eval/dev || exit 1
NET=/home/sa_105440658512316303279/net_v2_absval.pt
OUT=~/minimal_config.jsonl
: > "$OUT"
echo "=== minimal-config n256 seed300000 $(date -u +%Y%m%dT%H%M%SZ) ===" >> "$OUT"

# A: pure value-net argmax, NO rollout (every D=8 frames)
sm=$(python3 eval_value.py --net "$NET" --n 256 --seedStart 300000 --frames 1500 \
     --Hs 0 --device cuda --weights ../js/predator_weights.json --skip-planner 2>/dev/null \
     | grep -oE '"student_mean": [0-9.]+' | grep -oE '[0-9.]+$')
echo "{\"mode\": \"no_rollout_pureV\", \"K_roll\": 0, \"roll_M\": 0, \"Hs\": 0, \"n\": 256, \"student_mean\": $sm}" >> "$OUT"
echo "A no_rollout_pureV -> $sm"

# B1: ballistic kr=1, vary rollout depth Hs at full M=120
for hs in 60 40 30 20 10; do
  sm=$(python3 eval_value.py --net "$NET" --n 256 --seedStart 300000 --frames 1500 \
       --Hs "$hs" --lookahead --roll_M 120 --K_roll 1 --prune_by ball \
       --device cuda --weights ../js/predator_weights.json --skip-planner 2>/dev/null \
       | grep -oE '"student_mean": [0-9.]+' | grep -oE '[0-9.]+$')
  echo "{\"mode\": \"ball_kr1_Hs\", \"K_roll\": 1, \"roll_M\": 120, \"Hs\": $hs, \"n\": 256, \"student_mean\": $sm}" >> "$OUT"
  echo "B1 ball kr1 Hs=$hs -> $sm"
done

# B2: ballistic kr=1, vary boid count roll_M at Hs=60
for m in 64 32 16 8; do
  sm=$(python3 eval_value.py --net "$NET" --n 256 --seedStart 300000 --frames 1500 \
       --Hs 60 --lookahead --roll_M "$m" --K_roll 1 --prune_by ball \
       --device cuda --weights ../js/predator_weights.json --skip-planner 2>/dev/null \
       | grep -oE '"student_mean": [0-9.]+' | grep -oE '[0-9.]+$')
  echo "{\"mode\": \"ball_kr1_M\", \"K_roll\": 1, \"roll_M\": $m, \"Hs\": 60, \"n\": 256, \"student_mean\": $sm}" >> "$OUT"
  echo "B2 ball kr1 M=$m -> $sm"
done
echo "MINIMAL_CONFIG_DONE" >> "$OUT"
setsid bash ~/watchdog.sh > ~/watchdog.log 2>&1 < /dev/null &
echo "watchdog resumed pid $!" >> "$OUT"

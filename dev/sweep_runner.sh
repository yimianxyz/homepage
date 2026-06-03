#!/usr/bin/env bash
# Load-bearing-lookahead frontier sweep: planner catches vs rollout horizon H.
# usage: sweep_runner.sh <tag> <pass> <H1> <H2> ...   (pass = single|two)
set -u
tag="$1"; shift
pass="$1"; shift
cd ~/js_eval/dev || exit 1
TP=""
[ "$pass" = "two" ] && TP="--twopass"
for H in "$@"; do
  PYTHONWARNINGS=ignore python3 planner_probe.py --controller planner \
    --n 128 --seedStart 200000 --frames 5000 --K 16 --H "$H" --D 8 \
    --device cuda $TP --weights ../js/predator_weights.json \
    --out ~/sweep_${pass}_H${H}_${tag}.json >> ~/sweep_${tag}.log 2>&1
  echo "DONE_H=${H} pass=${pass} $(date -u +%H:%M:%S)" >> ~/sweep_${tag}.log
done
echo "ALL_DONE_${tag}" >> ~/sweep_${tag}.log

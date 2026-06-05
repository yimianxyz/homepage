#!/bin/bash
# Per-VM two-pass data-gen watchdog. Runs ON the VM. Keeps gen_feat.py running
# until its output .pt exists (relaunches on crash/OOM), then exits. Deliberately
# does NOT touch VM power state — it dies when the VM stops. This is the safe
# never-idle pattern (the old local es_supervisor that restarted VMs caused a
# cost runaway and is gone).
#   wd_gen.sh <seedStart> <outName> <n> <frames>
set -u
cd ~/js_eval/dev || exit 1
SEED=${1:?}; OUT=${2:?}; N=${3:-128}; F=${4:-1500}
log=~/wd_gen.log
echo "$(date -u +%H:%M:%S) wd_gen start seed=$SEED out=$OUT n=$N frames=$F" >> "$log"
while true; do
  if [ -f ~/"$OUT" ]; then
    echo "$(date -u +%H:%M:%S) gen DONE: $OUT ($(stat -c%s ~/"$OUT") bytes)" >> "$log"
    break
  fi
  if ! pgrep -f "gen_feat.py.*$OUT" >/dev/null 2>&1; then
    echo "$(date -u +%H:%M:%S) launching gen for $OUT" >> "$log"
    setsid python3 gen_feat.py --twopass --seedStart "$SEED" --n "$N" --frames "$F" \
      --out ~/"$OUT" --device cuda --weights ../js/predator_weights.json \
      >> ~/gen_tp.log 2>&1 < /dev/null &
  fi
  sleep 60
done

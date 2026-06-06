#!/bin/bash
# Parallel JS validation of cheap configs across CPU cores.
#   run_kroll_par.sh <kroll> <seed0> <nseed> <frames>
KROLL=${1:-2}; SEED0=${2:-200000}; NSEED=${3:-12}; FRAMES=${4:-1500}; HS=${5:-0}
cd /workspace/dev
seq $SEED0 $((SEED0+NSEED-1)) | xargs -P4 -I{} node eval_kroll_one.js --js /tmp/js_kroll --seed {} --kroll $KROLL --hs $HS --frames $FRAMES 2>&1 | grep -E "CATCHES|ERR" | sort -t' ' -k2 -n | \
awk '{c+=$6; n++; ms+=$8; print} END{printf "MEAN kroll='"$KROLL"' hs='"$HS"' n=%d catches=%.2f avg_ms=%.0f\n", n, c/n, ms/n}'

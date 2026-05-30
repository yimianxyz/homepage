#!/bin/bash
# Direction-targeted set-net training for the >99% PATROL-steer goal. The oracle proved
# production patrol force == seek-to-E3D-target (cos_med 1.0); the only job is computing the
# E3D target in a net. This trains with patrol-upweighted DIRECTION (cosine) loss + sharp gate
# temperature so the gate pool can realise E3D's softmax(9.25*log a) selection.
# Args: TAG MODE POOL D RHO [extra set_e2e train flags...]
# Env: TR/VA override datasets; EP epochs.
cd ~/js_eval/e2e
export PYTORCH_ALLOC_CONF=expandable_segments:True
TR=${TR:-setds_densA_train.pt}; VA=${VA:-setds_densA_val.pt}; EP=${EP:-400}
TAG=$1; MODE=$2; POOL=$3; D=$4; RHO=$5; shift 5
while pgrep -f 'set_e2e.py train\|gen_dataset\|eval_e2e' >/dev/null; do sleep 20; done
echo "=== TRAIN $TAG mode=$MODE pool=$POOL d=$D rho=$RHO TR=$TR ($*) ==="
python3 set_e2e.py train --train $TR --val $VA --mode $MODE --pool $POOL --d $D --rho $RHO \
  --epochs $EP --lr 2e-3 --bs 8192 --dirw 5 --patrolw 4 --tag $TAG --device cuda "$@" 2>&1 | tail -45
echo "=== STEER_MATCH $TAG ==="
python3 steer_match.py --net setnet_$TAG.pt --val $VA --device cuda 2>&1 | tail -8
echo "=== DONE $TAG ==="

#!/bin/bash
# Density ablations: richer 4-scale radii; deepsets vs attn; mean vs attn pool.
cd ~/js_eval/e2e
export PYTORCH_ALLOC_CONF=expandable_segments:True
W=predator_weights.json
echo "=== GEN density train (1024) radii 60,120,240,480 ==="
python3 set_e2e.py gen --weights $W --seeds 1024 --seedStart 50000 --frames 1500 --stride 5 \
  --tag densB_train --device cuda --density-radii 60,120,240,480 2>&1 | tail -2
echo "=== GEN density val (256) ==="
python3 set_e2e.py gen --weights $W --seeds 256 --seedStart 60000 --frames 1500 --stride 5 \
  --tag densB_val --device cuda --density-radii 60,120,240,480 2>&1 | tail -2
# tag mode pool d rho heads
for cfg in "ds4r deepsets mean 48 64 2" "dsattnpool deepsets attn 48 64 2" "attn1 attn mean 48 64 2"; do
  set -- $cfg; tag=$1; mode=$2; pool=$3; d=$4; rho=$5; heads=$6
  echo "=== TRAIN $tag (mode=$mode pool=$pool d=$d rho=$rho) ==="
  python3 set_e2e.py train --train setds_densB_train.pt --val setds_densB_val.pt \
    --mode $mode --pool $pool --d $d --rho $rho --heads $heads --nblocks 1 --epochs 300 --lr 2e-3 \
    --tag densB_$tag --device cuda --quiet 2>&1 | tail -2
  echo "=== DECOMP $tag ==="
  python3 set_e2e.py decompose --net setnet_densB_$tag.pt --weights $W \
    --seeds 512 --seedStart 70000 --device cuda 2>&1 | tail -6
done
echo "=== DONE VM3 density-ablations ==="

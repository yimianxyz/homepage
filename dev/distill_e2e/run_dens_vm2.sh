#!/bin/bash
# Density-DeepSets sweep: radii 80,178,320; sizes small->large (minimality vs capacity).
cd ~/js_eval/e2e
export PYTORCH_ALLOC_CONF=expandable_segments:True
W=predator_weights.json
echo "=== GEN density train (1024) radii 80,178,320 ==="
python3 set_e2e.py gen --weights $W --seeds 1024 --seedStart 50000 --frames 1500 --stride 5 \
  --tag densA_train --device cuda --density-radii 80,178,320 2>&1 | tail -2
echo "=== GEN density val (256) ==="
python3 set_e2e.py gen --weights $W --seeds 256 --seedStart 60000 --frames 1500 --stride 5 \
  --tag densA_val --device cuda --density-radii 80,178,320 2>&1 | tail -2
# tag d rho
for cfg in "ds48 48 64" "ds32 32 32" "ds64 64 128,64"; do
  set -- $cfg; tag=$1; d=$2; rho=$3
  echo "=== TRAIN deepsets $tag (d=$d rho=$rho) ==="
  python3 set_e2e.py train --train setds_densA_train.pt --val setds_densA_val.pt \
    --mode deepsets --pool mean --d $d --rho $rho --epochs 300 --lr 2e-3 \
    --tag dens_$tag --device cuda --quiet 2>&1 | tail -2
  echo "=== DECOMP deepsets $tag ==="
  python3 set_e2e.py decompose --net setnet_dens_$tag.pt --weights $W \
    --seeds 512 --seedStart 70000 --device cuda 2>&1 | tail -6
done
echo "=== DONE VM2 density-deepsets ==="

#!/bin/bash
# CONTROL: identical pipeline WITHOUT density features -> isolates the density-feature effect.
# Same gen settings as VM2 densA but no --density-radii, so feats are the base 5 dims.
cd ~/js_eval/e2e
export PYTORCH_ALLOC_CONF=expandable_segments:True
W=predator_weights.json
echo "=== GEN plain train (1024) NO density ==="
python3 set_e2e.py gen --weights $W --seeds 1024 --seedStart 50000 --frames 1500 --stride 5 \
  --tag ctrl_train --device cuda 2>&1 | tail -2
echo "=== GEN plain val (256) ==="
python3 set_e2e.py gen --weights $W --seeds 256 --seedStart 60000 --frames 1500 --stride 5 \
  --tag ctrl_val --device cuda 2>&1 | tail -2
# tag mode pool d rho
for cfg in "ds48 deepsets mean 48 64" "attn48 attn mean 48 64"; do
  set -- $cfg; tag=$1; mode=$2; pool=$3; d=$4; rho=$5
  echo "=== TRAIN ctrl $tag (mode=$mode pool=$pool d=$d) ==="
  python3 set_e2e.py train --train setds_ctrl_train.pt --val setds_ctrl_val.pt \
    --mode $mode --pool $pool --d $d --rho $rho --heads 2 --nblocks 1 --epochs 300 --lr 2e-3 \
    --tag ctrl_$tag --device cuda --quiet 2>&1 | tail -2
  echo "=== DECOMP ctrl $tag ==="
  python3 set_e2e.py decompose --net setnet_ctrl_$tag.pt --weights $W \
    --seeds 512 --seedStart 70000 --device cuda 2>&1 | tail -6
done
echo "=== DONE VM1 control ==="

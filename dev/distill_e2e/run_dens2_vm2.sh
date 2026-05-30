#!/bin/bash
# Round 2: density+attn-pool was the winner (patrol 7.12 @13k). Push selection sharper:
# self-attn block(s) BEFORE the attn-pool refine per-boid density; more heads/width.
# Reuses existing 3-radii dataset setds_densA_{train,val}.pt on VM2 (radii 80,178,320).
cd ~/js_eval/e2e
export PYTORCH_ALLOC_CONF=expandable_segments:True
W=predator_weights.json
TR=setds_densA_train.pt; VA=setds_densA_val.pt
# tag mode nblocks pool d rho heads
for cfg in \
  "ap_h4    deepsets 0 attn 64 64    4" \
  "sa1ap_h4 attn     1 attn 64 64    4" \
  "sa2ap_h4 attn     2 attn 64 64    4" \
  "sa2ap_w  attn     2 attn 96 128,64 4" ; do
  set -- $cfg; tag=$1; mode=$2; nb=$3; pool=$4; d=$5; rho=$6; heads=$7
  echo "=== TRAIN $tag (mode=$mode nb=$nb pool=$pool d=$d rho=$rho heads=$heads) ==="
  python3 set_e2e.py train --train $TR --val $VA --mode $mode --nblocks $nb --pool $pool \
    --d $d --rho $rho --heads $heads --epochs 300 --lr 1e-3 --tag r2_$tag --device cuda --quiet 2>&1 | tail -2
  echo "=== DECOMP $tag ==="
  python3 set_e2e.py decompose --net setnet_r2_$tag.pt --weights $W \
    --seeds 512 --seedStart 70000 --device cuda 2>&1 | tail -6
done
echo "=== DONE VM2 round2 ==="

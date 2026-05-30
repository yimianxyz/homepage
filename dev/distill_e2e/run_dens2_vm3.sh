#!/bin/bash
# Round 2 (VM3): data-scaling for the winner. Wait for round-1 to finish, then gen a LARGER
# 3-radii dataset (2048 seeds) and train density+attn-pool (with/without self-attn) on it —
# does more data push patrol past 7.18 toward prod 8.19?
cd ~/js_eval/e2e
export PYTORCH_ALLOC_CONF=expandable_segments:True
W=predator_weights.json
# wait for any round-1 set_e2e job to finish
while pgrep -f 'set_e2e.py' >/dev/null; do sleep 20; done
echo "=== GEN big 3-radii train (2048) ==="
python3 set_e2e.py gen --weights $W --seeds 2048 --seedStart 50000 --frames 1500 --stride 5 \
  --tag densA2k_train --device cuda --density-radii 80,178,320 2>&1 | tail -2
echo "=== GEN big 3-radii val (384) ==="
python3 set_e2e.py gen --weights $W --seeds 384 --seedStart 60000 --frames 1500 --stride 5 \
  --tag densA2k_val --device cuda --density-radii 80,178,320 2>&1 | tail -2
TR=setds_densA2k_train.pt; VA=setds_densA2k_val.pt
for cfg in "ap2k deepsets 0 attn 64 64 4" "sa2ap2k attn 2 attn 64 64 4"; do
  set -- $cfg; tag=$1; mode=$2; nb=$3; pool=$4; d=$5; rho=$6; heads=$7
  echo "=== TRAIN $tag (mode=$mode nb=$nb d=$d) ==="
  python3 set_e2e.py train --train $TR --val $VA --mode $mode --nblocks $nb --pool $pool \
    --d $d --rho $rho --heads $heads --epochs 300 --lr 1e-3 --tag $tag --device cuda --quiet 2>&1 | tail -2
  echo "=== DECOMP $tag ==="
  python3 set_e2e.py decompose --net setnet_$tag.pt --weights $W \
    --seeds 512 --seedStart 70000 --device cuda 2>&1 | tail -6
done
echo "=== DONE VM3 round2 ==="

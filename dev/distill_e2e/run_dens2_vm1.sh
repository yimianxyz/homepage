#!/bin/bash
# Round 2 (VM1): MINIMALITY sweep for the winner. Wait for round-1, gen a 3-radii dataset,
# then shrink density+attn-pool (no self-attn) to find the smallest net still at ~7.1 patrol.
cd ~/js_eval/e2e
export PYTORCH_ALLOC_CONF=expandable_segments:True
W=predator_weights.json
while pgrep -f 'set_e2e.py' >/dev/null; do sleep 20; done
echo "=== GEN 3-radii train (1024) ==="
python3 set_e2e.py gen --weights $W --seeds 1024 --seedStart 50000 --frames 1500 --stride 5 \
  --tag densA_train --device cuda --density-radii 80,178,320 2>&1 | tail -2
echo "=== GEN 3-radii val (256) ==="
python3 set_e2e.py gen --weights $W --seeds 256 --seedStart 60000 --frames 1500 --stride 5 \
  --tag densA_val --device cuda --density-radii 80,178,320 2>&1 | tail -2
TR=setds_densA_train.pt; VA=setds_densA_val.pt
# minimality: shrink d, rho, heads
for cfg in "min_d48h4 48 64 4" "min_d32h4 32 48 4" "min_d24h2 24 32 2" "min_d16h2 16 24 2"; do
  set -- $cfg; tag=$1; d=$2; rho=$3; heads=$4
  echo "=== TRAIN $tag (d=$d rho=$rho heads=$heads) ==="
  python3 set_e2e.py train --train $TR --val $VA --mode deepsets --nblocks 0 --pool attn \
    --d $d --rho $rho --heads $heads --epochs 300 --lr 1e-3 --tag $tag --device cuda --quiet 2>&1 | tail -2
  echo "=== DECOMP $tag ==="
  python3 set_e2e.py decompose --net setnet_$tag.pt --weights $W \
    --seeds 512 --seedStart 70000 --device cuda 2>&1 | tail -6
done
echo "=== DONE VM1 round2 ==="

#!/bin/bash
# GATE-POOL test (VM2): does ONE self-attn block before the gate pool help? Self-attn refines
# per-boid density (boid sees its neighbours) -> the gate score becomes sharper. Compare to VM1
# pure-gate. Also a head-to-head: gate vs OLD attn-pool on the SAME densA data at d48.
cd ~/js_eval/e2e
export PYTORCH_ALLOC_CONF=expandable_segments:True
W=predator_weights.json
while pgrep -f 'set_e2e.py' >/dev/null; do sleep 20; done
if [ ! -f setds_densA_train.pt ]; then
  echo "=== GEN densA train/val (3-radii 80,178,320) ==="
  python3 set_e2e.py gen --weights $W --seeds 1024 --seedStart 50000 --frames 1500 --stride 5 \
    --tag densA_train --device cuda --density-radii 80,178,320 2>&1 | tail -2
  python3 set_e2e.py gen --weights $W --seeds 256 --seedStart 60000 --frames 1500 --stride 5 \
    --tag densA_val --device cuda --density-radii 80,178,320 2>&1 | tail -2
fi
TR=setds_densA_train.pt; VA=setds_densA_val.pt
# mode attn nblocks=1 + gate pool (self-attn density refinement -> sharp gate); plus a control attn-pool
for cfg in "gatesa_d48a attn 1 gate" "gatesa_d48b attn 1 gate" "attnpool_ctrl deepsets 0 attn"; do
  set -- $cfg; tag=$1; mode=$2; nb=$3; pool=$4
  echo "=== TRAIN $tag (mode=$mode nblocks=$nb pool=$pool d48) ==="
  python3 set_e2e.py train --train $TR --val $VA --mode $mode --nblocks $nb --pool $pool \
    --d 48 --rho 64 --heads 2 --epochs 300 --lr 1e-3 --tag $tag --device cuda --quiet 2>&1 | tail -2
  echo "=== DECOMP $tag ==="
  python3 set_e2e.py decompose --net setnet_$tag.pt --weights $W \
    --seeds 512 --seedStart 70000 --device cuda 2>&1 | tail -6
done
echo "=== DONE VM2 gate+selfattn ==="

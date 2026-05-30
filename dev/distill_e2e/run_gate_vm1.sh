#!/bin/bash
# GATE-POOL test (VM1): the decisive density-gated pool experiment. Production patrol selection
# is predator-INDEPENDENT (weight each boid by its own density^2.37), so a self-weighted softmax
# pool with learnable temperature should match it. Compare val patrol ANGLE (reliable metric)
# vs old attn-pool 36-38 and grid 10.8. 3-radii densA [80,178,320] incl prod cluster_r=178.
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
# pure density-gated pool (no self-attn), vary d for minimality + variance seeds at d48
for cfg in "gate_d48a 48 64" "gate_d48b 48 64" "gate_d32 32 48" "gate_d24 24 32"; do
  set -- $cfg; tag=$1; d=$2; rho=$3
  echo "=== TRAIN $tag (3-radii GATE pool d=$d) ==="
  python3 set_e2e.py train --train $TR --val $VA --mode deepsets --nblocks 0 --pool gate \
    --d $d --rho $rho --heads 2 --epochs 300 --lr 1e-3 --tag $tag --device cuda --quiet 2>&1 | tail -2
  echo "=== DECOMP $tag ==="
  python3 set_e2e.py decompose --net setnet_$tag.pt --weights $W \
    --seeds 512 --seedStart 70000 --device cuda 2>&1 | tail -6
done
echo "=== DONE VM1 gate-pool ==="

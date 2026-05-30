#!/bin/bash
# GATE-POOL test (VM3): radii sensitivity for the gate score. The gate needs the RIGHT density
# feature to weight on. Test 4-radii densB [60,120,240,480] and a single-radius-at-prod-cluster_r
# variant, pure gate pool d48. Plus a variance seed. Compares which density input the gate likes.
cd ~/js_eval/e2e
export PYTORCH_ALLOC_CONF=expandable_segments:True
W=predator_weights.json
while pgrep -f 'set_e2e.py' >/dev/null; do sleep 20; done
# densB (4-radii) should already exist from round-1; gen if missing
if [ ! -f setds_densB_train.pt ]; then
  python3 set_e2e.py gen --weights $W --seeds 1024 --seedStart 50000 --frames 1500 --stride 5 \
    --tag densB_train --device cuda --density-radii 60,120,240,480 2>&1 | tail -2
  python3 set_e2e.py gen --weights $W --seeds 256 --seedStart 60000 --frames 1500 --stride 5 \
    --tag densB_val --device cuda --density-radii 60,120,240,480 2>&1 | tail -2
fi
# single-radius at production cluster_r=178 — minimal density input the gate could need
if [ ! -f setds_densC_train.pt ]; then
  echo "=== GEN densC (1-radius 178 = prod cluster_r) ==="
  python3 set_e2e.py gen --weights $W --seeds 1024 --seedStart 50000 --frames 1500 --stride 5 \
    --tag densC_train --device cuda --density-radii 178 2>&1 | tail -2
  python3 set_e2e.py gen --weights $W --seeds 256 --seedStart 60000 --frames 1500 --stride 5 \
    --tag densC_val --device cuda --density-radii 178 2>&1 | tail -2
fi
for cfg in "gate4r_a setds_densB" "gate4r_b setds_densB" "gate1r_a setds_densC" "gate1r_b setds_densC"; do
  set -- $cfg; tag=$1; ds=$2
  echo "=== TRAIN $tag (GATE pool d48 on ${ds}) ==="
  python3 set_e2e.py train --train ${ds}_train.pt --val ${ds}_val.pt --mode deepsets --nblocks 0 \
    --pool gate --d 48 --rho 64 --heads 2 --epochs 300 --lr 1e-3 --tag $tag --device cuda --quiet 2>&1 | tail -2
  echo "=== DECOMP $tag ==="
  python3 set_e2e.py decompose --net setnet_$tag.pt --weights $W \
    --seeds 512 --seedStart 70000 --device cuda 2>&1 | tail -6
done
echo "=== DONE VM3 gate-radii ==="

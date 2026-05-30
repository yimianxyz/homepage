#!/bin/bash
# CAPACITY test (VM1): grid plateaus at ~7.0 catches / 21deg patrol angle, resolution-saturated
# (G17~=G21>G25). Is 21deg capacity-bound or representational? Reuse the G21 dataset, throw a much
# BIGGER/DEEPER head + longer training at it. If patrol angle stays ~21deg -> representational
# ceiling confirmed (the histogram MLP simply cannot sharpen the cluster-selection further).
cd ~/js_eval/e2e
export PYTORCH_ALLOC_CONF=expandable_segments:True
W=predator_weights.json
G=21; K=8
while pgrep -f 'train_e2e\|gen_dataset\|eval_e2e\|measure_patrol' >/dev/null; do sleep 20; done
TR=dataset_gb21_train.pt; VA=dataset_gb21_val.pt
for cfg in "cap_512 512,256,128 400" "cap_1024 1024,512,256 500"; do
  set -- $cfg; tag=$1; hid=$2; ep=$3
  echo "=== TRAIN $tag (G21 hidden=$hid reynolds force ${ep}ep) ==="
  python3 train_e2e.py --train $TR --val $VA --hidden $hid --head reynolds --target force \
    --epochs $ep --lr 1e-3 --bs 4096 --tag $tag --device cuda --quiet 2>&1 | tail -3
  echo "=== PATROL ANGLE $tag (reliable) ==="
  python3 measure_patrol_angle.py --net net_$tag.pt --data $VA --device cuda 2>&1 | tail -4
  echo "=== EVAL $tag catches (512 seeds) ==="
  python3 eval_e2e.py --net net_$tag.pt --weights $W --seeds 512 --seedStart 70000 \
    --G $G --K $K --device cuda 2>&1 | tail -12
done
echo "=== DONE VM1 gridcap ==="

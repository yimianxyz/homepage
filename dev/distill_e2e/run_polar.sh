#!/bin/bash
# LOG-POLAR encoder test. Args: nr nt (radial x angular bins). The Cartesian grid plateaus at
# ~7.0 catches / 21deg patrol angle. Log-polar puts ANGULAR bins aligned with the output direction
# (densest angular sector = patrol heading), so if 21deg is a readout/geometry limit (not an info
# limit) this should break it. Matched recipe (hidden 256,128, reynolds, force, NO dirw, 1280
# seeds, 300ep), TWO training seeds for variance. Usage: bash run_polar.sh 8 48
cd ~/js_eval/e2e
export PYTORCH_ALLOC_CONF=expandable_segments:True
W=predator_weights.json
NR=${1:-8}; NT=${2:-48}; K=8
while pgrep -f 'train_e2e\|gen_dataset\|eval_e2e\|measure_patrol\|set_e2e.py' >/dev/null; do sleep 20; done
TAG=p${NR}x${NT}
TR=dataset_${TAG}_train.pt; VA=dataset_${TAG}_val.pt
if [ ! -f $TR ]; then
  echo "=== GEN polar $NR x $NT train (1280 seeds) ==="
  python3 gen_dataset_e2e.py --weights $W --seeds 1280 --seedStart 50000 --frames 1500 --stride 5 \
    --polar $NR,$NT --K $K --tag ${TAG}_train --device cuda 2>&1 | tail -2
  echo "=== GEN polar val (256) ==="
  python3 gen_dataset_e2e.py --weights $W --seeds 256 --seedStart 60000 --frames 1500 --stride 5 \
    --polar $NR,$NT --K $K --tag ${TAG}_val --device cuda 2>&1 | tail -2
fi
for s in a b; do
  tag=${TAG}_$s
  echo "=== TRAIN $tag (polar $NR x $NT hidden256,128 reynolds force 300ep) ==="
  python3 train_e2e.py --train $TR --val $VA --hidden 256,128 --head reynolds --target force \
    --epochs 300 --lr 1.5e-3 --bs 4096 --tag $tag --device cuda --quiet 2>&1 | tail -3
  echo "=== PATROL ANGLE $tag (reliable) ==="
  python3 measure_patrol_angle.py --net net_$tag.pt --data $VA --device cuda 2>&1 | tail -4
  echo "=== EVAL $tag catches (512 seeds) ==="
  python3 eval_e2e.py --net net_$tag.pt --weights $W --seeds 512 --seedStart 70000 \
    --K $K --device cuda 2>&1 | tail -12
done
echo "=== DONE polar $NR x $NT ==="

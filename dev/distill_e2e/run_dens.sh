#!/bin/bash
# DENSITY-AUGMENTED GRID test. Args: G densr (grid size, local-density Gaussian sigma).
# The CONVERGED CEILING (~7.1 catches / ~18deg) is an INFORMATION limit: a COUNT-based
# histogram cannot recover production's per-boid local-density^2.37 cluster SELECTION
# (10 tight vs 10 spread boids = same cell counts, very different production weight).
# This adds ONE extra grid channel: each boid deposits its Gaussian local-neighbour
# density (neutral sigma, no dens_pow) so the grid CAN distinguish tight vs spread.
# If patrol angle breaks below ~17deg toward prod 8.19, density is the missing feature.
# Matched recipe (hidden 256,128, reynolds force, 1280 seeds, 300ep), TWO seeds for variance.
cd ~/js_eval/e2e
export PYTORCH_ALLOC_CONF=expandable_segments:True
W=predator_weights.json
G=${1:-13}; DR=${2:-120}; K=8
while pgrep -f 'train_e2e\|gen_dataset\|eval_e2e\|measure_patrol\|set_e2e.py' >/dev/null; do sleep 20; done
TAG=d${G}r${DR%.*}
TR=dataset_${TAG}_train.pt; VA=dataset_${TAG}_val.pt
if [ ! -f $TR ]; then
  echo "=== GEN dens G$G r$DR train (1280 seeds) ==="
  python3 gen_dataset_e2e.py --weights $W --seeds 1280 --seedStart 50000 --frames 1500 --stride 5 \
    --G $G --densr $DR --K $K --tag ${TAG}_train --device cuda 2>&1 | tail -2
  echo "=== GEN dens val (256) ==="
  python3 gen_dataset_e2e.py --weights $W --seeds 256 --seedStart 60000 --frames 1500 --stride 5 \
    --G $G --densr $DR --K $K --tag ${TAG}_val --device cuda 2>&1 | tail -2
fi
for s in a b; do
  tag=${TAG}_$s
  echo "=== TRAIN $tag (G$G dens r$DR hidden256,128 reynolds force 300ep) ==="
  python3 train_e2e.py --train $TR --val $VA --hidden 256,128 --head reynolds --target force \
    --epochs 300 --lr 1.5e-3 --bs 4096 --tag $tag --device cuda --quiet 2>&1 | tail -3
  echo "=== PATROL ANGLE $tag (reliable) ==="
  python3 measure_patrol_angle.py --net net_$tag.pt --data $VA --device cuda 2>&1 | tail -4
  echo "=== EVAL $tag catches (512 seeds) ==="
  python3 eval_e2e.py --net net_$tag.pt --weights $W --seeds 512 --seedStart 70000 \
    --K $K --device cuda 2>&1 | tail -12
done
echo "=== DONE dens G$G r$DR ==="

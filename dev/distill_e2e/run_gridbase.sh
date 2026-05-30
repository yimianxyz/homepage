#!/bin/bash
# CLEAN matched grid baseline. Arg: G (grid resolution). Recipe = the one that gave the best
# grid (hidden 256,128, reynolds head, force target, NO dirw, 1280 seeds, 300 ep). Trains TWO
# seeds (a/b) for variance on BOTH the reliable patrol ANGLE and the noisy catch count, so we
# get a defensible grid ceiling number instead of single-run anecdotes. Usage: bash run_gridbase.sh 21
cd ~/js_eval/e2e
export PYTORCH_ALLOC_CONF=expandable_segments:True
W=predator_weights.json
G=${1:-21}; HID=${2:-256,128}; K=8
while pgrep -f 'set_e2e.py\|train_e2e\|gen_dataset_e2e\|eval_e2e' >/dev/null; do sleep 20; done
HTAG=$(echo $HID | tr ',' '-')
TR=dataset_gb${G}_train.pt; VA=dataset_gb${G}_val.pt
if [ ! -f $TR ]; then
  echo "=== GEN G$G train (1280 seeds) ==="
  python3 gen_dataset_e2e.py --weights $W --seeds 1280 --seedStart 50000 --frames 1500 --stride 5 \
    --G $G --K $K --tag gb${G}_train --device cuda 2>&1 | tail -2
  echo "=== GEN G$G val (256) ==="
  python3 gen_dataset_e2e.py --weights $W --seeds 256 --seedStart 60000 --frames 1500 --stride 5 \
    --G $G --K $K --tag gb${G}_val --device cuda 2>&1 | tail -2
fi
for s in a b; do
  tag=gb${G}h${HTAG}_$s
  echo "=== TRAIN $tag (G$G hidden$HID reynolds force NO-dirw 300ep) ==="
  python3 train_e2e.py --train $TR --val $VA --hidden $HID --head reynolds --target force \
    --epochs 300 --lr 1.5e-3 --bs 4096 --tag $tag --device cuda --quiet 2>&1 | tail -3
  echo "=== PATROL ANGLE $tag (reliable) ==="
  python3 measure_patrol_angle.py --net net_$tag.pt --data $VA --device cuda 2>&1 | tail -4
  echo "=== EVAL $tag catches (512 seeds) ==="
  python3 eval_e2e.py --net net_$tag.pt --weights $W --seeds 512 --seedStart 70000 \
    --G $G --K $K --device cuda 2>&1 | tail -12
done
echo "=== DONE gridbase G$G ==="

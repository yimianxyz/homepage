#!/bin/bash
# GRID PLATEAU test (VM1): the grid is the best patrol encoder (10.8 deg @ G21, full ~7.16, 430k).
# Scaling curve G9->G21 plateaued (+0.53,+0.31,+0.10). Push ALL levers in one decisive run:
# G25 resolution + big head + 1536 seeds + dir-weighted loss + 350 ep. If val patrol angle drops
# well below 10.8 -> plateau was soft, pursue grid. If it stays ~10-11 -> representational CEILING
# confirmed for raw-obs encoders, and we write the ceiling report. Reliable metric = patrol angle.
cd ~/js_eval/e2e
export PYTORCH_ALLOC_CONF=expandable_segments:True
W=predator_weights.json
G=25; K=8
echo "=== GEN G$G train (1536 seeds) ==="
python3 gen_dataset_e2e.py --weights $W --seeds 1536 --seedStart 50000 --frames 1500 --stride 5 \
  --G $G --K $K --tag g25_train --device cuda 2>&1 | tail -2
echo "=== GEN G$G val (256) ==="
python3 gen_dataset_e2e.py --weights $W --seeds 256 --seedStart 60000 --frames 1500 --stride 5 \
  --G $G --K $K --tag g25_val --device cuda 2>&1 | tail -2
TR=dataset_g25_train.pt; VA=dataset_g25_val.pt
echo "=== TRAIN g25 big-head (256,128,64) reynolds dirw0.3 350ep ==="
python3 train_e2e.py --train $TR --val $VA --hidden 256,128,64 --head reynolds --target force \
  --dirw 0.3 --epochs 350 --lr 1.5e-3 --bs 4096 --tag g25big --device cuda --quiet 2>&1 | tail -3
echo "=== PATROL ANGLE g25big (reliable metric) ==="
python3 measure_patrol_angle.py --net net_g25big.pt --data $VA --device cuda 2>&1 | tail -4
echo "=== EVAL g25big catches (noisy, 512 seeds) ==="
python3 eval_e2e.py --net net_g25big.pt --weights $W --seeds 512 --seedStart 70000 \
  --G $G --K $K --device cuda 2>&1 | tail -6
echo "=== DONE VM1 gridmax ==="

#!/usr/bin/env bash
# Server-side DAgger round 1 chain: wait for the relabel dataset, retrain on
# merged base+DAgger, eval the new net with an E3D-bias sweep.
cd ~/js_eval/dev || exit 1
until grep -q '"saved"' ~/gen_dagger1.log 2>/dev/null; do sleep 20; done
PYTHONWARNINGS=ignore python3 train_value.py \
  --data ~/ds_feat_train_n64.pt,~/ds_dagger1.pt \
  --loss value,listnet --hidden 48 --epochs 150 --device cuda --out ~/net_value_v2.pt
echo V2_TRAINED
PYTHONWARNINGS=ignore python3 eval_value.py --net ~/net_value_v2.pt --n 128 \
  --seedStart 200000 --frames 5000 --Hs 0 --bias_sweep "0,0.5,1,2,4" \
  --device cuda --weights ../js/predator_weights.json --out ~/eval_v2.json
echo V2_DONE

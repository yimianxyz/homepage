#!/bin/bash
# sweep.sh {A|B|C}  — self-driving architecture search for the patrol-distill 100% target.
# Runs a slice of the config grid sequentially, appends best patrol cos to sweep_results.tsv.
# Designed to run under nohup on a VM so it survives IAP flakiness; poll the tsv to read progress.
cd ~/js_eval/e2e || exit 1
TR=setds_densAnt_train.pt; VA=setds_densAnt_val.pt
EP=${EP:-90}
run(){ local name="$1"; shift
  echo "[$(date -u +%H:%M:%S)] START $name :: $*" >> sweep.log
  python3 -u rel_train.py --train $TR --val $VA --bs 2048 --epochs $EP --device cuda --tag sw_$name "$@" > sw_$name.log 2>&1
  local best=$(grep -o 'best_pat_cosM=[0-9.eE+-]*' sw_$name.log | tail -1)
  local prm=$(grep -o 'params=[0-9]*' sw_$name.log | tail -1)
  printf "%s\t%s\t%s\t%s\n" "$name" "$best" "$prm" "$*" >> sweep_results.tsv
  echo "[$(date -u +%H:%M:%S)] DONE $name $best $prm" >> sweep.log
}
case "$1" in
 A) # cfrad: transformer encoder + E3D-style sharp selection head (top bet for ~1.0)
   run cfrad_l1   --mode cfrad --d 64 --nblocks 2 --heads 4 --K 6 --count_nbhd --logt_init 1.0
   run cfrad_l2   --mode cfrad --d 64 --nblocks 2 --heads 4 --K 6 --count_nbhd --logt_init 2.0
   run cfrad_deep --mode cfrad --d 64 --nblocks 3 --heads 4 --K 8 --count_nbhd --logt_init 2.0
   run cfrad_big  --mode cfrad --d 96 --nblocks 3 --heads 6 --K 8 --count_nbhd --logt_init 2.0 --lr 1.5e-3
   run cfrad_foc  --mode cfrad --d 64 --nblocks 3 --heads 4 --K 8 --count_nbhd --logt_init 2.0 --focal
   ;;
 B) # countformer + PMA pooling (Graphormer-style), nbhd vs base, depth
   run cf_nb_l1   --mode countformer --d 64 --nblocks 2 --heads 4 --K 6 --count_nbhd --logt_init 1.0
   run cf_nb_l2   --mode countformer --d 64 --nblocks 3 --heads 4 --K 6 --count_nbhd --logt_init 2.0
   run cf_base    --mode countformer --d 64 --nblocks 2 --heads 4 --K 6 --logt_init 1.0
   run cf_big     --mode countformer --d 96 --nblocks 3 --heads 6 --K 8 --count_nbhd --logt_init 2.0 --lr 1.5e-3
   run cf_seeds4  --mode countformer --d 64 --nblocks 3 --heads 4 --K 8 --n_seeds 4 --count_nbhd --logt_init 2.0
   ;;
 C) # radial pure (sharp-gate control: does count precision alone break 0.983?) + extras
   run rad_l1     --mode radial --d 64 --K 8 --logt_init 1.0
   run rad_l2     --mode radial --d 64 --K 8 --logt_init 2.0
   run rad_l3     --mode radial --d 80 --K 10 --logt_init 3.0
   run cfrad_l3   --mode cfrad --d 64 --nblocks 2 --heads 4 --K 8 --count_nbhd --logt_init 3.0
   run cfrad_k10  --mode cfrad --d 80 --nblocks 3 --heads 5 --K 10 --count_nbhd --logt_init 2.0 --lr 1.5e-3
   ;;
 *) echo "usage: sweep.sh {A|B|C}"; exit 1;;
esac
echo "[$(date -u +%H:%M:%S)] SWEEP $1 COMPLETE" >> sweep.log

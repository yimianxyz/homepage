#!/bin/bash
# sweep_min2.sh {A|B|C} — MINIMALITY stage 2 (radial), informed by stage 1.
# Stage-1 finding: shrinking rho head from (128,64) to a SINGLE layer (32,) collapses
# patrol cos 0.9872 -> 0.94 (m_k8h64). The output head needs DEPTH, not width. So stage 2
# holds rho at TWO layers and shrinks its width + RadialPool score_hidden(H) + radii(K),
# hunting the smallest net >= 0.9866 (99.9% of the 0.9876 cfrad ceiling).
# Appends to the same sweep.log / sweep_results.tsv the monitor polls. EP=120, logt_init=2.0.
cd ~/js_eval/e2e || exit 1
TR=${TR:-setds_densAnt_train.pt}; VA=${VA:-setds_densAnt_val.pt}
EP=${EP:-120}
run(){ local name="$1"; shift
  echo "[$(date -u +%H:%M:%S)] START $name :: $*" >> sweep.log
  python3 -u rel_train.py --train $TR --val $VA --bs 2048 --epochs $EP --device cuda --tag sw_$name "$@" > sw_$name.log 2>&1
  local best=$(grep -o 'best_pat_cosM=[0-9.eE+-]*' sw_$name.log | tail -1)
  local prm=$(grep -o 'params=[0-9]*' sw_$name.log | tail -1)
  printf "%s\t%s\t%s\t%s\n" "$name" "$best" "$prm" "$*" >> sweep_results.tsv
  echo "[$(date -u +%H:%M:%S)] DONE $name $best $prm" >> sweep.log
}
R="--mode radial --logt_init 2.0"
case "$1" in
 A) # rho-WIDTH ladder at full H64/K8 (isolate how narrow a 2-layer head can go)
   run s_r9648    $R --K 8 --edge_hidden 64 --rho 96,48
   run s_r6432    $R --K 8 --edge_hidden 64 --rho 64,32
   run s_r4824    $R --K 8 --edge_hidden 64 --rho 48,24
   run s_r3216    $R --K 8 --edge_hidden 64 --rho 32,16
   run s_r2412    $R --K 8 --edge_hidden 64 --rho 24,12
   run s_r6432k6  $R --K 6 --edge_hidden 64 --rho 64,32
   ;;
 B) # rho 64,32 / 48,24 with H,K shrink
   run s6432_h48  $R --K 8 --edge_hidden 48 --rho 64,32
   run s6432_h32  $R --K 8 --edge_hidden 32 --rho 64,32
   run s6432_k6   $R --K 6 --edge_hidden 48 --rho 64,32
   run s6432_k4   $R --K 4 --edge_hidden 64 --rho 64,32
   run s4824_h48  $R --K 8 --edge_hidden 48 --rho 48,24
   run s4824_h32  $R --K 8 --edge_hidden 32 --rho 48,24
   ;;
 C) # smallest 2-layer-rho tier
   run s4824_k6   $R --K 6 --edge_hidden 48 --rho 48,24
   run s4824_k4   $R --K 4 --edge_hidden 48 --rho 48,24
   run s3216_h48  $R --K 8 --edge_hidden 48 --rho 32,16
   run s3216_k6   $R --K 6 --edge_hidden 48 --rho 32,16
   run s3216_h32  $R --K 8 --edge_hidden 32 --rho 32,16
   run s2412_h48  $R --K 8 --edge_hidden 48 --rho 24,12
   ;;
 *) echo "usage: sweep_min2.sh {A|B|C}"; exit 1;;
esac
echo "[$(date -u +%H:%M:%S)] SWEEP_MIN2 $1 COMPLETE" >> sweep.log

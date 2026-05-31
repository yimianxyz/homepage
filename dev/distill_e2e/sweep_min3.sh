#!/bin/bash
# sweep_min3.sh {A|B|C} — MINIMALITY stage 3 (radial), informed by stages 1-2.
# Findings so far:
#   * rho head must be 2-layer; its WIDTH is the dominant lever (128,64=0.9872, 96,48=0.9854,
#     64,32=~0.983). Only the wide 128,64 head clears the strict 0.9866 bar.
#   * score_hidden H is nearly FREE: H32 (0.9828) ~= H48 (0.9818) at rho 64,32.
#   * K matters only low (K6=0.9815 vs K4=0.9777 at rho 48,24).
# So the untested sweet spot is the WIDE rho 128,64 head + SHRUNK H/K — likely holds 0.9866
# at ~10.5k (vs the 15188 H64/K8 baseline). rho 96,48 internal-shrink probed too (cheaper,
# may nudge over the bar on a lucky seed). re-runs = fresh seeds (randperm unseeded).
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
R="--mode radial --logt_init 2.0 --rho 128,64"
Q="--mode radial --logt_init 2.0 --rho 96,48"
case "$1" in
 A) # WIDE rho 128,64 + shrunk H/K — the strict-bar minimum hunt
   run t_r128_h32k6 $R --K 6 --edge_hidden 32
   run t_r128_h24k6 $R --K 6 --edge_hidden 24
   run t_r128_h16k6 $R --K 6 --edge_hidden 16
   run t_r128_h24k8 $R --K 8 --edge_hidden 24
   run t_r128_h16k8 $R --K 8 --edge_hidden 16
   run t_r128_h32k4 $R --K 4 --edge_hidden 32
   ;;
 B) # rho 96,48 internal-shrink (cheaper tier) + rho128 reseeds near the bar
   run t_r96_h32k6  $Q --K 6 --edge_hidden 32
   run t_r96_h24k6  $Q --K 6 --edge_hidden 24
   run t_r96_h32k8  $Q --K 8 --edge_hidden 32
   run t_r96_h16k6  $Q --K 6 --edge_hidden 16
   run t_r128_h32k6_s2 $R --K 6 --edge_hidden 32
   run t_r128_h24k8_s2 $R --K 8 --edge_hidden 24
   ;;
 C) # reseeds for robustness of the leading candidates (near-bar => seed variance matters)
   run t_r128_h32k8_s2 $R --K 8 --edge_hidden 32
   run t_r128_h16k6_s2 $R --K 6 --edge_hidden 16
   run t_r128_h24k6_s2 $R --K 6 --edge_hidden 24
   run t_r96_h32k6_s2  $Q --K 6 --edge_hidden 32
   run t_r128_h32k6_s3 $R --K 6 --edge_hidden 32
   run t_r96_h24k8     $Q --K 8 --edge_hidden 24
   ;;
 *) echo "usage: sweep_min3.sh {A|B|C}"; exit 1;;
esac
echo "[$(date -u +%H:%M:%S)] SWEEP_MIN3 $1 COMPLETE" >> sweep.log

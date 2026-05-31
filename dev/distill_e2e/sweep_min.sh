#!/bin/bash
# sweep_min.sh {A|B|C} — MINIMALITY search over the radial (RadialPool) policy.
# Goal: smallest net holding >= 0.9866 patrol cos_med (= 99.9% of the 0.9876 cfrad best).
# radial mode now drops the dead phi MLP (rel_net.py), so these counts are live params only.
# Levers (largest first): rho width (128,64 -> 32 -> 16), score_hidden H (64->48->32->24),
# radii K (8->6->5->4). All logt_init=2.0 (sharpest gate from the cfrad sweep). EP=120.
# Appends to the SAME sweep.log / sweep_results.tsv / sw_*.log the monitor already polls.
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
 A) # reference (no-phi baseline) + high-H frontier
   run m_ref      $R --K 8 --edge_hidden 64 --rho 128,64
   run m_k8h64    $R --K 8 --edge_hidden 64 --rho 32
   run m_k6h64    $R --K 6 --edge_hidden 64 --rho 32
   run m_k6h64r16 $R --K 6 --edge_hidden 64 --rho 16
   run m_k6h48r64 $R --K 6 --edge_hidden 48 --rho 64,32
   run m_k8h48    $R --K 8 --edge_hidden 48 --rho 32
   ;;
 B) # mid frontier: score_hidden 48/32 x K
   run m_k6h48    $R --K 6 --edge_hidden 48 --rho 32
   run m_k5h48    $R --K 5 --edge_hidden 48 --rho 32
   run m_k4h48    $R --K 4 --edge_hidden 48 --rho 32
   run m_k6h48r16 $R --K 6 --edge_hidden 48 --rho 16
   run m_k8h32    $R --K 8 --edge_hidden 32 --rho 32
   run m_k6h32    $R --K 6 --edge_hidden 32 --rho 32
   ;;
 C) # smallest tier: H 32/24, push K down
   run m_k5h32    $R --K 5 --edge_hidden 32 --rho 32
   run m_k4h32    $R --K 4 --edge_hidden 32 --rho 32
   run m_k8h24    $R --K 8 --edge_hidden 24 --rho 32
   run m_k6h24    $R --K 6 --edge_hidden 24 --rho 32
   run m_k4h24    $R --K 4 --edge_hidden 24 --rho 32
   run m_k6h24r16 $R --K 6 --edge_hidden 24 --rho 16
   ;;
 *) echo "usage: sweep_min.sh {A|B|C}"; exit 1;;
esac
echo "[$(date -u +%H:%M:%S)] SWEEP_MIN $1 COMPLETE" >> sweep.log

#!/bin/bash
# run_sealed_sharded.sh — run a Phase-2 verdict_moe sweep CELL-SHARDED across the
# container's cores (one verdict_moe process per device cell, capped concurrency),
# then merge into one exact pooled verdict. Cuts the (slow) natural full-game
# sealed sweep from hours to ~(total/ncores). Each shard is independent; the merge
# recomputes S_dec from summed counts (exact).
#
#   MODE=nn DIST=natural SEEDS=12 OFFSET=60 TAG=sealed_natural \
#   STUDENT=/path/moePolicy.js WEIGHTS=/path/moe_weights.json \
#   bash verifier/run_sealed_sharded.sh
# (DIST=scatter|natural; for --calibration set OFFSET=calib)
set -u
cd /workspace/.team/wt-exact-nn-moe/dev/exact_nn
MODE="${MODE:-nn}"; DIST="${DIST:-natural}"; SEEDS="${SEEDS:-12}"; OFFSET="${OFFSET:-60}"
TAG="${TAG:-${MODE}_${DIST}}"; NPAR="${NPAR:-4}"
EV=evidence/phase2; mkdir -p $EV/shards
[ -n "${STUDENT:-}" ] && export EXACTNN_MOE_STUDENT="$STUDENT"
[ -n "${WEIGHTS:-}" ] && export EXACTNN_MOE_WEIGHTS="$WEIGHTS"
export EXACTNN_MOE_MODE="$MODE"
CELLS="390x844 820x1180 1024x768 1512x982 1680x1050 2560x1440"
DISTFLAG="--$DIST"; [ "$DIST" = "natural" ] && DISTFLAG="--natural" || DISTFLAG="--scatter"
OFFFLAG="--sealOffset $OFFSET"; [ "$OFFSET" = "calib" ] && OFFFLAG="--calibration"

echo "=== sharded $MODE $DIST seeds=$SEEDS offset=$OFFSET (cap $NPAR) $(date -u +%H:%M:%S) ==="
pids=""; n=0
for C in $CELLS; do
  OUT=$EV/shards/${TAG}_${C}.json
  ERR=$EV/shards/${TAG}_${C}.err
  ( node verifier/verdict_moe.js --mode "$MODE" $DISTFLAG --seeds "$SEEDS" $OFFFLAG \
      ${STUDENT:+--student "$STUDENT"} ${WEIGHTS:+--weights "$WEIGHTS"} \
      --cells "$C" --out "$OUT" > /dev/null 2>"$ERR"; echo "  [done $C] $(grep -oE 'POOLED  S_dec = [0-9.]+%' "$ERR" | head -1) $(date -u +%H:%M:%S)" ) &
  pids="$pids $!"; n=$((n+1))
  # cap concurrency at NPAR
  while [ "$(jobs -r | wc -l)" -ge "$NPAR" ]; do wait -n 2>/dev/null || sleep 1; done
done
wait
echo "=== all $n cells done; merging $(date -u +%H:%M:%S) ==="
node verifier/merge_moe_reports.js $EV/${TAG}_MERGED.json $EV/shards/${TAG}_*.json
echo "merged -> $EV/${TAG}_MERGED.json"

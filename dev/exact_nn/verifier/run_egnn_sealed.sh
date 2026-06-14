#!/bin/bash
# run_egnn_sealed.sh — sealed verdict for the PURE ENDGAME NN (candidates/egnn.js),
# cell-sharded across cores, on the FRESH Phase-2 salt. Merges to one exact verdict.
#   MODE=nn DIST=scatter SEEDS=512 OFFSET=0 TAG=egnn_sealed_scatter \
#   STUDENT=/path/endgamePolicy.js WEIGHTS=/path/eg_weights.json bash verifier/run_egnn_sealed.sh
set -u
cd /workspace/.team/wt-exact-nn-moe/dev/exact_nn
MODE="${MODE:-nn}"; DIST="${DIST:-scatter}"; SEEDS="${SEEDS:-512}"; OFFSET="${OFFSET:-0}"
TAG="${TAG:-egnn_${MODE}_${DIST}}"; NPAR="${NPAR:-4}"
EV=evidence/phase2; mkdir -p $EV/shards
export EXACTNN_SALT_PATH="${EXACTNN_SALT_PATH:-$HOME/.exactnn_seal_salt_p2}"
export EXACTNN_COMMIT_PATH="${EXACTNN_COMMIT_PATH:-verifier/seal_commitment_p2.json}"
export EXACTNN_EGNN_MODE="$MODE"
[ -n "${STUDENT:-}" ] && export EXACTNN_EGNN_STUDENT="$STUDENT"
[ -n "${WEIGHTS:-}" ] && export EXACTNN_EGNN_WEIGHTS="$WEIGHTS"
[ -n "${EXTRA_ENV:-}" ] && export ${EXTRA_ENV}   # e.g. EXACTNN_EGNN_ABLATE=wa0
CELLS="390x844 820x1180 1024x768 1512x982 1680x1050 2560x1440"
DISTFLAG="--scatter"; [ "$DIST" = "natural" ] && DISTFLAG="--natural"
OFFFLAG="--sealOffset $OFFSET"; [ "$OFFSET" = "calib" ] && OFFFLAG="--calibration"
echo "=== egnn sealed $MODE $DIST seeds=$SEEDS offset=$OFFSET salt=$(basename $EXACTNN_SALT_PATH) (cap $NPAR) $(date -u +%H:%M:%S) ==="
for C in $CELLS; do
  OUT=$EV/shards/${TAG}_${C}.json; ERR=$EV/shards/${TAG}_${C}.err
  ( node verifier/verdict_moe.js --candidate candidates/egnn.js --mode "$MODE" $DISTFLAG \
      --seeds "$SEEDS" $OFFFLAG ${WEIGHTS:+--weights "$WEIGHTS"} --cells "$C" --out "$OUT" \
      > /dev/null 2>"$ERR"; echo "  [done $C] $(grep -oE 'ENDGAME  S_dec = [0-9.]+%' "$ERR" | head -1) $(date -u +%H:%M:%S)" ) &
  while [ "$(jobs -r | wc -l)" -ge "$NPAR" ]; do wait -n 2>/dev/null || sleep 1; done
done
wait
echo "=== merging $(date -u +%H:%M:%S) ==="
node verifier/merge_moe_reports.js $EV/${TAG}_MERGED.json $EV/shards/${TAG}_*.json
echo "merged -> $EV/${TAG}_MERGED.json"

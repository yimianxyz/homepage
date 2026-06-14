#!/bin/bash
# run_egnn_ablation.sh — the endgame-NN HONESTY decomposition (held-out scatter).
# Is the NN genuine (decides from allowed cheap closed-form geometry), or does it
# lean on the exact answer? There is NO exact scan-t input (source-confirmed), so we
# ablate the strongest cheap-geom proxy (the wrap-aware analytic reach time) and
# watch S_dec. Graceful degradation = genuine; ~100% survival = the exact answer is
# leaking; collapse to noise = degenerate.
#   STUDENT=/path/endgamePolicy.js WEIGHTS=/path/eg_weights.json SEEDS=300 \
#   bash verifier/run_egnn_ablation.sh
set -u
cd /workspace/.team/wt-exact-nn-moe/dev/exact_nn
STUDENT="${STUDENT:?set STUDENT}"; WEIGHTS="${WEIGHTS:?set WEIGHTS}"; SEEDS="${SEEDS:-300}"
EV=evidence/phase2/ablation; mkdir -p "$EV"
export EXACTNN_EGNN_STUDENT="$STUDENT" EXACTNN_EGNN_WEIGHTS="$WEIGHTS"

run_one() {   # $1=name  $2=mode  $3=ablate(optional)
  local name="$1" mode="$2" ablate="${3:-}"
  local out="$EV/egnn_abl_${name}.json"
  if [ -n "$ablate" ]; then export EXACTNN_EGNN_ABLATE="$ablate"; else unset EXACTNN_EGNN_ABLATE; fi
  EXACTNN_EGNN_MODE="$mode" node verifier/verdict_moe.js --candidate candidates/egnn.js --mode "$mode" \
    --scatter --calibration --seeds "$SEEDS" --weights "$WEIGHTS" --out "$out" >/dev/null 2>"$EV/egnn_abl_${name}.err"
  node -e "const r=require('./$out');console.log('  ${name}'.padEnd(15)+' endgame='+(r.S_dec_endgame*100).toFixed(3)+'%  (egC='+r.egCommits_total+' egDis='+r.egDisagree_total+' malf='+r.gateMalformed_total+')')"
}

echo "=== egnn HONESTY ablation (held-out scatter, $SEEDS seeds, 6 cells) === $(date -u +%H:%M:%S)"
run_one oracle      oracle
run_one nn          nn
run_one nn_wa0      nn       wa0
run_one nn_analytic nn       analytic
run_one nn_reach    nn       reach
run_one raw_geom    raw_geom
echo "=== egnn ablation DONE $(date -u +%H:%M:%S) ==="

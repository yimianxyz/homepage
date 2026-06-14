#!/bin/bash
# run_ablation.sh — the Phase-2 NN-vs-raw-argmax HONESTY decomposition (audit angle 3),
# on HELD-OUT calibration [270000,280000) (never the sealed pool). Answers "is the NN
# doing real work or argmax-of-given-scores?" by comparing S_dec across:
#   nn        — the real MoE (gate+experts+shared head + w_skip·dec)
#   oracle    — commits prod's exact pick = CEILING (perfect argmax of visible scores)
#   raw_prior — argmax vprior / argmin analytic-time = FLOOR (no rollout / no NN)
#   noskip    — nn with w_skip=0 → head(e) ONLY (do the expert+head alone decide?)
#   nohead    — nn with H.out=0 → w_skip·dec ONLY (does the raw decisive-score skip alone decide?)
#
#   STUDENT=/path/moePolicy.js W0=/path/moe_weights.json \
#   WNOSKIP=/path/moe_weights_noskip.json WNOHEAD=/path/moe_weights_nohead.json \
#   SEEDS_S=300 SEEDS_N=6 bash verifier/run_ablation.sh
set -u
cd /workspace/.team/wt-exact-nn-moe/dev/exact_nn
EV=evidence/phase2; mkdir -p $EV/ablation
STUDENT="${STUDENT:?set STUDENT}"; W0="${W0:?set W0}"
WNOSKIP="${WNOSKIP:-$EV/moe_weights_noskip.json}"; WNOHEAD="${WNOHEAD:-$EV/moe_weights_nohead.json}"
SEEDS_S="${SEEDS_S:-300}"; SEEDS_N="${SEEDS_N:-6}"
NCELLS="390x844,1024x768,1680x1050"   # 3-cell natural subset (planner power is huge per game)

run() {  # name mode weights dist seeds cells
  local name="$1" mode="$2" wts="$3" dist="$4" seeds="$5" cells="$6"
  local out=$EV/ablation/abl_${name}_${dist}.json
  EXACTNN_MOE_MODE=$mode EXACTNN_MOE_STUDENT=$STUDENT EXACTNN_MOE_WEIGHTS=$wts \
    node verifier/verdict_moe.js --mode $mode --$dist --calibration --seeds $seeds ${cells:+--cells $cells} \
    --student $STUDENT --weights $wts --out $out > /dev/null 2>$EV/ablation/abl_${name}_${dist}.err || true
  node -e 'const r=require("./'$out'");console.log("  '$name' ['$dist']  pooled="+(r.S_dec_pooled==null?"-":(r.S_dec_pooled*100).toFixed(3)+"%")+"  planner="+(r.S_dec_planner==null?"-":(r.S_dec_planner*100).toFixed(3)+"%")+"  endgame="+(r.S_dec_endgame==null?"-":(r.S_dec_endgame*100).toFixed(3)+"%")+"  (plans="+r.plans_total+" egC="+r.egCommits_total+" malformed="+r.gateMalformed_total+")")'
}

echo "=== ABLATION (held-out) — endgame scatter ($SEEDS_S seeds, 6 cells) ===  $(date -u +%H:%M:%S)"
run oracle    oracle    $W0      scatter $SEEDS_S ""
run raw_prior raw_prior $W0      scatter $SEEDS_S ""
run nn        nn        $W0      scatter $SEEDS_S ""
run noskip    nn        $WNOSKIP scatter $SEEDS_S ""
run nohead    nn        $WNOHEAD scatter $SEEDS_S ""
echo "=== ABLATION (held-out) — natural full-game ($SEEDS_N seeds, 3 cells) ===  $(date -u +%H:%M:%S)"
run oracle    oracle    $W0      natural $SEEDS_N "$NCELLS"
run raw_prior raw_prior $W0      natural $SEEDS_N "$NCELLS"
run nn        nn        $W0      natural $SEEDS_N "$NCELLS"
run noskip    nn        $WNOSKIP natural $SEEDS_N "$NCELLS"
run nohead    nn        $WNOHEAD natural $SEEDS_N "$NCELLS"
echo "=== ABLATION DONE $(date -u +%H:%M:%S) ==="

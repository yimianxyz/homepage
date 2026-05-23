#!/bin/bash
# Full eval batch: 16 seeds × 5000 frames × flock_centroid patrol.
# Each takes ~7 min with 4 workers. Run one at a time to avoid contention.

set -e
mkdir -p dev/reports/rule_variants

run_full() {
    local label="$1"; shift
    local report="dev/reports/rule_variants/${label}.json"
    local mean=$(node dev/eval_tte.js "$@" --seeds 16 --seedStart 100 --maxFrames 5000 --workers 4 --autoTarget flock_centroid --report "$report" 2>&1 | grep meanCatches | head -1 | sed 's/.*: //; s/,$//' | tr -d ' ')
    echo "$label  meanCatches=$mean"
}

echo "=== Full evals (16 seeds × 5000 frames × flock_centroid) ==="
echo "shipped reference: 24.25"
echo

# Baselines
run_full "shipped_NN"    js/predator_weights.json
run_full "rule_v1"       --policy rule
run_full "rule_v2_a5"    --policy rule_v2 --alpha 5
run_full "rule_v2_a8"    --policy rule_v2 --alpha 8

# rule_v3 with α=0 (most likely to differ from v2)
run_full "v3_smd_w005_a0"  --policy rule_v3 --mode score_minus_dist --distW 0.05 --alpha 0
run_full "v3_closing_a0"   --policy rule_v3 --mode closing_only --alpha 0
run_full "v3_tte_a0"       --policy rule_v3 --mode time_to_catch --alpha 0

# rule_v3 + lookahead (combine smart target + prediction)
run_full "v3_smd_w005_a5"  --policy rule_v3 --mode score_minus_dist --distW 0.05 --alpha 5
run_full "v3_tte_a5"       --policy rule_v3 --mode time_to_catch --alpha 5

# rule_v4 (perfect intercept)
run_full "v4_intercept"    --policy rule_v4

echo
echo "=== DONE ==="

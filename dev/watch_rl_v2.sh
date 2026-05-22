#!/bin/bash
# Continuous watcher: every INTERVAL seconds, pull each VM's best.json,
# run JS eval (16 seeds × 5000 frames, flock_centroid patrol), and log
# the JS-verified mean catches alongside the sim_torch baseline.
#
# Banner if a candidate JS-verifies above the shipped baseline of 24.25.

set -e
INTERVAL="${1:-900}"          # 15 min default
THRESHOLD="${2:-24.25}"       # flock_centroid baseline
DIR="${RL_V2_DIR:-rl_v2_$(date +%s)}"

if [ -f /tmp/rl_v2_dir ]; then DIR=$(cat /tmp/rl_v2_dir); fi

WATCH="dev/watch/rl_v2"
mkdir -p "$WATCH" dev/reports/rl_v2

VMS=(
    "1:us-central1-a:es_shipped_sigma05"
    "2:us-central1-a:es_shipped_sigma15"
)

PROJECT=data-analytics-prod-aegis

while true; do
    DATESTR=$(date -u +%Y%m%dT%H%M%SZ)
    echo "===== watcher round $DATESTR ====="
    for entry in "${VMS[@]}"; do
        IFS=':' read -r vm zone run <<<"$entry"
        echo "--- VM$vm/$run ---"

        BEST="$WATCH/vm${vm}_${run}_${DATESTR}.json"
        REPORT="dev/reports/rl_v2/vm${vm}_${run}_${DATESTR}_jseval.json"

        # Pull best.json
        gcloud compute scp --zone=$zone --project=$PROJECT --tunnel-through-iap \
            ml-forecast-$vm:~/$DIR/$run/best.json "$BEST" 2>&1 | tail -1

        if [ ! -s "$BEST" ]; then
            echo "  no best.json yet on VM$vm/$run"
            continue
        fi

        # Pull latest training log line for context
        gcloud compute ssh ml-forecast-$vm --zone=$zone --project=$PROJECT \
            --tunnel-through-iap \
            --command="tail -1 ~/$DIR/$run/es_log.jsonl 2>/dev/null" 2>&1 \
            | grep -v Warning | grep -v "please see" | tail -1

        # JS eval — 16 seeds × 5000 frames, flock_centroid
        echo "  JS eval running…"
        node dev/eval_tte.js "$BEST" \
            --seeds 16 --seedStart 100 --maxFrames 5000 \
            --autoTarget flock_centroid --workers 4 \
            --report "$REPORT" 2>&1 | grep -E 'meanCatches|elapsedSec' | tail -2

        MEAN=$(python3 -c "import json; print(json.load(open('$REPORT')).get('meanCatches', '?'))" 2>/dev/null)
        echo "  VM$vm/$run JS mean catches: $MEAN"

        if [ "$MEAN" != "?" ] && python3 -c "import sys; sys.exit(0 if $MEAN > $THRESHOLD else 1)" 2>/dev/null; then
            echo ""
            echo "  #####################################################"
            echo "  ##  VM$vm/$run BEATS BASELINE 24.25: JS = $MEAN  ##"
            echo "  #####################################################"
            echo ""
        fi
    done
    echo "===== sleep $INTERVAL seconds ====="
    sleep "$INTERVAL"
done

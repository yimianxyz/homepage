#!/bin/bash
# Continuously pulls every new ckpt_gen*.json from VM 3 finer ES run
# and JS-verifies it. Logs JS score per ckpt for later analysis.
set -e
DIR="${RL_V2_DIR:-rl_v2_$(date +%s)}"
[ -f /tmp/rl_v2_dir ] && DIR=$(cat /tmp/rl_v2_dir)
PROJECT=data-analytics-prod-aegis
ZONE=us-central1-c
VM=ml-forecast-3
RUN=es_v6gen10_finer
LOCAL_DIR=/tmp/${RUN}_ckpts
mkdir -p "$LOCAL_DIR"
RESULTS="$LOCAL_DIR/results.tsv"
[ ! -s "$RESULTS" ] && echo -e "gen\tjs_mean\tts" > "$RESULTS"

last_processed_gen=-1
while true; do
    # List ckpt files on VM, parse gens
    gens=$(gcloud compute ssh $VM --zone=$ZONE --project=$PROJECT --tunnel-through-iap \
        --command="ls ~/$DIR/$RUN/ckpt_gen*.json 2>/dev/null | xargs -I{} basename {} | sed 's/ckpt_gen0*\([0-9]*\).json/\1/'" 2>/dev/null \
        | grep -v Warning | grep -v "please see" | grep -v NumPy | sort -n)
    for g in $gens; do
        if [ -n "$g" ] && [ "$g" -gt "$last_processed_gen" ]; then
            local_ckpt="$LOCAL_DIR/ckpt_gen$(printf '%04d' $g).json"
            jsout="$LOCAL_DIR/ckpt_gen$(printf '%04d' $g)_jseval.json"
            if [ ! -f "$jsout" ]; then
                echo "$(date +%H:%M:%S) pulling gen $g ckpt..."
                gcloud compute scp --zone=$ZONE --project=$PROJECT --tunnel-through-iap \
                    $VM:~/$DIR/$RUN/ckpt_gen$(printf '%04d' $g).json "$local_ckpt" 2>&1 | tail -1
                echo "$(date +%H:%M:%S) JS evaling gen $g..."
                node dev/eval_tte.js "$local_ckpt" --seeds 16 --seedStart 100 \
                    --maxFrames 5000 --autoTarget flock_centroid --workers 4 \
                    --report "$jsout" 2>&1 | grep -E 'meanCatches' | tail -1
                MEAN=$(python3 -c "import json; print(json.load(open('$jsout')).get('meanCatches'))" 2>/dev/null)
                if [ -n "$MEAN" ]; then
                    ts=$(date +%s)
                    echo -e "$g\t$MEAN\t$ts" >> "$RESULTS"
                    # Banner if > 24.25
                    if python3 -c "import sys; sys.exit(0 if $MEAN > 24.25 else 1)" 2>/dev/null; then
                        echo ""
                        echo "  #################################################"
                        echo "  ##  GEN $g BEATS SHIPPED 24.25:  JS = $MEAN  ##"
                        echo "  #################################################"
                        echo ""
                    fi
                fi
                last_processed_gen=$g
            fi
        fi
    done
    sleep 120
done

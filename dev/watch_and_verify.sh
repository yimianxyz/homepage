#!/bin/bash
# Continuous watcher: every N seconds, pull each VM's best.pt + log line,
# export to JS, run JS eval, log result. If best JS catches > threshold,
# print prominent banner.
set -e
INTERVAL="${1:-1800}"     # seconds between rounds (default 30 min)
THRESHOLD="${2:-25.0}"    # JS catches > this triggers banner

WATCH_DIR="dev/watch"
mkdir -p "$WATCH_DIR" dev/weights/teachers dev/reports/teachers

while true; do
    NOW=$(date +%s)
    DATESTR=$(date -u +%Y%m%dT%H%M%SZ)
    echo "===== watcher round $DATESTR ====="

    for vm in 1 2 3; do
        z="us-central1-a"; [ "$vm" = "3" ] && z="us-central1-c"
        echo "--- pulling VM$vm best.pt ---"
        BEST="$WATCH_DIR/vm${vm}_${DATESTR}_best.pt"
        WEIGHTS="dev/weights/teachers/vm${vm}_${DATESTR}.json"
        REPORT="dev/reports/teachers/vm${vm}_${DATESTR}_eval.json"
        # Try pulling best.pt; ignore if not yet present
        gcloud compute scp --zone=$z --project=data-analytics-prod-aegis --tunnel-through-iap \
            ml-forecast-$vm:~/predator-rl/dev/checkpoints/teacher_vm${vm}/best.pt \
            "$BEST" 2>&1 | tail -1
        if [ ! -f "$BEST" ]; then
            echo "no best.pt yet on VM$vm"
            continue
        fi
        # Latest gen + baseline from log
        gcloud compute ssh ml-forecast-$vm --zone=$z --project=data-analytics-prod-aegis --tunnel-through-iap \
            --command='tail -1 ~/predator-rl/dev/checkpoints/teacher_vm'$vm'.log 2>/dev/null' 2>&1 | grep -v Warn | grep -v "please see" | tail -1
        # Export + eval
        python3 -W ignore dev/export_teacher.py --ckpt "$BEST" --out "$WEIGHTS" 2>&1 | tail -1
        echo "running JS eval (16 seeds × 5000 frames)..."
        node dev/eval_tte.js "$WEIGHTS" \
            --seeds 16 --seedStart 100 --maxFrames 5000 --workers 4 \
            --autoTarget flock_centroid \
            --report "$REPORT" 2>&1 | grep -E 'meanCatches|elapsedSec' | tail -2
        MEAN=$(python3 -c "import json; print(json.load(open('$REPORT'))['meanCatches'])" 2>/dev/null || echo "?")
        echo "VM$vm JS mean catches: $MEAN"
        # Compare to threshold
        if [ "$MEAN" != "?" ]; then
            if python3 -c "import sys; sys.exit(0 if $MEAN > $THRESHOLD else 1)" 2>/dev/null; then
                echo ""
                echo "  ##################################"
                echo "  # VM$vm BEATS THRESHOLD: $MEAN  #"
                echo "  ##################################"
                echo ""
            fi
        fi
    done

    echo "===== sleep $INTERVAL seconds ====="
    sleep $INTERVAL
done

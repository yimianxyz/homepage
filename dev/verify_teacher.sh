#!/bin/bash
# Verify a teacher checkpoint by exporting to JS weights and running JS eval.
#
# Usage: dev/verify_teacher.sh <ckpt_path> [seeds] [frames]
# Example: dev/verify_teacher.sh /tmp/teacher_vm1_best.pt 16 5000
set -e
CKPT="$1"
SEEDS="${2:-16}"
FRAMES="${3:-5000}"

if [ -z "$CKPT" ]; then
    echo "Usage: $0 <ckpt_path> [seeds] [frames]"
    exit 1
fi

NAME="$(basename "$CKPT" .pt)"
WEIGHTS="dev/weights/teachers/${NAME}.json"
REPORT="dev/reports/teachers/${NAME}_eval.json"

mkdir -p dev/weights/teachers dev/reports/teachers

echo "=== Exporting $CKPT → $WEIGHTS ==="
python3 dev/export_teacher.py --ckpt "$CKPT" --out "$WEIGHTS"

echo
echo "=== JS eval: $SEEDS seeds × $FRAMES frames, flock_centroid ==="
date
node dev/eval_tte.js "$WEIGHTS" \
    --seeds "$SEEDS" --seedStart 100 --maxFrames "$FRAMES" --workers 4 \
    --autoTarget flock_centroid \
    --report "$REPORT" 2>&1 | tail -16
date

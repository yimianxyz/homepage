#!/bin/bash
# EXACT-NN training-pipeline shakedown on VM2 (L4 GPU).
# Generates the throwaway synth dataset once, then runs the full matrix:
#   3 arch families x 2 sizes x 3 tasks  (+ f64-head variant per arch)
# ~400 steps each -- pipeline sanity + samples/sec benchmark, NOT model quality.
set -u
cd "$(dirname "$0")"
DATA=data
RESULTS=bench.jsonl
STEPS=400
BS=512

if [ ! -d "$DATA" ] || [ -z "$(ls "$DATA" 2>/dev/null)" ]; then
  echo "=== generating synth data (throwaway) ==="
  python3 synth_data.py --out "$DATA" --n 60000 --shard 8192 2>&1 | tail -3
fi

rm -f "$RESULTS"
for arch in deepset transformer pointer; do
  for size in small medium; do
    for task in l1r l1s l1p; do
      echo "=== $task $arch $size ==="
      python3 train.py --task "$task" --arch "$arch" --size "$size" \
        --data "$DATA" --steps "$STEPS" --bs "$BS" --val-every 200 \
        --results "$RESULTS" || echo "FAILED: $task $arch $size"
    done
  done
  echo "=== l1s $arch small f64-head ==="
  python3 train.py --task l1s --arch "$arch" --size small --f64-head \
    --data "$DATA" --steps "$STEPS" --bs "$BS" --val-every 200 \
    --results "$RESULTS" || echo "FAILED: f64 $arch"
done
echo "=== ALL DONE ==="
cat "$RESULTS"

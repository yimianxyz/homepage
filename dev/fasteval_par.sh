#!/bin/bash
# Parallel fasteval: shard `seeds` across `workers` node processes, aggregate.
# Usage: fasteval_par.sh <policyDir> <W> <H> <seedStart> <seeds> <frames> [workers]
set -u
POLICY=$1; W=$2; H=$3; SEED0=$4; SEEDS=$5; FRAMES=$6; WK=${7:-4}
TMP=$(mktemp -d)
per_worker=$(( (SEEDS + WK - 1) / WK ))
pids=()
for ((w=0; w<WK; w++)); do
  s0=$(( SEED0 + w*per_worker ))
  cnt=$per_worker
  rem=$(( SEEDS - w*per_worker ))
  if (( rem <= 0 )); then break; fi
  if (( cnt > rem )); then cnt=$rem; fi
  node /workspace/dev/fasteval.js --policyDir "$POLICY" --W "$W" --H "$H" \
       --seedStart "$s0" --seeds "$cnt" --frames "$FRAMES" --perseed \
       > "$TMP/w$w.json" 2>/dev/null &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python3 - "$TMP" "$POLICY" "$W" "$H" "$SEEDS" "$FRAMES" <<'PY'
import sys, json, glob, math
tmp, policy, W, H, seeds, frames = sys.argv[1:7]
per = []
for f in sorted(glob.glob(tmp + '/w*.json')):
    try:
        d = json.load(open(f)); per += d.get('per', [])
    except Exception:
        pass
n = len(per)
mean = sum(per)/n if n else 0.0
sd = (sum((x-mean)**2 for x in per)/(n-1))**0.5 if n > 1 else 0.0
se = sd/math.sqrt(n) if n else 0.0
print(json.dumps(dict(policy=policy, W=int(W), H=int(H), seeds=n, frames=int(frames),
    mean=round(mean,3), se=round(se,3))))
PY
rm -rf "$TMP"

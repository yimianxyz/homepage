#!/bin/bash
# run_throughput_search.sh — shard the paired-seed throughput farm by SCREEN across
# cores. Late-game scatter (startBoids above the max split threshold) isolates the
# config-dependent regime at high power. Held-out search block (seedBase 270000).
#   CONFIGS="count:T=3;count:T=5;count:T=8;density:Tref=5;horizon:H=90" SEEDS=120 bash verifier/run_throughput_search.sh
set -u
cd /workspace/.team/wt-exact-nn-moe/dev/exact_nn
CONFIGS="${CONFIGS:-count:T=3;count:T=5;count:T=8;density:Tref=5;horizon:H=90}"
SEEDS="${SEEDS:-120}"; SB="${SB:-28}"; BASE="${BASE:-270000}"; MF="${MF:-50000}"; NPAR="${NPAR:-4}"
TAG="${TAG:-search}"; EV=evidence/phase2/throughput; mkdir -p "$EV"
CELLS="${CELLS:-390x844 820x1180 1024x768 1512x982 1680x1050 2560x1440}"
echo "=== throughput search ($TAG): [$CONFIGS] x screens, $SEEDS paired seeds, scatter-$SB $(date -u +%H:%M:%S) ==="
for cell in $CELLS; do
  ( node verifier/throughput_farm.js --configs "$CONFIGS" --cells "$cell" --seeds "$SEEDS" --seedBase "$BASE" \
      --scatter --startBoids "$SB" --maxFrames "$MF" --out "$EV/${TAG}_${cell}.json" > "$EV/${TAG}_${cell}.log" 2>&1
    echo "  [done $cell] $(date -u +%H:%M:%S) $(tail -1 $EV/${TAG}_${cell}.log)" ) &
  while [ "$(jobs -r | wc -l)" -ge "$NPAR" ]; do wait -n 2>/dev/null || sleep 1; done
done
wait
echo "=== merging per-screen → $EV/${TAG}_ALL.json $(date -u +%H:%M:%S) ==="
node -e '
const fs=require("fs"),path=require("path");const EV="'"$EV"'",TAG="'"$TAG"'";
const cells="'"$CELLS"'".split(" ");const merged=null;let out=null;
for(const c of cells){const f=path.join(EV,TAG+"_"+c+".json");if(!fs.existsSync(f)){console.error("missing "+f);continue;}
  const r=JSON.parse(fs.readFileSync(f));if(!out){out={metric:r.metric,configs:r.configs,cells:[],seeds:r.seeds,seedSet:r.seedSet,maxFrames:r.maxFrames,R:{}};}
  out.cells.push(c);out.R[c]=r.R[c];}
fs.writeFileSync(path.join(EV,TAG+"_ALL.json"),JSON.stringify(out));console.log("merged "+out.cells.length+" screens");
'
echo "=== DONE $(date -u +%H:%M:%S) ==="

#!/bin/bash
# run_throughput_search.sh — paired-seed throughput farm sharded by (SCREEN×CONFIG) for
# even core use (the big screens are slow; per-screen sharding starves on 2560). Pairing
# is preserved: every (screen,config) job runs the SAME seed list (seedBase+i), so configs
# are paired per seed for the Wilcoxon. Late-game scatter isolates the config-dependent regime.
#   CONFIGS="count:T=3 count:T=5 count:T=8 density:Tref=5 horizon:H=40 horizon:H=90" SEEDS=80 bash verifier/run_throughput_search.sh
set -u
cd /workspace/.team/wt-exact-nn-moe/dev/exact_nn
CONFIGS="${CONFIGS:-count:T=3 count:T=5 count:T=8 density:Tref=5 horizon:H=40 horizon:H=90}"
SEEDS="${SEEDS:-80}"; SB="${SB:-28}"; BASE="${BASE:-270000}"; MF="${MF:-50000}"; NPAR="${NPAR:-4}"
TAG="${TAG:-search}"; EV=evidence/phase2/throughput; mkdir -p "$EV/shards"
CELLS="${CELLS:-390x844 820x1180 1024x768 1512x982 1680x1050 2560x1440}"
# SEALED=1 → confirm on the sealed (p2) block instead of the held-out search block
SEEDFLAG="--seedBase $BASE"; [ "${SEALED:-0}" = "1" ] && { SEEDFLAG="--sealed --sealOffset ${SOFF:-0}"; export EXACTNN_SALT_PATH="${EXACTNN_SALT_PATH:-$HOME/.exactnn_seal_salt_p2}" EXACTNN_COMMIT_PATH="${EXACTNN_COMMIT_PATH:-verifier/seal_commitment_p2.json}"; }
echo "=== throughput search ($TAG): {$CONFIGS} x {$CELLS}, $SEEDS paired seeds, scatter-$SB, base $BASE $(date -u +%H:%M:%S) ==="
# launch (screen,config) jobs, big screens first (longest), 4-parallel
for cell in 2560x1440 1680x1050 1512x982 1024x768 820x1180 390x844; do
  case " $CELLS " in *" $cell "*) ;; *) continue;; esac
  for cfg in $CONFIGS; do
    lab=$(echo "$cfg" | tr ':,=' '___')
    ( node verifier/throughput_farm.js --configs "$cfg" --cells "$cell" --seeds "$SEEDS" $SEEDFLAG \
        --scatter --startBoids "$SB" --maxFrames "$MF" --out "$EV/shards/${TAG}_${cell}_${lab}.json" >/dev/null 2>"$EV/shards/${TAG}_${cell}_${lab}.err"
      echo "  [done $cell $cfg] $(date -u +%H:%M:%S)" ) &
    while [ "$(jobs -r | wc -l)" -ge "$NPAR" ]; do wait -n 2>/dev/null || sleep 1; done
  done
done
wait
echo "=== merge (screen,config) shards → $EV/${TAG}_ALL.json $(date -u +%H:%M:%S) ==="
node -e '
const fs=require("fs"),path=require("path");const EV="'"$EV"'",TAG="'"$TAG"'";
const cells="'"$CELLS"'".split(" ").filter(Boolean);
const out={metric:"throughput=caught/frames (paired)",configs:[],cells:[],seeds:'"$SEEDS"',seedSet:"held-out@'"$BASE"'",startBoids:'"$SB"',R:{}};
const cfgSet=new Set();
for(const c of cells){ const R={}; let any=false;
  for(const f of fs.readdirSync(path.join(EV,"shards"))){ if(!f.startsWith(TAG+"_"+c+"_")||!f.endsWith(".json"))continue;
    const r=JSON.parse(fs.readFileSync(path.join(EV,"shards",f))); for(const lab of r.configs){ R[lab]=r.R[c][lab]; cfgSet.add(lab); any=true; } }
  if(any){ out.cells.push(c); out.R[c]=R; } }
out.configs=[...cfgSet];
fs.writeFileSync(path.join(EV,TAG+"_ALL.json"),JSON.stringify(out));
console.log("merged "+out.cells.length+" screens x "+out.configs.length+" configs");
'
echo "=== DONE $(date -u +%H:%M:%S) ==="

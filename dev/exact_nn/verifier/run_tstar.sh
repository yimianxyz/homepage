#!/bin/bash
# run_tstar.sh — shard the T*(screen,N0) farm by (cell × seed-block) across cores.
# Each job = ONE cell, a contiguous seed block; pairing is within-seed (across T) so
# blocks are independent and merge cleanly. Big/slow cells launched first.
#   CELLS="..." SEEDS=200 BLK=25 NPAR=4 TAG=mapA bash verifier/run_tstar.sh
set -u
cd /workspace/.team/wt-exact-nn-moe/dev/exact_nn
TS="${TS:-1,2,3,4,5,6,7,8,9,10,11,12}"
SEEDS="${SEEDS:-200}"; BLK="${BLK:-25}"; BASE="${BASE:-270000}"; NPAR="${NPAR:-4}"
FORKN="${FORKN:-13}"; MF="${MF:-60000}"; RULE="${RULE:-count}"
TAG="${TAG:-mapA}"; EV="evidence/phase2/tstar"; mkdir -p "$EV/shards"
# cells: "W,H,ua,N0;..."  (semicolon list). Default = full deployment grid.
CELLS="${CELLS:-360,800,1,0;390,844,1,0;393,852,1,0;412,915,1,0;414,896,1,0;1280,720,0,0;1366,768,0,0;1440,900,0,0;1536,864,0,0;1512,982,0,0;1600,900,0,0;1680,1050,0,0;1920,1080,0,0;2560,1440,0,0}"
SEALED="${SEALED:-0}"; SOFF="${SOFF:-0}"
nblk=$(( (SEEDS + BLK - 1) / BLK ))
echo "=== tstar farm ($TAG): $(echo "$CELLS" | tr ';' '\n' | wc -l) cells x $SEEDS seeds ($nblk blocks of $BLK) x T={$TS} sealed=$SEALED NPAR=$NPAR $(date -u +%H:%M:%S) ==="
# order cells slow-first: bigger W*H and N0=120 first (rough proxy = W*H*N0eff)
order_cells() { echo "$1" | tr ';' '\n' | awk -F, '{n=$4; if(n==0){ if($3==1)n=60; else n=120 } print $1*$2*n, $0}' | sort -rn | awk '{print $2}'; }
jobs_running() { jobs -r | wc -l; }
for cell in $(order_cells "$CELLS"); do
  ckey=$(echo "$cell" | tr ',' '_')
  for b in $(seq 0 $((nblk-1))); do
    bs=$(( b * BLK )); cnt=$BLK; [ $(( bs + cnt )) -gt "$SEEDS" ] && cnt=$(( SEEDS - bs ))
    [ "$cnt" -le 0 ] && continue
    if [ "$SEALED" = "1" ]; then
      SF="--sealed --sealOffset $(( SOFF + bs ))"
      export EXACTNN_SALT_PATH="${EXACTNN_SALT_PATH:-$HOME/.exactnn_seal_salt_p2}" EXACTNN_COMMIT_PATH="${EXACTNN_COMMIT_PATH:-verifier/seal_commitment_p2.json}"
    else
      SF="--seedBase $(( BASE + bs )) --seeds $cnt"
    fi
    [ "$SEALED" = "1" ] && SF="$SF --seeds $cnt"
    out="$EV/shards/${TAG}_${ckey}_b${b}.json"
    ( node verifier/tstar_farm.js --cells "$cell" --Ts "$TS" --rule "$RULE" $SF \
        --forkN "$FORKN" --maxFrames "$MF" --out "$out" >/dev/null 2>"$EV/shards/${TAG}_${ckey}_b${b}.err" \
      && echo "  [done $ckey b$b] $(date -u +%H:%M:%S)" || echo "  [FAIL $ckey b$b] see .err" ) &
    while [ "$(jobs_running)" -ge "$NPAR" ]; do wait -n 2>/dev/null || sleep 1; done
  done
done
wait
echo "=== merge shards → $EV/${TAG}_ALL.json $(date -u +%H:%M:%S) ==="
node verifier/tstar_merge.js "$EV/shards" "$TAG" "$EV/${TAG}_ALL.json"
echo "=== DONE $(date -u +%H:%M:%S) ==="

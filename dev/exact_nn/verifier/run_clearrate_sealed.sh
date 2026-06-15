#!/bin/bash
# run_clearrate_sealed.sh — sealed full-game clear-rate for the 3 endgame policies,
# sharded by (policy,cell) across cores, on the FRESH salt. The panel's OUTCOME metric.
#   SEEDS=20 bash verifier/run_clearrate_sealed.sh
set -u
cd /workspace/.team/wt-exact-nn-moe/dev/exact_nn
export EXACTNN_SALT_PATH="${EXACTNN_SALT_PATH:-$HOME/.exactnn_seal_salt_p2}"
export EXACTNN_COMMIT_PATH="${EXACTNN_COMMIT_PATH:-verifier/seal_commitment_p2.json}"
SEEDS="${SEEDS:-20}"; PROD_SEEDS="${PROD_SEEDS:-12}"; NPAR="${NPAR:-4}"; MF="${MF:-30000}"
EV=evidence/phase2/clearrate; mkdir -p "$EV"
CELLS="390x844 820x1180 1024x768 1512x982 1680x1050 2560x1440"
echo "=== sealed clear-rate: {prod,rawnn,analytic} x 6 cells, fresh salt $(date -u +%H:%M:%S) ==="
launch() { # policy cell seeds
  local pol="$1" cell="$2" sd="$3"
  ( node verifier/clearrate_verdict.js --policy "$pol" --seeds "$sd" --sealOffset 0 --cells "$cell" --maxFrames "$MF" \
      --out "$EV/${pol}_${cell}.json" >/dev/null 2>"$EV/${pol}_${cell}.err"
    echo "  [done $pol $cell] $(grep -oE 'clear-rate [0-9.]+%' "$EV/${pol}_${cell}.err" | head -1) stuck=$(grep -oE 'stuck [0-9]+' "$EV/${pol}_${cell}.err" | head -1) $(date -u +%H:%M:%S)" ) &
  while [ "$(jobs -r | wc -l)" -ge "$NPAR" ]; do wait -n 2>/dev/null || sleep 1; done
}
for cell in $CELLS; do launch rawnn    "$cell" "$SEEDS"; done
for cell in $CELLS; do launch analytic "$cell" "$SEEDS"; done
for cell in $CELLS; do launch prod     "$cell" "$PROD_SEEDS"; done
wait
echo "=== clear-rate sweep DONE $(date -u +%H:%M:%S) ==="
# summary
node -e '
const fs=require("fs"),path=require("path");const dir="'"$EV"'";
const byPol={};
for(const f of fs.readdirSync(dir)){ if(!f.endsWith(".json"))continue; const r=JSON.parse(fs.readFileSync(path.join(dir,f)));
  const pol=r.policy; byPol[pol]=byPol[pol]||{g:0,c:0,stuck:0,cells:[]};
  byPol[pol].g+=r.games; byPol[pol].c+=r.cleared; byPol[pol].stuck+=r.stuck_total;
  byPol[pol].cells.push(r.perCell[0].cell+":"+(r.perCell[0].clearRate*100).toFixed(0)+"%"+(r.perCell[0].stuck?("(stuck"+r.perCell[0].stuck+")"):"")); }
for(const p of Object.keys(byPol)){const b=byPol[p];console.log(p.padEnd(9)+" clear "+(100*b.c/b.g).toFixed(1)+"% ("+b.c+"/"+b.g+") stuck="+b.stuck+" | "+b.cells.sort().join(" "));}
'

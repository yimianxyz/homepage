#!/usr/bin/env bash
# eg_cert_evidence.sh — large multi-seed independent soundness run on the FINAL
# eg_bound.js certificate (U<=TMAX guard). Aggregates falseCerts across seeds into
# evidence/eg_cert_soundness_<total>.json. Cert is weights-independent (pure torus
# geometry) so this evidence is valid regardless of which NN weights side-a ships.
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-500000}
SEEDS=${2:-"3 17 101 9973"}
TOTAL=0; FALSE=0; CSUM=0; PARTS=""
for s in $SEEDS; do
  R=$(node verifier/eg_cert_verify.js "$N" "$s" 2>/dev/null)
  c=$(echo "$R" | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const r=JSON.parse(s);console.log(r.commits+" "+r.falseCertifications+" "+r.certified);})')
  read -r cc ff certd <<< "$c"
  TOTAL=$((TOTAL+cc)); FALSE=$((FALSE+ff)); CSUM=$((CSUM+certd))
  PARTS="$PARTS {\"seed\":$s,\"commits\":$cc,\"falseCerts\":$ff,\"certified\":$certd},"
  echo "  seed $s: commits=$cc falseCerts=$ff certified=$certd"
done
FRAC=$(node -e "console.log((${CSUM}/${TOTAL}).toFixed(5))")
OUT="evidence/eg_cert_soundness_${TOTAL}.json"
mkdir -p evidence
cat > "$OUT" <<JSON
{
 "instrument": "verifier/eg_cert_verify.js (independent; egPick ground truth, exact 1400-step scan)",
 "target": "endgame/eg_bound.js FINAL (U<=TMAX guard)",
 "totalCommits": ${TOTAL},
 "falseCertifications": ${FALSE},
 "certifiedFraction_broadRandom": ${FRAC},
 "verdict": "$( [ "$FALSE" -eq 0 ] && echo "SOUND — 0 false certifications over ${TOTAL} commits" || echo "UNSOUND — ${FALSE} false certs" )",
 "perSeed": [ ${PARTS%,} ]
}
JSON
echo "=== AGGREGATE: ${TOTAL} commits, ${FALSE} false certs, certFrac=${FRAC} -> ${OUT} ==="

#!/bin/bash
# finalize_holdout.sh — fetch VM3 shards, merge with local, run stats + fit.
#   bash verifier/finalize_holdout.sh
set -u
cd /workspace/.team/wt-exact-nn-moe/dev/exact_nn
export CLOUDSDK_CONFIG=$HOME/.gcloud-mlf PATH="$HOME/google-cloud-sdk/bin:$PATH"
EV=evidence/phase2/tstar
echo "=== fetch VM3 shards (md5-verified) $(date -u +%H:%M:%S) ==="
# tar VM3 shards on the remote, scp, verify md5, extract
for attempt in 1 2 3 4 5; do
  timeout 90 gcloud compute ssh ml-forecast-3 --zone us-central1-c --tunnel-through-iap \
    --command 'cd ~/tstar/dev/exact_nn/evidence/phase2/tstar/shards && tar czf /tmp/vm3_shards.tgz mapHO_vm3*.json 2>/dev/null && md5sum /tmp/vm3_shards.tgz' 2>/dev/null | grep -E '[0-9a-f]{32}' > /tmp/vm3_md5.txt && break
  echo "  ssh retry $attempt..."; sleep 5
done
REMOTE_MD5=$(awk '{print $1}' /tmp/vm3_md5.txt)
echo "  remote md5=$REMOTE_MD5"
for attempt in 1 2 3 4 5; do
  timeout 120 gcloud compute scp ml-forecast-3:/tmp/vm3_shards.tgz /tmp/vm3_shards.tgz --zone us-central1-c --tunnel-through-iap 2>/dev/null
  LOCAL_MD5=$(md5sum /tmp/vm3_shards.tgz 2>/dev/null | awk '{print $1}')
  echo "  local md5=$LOCAL_MD5 (attempt $attempt)"
  [ "$REMOTE_MD5" = "$LOCAL_MD5" ] && { echo "  MD5 MATCH"; break; }
  echo "  mismatch, retry..."; sleep 5
done
tar xzf /tmp/vm3_shards.tgz -C "$EV/shards/" && echo "  extracted VM3 shards: $(ls $EV/shards/mapHO_vm3*.json 2>/dev/null | wc -l)"
echo "=== merge ALL shards (local + vm3 r1 + vm3 r2 + 1512 local) $(date -u +%H:%M:%S) ==="
# merge each TAG family then combine: tstar_merge groups by cell key, so just point it at all
node -e '
const fs=require("fs"),path=require("path");const dir="'"$EV"'/shards";
const files=fs.readdirSync(dir).filter(f=>f.endsWith(".json")&&(f.startsWith("mapHO_local_")||f.startsWith("mapHO_vm3")));
const m={metric:null,rule:null,Ts:null,cells:[],seeds:0,seedSet:"held-out@270000",forkN:null,maxFrames:null,R:{}};const ord=[];
for(const f of files.sort()){const r=JSON.parse(fs.readFileSync(path.join(dir,f)));
  m.metric=m.metric||r.metric;m.rule=m.rule||r.rule;m.Ts=m.Ts||r.Ts;m.forkN=m.forkN||r.forkN;m.maxFrames=m.maxFrames||r.maxFrames;
  for(const k in r.R){const s=r.R[k];if(!m.R[k]){m.R[k]={W:s.W,H:s.H,uaMobile:s.uaMobile,N0eff:s.N0eff,forcedN0:s.forcedN0,byT:{},prefix:[]};ord.push(k);for(const T of m.Ts)m.R[k].byT[T]=[];}
    for(const T of m.Ts)if(s.byT[T])m.R[k].byT[T].push(...s.byT[T]);if(s.prefix)m.R[k].prefix.push(...s.prefix);}}
for(const k in m.R)for(const T of m.Ts){const seen=new Set(),u=[];for(const row of m.R[k].byT[T]){if(!seen.has(row.seed)){seen.add(row.seed);u.push(row);}}m.R[k].byT[T]=u.sort((a,b)=>a.seed-b.seed);}
m.cells=ord;m.seeds=Math.max(...ord.map(c=>m.R[c].byT[m.Ts[0]].length));
fs.writeFileSync("'"$EV"'/mapHO_ALL.json",JSON.stringify(m));
console.log("merged "+files.length+" shards -> "+ord.length+" cells, up to "+m.seeds+" seeds");
'
echo "=== STATS -> surface $(date -u +%H:%M:%S) ==="
node verifier/tstar_stats.js "$EV/mapHO_ALL.json" --out "$EV/mapHO_surface.json" 2> "$EV/mapHO_stats.txt"
cat "$EV/mapHO_stats.txt"
echo "=== FIT $(date -u +%H:%M:%S) ==="
node verifier/tstar_fit.js "$EV/mapHO_surface.json" | tee "$EV/mapHO_fit.txt"
echo "=== DONE $(date -u +%H:%M:%S) ==="

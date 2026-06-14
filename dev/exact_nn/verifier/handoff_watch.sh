#!/usr/bin/env bash
# handoff_watch.sh — poll for side-a's L1e egboidPick handoff signal, exit when
# detected (so the agent is re-invoked to run the verdict). Signals (any one):
#   (1) a new comment on issue #5 or #6 (the lead's handoff protocol);
#   (2) side-a/exact-nn-oracle branch tip advances (local or origin);
#   (3) a new/changed egboidPick.js / eg_weights*.json in side-a's worktree or .team.
# Polls every POLL seconds, max MAXMIN minutes (then exits 'timeout' as a heartbeat).
set -uo pipefail
cd /workspace/.team/wt-exact-nn
POLL=${POLL:-180}; MAXMIN=${MAXMIN:-150}
SA_WT=/workspace/.team/wt-exact-nn-oracle/dev/exact_nn/endgame

base_sa_local=$(git rev-parse refs/heads/side-a/exact-nn-oracle 2>/dev/null || echo none)
git fetch origin --quiet 2>/dev/null || true
base_sa_orig=$(git rev-parse origin/side-a/exact-nn-oracle 2>/dev/null || echo none)
base_c5=$(gh issue view 5 --json comments --jq '.comments|length' 2>/dev/null || echo -1)
base_c6=$(gh issue view 6 --json comments --jq '.comments|length' 2>/dev/null || echo -1)
base_w=$(ls -la --time-style=+%s "$SA_WT"/egboidPick.js "$SA_WT"/eg_weights*.json 2>/dev/null | awk '{print $6,$7}' | md5sum | cut -d' ' -f1)
echo "[watch] baselines: saLocal=$base_sa_local saOrig=$base_sa_orig c5=$base_c5 c6=$base_c6 w=$base_w"

iters=$(( MAXMIN*60/POLL ))
for ((i=0;i<iters;i++)); do
  sleep "$POLL"
  git fetch origin --quiet 2>/dev/null || true
  sa_local=$(git rev-parse refs/heads/side-a/exact-nn-oracle 2>/dev/null || echo none)
  sa_orig=$(git rev-parse origin/side-a/exact-nn-oracle 2>/dev/null || echo none)
  c5=$(gh issue view 5 --json comments --jq '.comments|length' 2>/dev/null || echo -1)
  c6=$(gh issue view 6 --json comments --jq '.comments|length' 2>/dev/null || echo -1)
  w=$(ls -la --time-style=+%s "$SA_WT"/egboidPick.js "$SA_WT"/eg_weights*.json 2>/dev/null | awk '{print $6,$7}' | md5sum | cut -d' ' -f1)
  hit=""
  [ "$sa_local" != "$base_sa_local" ] && hit="$hit side-a-local-tip($base_sa_local->$sa_local)"
  [ "$sa_orig" != "$base_sa_orig" ] && hit="$hit side-a-origin-tip($base_sa_orig->$sa_orig)"
  [ "$c5" != "$base_c5" ] && hit="$hit issue#5-comments($base_c5->$c5)"
  [ "$c6" != "$base_c6" ] && hit="$hit issue#6-comments($base_c6->$c6)"
  [ "$w" != "$base_w" ] && hit="$hit egboidPick/weights-changed"
  if [ -n "$hit" ]; then
    echo "[watch] HANDOFF SIGNAL:$hit"
    exit 0
  fi
done
echo "[watch] timeout after ${MAXMIN}m — heartbeat exit (re-arm)"
exit 0

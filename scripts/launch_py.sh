#!/bin/bash
# Generic detached launcher for a python job on a VM.
#   LOGOUT=~/ckpt/run.out MATCH=evolve_patrol bash launch_py.sh dev/evolve_patrol.py --args...
# Kills stale python3 procs whose cmdline matches MATCH (never the bash
# launcher, since we filter on `ps -C python3`), then starts detached.
set -u
: "${LOGOUT:?set LOGOUT}"
: "${MATCH:?set MATCH (pattern identifying this job's stale procs)}"

for pid in $(ps -C python3 -o pid= 2>/dev/null); do
  if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "$MATCH"; then
    kill "$pid" 2>/dev/null && echo "killed stale pid $pid"
  fi
done
sleep 2
cd ~/js_eval || exit 1
setsid nohup python3 "$@" </dev/null >"$LOGOUT" 2>&1 &
disown
sleep 12
if pgrep -f "$MATCH" >/dev/null; then echo LAUNCH_ALIVE; else echo LAUNCH_DIED; tail -8 "$LOGOUT" 2>/dev/null; fi

#!/bin/bash
# Robustly (re)launch an e2e_ppo run on a VM.
#   LOGOUT=~/ckpt/run.out bash launch_ppo.sh <e2e_ppo.py args...>
# Kills any existing python3 e2e_ppo process WITHOUT matching this launcher
# shell (filters on /proc/<pid>/comm == python3, never bash), then starts the
# new run fully detached (setsid + nohup + disown) and confirms it is alive.
set -u
: "${LOGOUT:?set LOGOUT to the stdout/stderr file}"

for pid in $(ps -C python3 -o pid= 2>/dev/null); do
  if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "e2e_ppo.py"; then
    kill "$pid" 2>/dev/null && echo "killed old e2e_ppo pid $pid"
  fi
done
sleep 3

cd ~/js_eval || exit 1
setsid nohup python3 dev/e2e_ppo.py "$@" </dev/null >"$LOGOUT" 2>&1 &
disown
sleep 16
if pgrep -f "python3 dev/e2e_ppo.py" >/dev/null; then
  echo "LAUNCH_ALIVE"
else
  echo "LAUNCH_DIED"; tail -5 "$LOGOUT" 2>/dev/null
fi

#!/bin/bash
# Preemption-resilient ES supervisor.
#
# Loops forever (run with `nohup` or under tmux). Each iteration:
#   1. Pick the first running ml-forecast VM (any zone). If none, try to start
#      each VM in turn (handles per-zone stockouts).
#   2. If a VM is up, ensure the latest code + a launcher script live on it.
#   3. Check if rl_train_v2.py is already running. If not, launch it with
#      --resume <last.pt> (when present) or from scratch.
#   4. Pull best.json + run_state.json back to /workspace/checkpoints/ so we
#      have a local backup even if the VM disk is lost.
#
# Termination: kill the script, or write `~/STOP_SUPERVISOR` on the VM.
set -u

GCLOUD=$HOME/gc/google-cloud-sdk/bin/gcloud
PROJECT=data-analytics-prod-aegis
CSEK=$HOME/.config/ml-forecast/csek.json
CKPT_DIR_REMOTE='$HOME/checkpoints/es_h8_v3'
CKPT_DIR_LOCAL=/workspace/checkpoints/es_h8_v3
INIT_FROM='weights/rule_v3_smd_a5_K4_H8.json'
LOG=/workspace/scripts/supervisor.log

# Stage configuration: small-first → grow if signal. Pick by stage file.
STAGE_FILE=/workspace/scripts/es_stage
if [ ! -f "$STAGE_FILE" ]; then echo 1 > "$STAGE_FILE"; fi

ZONES_A="us-central1-a"
ZONES_C="us-central1-c"

mkdir -p "$CKPT_DIR_LOCAL"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG" >&2; }

find_running_vm() {
    # Print "<name> <zone>" for the first running ml-forecast VM, or empty.
    $GCLOUD compute instances list --project=$PROJECT \
        --filter="name~ml-forecast AND status=RUNNING" \
        --format="value(name,zone)" 2>/dev/null | head -1
}

try_start_vm() {
    local name=$1 zone=$2
    $GCLOUD compute instances start "$name" --zone=$zone --project=$PROJECT \
        --csek-key-file="$CSEK" 2>&1 | tail -3
}

bring_a_vm_online() {
    # Best-effort: try all 3 VMs in their known zones. Return 0 if any succeed.
    for spec in "ml-forecast-1 $ZONES_A" "ml-forecast-2 $ZONES_A" "ml-forecast-3 $ZONES_C"; do
        local name=${spec%% *} zone=${spec##* }
        log "trying to start $name in $zone"
        out=$(try_start_vm "$name" "$zone")
        if echo "$out" | grep -q "RUNNING\|status: RUNNING"; then
            log "$name started in $zone"
            return 0
        fi
        if ! echo "$out" | grep -q "stockout\|enough resources"; then
            log "$name: unexpected: $out"
        fi
    done
    return 1
}

ssh_run() {
    local vm=$1 zone=$2; shift 2
    $GCLOUD compute ssh "$vm" --zone=$zone --project=$PROJECT \
        --tunnel-through-iap --command="$*" 2>&1 | grep -v "^WARNING:" \
        | grep -v "please see https" | grep -v "increasing_the_tcp"
}

scp_to() {
    local vm=$1 zone=$2 src=$3 dst=$4
    $GCLOUD compute scp "$src" "$vm":"$dst" --zone=$zone --project=$PROJECT \
        --tunnel-through-iap 2>&1 | grep -v "^WARNING:" | grep -v "please see https"
}

scp_from() {
    local vm=$1 zone=$2 src=$3 dst=$4
    $GCLOUD compute scp "$vm":"$src" "$dst" --zone=$zone --project=$PROJECT \
        --tunnel-through-iap 2>&1 | grep -v "^WARNING:" | grep -v "please see https"
}

push_latest_code() {
    local vm=$1 zone=$2
    log "syncing dev/*.py to $vm"
    scp_to "$vm" "$zone" /workspace/dev/sim_torch.py "/home/sa_105440658512316303279/js_eval/dev/sim_torch.py"
    scp_to "$vm" "$zone" /workspace/dev/rl_train_v2.py "/home/sa_105440658512316303279/js_eval/dev/rl_train_v2.py"
}

stage_args() {
    # Map stage number -> ES args. Small-first.
    local stage=$(cat "$STAGE_FILE")
    case "$stage" in
        1) echo "--K 64 --S 16 --frames 5000 --sigma 0.02 --lr 0.5 --top_k 8 --gens 10 --ckpt_every 1" ;;
        2) echo "--K 128 --S 32 --frames 5000 --sigma 0.02 --lr 0.5 --top_k 16 --gens 30 --ckpt_every 1" ;;
        3) echo "--K 128 --S 64 --frames 5000 --sigma 0.015 --lr 0.4 --top_k 16 --gens 60 --ckpt_every 1" ;;
        *) echo "--K 128 --S 64 --frames 5000 --sigma 0.01 --lr 0.3 --top_k 16 --gens 100 --ckpt_every 1" ;;
    esac
}

launch_es() {
    local vm=$1 zone=$2
    local stage=$(cat "$STAGE_FILE")
    local args=$(stage_args)
    local resume_arg=""
    if ssh_run "$vm" "$zone" "test -f $CKPT_DIR_REMOTE/last.pt" 2>/dev/null \
        | tail -1 | grep -qv "No such"; then
        # try detecting existence
        local exists=$(ssh_run "$vm" "$zone" "test -f $CKPT_DIR_REMOTE/last.pt && echo YES")
        if echo "$exists" | grep -q YES; then
            resume_arg="--resume $CKPT_DIR_REMOTE/last.pt"
            log "resuming from last.pt (stage $stage)"
        else
            log "starting fresh (stage $stage)"
        fi
    fi
    local cmd="cd ~/js_eval && mkdir -p $CKPT_DIR_REMOTE && \
        nohup python3 dev/rl_train_v2.py --init_from $INIT_FROM $args \
            --device cuda --out $CKPT_DIR_REMOTE --seed 1234 \
            $resume_arg > ~/es_h8_v3.log 2>&1 < /dev/null & echo ES_PID \$!"
    ssh_run "$vm" "$zone" "$cmd"
}

is_es_running() {
    local vm=$1 zone=$2
    local out=$(ssh_run "$vm" "$zone" "pgrep -af rl_train_v2.py | grep -v grep | wc -l")
    [ "$(echo "$out" | tail -1 | tr -d ' \r')" != "0" ]
}

backup_local() {
    local vm=$1 zone=$2
    scp_from "$vm" "$zone" "$CKPT_DIR_REMOTE/last.pt" "$CKPT_DIR_LOCAL/last.pt" 2>/dev/null
    scp_from "$vm" "$zone" "$CKPT_DIR_REMOTE/best.pt" "$CKPT_DIR_LOCAL/best.pt" 2>/dev/null
    scp_from "$vm" "$zone" "$CKPT_DIR_REMOTE/best.json" "$CKPT_DIR_LOCAL/best.json" 2>/dev/null
    scp_from "$vm" "$zone" "$CKPT_DIR_REMOTE/run_state.json" "$CKPT_DIR_LOCAL/run_state.json" 2>/dev/null
    scp_from "$vm" "$zone" "$CKPT_DIR_REMOTE/es_log.jsonl" "$CKPT_DIR_LOCAL/es_log.jsonl" 2>/dev/null
}

while true; do
    if [ -f ~/STOP_SUPERVISOR ]; then log "STOP file found; exiting"; exit 0; fi

    info=$(find_running_vm)
    if [ -z "$info" ]; then
        log "no VM running; attempting starts..."
        if ! bring_a_vm_online; then
            log "all VMs stockout; retrying in 120s"
            sleep 120
            continue
        fi
        sleep 30   # wait for VM SSH to come up
        continue
    fi

    vm=$(echo "$info" | awk '{print $1}')
    zone=$(echo "$info" | awk '{print $2}')
    # zone might be a path; trim
    zone=$(basename "$zone")
    log "active VM: $vm in $zone"

    push_latest_code "$vm" "$zone"

    if ! is_es_running "$vm" "$zone"; then
        log "ES not running; launching"
        launch_es "$vm" "$zone"
    else
        log "ES running"
    fi

    backup_local "$vm" "$zone"
    sleep 120
done

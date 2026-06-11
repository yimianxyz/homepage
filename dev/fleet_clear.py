#!/usr/bin/env python3
"""Fleet clearance dispatcher: run full-game time-to-extinction (clear_eval.js)
across the 3 GPU VMs' 12 CPU cores. Each job = (policy, device) cell; remote_clear.sh
shards it over the host's 4 cores. Jobs are balanced so each host gets exactly one
slow deployed-on-big-screen cell. Prints a markdown table.
"""
import json, base64, subprocess, threading, queue, os, re

GCLOUD = os.path.expanduser("~/google-cloud-sdk/bin/gcloud")
HOSTS = [("ml-forecast-1", "us-central1-a"),
         ("ml-forecast-2", "us-central1-a"),
         ("ml-forecast-3", "us-central1-c")]

DEVICES = {  # name -> (W, H, maxFrames)
    "phone":  (390, 844, 14000),
    "ipad":   (820, 1180, 26000),
    "laptop": (1440, 900, 26000),
}
POLICIES = {  # label -> (policyDir, config or None)
    "deployed": ("js", None),
    "old_teri": ("ship_teri", None),
    "new":      ("exp/js", {"endgame": True, "egFreeze": 0, "egDt": 1, "egSlack": 1.0}),
}
SEED0, SEEDS = 270000, 64

# build 9 jobs, assign to hosts balanced (each host one deployed-bigscreen)
JOBS = []
for dev in DEVICES:
    for pol in POLICIES:
        JOBS.append({"dev": dev, "pol": pol})
# host assignment: spread slow cells
ASSIGN = {
    0: [("phone", "deployed"), ("ipad", "old_teri"), ("laptop", "new")],
    1: [("phone", "old_teri"), ("ipad", "new"), ("laptop", "deployed")],
    2: [("phone", "new"), ("ipad", "deployed"), ("laptop", "old_teri")],
}


def run_cell(host, zone, dev, pol):
    W, H, MF = DEVICES[dev]
    policyDir, cfg = POLICIES[pol]
    cfgb = base64.b64encode(json.dumps(cfg).encode()).decode() if cfg else '""'
    cmd = f"bash ~/eval/remote_clear.sh {policyDir} {cfgb} {W} {H} {SEEDS} {MF} {SEED0} 4"
    full = [GCLOUD, "compute", "ssh", host, "--zone", zone, "--tunnel-through-iap", "--command", cmd]
    for attempt in range(2):
        try:
            out = subprocess.run(full, capture_output=True, text=True, timeout=2400)
            for line in reversed(out.stdout.strip().splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    return json.loads(line)
        except Exception as e:
            pass
    return None


def worker(hi, results, lock):
    host, zone = HOSTS[hi]
    for dev, pol in ASSIGN[hi]:
        r = run_cell(host, zone, dev, pol)
        with lock:
            results[(dev, pol)] = r
        print(f"  [{host}] {dev}/{pol}: " + (json.dumps(r) if r else "FAILED"), flush=True)


def main():
    results, lock = {}, threading.Lock()
    threads = [threading.Thread(target=worker, args=(hi, results, lock)) for hi in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()
    # table
    print("\n| device | deployed | old_teri | new | new vs deployed |")
    print("|---|---|---|---|---|")
    for dev in DEVICES:
        cells = {}
        for pol in POLICIES:
            r = results.get((dev, pol))
            cells[pol] = f"{r['tClear']}f/{int(r['clearRate']*100)}%" if r else "—"
        dr, nr = results.get((dev, "deployed")), results.get((dev, "new"))
        spd = f"{dr['tClear']/nr['tClear']:.2f}x" if (dr and nr and nr['tClear']) else "—"
        print(f"| {dev} | {cells['deployed']} | {cells['old_teri']} | {cells['new']} | {spd} |")
    print("\nRAW:", json.dumps({f"{d}/{p}": results.get((d, p)) for d in DEVICES for p in POLICIES}))


if __name__ == "__main__":
    main()

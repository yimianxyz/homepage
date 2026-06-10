#!/usr/bin/env python3
"""Fleet eval dispatcher: run fasteval jobs across the 3 GPU VMs (CPU) in parallel.

Each job = a (policyDir, config, device) cell. A pool of one worker thread per host
pulls jobs from a queue and runs ~/eval/remote_shard.sh (4 node workers) over SSH,
returning per-seed catches. Aggregates mean/se per job.

Jobs spec: JSON list on stdin (or --jobs file), each:
  {"label":..., "policy":"js|exp|wrap", "config":{...}|null, "W":390,"H":844,
   "seeds":64, "frames":1500, "seed0":200000}

  echo '[{...},...]' | python3 dev/fleet_eval.py
"""
import sys, json, base64, subprocess, threading, queue, math, time

HOSTS = [("ml-forecast-1", "us-central1-a"),
         ("ml-forecast-2", "us-central1-a"),
         ("ml-forecast-3", "us-central1-c")]
GCLOUD = __import__('os').path.expanduser("~/google-cloud-sdk/bin/gcloud")


def run_job(host, zone, job):
    cfg = job.get("config")
    cfgb = base64.b64encode(json.dumps(cfg).encode()).decode() if cfg else ""
    pol = job["policy"]
    cmd = (f"bash ~/eval/remote_shard.sh {pol} '{cfgb}' {job['W']} {job['H']} "
           f"{job.get('seed0',200000)} {job['seeds']} {job.get('frames',1500)} {job.get('workers',4)}")
    full = [GCLOUD, "compute", "ssh", host, "--zone", zone, "--tunnel-through-iap",
            "--command", cmd]
    for attempt in range(3):
        try:
            out = subprocess.run(full, capture_output=True, text=True, timeout=1200)
            txt = out.stdout.strip()
            # the per-seed JSON array is the last bracketed line
            for line in reversed(txt.splitlines()):
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    per = json.loads(line)
                    return per
        except Exception as e:
            last = str(e)
            time.sleep(2)
    return None


def worker(host, zone, q, results, lock):
    while True:
        try:
            idx, job = q.get_nowait()
        except queue.Empty:
            return
        per = run_job(host, zone, job)
        with lock:
            results[idx] = (job, per)
        q.task_done()


def main():
    if "--jobs" in sys.argv:
        jobs = json.load(open(sys.argv[sys.argv.index("--jobs") + 1]))
    else:
        jobs = json.load(sys.stdin)
    q = queue.Queue()
    for i, j in enumerate(jobs):
        q.put((i, j))
    results = {}
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(h, z, q, results, lock))
               for h, z in HOSTS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out = []
    for i in range(len(jobs)):
        job, per = results.get(i, (jobs[i], None))
        if not per:
            out.append(dict(label=job.get("label"), mean=None, se=None, n=0, err=True))
            continue
        n = len(per)
        mean = sum(per) / n
        sd = (sum((x - mean) ** 2 for x in per) / (n - 1)) ** 0.5 if n > 1 else 0.0
        se = sd / math.sqrt(n) if n else 0.0
        out.append(dict(label=job.get("label"), policy=job["policy"], config=job.get("config"),
                        W=job["W"], H=job["H"], n=n, mean=round(mean, 3), se=round(se, 3)))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

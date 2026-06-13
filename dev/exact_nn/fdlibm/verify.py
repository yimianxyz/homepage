#!/usr/bin/env python3
# EXACT-NN fdlibm port — verifier.
# Loads ref.jsonl (from gen_ref.js, node = ground truth), runs js_exp/js_pow
# on CPU (and CUDA when available), reports per-range bitexact% and max ulp.
#
# Acceptance: POLICY ranges must be 100.0000% bitexact (the task bar).
# Exit code 1 if any policy range is below 100% on any tested device.
#
# Usage: python3 verify.py [ref.jsonl]
import json
import struct
import sys
from collections import defaultdict

import torch

from fdlibm_torch import js_exp, js_pow

POLICY = {"exp_p64", "exp_p3", "exp_edge", "pow_dens", "pow_sharp"}
EXP_OPS_FIRST = ("exp_p64", "exp_p3", "exp_edge", "exp_subn", "exp_stress",
                 "pow_dens", "pow_sharp", "pow_mid", "pow_edge", "pow_stress")


def u64_to_i64(u):
    return u - (1 << 64) if u >= (1 << 63) else u


def i64_to_u64(i):
    return i + (1 << 64) if i < 0 else i


def ulp_dist_u64(a, b):
    """ulp distance between two doubles given as u64 bit patterns
    (monotonic remap of the IEEE total order; exact, arbitrary precision)."""
    def m(u):
        return u if u < (1 << 63) else (1 << 63) - u
    return abs(m(a) - m(b))


def load(path):
    groups = defaultdict(lambda: ([], []))  # op -> (list[list[u64 ins]], list[u64 out])
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            ins, outs = groups[rec["op"]]
            ins.append([int(h, 16) for h in rec["in"]])
            outs.append(int(rec["out"], 16))
    return groups


def tensors_for(op, ins, device):
    cols = list(zip(*ins))
    ts = [torch.tensor([u64_to_i64(u) for u in col], dtype=torch.int64,
                       device=device).view(torch.float64) for col in cols]
    return ts


def main(path):
    groups = load(path)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows = []
    failures = []
    for device in devices:
        for op in sorted(groups, key=lambda o: EXP_OPS_FIRST.index(o)
                         if o in EXP_OPS_FIRST else 99):
            ins, outs = groups[op]
            ts = tensors_for(op, ins, device)
            if op.startswith("exp"):
                got = js_exp(ts[0])
            else:
                got = js_pow(ts[0], ts[1])
            got_u = [i64_to_u64(v) for v in got.view(torch.int64).cpu().tolist()]
            n = len(outs)
            mism = [(i, g, w) for i, (g, w) in enumerate(zip(got_u, outs)) if g != w]
            exact = n - len(mism)
            maxulp = max((ulp_dist_u64(g, w) for _, g, w in mism), default=0)
            pct = 100.0 * exact / n
            tag = "POLICY" if op in POLICY else "stress"
            rows.append((device, op, tag, n, exact, pct, maxulp))
            if op in POLICY and exact != n:
                failures.append((device, op, mism[:10], ins))
    print(f"{'dev':5} {'range':11} {'class':7} {'n':>8} {'bitexact':>9} "
          f"{'bitexact%':>10} {'maxulp':>7}")
    print("-" * 64)
    for device, op, tag, n, exact, pct, maxulp in rows:
        print(f"{device:5} {op:11} {tag:7} {n:8d} {exact:9d} {pct:10.4f} {maxulp:7d}")
    print("-" * 64)
    for device in devices:
        pol = [r for r in rows if r[0] == device and r[2] == "POLICY"]
        ok = all(r[4] == r[3] for r in pol)
        npol = sum(r[3] for r in pol)
        print(f"{device}: policy ranges {'100.0000% BITEXACT' if ok else 'FAILED'} "
              f"over {npol} samples")
    if failures:
        print("\nfirst policy mismatches:")
        for device, op, mism, ins in failures:
            for i, g, w in mism:
                arg = " ".join(f"{u:016x}" for u in ins[i])
                print(f"  {device} {op} in={arg} got={g:016x} want={w:016x} "
                      f"ulp={ulp_dist_u64(g, w)}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ref.jsonl")

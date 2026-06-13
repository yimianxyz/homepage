#!/usr/bin/env python3
# EXACT-NN float64 spike — stage 2 (torch side).
# Reads vectors.jsonl (from gen_vectors.js), recomputes every op class in
# torch float64 on CPU and CUDA, and reports per class: bitwise match rate,
# max ulp distance. Decides whether the GPU can be a BITWISE engine for the
# policy (rollout physics) and where it can't (transcendentals -> JS-side or
# fdlibm port).
import json, struct, sys
import torch

def bits(x):  # float64 -> u64
    return struct.unpack('<Q', struct.pack('<d', x))[0]

def from_hex(h):
    return struct.unpack('>d', bytes.fromhex(h))[0]

def ulp_dist(a, b):
    ia, ib = bits(a), bits(b)
    # map to monotonic integer line (two's-complement style for negatives)
    def m(i):
        return i if i < (1 << 63) else (1 << 64) - i + (1 << 63) - 1
    return abs(m(ia) - m(ib))

def fastmag(x, y):
    ax, ay = abs(x), abs(y)
    hi = ax if ax > ay else ay
    lo = ax if ax < ay else ay
    return hi * 0.96 + lo * 0.398

def compute(op, ins, dev):
    t = lambda v: torch.tensor(v, dtype=torch.float64, device=dev)
    if op == 'mul_add':
        a, c, b, d = ins
        return (t(a) * t(c) + t(b) * t(d)).item()
    if op == 'div':
        a, b = ins
        return (t(a) / t(b)).item()
    if op == 'sqrt_hyp':
        a, b = ins
        return torch.sqrt(t(a) * t(a) + t(b) * t(b)).item()
    if op == 'fastmag':
        c, d = ins
        return fastmag(c, d)  # pure-python IEEE float64 ops == JS semantics
    if op == 'round':
        a, = ins
        # JS Math.round = floor(x+0.5); torch.round is banker's — test the JS-faithful form
        return torch.floor(t(a) + 0.5).item()
    if op in ('exp_negsq', 'exp_reach'):
        a, = ins
        return torch.exp(t(a)).item()
    if op == 'pow_dens' or op == 'pow_sharp':
        a, b = ins
        return torch.pow(t(a), t(b)).item()
    if op == 'erf':
        z, = ins
        s = -1.0 if z < 0 else 1.0
        x = t(abs(z))
        tt = 1 / (1 + 0.3275911 * x)
        y = 1 - (((((1.061405429 * tt - 1.453152027) * tt) + 1.421413741) * tt - 0.284496736) * tt + 0.254829592) * tt * torch.exp(-x * x)
        return (s * y).item()
    if op == 'gelu':
        z, = ins
        x = t(z) / (2 ** 0.5)
        s = torch.sign(x) + (x == 0).double()
        ax = torch.abs(x)
        tt = 1 / (1 + 0.3275911 * ax)
        y = 1 - (((((1.061405429 * tt - 1.453152027) * tt) + 1.421413741) * tt - 0.284496736) * tt + 0.254829592) * tt * torch.exp(-ax * ax)
        return (0.5 * t(z) * (1 + s * y)).item()
    raise ValueError(op)

def main(path):
    rows = [json.loads(l) for l in open(path)]
    devs = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    stats = {}
    for dev in devs:
        for rec in rows:
            ins = [from_hex(h) for h in rec['in']]
            ref = from_hex(rec['out'])
            got = compute(rec['op'], ins, dev)
            key = (dev, rec['op'])
            s = stats.setdefault(key, [0, 0, 0])  # n, exact, maxulp
            s[0] += 1
            if bits(got) == bits(ref):
                s[1] += 1
            else:
                s[2] = max(s[2], ulp_dist(got, ref))
    print(f"{'dev':5} {'op':10} {'n':>7} {'bitexact%':>10} {'maxulp':>8}")
    for (dev, op), (n, ex, mu) in sorted(stats.items()):
        print(f"{dev:5} {op:10} {n:7d} {100.0*ex/n:10.4f} {mu:8d}")

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'vectors.jsonl')

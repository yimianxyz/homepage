#!/usr/bin/env python3
"""Compare GPU vs JS count-rule throughput-vs-T (cross-check). Prints the two curves
per screen + Spearman rank-corr + argmax agreement.
  python3 xcc_compare.py xcc_gpu.jsonl /tmp/xcc_js
"""
import json, sys, glob, os
from collections import defaultdict

gpu_path, js_dir = sys.argv[1], sys.argv[2]

def load_jsonl(p):
    out = []
    for line in open(p):
        line = line.strip()
        if line.startswith('{'):
            out.append(json.loads(line))
    return out

gpu = load_jsonl(gpu_path)
js = []
for f in glob.glob(os.path.join(js_dir, '*.json')):
    txt = open(f).read().strip()
    if txt.startswith('{'):
        js.append(json.loads(txt))

def curve(rows, cell):
    d = {}
    for r in rows:
        if r['cell'] == cell and r['rule'] == 'count':
            d[r['param']] = r['throughput']
    return d

def spearman(xs, ys):
    n = len(xs)
    if n < 2: return float('nan')
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0]*n
        for pos, i in enumerate(order): rk[i] = pos
        return rk
    rx, ry = ranks(xs), ranks(ys)
    dx = [rx[i]-ry[i] for i in range(n)]
    return 1 - 6*sum(d*d for d in dx)/(n*(n*n-1))

cells = sorted(set(r['cell'] for r in gpu) | set(r['cell'] for r in js))
for cell in cells:
    g, j = curve(gpu, cell), curve(js, cell)
    Ts = sorted(set(g) & set(j))
    if not Ts: continue
    gv = [g[t] for t in Ts]; jv = [j[t] for t in Ts]
    print(f'\n=== {cell} (count) ===')
    print('  T:    ' + '  '.join(f'{int(t):>6}' for t in Ts))
    print('  GPU:  ' + '  '.join(f'{v*1e4:6.1f}' for v in gv) + '  (x1e-4)')
    print('  JS:   ' + '  '.join(f'{v*1e4:6.1f}' for v in jv) + '  (x1e-4)')
    gstar = Ts[max(range(len(Ts)), key=lambda i: gv[i])]
    jstar = Ts[max(range(len(Ts)), key=lambda i: jv[i])]
    rho = spearman(gv, jv)
    ratio = sum(gv)/sum(jv) if sum(jv) else float('nan')
    print(f'  GPU T*={int(gstar)}  JS T*={int(jstar)}  spearman={rho:+.3f}  GPU/JS level={ratio:.3f}')

#!/usr/bin/env python3
"""Analyze the GPU throughput surface -> per-screen T* (count rule) + whether ONE
horizon_scaled setting auto-captures the per-screen optimum.

  python3 analyze_surface.py cfgs_vm1.out cfgs_vm2.out
"""
import json, sys
from collections import defaultdict

rows = []
for path in sys.argv[1:]:
    for line in open(path):
        line = line.strip()
        if not line.startswith('{'):
            continue
        rows.append(json.loads(line))

# index: (cell, rule) -> [(param, throughput, clearRate, medClearFrames)]
by = defaultdict(list)
screens_order = []
for r in rows:
    by[(r['cell'], r['rule'])].append((r['param'], r['throughput'], r['clearRate'], r.get('medClearFrames', -1)))
    if r['cell'] not in screens_order:
        screens_order.append(r['cell'])

def area(cell):
    w, h = map(int, cell.split('x')); return w * h

screens = sorted(set(r['cell'] for r in rows), key=area)

print('=== COUNT rule: throughput vs T, per screen (T* = argmax) ===')
count_star = {}
for cell in screens:
    pts = sorted(by.get((cell, 'count'), []))
    if not pts: continue
    best = max(pts, key=lambda x: x[1])
    count_star[cell] = best
    cr_min = min(p[2] for p in pts)
    curve = '  '.join(f'T{int(p[0])}:{p[1]*1000:.3f}' for p in pts)
    print(f'{cell:>10} (area {area(cell)/1e6:.2f}M)  T*={int(best[0])} thru={best[1]*1000:.3f}e-3 clrMin={cr_min:.2f}')
    print(f'             {curve}')

print('\n=== HORIZON_SCALED: throughput vs h, per screen (h* = argmax) ===')
for cell in screens:
    pts = sorted(by.get((cell, 'horizon_scaled'), []))
    if not pts: continue
    best = max(pts, key=lambda x: x[1])
    curve = '  '.join(f'h{int(p[0])}:{p[1]*1000:.3f}' for p in pts)
    cs = count_star.get(cell)
    gap = (cs[1] - best[1]) / cs[1] * 100 if cs else float('nan')
    print(f'{cell:>10}  h*={int(best[0])} thru={best[1]*1000:.3f}e-3   vs countT* gap={gap:+.1f}%')
    print(f'             {curve}')

print('\n=== HORIZON AUTO-CAPTURE TEST: for each fixed h, worst-screen gap vs per-screen count-T* ===')
hvals = sorted(set(p[0] for cell in screens for p in by.get((cell, 'horizon_scaled'), [])))
for h in hvals:
    gaps = []
    for cell in screens:
        cs = count_star.get(cell)
        hp = [p for p in by.get((cell, 'horizon_scaled'), []) if p[0] == h]
        if cs and hp:
            gaps.append((cell, (cs[1] - hp[0][1]) / cs[1] * 100))
    if gaps:
        worst = max(gaps, key=lambda x: x[1])
        mean = sum(g for _, g in gaps) / len(gaps)
        print(f'h={int(h)}: mean gap {mean:+.1f}%  worst {worst[1]:+.1f}% @ {worst[0]}')

print('\n=== DENSITY rule: throughput vs rho, per screen ===')
for cell in screens:
    pts = sorted(by.get((cell, 'density'), []))
    if not pts: continue
    best = max(pts, key=lambda x: x[1])
    curve = '  '.join(f'r{int(p[0])}:{p[1]*1000:.3f}' for p in pts)
    print(f'{cell:>10}  rho*={int(best[0])} thru={best[1]*1000:.3f}e-3   {curve}')

print('\n=== T*-vs-screen summary (count) ===')
for cell in screens:
    cs = count_star.get(cell)
    if cs: print(f'{cell:>10}  area={area(cell)/1e6:.2f}M  T*={int(cs[0])}  medClear={cs[3]:.0f}')

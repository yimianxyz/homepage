#!/usr/bin/env python3
"""Fit the best dynamic gate formula T(W,Hc,N0) to the GPU throughput surface (#5).

Loads any number of surface JSONL files (rows: {cell:"WxHc:N0", rule, param=T, throughput}),
builds the throughput table over (W,Hc,N0,T), and fits the panel's candidate forms to the
SURFACE (loss = -prevalence-weighted achieved throughput when the formula picks T per cell),
NOT the noisy argmax. Reports each form's params, the achieved-vs-per-cell-optimal gap, and
the gain vs fixed-T=8 / fixed-T=5. Disentangles the area-vs-N0 confound (uses N0 explicitly).

  python3 fit_formula.py surface1.jsonl surface2.jsonl ...
"""
import json, sys, math
from collections import defaultdict

BORDER = 20  # (W+20)(Hc+20) torus area, matches A_ref convention
A_REF = (1024 + BORDER) * (768 + BORDER)

def load(paths):
    tab = {}  # (W,H,N0,T) -> throughput  (count rule only)
    for p in paths:
        for L in open(p):
            L = L.strip()
            if not L.startswith('{'): continue
            r = json.loads(L)
            if r.get('rule') != 'count': continue
            W, H = map(int, r['cell'].split('x'))
            n0 = int(r['boids'])
            tab[(W, H, n0, int(round(r['param'])))] = r['throughput']
    return tab

def cells(tab):
    return sorted({(W, H, N0) for (W, H, N0, T) in tab})

def Ts_for(tab, c):
    W, H, N0 = c
    return sorted(T for (w, h, n, T) in tab if (w, h, n) == c)

def thru(tab, c, T):
    W, H, N0 = c
    Ts = Ts_for(tab, c)
    Tn = min(Ts, key=lambda t: abs(t - T))   # snap to nearest MEASURED T
    return tab[(W, H, N0, Tn)]

def area(W, H): return (W + BORDER) * (H + BORDER)

# ---- candidate formulas: name -> (param-grid-fn, T(c,params)) ----
def clampT(x): return int(min(12, max(1, round(x))))

FORMS = {
    # (a) power of area
    'power_A':   (lambda: [(a, p) for a in [x/1000 for x in range(2, 200, 2)] for p in [x/100 for x in range(10, 90, 5)]],
                  lambda c, pr: clampT(pr[0] * area(c[0], c[1]) ** pr[1])),
    # (b) linear in sqrt(area)
    'lin_sqrtA': (lambda: [(a, b) for a in range(-6, 8) for b in [x/1000 for x in range(1, 30)]],
                  lambda c, pr: clampT(pr[0] + pr[1] * math.sqrt(area(c[0], c[1])))),
    # (d) fraction of initial boids
    'frac_N0':   (lambda: [(th,) for th in [x/100 for x in range(3, 25)]],
                  lambda c, pr: clampT(pr[0] * c[2])),
    # (e) density N0/A (boids per Mpx^2) scaled
    'density':   (lambda: [(a, b) for a in range(-2, 10) for b in [x/10 for x in range(0, 60, 2)]],
                  lambda c, pr: clampT(pr[0] + pr[1] * (c[2] / (area(c[0], c[1]) / 1e6)))),
    # (f) min-dimension (short axis)
    'minDim':    (lambda: [(a, b) for a in range(-4, 8) for b in [x/1000 for x in range(1, 40)]],
                  lambda c, pr: clampT(pr[0] + pr[1] * min(c[0], c[1]))),
    # combined: theta*N0 * (A/Aref)^q
    'frac_x_Aq': (lambda: [(th, q) for th in [x/100 for x in range(4, 18)] for q in [x/100 for x in range(0, 60, 5)]],
                  lambda c, pr: clampT(pr[0] * c[2] * (area(c[0], c[1]) / A_REF) ** pr[1])),
    # fixed baselines
    'fixed8':    (lambda: [(8,)], lambda c, pr: 8),
    'fixed5':    (lambda: [(5,)], lambda c, pr: 5),
}

def evaluate(tab, Tfn, params, cs, weights):
    """weighted achieved throughput + mean/worst gap vs per-cell optimal."""
    tot_w = sum(weights[c] for c in cs)
    ach = 0.0; gaps = []
    for c in cs:
        opt = max(thru(tab, c, T) for T in Ts_for(tab, c))
        got = thru(tab, c, Tfn(c, params))
        ach += weights[c] * got
        gaps.append((opt - got) / opt * 100)
    return ach / tot_w, sum(gaps) / len(gaps), max(gaps)

def main():
    tab = load(sys.argv[1:])
    cs = cells(tab)
    print(f"loaded {len(tab)} (W,H,N0,T) points over {len(cs)} (W,H,N0) cells")
    # approx real-world screen prevalence (StatCounter-ish; complexity bar = COMMON screens).
    # keyed by "WxH"; unlisted -> small default. side-b pins exact shares for the sealed set.
    SHARE = {
        '1920x1080': 22, '1366x768': 10, '1536x864': 9, '1440x900': 4, '2560x1440': 4,
        '1600x900': 3, '1280x720': 2, '1512x982': 3, '1680x1050': 2,
        '390x844': 6, '393x852': 5, '412x915': 5, '360x800': 5, '414x896': 4,
    }
    weights = {c: SHARE.get(f'{c[0]}x{c[1]}', 1) for c in cs}
    print(f"\n{'form':>11} {'params':>22} {'achThru(e-3)':>12} {'meanGap%':>9} {'worstGap%':>10}")
    results = {}
    for name, (grid, Tfn) in FORMS.items():
        best = None
        for pr in grid():
            ach, mean, worst = evaluate(tab, Tfn, pr, cs, weights)
            if best is None or mean < best[2]:   # minimize mean gap
                best = (pr, ach, mean, worst)
        results[name] = best
        pr, ach, mean, worst = best
        prs = '(' + ','.join(f'{x:.3f}' if isinstance(x, float) else str(x) for x in pr) + ')'
        print(f"{name:>11} {prs:>22} {ach*1e3:>12.4f} {mean:>9.2f} {worst:>10.2f}")
    # best non-fixed form: per-screen gain vs fixed-8 (the complexity-bar table)
    best_name = min((n for n in FORMS if not n.startswith('fixed')), key=lambda n: results[n][2])
    bpr = results[best_name][0]; bTfn = FORMS[best_name][1]
    print(f"\n=== per-screen: BEST form '{best_name}' {bpr} vs fixed-8 (gain = formula/fixed8 -1) ===")
    print(f"{'screen':>11} {'N0':>4} {'area(M)':>7} {'wt':>4} {'Tf':>3} {'fThru':>7} {'T8thru':>7} {'gain%':>7} {'optThru':>7} {'fGap%':>6}")
    tw = sum(weights.values()); wgain = 0.0
    for c in sorted(cs, key=lambda c: -weights[c]):
        Tf = bTfn(c, bpr); ff = thru(tab, c, Tf); t8 = thru(tab, c, 8)
        opt = max(thru(tab, c, T) for T in Ts_for(tab, c))
        g = (ff - t8) / t8 * 100; wgain += weights[c] * (ff - t8) / t8
        print(f"{c[0]}x{c[1]:>4} {c[2]:>4} {area(c[0],c[1])/1e6:>7.2f} {weights[c]:>4} {Tf:>3} {ff*1e3:>7.3f} {t8*1e3:>7.3f} {g:>+7.2f} {opt*1e3:>7.3f} {(opt-ff)/opt*100:>6.2f}")
    print(f"  prevalence-weighted gain of '{best_name}' vs fixed-8: {wgain/tw*100:+.2f}%")
    # per-cell T* table
    print(f"\n{'screen':>11} {'N0':>4} {'area(M)':>8} {'T*':>4}  throughput-by-T (T*=max)")
    for c in cs:
        Ts = Ts_for(tab, c); vals = [thru(tab, c, T) for T in Ts]
        tstar = Ts[vals.index(max(vals))]
        curve = ' '.join(f'{T}:{v*1e3:.2f}' for T, v in zip(Ts, vals))
        print(f"{c[0]}x{c[1]:>4} {c[2]:>4} {area(c[0],c[1])/1e6:>8.2f} {tstar:>4}  {curve}")

if __name__ == '__main__':
    main()

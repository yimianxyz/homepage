import json, sys, math, random

paths = sys.argv[1:] if len(sys.argv)>1 else ['/tmp/altsig_recs_1024.json','/tmp/altsig_recs_2560.json']
recs=[]
for p in paths:
    try: recs+=json.load(open(p))
    except Exception as e: print("skip",p,e)
N=len(recs)
dis=sum(1 for r in recs if not r['agree'])
print(f"=== combined plans={N} disagrees={dis} agree={1-dis/N:.4f} ===")
for r in recs:
    if r['margin'] is None: r['margin']=1e18
def fin(v): return v is not None and isinstance(v,(int,float)) and math.isfinite(v) and abs(v)<1e17

from collections import Counter
print("cells:",dict(Counter(r['cell'] for r in recs)))

hi=[r for r in recs if 1<=r['margin']<1e17]
print(f"margin>=1 band: n={len(hi)} disagreeRate={sum(1 for r in hi if not r['agree'])/max(1,len(hi)):.4f}")

FD=[('margin','higher'),('winSoftmaxProb','higher'),('winEntropy','lower'),
    ('ecArgmaxGap','lower'),('bootMargin','higher'),('scoreVsBoot','higher')]

def share1d(pool,f,d,ref):
    bad=[r[f] for r in pool if not r['agree'] and fin(r[f])]
    if any(not fin(r[f]) for r in pool if not r['agree']): return 0.0,None
    if d=='higher':
        thr=max(bad) if bad else -1e18; t=[r for r in pool if fin(r[f]) and r[f]>thr]
    else:
        thr=min(bad) if bad else 1e18; t=[r for r in pool if fin(r[f]) and r[f]<thr]
    return len(t)/len(ref),thr

print("\n[1D IN-SAMPLE NN-share, 0 disagree, frac of ALL]:")
for f,d in sorted(FD,key=lambda x:-share1d(recs,x[0],x[1],recs)[0]):
    s,thr=share1d(recs,f,d,recs); print(f"  {f:16s}({d:6s}) NN-share={s:.6f}")

# 2D fast: subsampled margin grid
def best2d(field,dirn):
    ms=sorted(set(r['margin'] for r in recs))
    grid=[ms[min(len(ms)-1,int(q*(len(ms)-1)))] for q in [i/200 for i in range(201)]]
    best=0.0;cfg=None
    for m in sorted(set(grid)):
        pool=[r for r in recs if r['margin']>=m]
        if not pool: continue
        bad=[r[field] for r in pool if not r['agree'] and fin(r[field])]
        if any(not fin(r[field]) for r in pool if not r['agree']): continue
        if dirn=='higher':
            thr=max(bad) if bad else -1e18; t=[r for r in pool if fin(r[field]) and r[field]>thr]
        else:
            thr=min(bad) if bad else 1e18; t=[r for r in pool if fin(r[field]) and r[field]<thr]
        if len(t)/N>best: best=len(t)/N;cfg=(round(m,4),round(thr,4),len(t))
    return best,cfg
print("\n[2D IN-SAMPLE margin&altsig NN-share (subsampled grid)]:")
for f,d in [('winSoftmaxProb','higher'),('winEntropy','lower'),('ecArgmaxGap','lower'),('bootMargin','higher'),('scoreVsBoot','higher')]:
    b,c=best2d(f,d); print(f"  margin&{f:16s} NN-share={b:.6f} cfg={c}")

# ---- HELD-OUT (the overfitting test) ----
random.seed(7); idx=list(range(N)); random.shuffle(idx); h=N//2
calib=[recs[i] for i in idx[:h]]; test=[recs[i] for i in idx[h:]]
def heldout(field,dirn,mfloor=0.0):
    cp=[r for r in calib if r['margin']>=mfloor]
    bad=[r[field] for r in cp if not r['agree'] and fin(r[field])]
    if any(not fin(r[field]) for r in cp if not r['agree']): return None
    if dirn=='higher':
        thr=max(bad) if bad else -1e18; tt=[r for r in test if r['margin']>=mfloor and fin(r[field]) and r[field]>thr]
    else:
        thr=min(bad) if bad else 1e18; tt=[r for r in test if r['margin']>=mfloor and fin(r[field]) and r[field]<thr]
    td=sum(1 for r in tt if not r['agree'])
    return dict(thr=round(thr,4),test_frac=round(len(tt)/len(test),6),test_n=len(tt),test_disagree=td)
print("\n[HELD-OUT: threshold from calib half, eval on test half. disagree>0 => overfit]")
for f,d in FD:
    print(f"  1D {f:16s}({d:6s}): {heldout(f,d)}")
# 2D held-out: best calib margin floor
for f,d in [('winSoftmaxProb','higher'),('winEntropy','lower'),('bootMargin','higher')]:
    ms=sorted(set(r['margin'] for r in calib))
    bestm,bf=0.0,-1
    for m in [ms[min(len(ms)-1,int(q*(len(ms)-1)))] for q in [i/100 for i in range(101)]]:
        cp=[r for r in calib if r['margin']>=m]
        if not cp: continue
        bad=[r[f] for r in cp if not r['agree'] and fin(r[f])]
        if any(not fin(r[f]) for r in cp if not r['agree']): continue
        if d=='higher':
            thr=max(bad) if bad else -1e18; t=[r for r in cp if fin(r[f]) and r[f]>thr]
        else:
            thr=min(bad) if bad else 1e18; t=[r for r in cp if fin(r[f]) and r[f]<thr]
        if len(t)/len(calib)>bf: bf=len(t)/len(calib);bestm=m
    print(f"  2D margin&{f:16s}({d:6s}) calibFloor={bestm:.4g}: {heldout(f,d,bestm)}")

# ---- MULTI-SIGNAL held-out (richest construction) ----
sigs=[('margin','higher'),('bootMargin','higher'),('winSoftmaxProb','higher'),('winEntropy','lower'),('ecArgmaxGap','lower')]
def passes(r,thrs):
    for (f,d),t in zip(sigs,thrs):
        if not fin(r[f]): return False
        if d=='higher' and not r[f]>=t: return False
        if d=='lower' and not r[f]<=t: return False
    return True
def pcts(field,dirn):
    vals=sorted(v for v in (r[field] for r in calib) if fin(v))
    if not vals: return [-1e18 if dirn=='higher' else 1e18]
    qs=[0.0,0.5,0.8,0.9,0.95,0.98,0.99]
    out=[(vals[min(len(vals)-1,int(q*(len(vals)-1)))] if dirn=='higher' else vals[len(vals)-1-min(len(vals)-1,int(q*(len(vals)-1)))]) for q in qs]
    out.append(-1e18 if dirn=='higher' else 1e18)
    return sorted(set(out))
grids=[pcts(f,d) for f,d in sigs]
best=None
for tm in grids[0]:
  for tb in grids[1]:
    for tp in grids[2][::2]:
      for te in grids[3][::2]:
        for tg in grids[4][::2]:
          thrs=(tm,tb,tp,te,tg)
          ct=[r for r in calib if passes(r,thrs)]
          if not ct or any(not r['agree'] for r in ct): continue
          frac=len(ct)/len(calib)
          if best is None or frac>best[0]:
            tt=[r for r in test if passes(r,thrs)]
            td=sum(1 for r in tt if not r['agree'])
            best=(frac,thrs,len(ct),len(tt)/len(test),len(tt),td)
print("\n[MULTI-SIGNAL held-out: max calib coverage at 0 calib-disagree, eval on test]")
if best:
    frac,thrs,cn,tf,tn,td=best
    nm=['margin','bootMargin','prob','entropy','ecGap']
    print(f"  calib_frac={frac:.5f}(n={cn}) thr={dict(zip(nm,[round(x,4) for x in thrs]))}")
    print(f"  -> TEST frac={tf:.5f}(n={tn}) TEST_disagree={td}  {'<<OVERFIT' if td>0 else '(0 on held-out)'}")
else:
    print("  no gate")

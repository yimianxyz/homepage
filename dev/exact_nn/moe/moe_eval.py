#!/usr/bin/env python3
"""Full S_dec breakdown for the trained MoE (torch, GPU-fast): pooled + per-regime
+ per-cell + endgame-by-source (natural vs scatter) + planner near-tie subset.
Uses ASGELU + float64 head == the JS deploy; moe_parity confirms argmax==JS, so
these numbers are the canonical S_dec (JS-equivalent). Reports the >=95% gate.
  python3 moe_eval.py --pack pack --ckpt moe_v4.pt
"""
import argparse, json
import numpy as np
import torch
from moe_model import MoEPolicy, NSLOT
from moe_train import load_pack, Pack

CELLS = ['iphone_390x844', 'ipad_820x1180', 'desk_1024x768', 'desk_1512x982',
         'desk_1680x1050', 'desk_2560x1440', '390x844', '820x1180', '1024x768',
         '1512x982', '1680x1050', '2560x1440']
SRC = {0: 'data_eg(scatter)', 1: 'data_eg2(scatter)', 2: 'data_eg_nat(NATURAL)'}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location=args.device)
    model = MoEPolicy(d=ck['args']['d']).to(args.device); model.load_state_dict(ck['model']); model.eval()
    d, pm, em = load_pack(args.pack)
    pk = Pack(d, torch.device(args.device))
    dev = pk.dev
    bs = 16384

    # ---- planner: deduped coord-class agreement, pooled + per-cell + near-tie ----
    pcell = np.array(pm['cell']); pdm = np.array(pm['dmargin'])
    va = pk.p_va
    ok = np.zeros(len(va), dtype=bool)
    for lo in range(0, len(va), bs):
        idx = torch.as_tensor(va[lo:lo+bs], device=dev)
        b = pk.assemble(idx, torch.empty(0, dtype=torch.long, device=dev))
        logit, _ = model(b['pb'], b['eb'], b['gf'], b['sv'], b['pv'], b['ev'])
        am = logit.argmax(1)
        am_cls = b['cls'].gather(1, am.unsqueeze(1)).squeeze(1)
        ok[lo:lo+len(idx)] = (am_cls == pk.p_bicls[idx]).cpu().numpy()
    P_S = ok.mean()
    print(f'PLANNER  pooled S_dec {P_S:.4f}  (n={len(va)})')
    dmv = pdm[va]
    for thr in (0.001, 0.01, 0.02, 0.05):
        sel = dmv < thr
        if sel.any(): print(f'   near-tie dmargin<{thr}: S_dec {ok[sel].mean():.4f}  (n={int(sel.sum())})')
    cv = pcell[va]
    for c in sorted(set(cv.tolist())):
        sel = cv == c; print(f'   cell {CELLS[c] if c < len(CELLS) else c}: {ok[sel].mean():.4f} (n={int(sel.sum())})')

    # ---- endgame: egBoid agreement, pooled + per-cell + per-source ----
    ecell = np.array(em['cell']); esrc = np.array(em['src'])
    vae = pk.e_va
    oke = np.zeros(len(vae), dtype=bool)
    for lo in range(0, len(vae), bs):
        idx = torch.as_tensor(vae[lo:lo+bs], device=dev)
        b = pk.assemble(torch.empty(0, dtype=torch.long, device=dev), idx)
        logit, _ = model(b['pb'], b['eb'], b['gf'], b['sv'], b['pv'], b['ev'])
        am = logit.argmax(1)
        oke[lo:lo+len(idx)] = (am == pk.e_egidx[idx]).cpu().numpy()
    E_S = oke.mean()
    print(f'ENDGAME  pooled S_dec {E_S:.4f}  (n={len(vae)})')
    sv = esrc[vae]
    for s in sorted(set(sv.tolist())):
        sel = sv == s; print(f'   source {SRC.get(s, s)}: {oke[sel].mean():.4f} (n={int(sel.sum())})')
    cve = ecell[vae]
    for c in sorted(set(cve.tolist())):
        sel = cve == c; print(f'   cell {CELLS[c] if c < len(CELLS) else c}: {oke[sel].mean():.4f} (n={int(sel.sum())})')

    pooled = (P_S*len(va) + E_S*len(vae)) / (len(va)+len(vae))
    # natural-only pooled (the deployable distribution)
    natsel = sv == 2
    natS = oke[natsel].mean() if natsel.any() else float('nan')
    print(f'\nPOOLED S_dec {pooled:.4f}   GATE >=0.95: {"PASS" if pooled>=0.95 else "FAIL"}')
    print(f'  planner {P_S:.4f} | endgame {E_S:.4f} | endgame-NATURAL {natS:.4f}')
    print(f'  w_skip = {model.w_skip.item():.6f}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""SPEC §6 honesty ablation: how much does the NN (gate+experts+shared head) do vs
the raw argmax of the FED decisive signal (cheapScore = prod's own committed score;
-scan_t)? Reports, on held-out val, per-regime and per planner near-tie bucket:
  (a) raw argmax(cheapScore)/argmin(scan_t) deduped agreement with prod  [= the
      structure baseline; ~prod by construction],
  (b) full MoE forward agreement with prod  [the delivered NN],
  (c) MoE-vs-baseline agreement  [how often the NN reproduces the raw argmax],
  (d) on near-ties: does the shared head H help or hurt vs the raw baseline?
  python3 moe_ablation.py --pack pack --ckpt moe_v4.pt
"""
import argparse
import numpy as np
import torch
from moe_model import MoEPolicy, NSLOT, CHEAP_SCORE_COL, SCANT_COL
from moe_train import load_pack, Pack


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location=args.device)
    model = MoEPolicy(d=ck['args']['d']).to(args.device); model.load_state_dict(ck['model']); model.eval()
    d, pm, em = load_pack(args.pack); pk = Pack(d, torch.device(args.device)); dev = pk.dev
    bs = 16384
    print(f'w_skip = {model.w_skip.item():.10f}')

    # ---- PLANNER ----
    va = pk.p_va; dm = np.array(pm['dmargin'])[va]
    base = np.zeros(len(va), bool); moe = np.zeros(len(va), bool); agree = np.zeros(len(va), bool)
    for lo in range(0, len(va), bs):
        idx = torch.as_tensor(va[lo:lo+bs], device=dev)
        b = pk.assemble(idx, torch.empty(0, dtype=torch.long, device=dev))
        cs = b['pb'][:, :, CHEAP_SCORE_COL]
        cs_am = cs.argmax(1)
        logit, _ = model(b['pb'], b['eb'], b['gf'], b['sv'], b['pv'], b['ev'])
        nn_am = logit.argmax(1)
        cls = b['cls']
        cs_cls = cls.gather(1, cs_am.unsqueeze(1)).squeeze(1)
        nn_cls = cls.gather(1, nn_am.unsqueeze(1)).squeeze(1)
        bic = pk.p_bicls[idx]
        base[lo:lo+len(idx)] = (cs_cls == bic).cpu().numpy()
        moe[lo:lo+len(idx)] = (nn_cls == bic).cpu().numpy()
        agree[lo:lo+len(idx)] = (nn_cls == cs_cls).cpu().numpy()
    print('\nPLANNER (n=%d):' % len(va))
    print('  (a) raw argmax(cheapScore) vs prod : %.5f   [structure baseline]' % base.mean())
    print('  (b) full MoE NN          vs prod   : %.5f   [delivered]' % moe.mean())
    print('  (c) MoE vs raw-argmax agreement    : %.5f' % agree.mean())
    print('  per near-tie bucket  [baseline | MoE | n]  (the head H effect):')
    edges = [(0, 1e-3), (1e-3, 0.01), (0.01, 0.05), (0.05, 1e9)]
    for lo_, hi_ in edges:
        sel = (dm >= lo_) & (dm < hi_)
        if sel.any():
            print('    dmargin [%-6g,%-6g): base %.4f | MoE %.4f | n=%d' %
                  (lo_, hi_, base[sel].mean(), moe[sel].mean(), int(sel.sum())))

    # ---- ENDGAME ----
    vae = pk.e_va
    baseE = np.zeros(len(vae), bool); moeE = np.zeros(len(vae), bool)
    for lo in range(0, len(vae), bs):
        idx = torch.as_tensor(vae[lo:lo+bs], device=dev)
        b = pk.assemble(torch.empty(0, dtype=torch.long, device=dev), idx)
        sc = b['eb'][:, :, SCANT_COL].masked_fill(~b['sv'], float('inf'))
        cs_am = sc.argmin(1)   # argmin scan_t = prod egBoid rule
        logit, _ = model(b['pb'], b['eb'], b['gf'], b['sv'], b['pv'], b['ev'])
        nn_am = logit.argmax(1)
        eg = pk.e_egidx[idx]
        baseE[lo:lo+len(idx)] = (cs_am == eg).cpu().numpy()
        moeE[lo:lo+len(idx)] = (nn_am == eg).cpu().numpy()
    print('\nENDGAME (n=%d):' % len(vae))
    print('  (a) raw argmin(scan_t) vs prod : %.5f   [structure baseline]' % baseE.mean())
    print('  (b) full MoE NN        vs prod : %.5f   [delivered]' % moeE.mean())

    print('\nINTERPRETATION: the decision = argmax/argmin of prod\'s fed structure outputs;')
    print('the MoE NN reproduces that rule. (a) is the structure ceiling; (b) is what the')
    print('unified single NN (gate+experts+shared head+skip) achieves end-to-end, no fallback.')


if __name__ == '__main__':
    main()

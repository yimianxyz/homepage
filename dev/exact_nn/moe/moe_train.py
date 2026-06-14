#!/usr/bin/env python3
"""Train the Phase-2 single MoE policy (SPEC_PHASE2_MOE §4).

  Phase A (pretrain experts, gate forced to regime): E_p+H on planner records,
           E_e+H on endgame records, in unified mixed batches (H shared, no
           forgetting). + gate BCE so the gate learns to route from gate_feat.
  Phase B (joint): real learned gate, lower LR, fine-tune everything end-to-end.

Loss per regime = CE(committed slot) + reg(logit -> prod score / -scan_t).
Selection metric = S_dec (deduped target-coord agreement / egBoid agreement) on
the held-out (by-seed) val split. Exports a float64 weights JSON for moePolicy.js.

  python3 moe_train.py --pack pack --stepsA 12000 --stepsB 6000 --out moe_weights.json
"""
import argparse, json, time, os
import numpy as np
import torch
import torch.nn.functional as Fn
from moe_model import MoEPolicy, PLANNER_DIM, ENDGAME_DIM, NSLOT, n_params

TMAX = 1400.0


def load_bin(p, dt):
    return np.fromfile(p, dtype=dt)


def load_pack(pack):
    pm = json.load(open(f'{pack}/planner_meta.json'))
    em = json.load(open(f'{pack}/endgame_meta.json'))
    P, E = pm['P'], em['E']
    d = dict(
        p_slots=load_bin(f'{pack}/planner_slots.f32', np.float32).reshape(P, NSLOT, PLANNER_DIM),
        p_score=load_bin(f'{pack}/planner_score.f32', np.float32).reshape(P, NSLOT),
        p_cls=load_bin(f'{pack}/planner_cls.i32', np.int32).reshape(P, NSLOT),
        p_bicls=load_bin(f'{pack}/planner_bicls.i32', np.int32),
        p_gate=load_bin(f'{pack}/planner_gate.f32', np.float32).reshape(P, -1),
        p_split=np.array(pm['split']), p_N=np.array(pm['N']),
        p_dmargin=np.array(pm['dmargin']), p_cell=np.array(pm['cell']),
        e_slots=load_bin(f'{pack}/endgame_slots.f32', np.float32).reshape(E, NSLOT, ENDGAME_DIM),
        e_scant=load_bin(f'{pack}/endgame_scant.f32', np.float32).reshape(E, NSLOT),
        e_valid=load_bin(f'{pack}/endgame_valid.i32', np.int32).reshape(E, NSLOT),
        e_egidx=load_bin(f'{pack}/endgame_egidx.i32', np.int32),
        e_gate=load_bin(f'{pack}/endgame_gate.f32', np.float32).reshape(E, -1),
        e_split=np.array(em['split']), e_N=np.array(em['N']),
        e_egmargin=np.array(em['egmargin']), e_cell=np.array(em['cell']),
        e_src=np.array(em['src']),
    )
    return d, pm, em


class Pack:
    """Holds GPU/CPU tensors + assembles unified mixed batches (both expert blocks,
    off-regime block zeroed) — the same single-forward shape the deploy uses."""
    def __init__(self, d, dev):
        self.dev = dev
        T = lambda a, t=torch.float32: torch.as_tensor(a, dtype=t, device=dev)
        # planner
        self.p_slots = T(d['p_slots']); self.p_score = T(d['p_score'])
        self.p_cls = T(d['p_cls'], torch.long); self.p_bicls = T(d['p_bicls'], torch.long)
        self.p_gate = T(d['p_gate'])
        self.p_tr = np.where(d['p_split'] == 0)[0]; self.p_va = np.where(d['p_split'] == 1)[0]
        # endgame
        self.e_slots = T(d['e_slots']); self.e_scant = T(d['e_scant'])
        self.e_valid = T(d['e_valid']) > 0.5; self.e_egidx = T(d['e_egidx'], torch.long)
        self.e_gate = T(d['e_gate'])
        self.e_tr = np.where(d['e_split'] == 0)[0]; self.e_va = np.where(d['e_split'] == 1)[0]
        self.GD = self.p_gate.shape[1]

    def batch(self, np_, ne, gen, split='tr'):
        pidx = self._draw(self.p_tr if split == 'tr' else self.p_va, np_, gen)
        eidx = self._draw(self.e_tr if split == 'tr' else self.e_va, ne, gen)
        return self.assemble(pidx, eidx)

    def _draw(self, pool, k, gen):
        if k <= 0:
            return torch.empty(0, dtype=torch.long, device=self.dev)
        ix = torch.randint(0, len(pool), (k,), generator=gen)
        return torch.as_tensor(pool, device=self.dev)[ix.to(self.dev)]

    def assemble(self, pidx, eidx):
        np_, ne = len(pidx), len(eidx)
        B = np_ + ne
        pb = torch.zeros(B, NSLOT, PLANNER_DIM, device=self.dev)
        eb = torch.zeros(B, NSLOT, ENDGAME_DIM, device=self.dev)
        gf = torch.zeros(B, self.GD, device=self.dev)
        pv = torch.zeros(B, NSLOT, dtype=torch.bool, device=self.dev)
        ev = torch.zeros(B, NSLOT, dtype=torch.bool, device=self.dev)
        sv = torch.zeros(B, NSLOT, dtype=torch.bool, device=self.dev)
        gt = torch.zeros(B, device=self.dev)                 # gate target
        label = torch.zeros(B, dtype=torch.long, device=self.dev)
        regt = torch.full((B, NSLOT), float('nan'), device=self.dev)  # reg target
        cls = torch.full((B, NSLOT), -1, dtype=torch.long, device=self.dev)
        is_p = torch.zeros(B, dtype=torch.bool, device=self.dev)
        if np_:
            pb[:np_] = self.p_slots[pidx]
            gf[:np_] = self.p_gate[pidx]
            pv[:np_] = True; sv[:np_] = True; gt[:np_] = 1.0
            label[:np_] = self.p_bicls[pidx]
            regt[:np_] = self.p_score[pidx]
            cls[:np_] = self.p_cls[pidx]
            is_p[:np_] = True
        if ne:
            eb[np_:] = self.e_slots[eidx]
            gf[np_:] = self.e_gate[eidx]
            ev[np_:] = self.e_valid[eidx]; sv[np_:] = self.e_valid[eidx]; gt[np_:] = 0.0
            label[np_:] = self.e_egidx[eidx]
            sc = self.e_scant[eidx].clamp(max=TMAX)
            regt[np_:] = torch.where(self.e_valid[eidx], -sc / 100.0, torch.full_like(sc, float('nan')))
        return dict(pb=pb, eb=eb, gf=gf, pv=pv, ev=ev, sv=sv, gt=gt, label=label,
                    regt=regt, cls=cls, is_p=is_p, np_=np_, ne=ne)


def step_loss(model, b, g_mode, w_reg, w_ce, w_gate):
    g_override = b['gt'] if g_mode == 'force' else None
    logit, g = model(b['pb'], b['eb'], b['gf'], b['sv'], b['pv'], b['ev'], g_override=g_override)
    label, regt, sv = b['label'], b['regt'], b['sv']
    ce = Fn.cross_entropy(logit.float(), label)
    fin = torch.isfinite(regt) & sv
    reg = ((logit[fin] - regt[fin]) ** 2).mean() if fin.any() else logit.sum() * 0.0
    # gate BCE from gate_feat (train the router even when g forced)
    graw = model.gate(b['gf']).squeeze(-1)
    gate_l = Fn.binary_cross_entropy(graw.clamp(1e-6, 1 - 1e-6), b['gt'])
    return w_ce * ce + w_reg * reg + w_gate * gate_l, ce.item(), reg.item(), gate_l.item()


@torch.no_grad()
def evaluate(model, pk, bs=16384):
    model.eval()
    out = {}
    # planner S_dec (deduped coord-class agreement)
    for split, pool in (('val', pk.p_va), ('train', pk.p_tr)):
        if len(pool) == 0:
            continue
        ok = tot = 0
        for lo in range(0, len(pool), bs):
            idx = torch.as_tensor(pool[lo:lo + bs], device=pk.dev)
            b = pk.assemble(idx, torch.empty(0, dtype=torch.long, device=pk.dev))
            logit, _ = model(b['pb'], b['eb'], b['gf'], b['sv'], b['pv'], b['ev'])
            am = logit.argmax(1)
            cls = b['cls']
            am_cls = cls.gather(1, am.unsqueeze(1)).squeeze(1)
            ok += int((am_cls == pk.p_bicls[idx]).sum()); tot += len(idx)
        out[f'planner_{split}'] = ok / max(tot, 1)
    # endgame S_dec (egBoid agreement), pooled + natural-only (src==2 in val)
    for split, pool in (('val', pk.e_va), ('train', pk.e_tr)):
        if len(pool) == 0:
            continue
        ok = tot = 0
        for lo in range(0, len(pool), bs):
            idx = torch.as_tensor(pool[lo:lo + bs], device=pk.dev)
            b = pk.assemble(torch.empty(0, dtype=torch.long, device=pk.dev), idx)
            logit, _ = model(b['pb'], b['eb'], b['gf'], b['sv'], b['pv'], b['ev'])
            am = logit.argmax(1)
            ok += int((am == pk.e_egidx[idx]).sum()); tot += len(idx)
        out[f'endgame_{split}'] = ok / max(tot, 1)
    # gate routing accuracy on val
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', required=True)
    ap.add_argument('--stepsA', type=int, default=12000)
    ap.add_argument('--stepsB', type=int, default=6000)
    ap.add_argument('--bs_p', type=int, default=512)
    ap.add_argument('--bs_e', type=int, default=512)
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--lrA', type=float, default=2e-3)
    ap.add_argument('--lrB', type=float, default=3e-4)
    ap.add_argument('--w_reg', type=float, default=1.0)
    ap.add_argument('--w_ce', type=float, default=0.1)
    ap.add_argument('--w_gate', type=float, default=0.5)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='moe_weights.json')
    ap.add_argument('--ckpt', default='moe_ckpt.pt')
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    print('[moe] loading pack...', flush=True)
    d, pm, em = load_pack(args.pack)
    pk = Pack(d, dev)
    print(f'[moe] planner {len(pk.p_tr)}tr/{len(pk.p_va)}va  endgame {len(pk.e_tr)}tr/{len(pk.e_va)}va', flush=True)
    model = MoEPolicy(d=args.d).to(dev)
    print(f'[moe] params {n_params(model)}', flush=True)
    gen = torch.Generator().manual_seed(args.seed + 1)

    def run(phase, steps, lr, g_mode):
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        t0 = time.time()
        for s in range(steps):
            b = pk.batch(args.bs_p, args.bs_e, gen, 'tr')
            loss, ce, reg, gl = step_loss(model, b, g_mode, args.w_reg, args.w_ce, args.w_gate)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if (s + 1) % 1000 == 0 or s + 1 == steps:
                ev = evaluate(model, pk)
                gp = model.gate(pk.p_gate[pk.p_va[:2048]]).mean().item() if len(pk.p_va) else float('nan')
                ge = model.gate(pk.e_gate[pk.e_va[:2048]]).mean().item() if len(pk.e_va) else float('nan')
                print('[%s %5d] loss %.4f ce %.3f reg %.4f gl %.4f | Sdec planner %.4f endgame %.4f | gate p%.3f e%.3f | %.0fs'
                      % (phase, s + 1, loss.item(), ce, reg, gl, ev.get('planner_val', -1),
                         ev.get('endgame_val', -1), gp, ge, time.time() - t0), flush=True)

    print('[moe] === Phase A: pretrain experts (gate forced) ===', flush=True)
    run('A', args.stepsA, args.lrA, 'force')
    print('[moe] === Phase B: joint (learned gate) ===', flush=True)
    run('B', args.stepsB, args.lrB, 'learn')

    ev = evaluate(model, pk)
    print('[moe] FINAL', ev, flush=True)
    export(model, args, ev)


def export(model, args, ev):
    sd = model.state_dict()
    weights = {k: v.double().cpu().tolist() for k, v in sd.items()}
    json.dump({'arch': 'moe', 'd': args.d, 'keys': list(sd.keys()),
               'shapes': {k: list(v.shape) for k, v in sd.items()},
               'weights': weights, 'metrics': ev}, open(args.out, 'w'))
    torch.save({'model': sd, 'args': vars(args), 'metrics': ev}, args.ckpt)
    print('[moe] exported ->', args.out, flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""One training entry point for the EXACT-NN L1 student tasks (SPEC section 4):

  --task l1r   regress the 4 rolled scores (primary rung). Inputs include
               feat + vprior (both exact at deploy time); the model output is
               gathered at pidx and MSE'd against rolled[4]. Agreement metric
               reconstructs the final score vector (exact vprior + student
               rolled scores) and compares the deduped argmax to bi.
  --task l1rs  split head: catch classification (CE, 24 classes) + boot MSE;
               deploy score = argmax(catches)+boot.
  --task l1rs2 side-a v2a, the BOOT-focused student. Same split head, but loss =
               w_ce·CE(catch) + w_boot·Huber(boot) + w_rank·pairwise-score-margin
               ranking over the 4 rolled (combined expected_catch+boot, mean-
               centered → matches prod's decisive score DIFFERENCES). Deploy score
               = EXPECTED catch + boot. Targets the ~69%-of-decisions boot lever.
               Tune --w-boot/--w-rank/--w-ce/--huber-delta.
  --task l1s   regress all 16 final scores; MSE over (B,16).
  --task l1p   pointer classification over the 16 candidates. Ties are real
               (E3D-padded duplicate candidates, SPEC section 3): candidates
               with bitwise-equal coordinates form one equivalence class, so
               the CE pools member logits per class (logsumexp) and targets
               the CLASS of bi; the agreement metric is class-level too.

Checkpointing every --ckpt-every steps (atomic, includes optimizer + sampler
RNG); --resume continues from the checkpoint. --bench reports steady-state
samples/sec (post-warmup, cuda-synchronized) and appends a JSON line to
--results for the REPORT table.

Example:
  python3 train.py --task l1r --arch deepset --size small \
      --data data/ --steps 400 --results bench.jsonl
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

import ds
import pack_oracle
from models import build_model, n_params

INPUT_KEYS = ('boid_tok', 'bmask', 'glob', 'cand_tok')
CATCH_CLASSES = 24   # l1rs catches head: classes 0..23 (max observed 13; slow-predator 90-step rollout won't exceed)
DEFAULT_W = {'ce': 1.0, 'boot': 5.0, 'rank': 2.0, 'huber_delta': 0.2}  # l1rs2 loss weights


def expected_catch(catch_logits):
    """(B,K,CATCH_CLASSES) -> (B,K) expected catch-count = Σ c·softmax_c. A smooth,
    differentiable stand-in for argmax(catches) — used as the l1rs2 deploy score AND
    in the ranking loss, so train and deploy use the SAME catch reduction (no
    argmax train/deploy mismatch). Confident (peaked) predictions ≈ argmax anyway."""
    p = catch_logits.float().softmax(-1)
    cc = torch.arange(CATCH_CLASSES, device=catch_logits.device, dtype=torch.float32)
    return (p * cc).sum(-1)


def class_pool_logits(logits, cls):
    """(B,16) member logits + (B,16) class ids -> (B,16) class logits at the
    canonical indices (logsumexp over members; ~-1e9 where class empty)."""
    B, K = logits.shape
    M = torch.full((B, K, K), -1e9, device=logits.device, dtype=logits.dtype)
    M = M.scatter(2, cls.unsqueeze(2), logits.unsqueeze(2))
    return torch.logsumexp(M, dim=1)


def class_pool_max(scores, cls):
    """(B,16) scores + (B,16) class ids -> (B,16) per-class MAX at the canonical
    index (−inf where no member maps). The deduped decision is argmax over this
    (SPEC §3). Used by the l1rs2 confident-wrong hinge."""
    B, K = scores.shape
    M = torch.full((B, K), float('-inf'), device=scores.device, dtype=scores.dtype)
    return M.scatter_reduce(1, cls, scores, reduce='amax', include_self=True)


def student_final_score(task, out, batch):
    """The student's implied 16-score vector (float64) for argmax agreement."""
    if task == 'l1r':
        s = batch['vprior'].clone()
        s.scatter_(1, batch['pidx'], out.double())
        return s
    if task == 'l1rs':
        idx = batch['pidx'].unsqueeze(2).expand(-1, -1, out.shape[2])
        g = out.gather(1, idx)                               # (B,4,CATCH_CLASSES+1)
        pred = g[..., :CATCH_CLASSES].argmax(-1).double() + g[..., CATCH_CLASSES].double()
        s = batch['vprior'].clone()
        s.scatter_(1, batch['pidx'], pred)
        return s
    if task == 'l1rs2':
        idx = batch['pidx'].unsqueeze(2).expand(-1, -1, out.shape[2])
        g = out.gather(1, idx)                               # (B,4,CATCH_CLASSES+1)
        # deploy score = EXPECTED catch + boot (smooth; matches the ranking loss).
        pred = expected_catch(g[..., :CATCH_CLASSES]).double() + g[..., CATCH_CLASSES].double()
        s = batch['vprior'].clone()
        s.scatter_(1, batch['pidx'], pred)
        return s
    return out.double()


def masked_mse(pred, target):
    """MSE over FINITE target entries only. Real labels contain -Infinity via
    prod's extermination path (rollout kills all boids -> bootstrap = -inf;
    arrives as JSON null); a regression target of -inf is meaningless, so it
    is masked out of the loss. How real training should treat these plans
    (mask / floor-clamp / separate catches+bootstrap labels) is an open
    oracle schema question (REPORT.md)."""
    t = target.to(pred.dtype)
    m = torch.isfinite(t)
    if bool(m.all()):
        return F.mse_loss(pred, t)
    d = (pred - t)[m]
    return (d * d).mean()


def task_loss(task, out, batch, wcfg=None):
    if task == 'l1r':
        pred = out.gather(1, batch['pidx'])
        return masked_mse(pred, batch['rolled'])
    if task == 'l1rs':
        idx = batch['pidx'].unsqueeze(2).expand(-1, -1, out.shape[2])
        g = out.gather(1, idx)                               # (B,4,CATCH_CLASSES+1)
        c = batch['catches']
        assert int(c.max()) < CATCH_CLASSES, 'catches %d >= CATCH_CLASSES %d -- raise it' % (int(c.max()), CATCH_CLASSES)
        ce = F.cross_entropy(g[..., :CATCH_CLASSES].reshape(-1, CATCH_CLASSES).float(), c.reshape(-1))
        return ce + masked_mse(g[..., CATCH_CLASSES], batch['boot'])
    if task == 'l1rs2':
        # side-a v2a: ~69% of prod decisions are decided by the BOOT difference
        # between two equal-catch rolled candidates (median margin 0.019), so boot
        # precision is the dominant lever. Loss = catch CE + heavily-weighted boot
        # Huber (robust to the rare high-boot tail) + a pairwise score-margin
        # RANKING loss on the combined (expected_catch + boot) among the 4 rolled,
        # which directly trains the deduped-argmax + margin the deploy commits on
        # (relative ordering, mean-removed → matches prod's score DIFFERENCES, the
        # thing the margin gate reads). The boot Huber anchors the absolute level
        # (needed for rolled-vs-vprior / vprior-top decisions, ~19%).
        w = wcfg or DEFAULT_W
        idx = batch['pidx'].unsqueeze(2).expand(-1, -1, out.shape[2])
        g = out.gather(1, idx)                               # (B,4,CATCH_CLASSES+1)
        c = batch['catches']
        assert int(c.max()) < CATCH_CLASSES, 'catches %d >= CATCH_CLASSES %d -- raise it' % (int(c.max()), CATCH_CLASSES)
        ce = F.cross_entropy(g[..., :CATCH_CLASSES].reshape(-1, CATCH_CLASSES).float(), c.reshape(-1))
        boot_pred = g[..., CATCH_CLASSES].float()
        boot_tgt = batch['boot'].float()
        bm = torch.isfinite(boot_tgt)
        boot_h = F.huber_loss(boot_pred[bm], boot_tgt[bm], delta=w['huber_delta']) if bool(bm.any()) else boot_pred.sum() * 0.0
        # pairwise score-margin ranking over the 4 rolled (mean-centered MSE ==
        # all-pairs squared-diff loss up to a constant): match prod's score gaps.
        pred_s = expected_catch(g[..., :CATCH_CLASSES]) + boot_pred           # (B,4)
        true_s = c.float() + torch.where(bm, boot_tgt, torch.zeros_like(boot_tgt))
        valid = bm.float()                                                    # exclude -inf-boot rolled from the mean
        vc = valid.sum(1, keepdim=True).clamp(min=1.0)
        pred_c = pred_s - (pred_s * valid).sum(1, keepdim=True) / vc
        true_c = true_s - (true_s * valid).sum(1, keepdim=True) / vc
        rank = (((pred_c - true_c) ** 2) * valid).sum() / valid.sum().clamp(min=1.0)
        loss = w['ce'] * ce + w['boot'] * boot_h + w['rank'] * rank
        # confident-wrong hinge (--w-clean): the deduped decision is argmax over
        # per-class-max student scores; penalize any wrong-coord class outscoring
        # prod's true-winner class. Pushes the student to NOT spread scores (large
        # margin) unless the winner is genuinely on top → cleaner high-confidence
        # band (the lever for a non-zero 0-mismatch NN-share). 0 by default.
        if w.get('clean', 0.0) > 0.0:
            full_s = batch['vprior'].float().clone()
            full_s.scatter_(1, batch['pidx'], pred_s.to(full_s.dtype))
            cm = class_pool_max(full_s, batch['cls'])                 # (B,16) per-class max
            bic = batch['bi_cls']
            win = cm.gather(1, bic.unsqueeze(1)).squeeze(1)           # true-winner class score
            other = cm.scatter(1, bic.unsqueeze(1), float('-inf')).max(1).values
            clean = F.relu(other - win).mean()                       # hinge: winner must top all
            loss = loss + w['clean'] * clean
        return loss
    if task == 'l1s':
        return masked_mse(out, batch['score'])
    if task == 'l1p':
        cl = class_pool_logits(out, batch['cls'])
        return F.cross_entropy(cl, batch['bi_cls'])
    raise ValueError(task)


@torch.no_grad()
def evaluate(model, task, val, bs, device, wcfg=None):
    model.eval()
    n = val['bi'].shape[0]
    tot_loss, agree_d, agree_raw, cnt = 0.0, 0, 0, 0
    for lo in range(0, n, bs):
        batch = {k: v[lo:lo + bs] for k, v in val.items()}
        out = model({k: batch[k] for k in INPUT_KEYS})
        loss = task_loss(task, out, batch, wcfg)
        b = batch['bi'].shape[0]
        tot_loss += loss.item() * b
        if task == 'l1p':
            sc = out.double()
        else:
            sc = student_final_score(task, out, batch)
        am = sc.argmax(dim=1)
        pred_cls = batch['cls'].gather(1, am.unsqueeze(1)).squeeze(1)
        agree_d += (pred_cls == batch['bi_cls']).sum().item()
        agree_raw += (am == batch['bi']).sum().item()
        cnt += b
    model.train()
    return tot_loss / cnt, agree_d / cnt, agree_raw / cnt


def save_ckpt(path, model, opt, sampler, step, args, rng_torch):
    tmp = path + '.tmp'
    torch.save({'step': step, 'model': model.state_dict(), 'opt': opt.state_dict(),
                'sampler': sampler.state_dict(), 'torch_rng': rng_torch.get_state()
                if rng_torch else None, 'args': vars(args)}, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['l1r', 'l1rs', 'l1rs2', 'l1s', 'l1p'])
    ap.add_argument('--arch', required=True, choices=['deepset', 'transformer', 'pointer'])
    ap.add_argument('--w-ce', type=float, default=DEFAULT_W['ce'], help='l1rs2 catch-CE weight')
    ap.add_argument('--w-boot', type=float, default=DEFAULT_W['boot'], help='l1rs2 boot-Huber weight')
    ap.add_argument('--w-rank', type=float, default=DEFAULT_W['rank'], help='l1rs2 score-margin ranking weight')
    ap.add_argument('--w-clean', type=float, default=0.0, help='l1rs2 confident-wrong hinge weight (0=off)')
    ap.add_argument('--huber-delta', type=float, default=DEFAULT_W['huber_delta'], help='l1rs2 boot Huber delta')
    ap.add_argument('--size', default='small', choices=['small', 'medium'])
    ap.add_argument('--f64-head', action='store_true')
    ap.add_argument('--data', required=True)
    ap.add_argument('--val-frac', type=float, default=0.1,
                    help='fraction of GAMES held out for val')
    ap.add_argument('--val-mode', default='seedtail', choices=['seedtail', 'percell'],
                    help='seedtail=top val-frac seeds overall (one cell); percell=top '
                         'val-frac seeds WITHIN each cell-block (seed//10000) — all cells in val')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--ckpt', default=None, help='checkpoint path (default per run name)')
    ap.add_argument('--ckpt-every', type=int, default=100)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--val-every', type=int, default=100)
    ap.add_argument('--warmup-steps', type=int, default=20, help='excluded from bench timing')
    ap.add_argument('--results', default=None, help='append JSON result line here')
    ap.add_argument('--max-records', type=int, default=None)
    ap.add_argument('--loader', default='synth', choices=['synth', 'oracle'],
                    help='synth=ds.load_packed (throwaway); oracle=pack_oracle.load_packed (real shards)')
    ap.add_argument('--prepacked', default=None,
                    help='npz cache of the packed dataset: load it if it exists (skips the ~8min '
                         'JSONL pack), else pack from --data then write it here. Reused across runs '
                         '(weight sweeps, arch sweeps, v2b) for the SAME --data.')
    args = ap.parse_args()

    run = '%s_%s_%s%s' % (args.task, args.arch, args.size, '_f64' if args.f64_head else '')
    ckpt_path = args.ckpt or os.path.join(os.path.dirname(args.data.rstrip('/')) or '.',
                                          'ckpt_%s.pt' % run)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- data: split by seed (game id) range, val = tail of the range ----
    t0 = time.time()
    if args.prepacked and os.path.exists(args.prepacked):
        z = np.load(args.prepacked)
        full = {k: z[k] for k in z.files}
        full['headers'] = []
        print('[%s] loaded prepacked npz %s' % (run, args.prepacked), flush=True)
    else:
        loader = pack_oracle if args.loader == 'oracle' else ds
        full = loader.load_packed(args.data, max_records=args.max_records)
        if args.prepacked:
            np.savez_compressed(args.prepacked, **{k: v for k, v in full.items() if k != 'headers'})
            print('[%s] wrote prepacked npz %s' % (run, args.prepacked), flush=True)
    if args.val_mode == 'percell':
        cellkey = full['seed'] // 10000          # device_matrix seedBase blocks
        va_sel = np.zeros(full['seed'].shape[0], dtype=bool)
        for ck in np.unique(cellkey):
            cs = np.unique(full['seed'][cellkey == ck])
            thr = cs[int(len(cs) * (1.0 - args.val_frac))]
            va_sel |= (cellkey == ck) & (full['seed'] >= thr)
        tr_sel = ~va_sel
    else:
        seeds = np.unique(full['seed'])
        split_seed = seeds[int(len(seeds) * (1.0 - args.val_frac))]
        tr_sel = full['seed'] < split_seed
        va_sel = ~tr_sel
    pack_secs = time.time() - t0
    n_all = full['seed'].shape[0]
    headers = full.pop('headers')
    inp = ds.build_inputs(full)
    if 'catches' in full:
        inp['catches'] = full['catches']; inp['boot'] = full['boot']
    tr = {k: v[tr_sel] for k, v in inp.items()}
    va = {k: v[va_sel] for k, v in inp.items()}
    dev = torch.device(args.device)
    tr_t, va_t = ds.to_torch(tr, dev), ds.to_torch(va, dev)
    # free the host-side float64 pack + float32 copies once tensors are on the GPU
    # (VM2 has only ~15 GB RAM; full + inp + tr + va held simultaneously OOMs).
    import gc
    del full, inp, tr, va
    gc.collect()
    # label-side sanity: argmax(score) class must equal bi's class everywhere
    lab_am = tr_t['score'].argmax(dim=1)
    lab_cls = tr_t['cls'].gather(1, lab_am.unsqueeze(1)).squeeze(1)
    assert bool((lab_cls == tr_t['bi_cls']).all()), 'label argmax != bi class -- schema bug'
    print('[%s] data: %d records (%d train / %d val), %d shards, pack %.1fs (%.0f rec/s)'
          % (run, n_all, tr_t['bi'].shape[0], va_t['bi'].shape[0], len(headers),
             pack_secs, n_all / pack_secs), flush=True)

    wcfg = {'ce': args.w_ce, 'boot': args.w_boot, 'rank': args.w_rank,
            'clean': args.w_clean, 'huber_delta': args.huber_delta}
    model = build_model(args.arch, args.size, args.f64_head,
                        head_out=(CATCH_CLASSES + 1) if args.task in ('l1rs', 'l1rs2') else 1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sampler = ds.BatchSampler(tr_t, args.bs, seed=args.seed + 1)
    step0 = 0
    if args.resume and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(sd['model'])
        opt.load_state_dict(sd['opt'])
        sampler.load_state_dict(sd['sampler'])
        if sd.get('torch_rng') is not None:
            torch.set_rng_state(sd['torch_rng'])
        step0 = sd['step']
        print('[%s] resumed from %s @ step %d' % (run, ckpt_path, step0), flush=True)
    print('[%s] params=%d device=%s' % (run, n_params(model), dev), flush=True)

    model.train()
    first_loss, last_loss = None, None
    bench_t0, bench_n = None, 0
    for step in range(step0, args.steps):
        batch = sampler.next()
        out = model({k: batch[k] for k in INPUT_KEYS})
        loss = task_loss(args.task, out, batch, wcfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        l = loss.item()
        if first_loss is None:
            first_loss = l
        last_loss = l
        if step - step0 == args.warmup_steps:
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            bench_t0 = time.time()
            bench_n = 0
        if bench_t0 is not None:
            bench_n += 1
        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            vl, ad, ar = evaluate(model, args.task, va_t, 2048, dev, wcfg)
            print('[%s] step %4d  train_loss %.5f  val_loss %.5f  '
                  'agree_dedup %.4f  agree_raw %.4f'
                  % (run, step + 1, l, vl, ad, ar), flush=True)
        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            save_ckpt(ckpt_path, model, opt, sampler, step + 1, args,
                      torch.default_generator)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    bench_secs = time.time() - bench_t0 if bench_t0 else None
    sps = bench_n * args.bs / bench_secs if bench_secs else None
    vl, ad, ar = evaluate(model, args.task, va_t, 2048, dev, wcfg)
    # per-cell S_dec (agree_dedup) on the val split — cell = seed//10000 block.
    percell = {}
    ck = (va_t['seed'] // 10000).cpu().numpy()
    for c in np.unique(ck):
        sel = np.where(ck == c)[0]
        sub = {k: v[sel] for k, v in va_t.items()}
        _, cad, _ = evaluate(model, args.task, sub, 2048, dev, wcfg)
        percell[int(c)] = round(cad, 4)
    print('[%s] per-cell S_dec(val): %s' % (run, json.dumps(percell)), flush=True)
    res = {'run': run, 'task': args.task, 'arch': args.arch, 'size': args.size,
           'f64_head': args.f64_head, 'params': n_params(model), 'bs': args.bs,
           'steps': args.steps, 'resumed_from': step0,
           'samples_per_sec': round(sps, 1) if sps else None,
           'pack_records_per_sec': round(n_all / pack_secs, 1),
           'first_train_loss': first_loss, 'final_train_loss': last_loss,
           'val_loss': vl, 'val_agree_dedup': ad, 'val_agree_raw': ar, 'val_mode': args.val_mode,
           'percell_sdec': percell,
           'loss_decreased': bool(last_loss is not None and first_loss is not None
                                  and last_loss < first_loss),
           'device': str(dev), 'epoch_secs_1e6': round(1e6 / sps, 1) if sps else None,
           'epoch_secs_1e7': round(1e7 / sps, 1) if sps else None}
    print('[%s] RESULT %s' % (run, json.dumps(res)), flush=True)
    if args.results:
        with open(args.results, 'a') as f:
            f.write(json.dumps(res) + '\n')


if __name__ == '__main__':
    main()

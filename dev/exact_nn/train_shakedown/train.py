#!/usr/bin/env python3
"""One training entry point for the EXACT-NN L1 student tasks (SPEC section 4):

  --task l1r   regress the 4 rolled scores (primary rung). Inputs include
               feat + vprior (both exact at deploy time); the model output is
               gathered at pidx and MSE'd against rolled[4]. Agreement metric
               reconstructs the final score vector (exact vprior + student
               rolled scores) and compares the deduped argmax to bi.
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


def class_pool_logits(logits, cls):
    """(B,16) member logits + (B,16) class ids -> (B,16) class logits at the
    canonical indices (logsumexp over members; ~-1e9 where class empty)."""
    B, K = logits.shape
    M = torch.full((B, K, K), -1e9, device=logits.device, dtype=logits.dtype)
    M = M.scatter(2, cls.unsqueeze(2), logits.unsqueeze(2))
    return torch.logsumexp(M, dim=1)


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


def task_loss(task, out, batch):
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
    if task == 'l1s':
        return masked_mse(out, batch['score'])
    if task == 'l1p':
        cl = class_pool_logits(out, batch['cls'])
        return F.cross_entropy(cl, batch['bi_cls'])
    raise ValueError(task)


@torch.no_grad()
def evaluate(model, task, val, bs, device):
    model.eval()
    n = val['bi'].shape[0]
    tot_loss, agree_d, agree_raw, cnt = 0.0, 0, 0, 0
    for lo in range(0, n, bs):
        batch = {k: v[lo:lo + bs] for k, v in val.items()}
        out = model({k: batch[k] for k in INPUT_KEYS})
        loss = task_loss(task, out, batch)
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
    ap.add_argument('--task', required=True, choices=['l1r', 'l1rs', 'l1s', 'l1p'])
    ap.add_argument('--arch', required=True, choices=['deepset', 'transformer', 'pointer'])
    ap.add_argument('--size', default='small', choices=['small', 'medium'])
    ap.add_argument('--f64-head', action='store_true')
    ap.add_argument('--data', required=True)
    ap.add_argument('--val-frac', type=float, default=0.1,
                    help='fraction of GAMES (seed range tail) held out for val')
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
    args = ap.parse_args()

    run = '%s_%s_%s%s' % (args.task, args.arch, args.size, '_f64' if args.f64_head else '')
    ckpt_path = args.ckpt or os.path.join(os.path.dirname(args.data.rstrip('/')) or '.',
                                          'ckpt_%s.pt' % run)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- data: split by seed (game id) range, val = tail of the range ----
    t0 = time.time()
    loader = pack_oracle if args.loader == 'oracle' else ds
    full = loader.load_packed(args.data, max_records=args.max_records)
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
    # label-side sanity: argmax(score) class must equal bi's class everywhere
    lab_am = tr_t['score'].argmax(dim=1)
    lab_cls = tr_t['cls'].gather(1, lab_am.unsqueeze(1)).squeeze(1)
    assert bool((lab_cls == tr_t['bi_cls']).all()), 'label argmax != bi class -- schema bug'
    print('[%s] data: %d records (%d train / %d val), %d shards, pack %.1fs (%.0f rec/s)'
          % (run, n_all, tr_t['bi'].shape[0], va_t['bi'].shape[0], len(headers),
             pack_secs, n_all / pack_secs), flush=True)

    model = build_model(args.arch, args.size, args.f64_head,
                        head_out=(CATCH_CLASSES + 1) if args.task == 'l1rs' else 1).to(dev)
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
        loss = task_loss(args.task, out, batch)
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
            vl, ad, ar = evaluate(model, args.task, va_t, 2048, dev)
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
    vl, ad, ar = evaluate(model, args.task, va_t, 2048, dev)
    res = {'run': run, 'task': args.task, 'arch': args.arch, 'size': args.size,
           'f64_head': args.f64_head, 'params': n_params(model), 'bs': args.bs,
           'steps': args.steps, 'resumed_from': step0,
           'samples_per_sec': round(sps, 1) if sps else None,
           'pack_records_per_sec': round(n_all / pack_secs, 1),
           'first_train_loss': first_loss, 'final_train_loss': last_loss,
           'val_loss': vl, 'val_agree_dedup': ad, 'val_agree_raw': ar,
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

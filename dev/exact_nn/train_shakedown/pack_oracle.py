#!/usr/bin/env python3
"""Pack REAL oracle decision shards (dev/exact_nn/oracle_logger.js schema) into
the exact numpy dict ds.load_packed produces, so models.py / train.py run
unchanged on real data. Throwaway synth (synth_data.py + ds.py) proved the
training code; this is the production loader.

Oracle record (one JSONL line in <shard>.decisions.jsonl.gz; the shard HEADER is
the sibling <shard>.meta.json, NOT line 0) vs ds.py's per-record fields:

  s.{px,py,pvx,pvy,psize, bx,by,bvx,bvy}   -> pred[5], boids[n,4]
  cfg {W,Hc,PREDATOR_RANGE,NUM_BOIDS}      -> cfg[4]   (as-evaluated, SPEC §1)
  cands[16,2] feat[16,19] ctx[4] vprior[16] pidx[4] score[16] bi
  rolled [[ci,catches,boot]...]            -> rolled[4] = catches+boot, aligned
                                              to pidx order (prod rolls pidx[0..3])
  cand_src                                  -> DERIVED from N: slot0=e3d(0);
                                              slot j=boid(1) if j<=N else e3d-pad(2)
                                              (candidates() pads k>=N with E3D copies)

Non-finite (JSON null), field-specific per the logger header:
  score[k] / rolled boot  null -> -inf   (extermination; masked in regression)
  (dmargin's +inf null is analysis-only, not a training input — not read here)

  python3 pack_oracle.py --data ../data --out packed.npz       # materialize
  # or import: load_packed(data_dir, seed_min, seed_max) -> dict (drop-in for ds)
"""
import argparse, glob, gzip, json, os
import numpy as np
import ds   # reuse dedup_classes, build_inputs, to_torch, BatchSampler, constants

MAXB, K, NFEAT = ds.MAXB, ds.K, ds.NFEAT


def _f(v):
    return -np.inf if v is None else v   # null in a score field == -Infinity


def pack_shard(dec_path):
    """One <shard>.decisions.jsonl.gz -> (header_from_meta, dict of arrays)."""
    meta_path = dec_path.replace('.decisions.jsonl.gz', '.meta.json')
    header = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    cols = {k: [] for k in ('seed', 'n', 'boids', 'bmask', 'pred', 'cfg', 'cands',
                            'cand_kind', 'cand_bref', 'feat', 'ctx', 'vprior',
                            'pidx', 'rolled', 'catches', 'boot', 'score', 'bi')}
    with gzip.open(dec_path, 'rt') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            s = r['s']
            n = len(s['bx'])
            assert n == r['N'], 'N mismatch'
            assert n <= MAXB, ('record with %d boids > MAXB=%d (spam game past the '
                               'cap — raise MAXB for the spam-profile corpus)' % (n, MAXB))
            b = np.zeros((MAXB, 4))
            b[:n, 0] = s['bx']; b[:n, 1] = s['by']; b[:n, 2] = s['bvx']; b[:n, 3] = s['bvy']
            m = np.zeros(MAXB, dtype=bool); m[:n] = True
            c = r['cfg']
            # the oracle record's pidx is the FULL 16-element ballistic-rank
            # permutation; train.py/ds.py want the 4 ROLLED indices (== pidx[0:4],
            # the candidates prod actually rolls). Store the 4 to match the synth
            # contract (gather/scatter over rolled scores).
            pidx = np.asarray(r['pidx'][:4], dtype=np.int64)
            tri = r['rolled']                         # [[ci, catches, boot]...]
            rolled = np.empty(len(tri)); catches = np.empty(len(tri), dtype=np.int64); boot = np.empty(len(tri))
            for i, (rci, rcatch, rboot) in enumerate(tri):   # NOT c/b — c is r['cfg'] above
                assert rci == pidx[i], 'rolled ci %d != pidx[%d]=%d (roll order broke)' % (rci, i, pidx[i])
                catches[i] = rcatch                    # integer rollout catch-count (split-head label)
                boot[i] = _f(rboot)                    # terminal value-net bootstrap (−inf if exterminated)
                rolled[i] = rcatch + _f(rboot)         # combined rolled score (single-head label)
            kind = np.empty(K, dtype=np.int64); kind[0] = 0
            for j in range(1, K):
                kind[j] = 1 if j <= n else 2          # boid vs E3D pad (candidates())
            cols['seed'].append(r['seed']); cols['n'].append(n)
            cols['boids'].append(b); cols['bmask'].append(m)
            cols['pred'].append([s['px'], s['py'], s['pvx'], s['pvy'], s['psize']])
            cols['cfg'].append([c['W'], c['Hc'], c['PREDATOR_RANGE'], c['NUM_BOIDS']])
            cols['cands'].append(np.asarray(r['cands'], dtype=np.float64))
            cols['cand_kind'].append(kind)
            cols['cand_bref'].append(np.full(K, -1, dtype=np.int64))   # unused by build_inputs
            cols['feat'].append(np.asarray(r['feat'], dtype=np.float64))
            cols['ctx'].append(np.asarray(r['ctx'], dtype=np.float64))
            cols['vprior'].append(np.asarray(r['vprior'], dtype=np.float64))
            cols['pidx'].append(pidx)
            cols['rolled'].append(rolled)
            cols['catches'].append(catches)
            cols['boot'].append(boot)
            cols['score'].append(np.asarray([_f(v) for v in r['score']], dtype=np.float64))
            cols['bi'].append(r['bi'])
    out = {k: np.asarray(v) for k, v in cols.items()}
    out['seed'] = out['seed'].astype(np.int64)
    out['n'] = out['n'].astype(np.int64)
    out['bi'] = out['bi'].astype(np.int64)
    return header, out


def iter_shard_paths(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, '*.decisions.jsonl.gz')))


def load_packed(data_dir, seed_min=None, seed_max=None, max_records=None):
    """Drop-in for ds.load_packed over oracle shards. Split by seed (game id)
    range so all plans of a game land on one side (train/calib/sealed
    discipline, SPEC §4c)."""
    parts, headers, total = [], [], 0
    for path in iter_shard_paths(data_dir):
        hdr, arr = pack_shard(path)
        sel = np.ones(arr['seed'].shape[0], dtype=bool)
        if seed_min is not None:
            sel &= arr['seed'] >= seed_min
        if seed_max is not None:
            sel &= arr['seed'] <= seed_max
        if not sel.any():
            continue
        arr = {k: v[sel] for k, v in arr.items()}
        parts.append(arr); headers.append(hdr); total += arr['seed'].shape[0]
        if max_records is not None and total >= max_records:
            break
    if not parts:
        raise RuntimeError('no records in %s for seed range [%s,%s]' % (data_dir, seed_min, seed_max))
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if max_records is not None:
        out = {k: v[:max_records] for k, v in out.items()}
    out['cls'] = ds.dedup_classes(out['cands'])
    out['bi_cls'] = out['cls'][np.arange(out['bi'].shape[0]), out['bi']]
    out['headers'] = headers
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default=None, help='write packed .npz here')
    ap.add_argument('--seed-min', type=int, default=None)
    ap.add_argument('--seed-max', type=int, default=None)
    ap.add_argument('--max-records', type=int, default=None)
    args = ap.parse_args()
    d = load_packed(args.data, args.seed_min, args.seed_max, args.max_records)
    n = d['bi'].shape[0]
    # label-side sanity (same invariant train.py asserts): argmax(score) class == bi class
    score = d['score']; cls = d['cls']
    am = np.argmax(np.where(np.isfinite(score), score, -np.inf), axis=1)
    am_cls = cls[np.arange(n), am]
    bad = int((am_cls != d['bi_cls']).sum())
    neg_inf = int((~np.isfinite(d['score'])).sum())
    print('packed %d records from %d shards; seeds %d..%d; -inf score entries %d; '
          'argmax-class != bi-class: %d'
          % (n, len(d['headers']), d['seed'].min(), d['seed'].max(), neg_inf, bad))
    assert bad == 0, 'label argmax class != bi class -- schema/parse bug'
    nb = np.bincount(d['n'].clip(0, MAXB))
    print('N distribution: 6-14=%d, 15+=%d, max N=%d' % (nb[6:15].sum(), nb[15:].sum(), d['n'].max()))
    if args.out:
        np.savez_compressed(args.out, **{k: v for k, v in d.items() if k != 'headers'})
        print('wrote ' + args.out)


if __name__ == '__main__':
    main()

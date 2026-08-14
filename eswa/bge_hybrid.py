# -*- coding: utf-8 -*-
"""BGE-backbone hybrids: does metadata help on the STRONGER dense backbone?

Answers the reviewer question: "metadata may fail only because it was built
on the weaker MiniLM backbone." Two additional configurations on all four
datasets, reusing cached BGE embeddings and BM25 score matrices:

  BGE-Hybrid   0.5 * minmax(BM25) + 0.5 * minmax(BGE)        (= Neural-Hybrid w/ BGE)
  BGE-CA-HR    top-100 of (0.6*BM25 + 0.4*BGE) reranked with
               + 0.15*C(d) + 0.10*R(d)                       (= CA-HR w/ BGE)

Output: results/{ds}_bge_hybrid_perquery.npz with per-query metrics.
"""
import json
import os
import sys

import numpy as np

from reproduce import (ART, RES, ALPHA, BETA, GAMMA, TOPK, METRICS,
                       load_qrels, build_eval_arrays, per_query_metrics,
                       minmax)

ROOT = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(ROOT, os.pardir, 'exp_v2')

DS = ['scidocs', 'scifact', 'nfcorpus', 'trec-covid']


def paths(ds):
    if ds in ('scidocs', 'scifact'):
        base = os.path.join(V2, ds)
        prep = os.path.join(V2, f'{ds}_prep')
        sm = os.path.join(V2, f'{ds}_scoremats.npz')
    else:
        base = os.path.join(ROOT, 'data', ds)
        prep = os.path.join(ART, f'{ds}_prep')
        sm = os.path.join(ART, f'{ds}_scoremats.npz')
    return base, prep, sm


def run(ds):
    base, prep, sm_path = paths(ds)
    doc_ids = json.load(open(os.path.join(prep, 'doc_ids.json')))
    didx = {d: i for i, d in enumerate(doc_ids)}
    C = np.load(os.path.join(prep, 'C.npy'))
    Rr = np.load(os.path.join(prep, 'R.npy'))
    z = np.load(sm_path)
    S_bm = z['S_bm']

    # BGE embeddings
    emb_dir = os.path.join(ART, f'{ds}_bge_emb')
    bge_ids = json.load(open(os.path.join(emb_dir, 'ids.json')))
    starts = sorted(int(f[6:-4]) for f in os.listdir(emb_dir)
                    if f.startswith('chunk_'))
    E = np.vstack([np.load(os.path.join(emb_dir, f'chunk_{s}.npy'))
                   for s in starts])
    bge_didx = {d: i for i, d in enumerate(bge_ids)}
    assert set(bge_ids) == set(doc_ids), f'{ds}: id set mismatch'
    # align BGE embedding rows to prep doc_ids order
    perm = np.array([bge_didx[d] for d in doc_ids])
    E = E[perm]

    qrels = load_qrels(os.path.join(base, 'qrels', 'test.tsv'))
    qids = sorted(qrels.keys())
    q_emb = np.load(os.path.join(ART, f'{ds}_bge_qemb.npy'))
    q_ids = json.load(open(os.path.join(ART, f'{ds}_bge_qids.json')))
    qmap = {q: q_emb[i] for i, q in enumerate(q_ids)}
    Qb = np.stack([qmap[q] for q in qids])

    # S_bm rows are in the same sorted-qid order (both pipelines do this)
    assert S_bm.shape[0] == len(qids), f'{ds}: scoremat/qid mismatch'

    rel_list, gains_list = build_eval_arrays(qrels, qids, didx, len(doc_ids))
    out = {}
    for m in ('BGE-Hybrid', 'BGE-CA-HR'):
        orders = []
        for qi in range(len(qids)):
            bm_n = minmax(S_bm[qi].astype(np.float64))
            sb_n = minmax((E @ Qb[qi]).astype(np.float64))
            if m == 'BGE-Hybrid':
                sc = 0.5 * bm_n + 0.5 * sb_n
            else:
                hyb = ALPHA * bm_n + (1 - ALPHA) * sb_n
                cand = np.argpartition(-hyb, TOPK)[:TOPK]
                sc = np.full_like(hyb, -1e18)
                sc[cand] = hyb[cand] + BETA * C[cand] + GAMMA * Rr[cand]
            orders.append(np.argsort(-sc)[:100])
        res = per_query_metrics(orders, rel_list, gains_list)
        out[m] = res
        print(ds, m, 'N@10 = %.4f' % float(np.mean(res['N@10'])), flush=True)

    np.savez_compressed(
        os.path.join(RES, f'{ds}_bge_hybrid_perquery.npz'),
        qids=np.array(qids),
        **{f'{m}||{k}': np.array(v) for m, r in out.items()
           for k, v in r.items()})


if __name__ == '__main__':
    targets = [sys.argv[1]] if len(sys.argv) > 1 else DS
    for ds in targets:
        run(ds)
    print('DONE')

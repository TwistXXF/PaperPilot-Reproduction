# -*- coding: utf-8 -*-
"""Reviewer-driven additions for the ESWA revision:

1. BGE-CA-HR metadata-weight sensitivity grid (beta x gamma) on all four
   datasets: does ANY reasonable metadata weight make CA-HR beat BGE-Hybrid
   on the stronger backbone?  Answers "fixed hyperparameters" objection.
2. RRF (k=60) hybrid baselines on both encoders (MiniLM + BGE): a standard,
   stronger hybrid reference that does not use min-max score normalisation.

Outputs:
  results/bge_sensitivity.json        grid means + Wilcoxon vs BGE-Hybrid
  results/{ds}_rrf_perquery.npz       per-query metrics for RRF-MiniLM / RRF-BGE
"""
import json
import os
import sys

import numpy as np

from reproduce import (ART, RES, ALPHA, TOPK,
                       load_qrels, build_eval_arrays, per_query_metrics,
                       minmax)

ROOT = os.path.dirname(os.path.abspath(__file__))

import _layout as L

DS = ['scidocs', 'scifact', 'nfcorpus', 'trec-covid']
BETAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
GAMMAS = [0.0, 0.05, 0.10, 0.15, 0.20]
RRF_K = 60


def paths(ds):
    return L.raw_ds(ds), L.prep_dir(ds), L.scoremats(ds)


def load_bge(ds, doc_ids):
    emb_dir = L.emb_dir(ds, bge=True)
    bge_ids = json.load(open(os.path.join(emb_dir, 'ids.json')))
    starts = sorted(int(f[6:-4]) for f in os.listdir(emb_dir)
                    if f.startswith('chunk_'))
    E = np.vstack([np.load(os.path.join(emb_dir, f'chunk_{s}.npy'))
                   for s in starts])
    bge_didx = {d: i for i, d in enumerate(bge_ids)}
    perm = np.array([bge_didx[d] for d in doc_ids])
    E = E[perm]
    q_emb = np.load(L.art_path(ds, f'{ds}_bge_qemb.npy'))
    q_ids = json.load(open(L.art_path(ds, f'{ds}_bge_qids.json')))
    return E, q_emb, {q: i for i, q in enumerate(q_ids)}


def rrf_orders(s_a, s_b, k=RRF_K):
    ra = np.empty_like(s_a, dtype=np.int64)
    rb = np.empty_like(s_b, dtype=np.int64)
    ra[np.argsort(-s_a)] = np.arange(len(s_a))
    rb[np.argsort(-s_b)] = np.arange(len(s_b))
    sc = 1.0 / (k + ra) + 1.0 / (k + rb)
    return np.argsort(-sc)[:100]


def run(ds):
    base, prep, sm_path = paths(ds)
    doc_ids = json.load(open(os.path.join(prep, 'doc_ids.json')))
    didx = {d: i for i, d in enumerate(doc_ids)}
    C = np.load(os.path.join(prep, 'C.npy'))
    Rr = np.load(os.path.join(prep, 'R.npy'))
    z = np.load(sm_path)
    S_bm, S_sb = z['S_bm'], z['S_sb']      # S_sb = MiniLM dense scores
    E, Qb_all, qmap = load_bge(ds, doc_ids)

    qrels = load_qrels(os.path.join(base, 'qrels', 'test.tsv'))
    qids = sorted(qrels.keys())
    assert S_bm.shape[0] == len(qids)
    Qb = np.stack([Qb_all[qmap[q]] for q in qids])
    rel_list, gains_list = build_eval_arrays(qrels, qids, didx, len(doc_ids))

    # precompute normalised per-query score vectors once
    bm_n = [minmax(S_bm[qi].astype(np.float64)) for qi in range(len(qids))]
    bge_n = [minmax((E @ Qb[qi]).astype(np.float64)) for qi in range(len(qids))]

    # ---- 1. beta x gamma sensitivity grid on the BGE backbone ----------
    # baseline: BGE-Hybrid = 0.5/0.5 min-max fusion (paper definition)
    hyb05 = [0.5 * bm_n[qi] + 0.5 * bge_n[qi] for qi in range(len(qids))]
    hyb_orders = [np.argsort(-h)[:100] for h in hyb05]
    hyb_res = per_query_metrics(hyb_orders, rel_list, gains_list)
    hyb_n = np.array(hyb_res['N@10'])
    # CA-HR candidate pool: alpha=0.6 hybrid (paper definition)
    hyb = [ALPHA * bm_n[qi] + (1 - ALPHA) * bge_n[qi] for qi in range(len(qids))]
    grid = {}
    for beta in BETAS:
        for gamma in GAMMAS:
            orders = []
            for qi in range(len(qids)):
                h = hyb[qi]
                cand = np.argpartition(-h, TOPK)[:TOPK]
                sc = np.full_like(h, -1e18)
                sc[cand] = h[cand] + beta * C[cand] + gamma * Rr[cand]
                orders.append(np.argsort(-sc)[:100])
            res = per_query_metrics(orders, rel_list, gains_list)
            grid[f'beta={beta}|gamma={gamma}'] = {
                'N@10': float(np.mean(res['N@10'])),
                'R@10': float(np.mean(res['R@10'])),
                '_pq': np.array(res['N@10']),
            }
    # Wilcoxon: each grid combo vs BGE-Hybrid (one-sided "greater")
    from scipy import stats as sstats
    keys = list(grid.keys())
    raw_p = []
    for k in keys:
        diff = grid[k]['_pq'] - hyb_n
        if np.all(np.abs(diff) < 1e-15):
            raw_p.append(1.0)
        else:
            raw_p.append(float(sstats.wilcoxon(grid[k]['_pq'], hyb_n,
                                               alternative='greater').pvalue))
    # Holm correction within this dataset's grid
    order = np.argsort(raw_p)
    m = len(raw_p)
    holm = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * raw_p[idx]))
        holm[idx] = running
    for k, p, ph in zip(keys, raw_p, holm):
        grid[k]['p_greater_than_hybrid'] = p
        grid[k]['p_holm'] = ph
        grid[k]['_pq'] = grid[k]['_pq'].tolist()
    best_key = max(keys, key=lambda k: grid[k]['N@10'])
    sens = {
        'bge_hybrid_N@10': float(np.mean(hyb_n)),
        'n_queries': len(qids),
        'betas': BETAS, 'gammas': GAMMAS,
        'grid': grid,
        'best_combo': best_key,
        'best_N@10': grid[best_key]['N@10'],
        'any_significant_gain_after_holm': bool(
            min(holm) < 0.05 and grid[keys[int(np.argmin(holm))]]['N@10']
            > float(np.mean(hyb_n))),
    }
    print(ds, 'BGE-Hybrid N@10 = %.4f | best grid %s N@10 = %.4f | '
              'min Holm p = %.3f' % (sens['bge_hybrid_N@10'], best_key,
                                     sens['best_N@10'], min(holm)),
          flush=True)

    # ---- 2. RRF baselines (MiniLM and BGE) ------------------------------
    rrf_out = {}
    for name, dense in [('RRF-MiniLM', None), ('RRF-BGE', None)]:
        orders = []
        for qi in range(len(qids)):
            s_dense = (S_sb[qi].astype(np.float64) if name == 'RRF-MiniLM'
                       else (E @ Qb[qi]).astype(np.float64))
            orders.append(rrf_orders(S_bm[qi].astype(np.float64), s_dense))
        res = per_query_metrics(orders, rel_list, gains_list)
        rrf_out[name] = res
        print(ds, name, 'N@10 = %.4f R@10 = %.4f'
              % (float(np.mean(res['N@10'])), float(np.mean(res['R@10']))),
              flush=True)
    np.savez_compressed(
        os.path.join(RES, f'{ds}_rrf_perquery.npz'),
        qids=np.array(qids),
        **{f'{m}||{k}': np.array(v) for m, r in rrf_out.items()
           for k, v in r.items()})
    sens['rrf'] = {m: {k: float(np.mean(v)) for k, v in r.items()}
                   for m, r in rrf_out.items()}
    return sens


if __name__ == '__main__':
    targets = [sys.argv[1]] if len(sys.argv) > 1 else DS
    out = {}
    for ds in targets:
        out[ds] = run(ds)
        json.dump(out, open(os.path.join(RES, 'bge_sensitivity.json'), 'w'),
                  indent=1)
    print('DONE')

# -*- coding: utf-8 -*-
"""Consolidate every experimental number into results/eswa_tables.json.

Single source of truth for the ESWA manuscript and for verify_eswa.py.
Sources:
  exp_v2/{ds}_perquery.npz, exp_v2/{ds}_{ablation,robust,router,latency}.json,
  exp_v2/tables.json                      (SCIDOCS, SciFact)
  exp_v3/results/{ds}_perquery.npz, {ds}_{ablation,robust,router}.json,
  exp_v3/results/{ds}_bge_perquery.npz, bge_{ds}.json,
  exp_v3/results/tables.json              (NFCorpus, TREC-COVID)
  exp_v3/results/gen_eval_summary.json    (generation-side)
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(ROOT, os.pardir, 'exp_v2')
R3 = os.path.join(ROOT, 'results')

METHODS7 = ['BM25', 'LSA-Dense', 'SBERT-Dense', 'Neural-Hybrid', 'UMA-RAG',
            'LP-RAG', 'CA-HR']
METHODS8 = ['BM25', 'LSA-Dense', 'SBERT-Dense', 'BGE-Dense', 'Neural-Hybrid',
            'UMA-RAG', 'LP-RAG', 'CA-HR']
METHODS10 = METHODS8 + ['BGE-Hybrid', 'BGE-CA-HR']
METRICS = ['R@1', 'R@5', 'R@10', 'N@10', 'MRR']
DS_META = {
    'scidocs': {'name': 'SCIDOCS', 'domain': 'computer science',
                'n_docs': 25657, 'n_queries': 1000,
                'meta_cov': {'citations': 0.997, 'year': 0.997, 'venue': 0.997}},
    'scifact': {'name': 'SciFact', 'domain': 'biomedical claims',
                'n_docs': 5183, 'n_queries': 300,
                'meta_cov': {'citations': 0.941, 'year': 0.941, 'venue': 0.941}},
    'nfcorpus': {'name': 'NFCorpus', 'domain': 'nutrition / medicine',
                 'n_docs': 3633, 'n_queries': 323,
                 'meta_cov': {'citations': 0.940, 'year': 0.940, 'venue': 0.940}},
    'trec-covid': {'name': 'TREC-COVID', 'domain': 'COVID-19 biomedicine',
                   'n_docs': 171332, 'n_queries': 50,
                   'meta_cov': {'citations': 0.698, 'year': 0.964,
                                'venue': 0.925}},
}


def wilcoxon_greater(a, b):
    from scipy import stats as sstats
    d = a - b
    d = d[d != 0]
    if len(d) < 5:
        return 1.0
    return float(sstats.wilcoxon(d, alternative='greater').pvalue)


def cohend(a, b):
    diff = a - b
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 1e-12 else 0.0


def load_perquery(ds):
    """Return {method: {metric: np.array}} merging MiniLM-family and BGE."""
    if ds in ('scidocs', 'scifact'):
        z = np.load(os.path.join(V2, f'{ds}_perquery.npz'), allow_pickle=True)
    else:
        z = np.load(os.path.join(R3, f'{ds}_perquery.npz'), allow_pickle=True)
    d = {m: {k: z[f'{m}||{k}'] for k in METRICS} for m in METHODS7}
    zb = np.load(os.path.join(R3, f'{ds}_bge_perquery.npz'))
    d['BGE-Dense'] = {k: zb[f'BGE-Dense||{k}'] for k in METRICS}
    zh = np.load(os.path.join(R3, f'{ds}_bge_hybrid_perquery.npz'))
    for m in ('BGE-Hybrid', 'BGE-CA-HR'):
        d[m] = {k: zh[f'{m}||{k}'] for k in METRICS}
    return d


def main():
    out = {'datasets': DS_META, 'main': {}, 'ablation': {}, 'robust': {},
           'router': {}, 'oracle': {}, 'latency': {}, 'generation': {}}

    for ds in DS_META:
        d = load_perquery(ds)
        avg = {m: {k: float(np.mean(v)) for k, v in d[m].items()}
               for m in METHODS10}
        tests = {}
        for base in METHODS10:
            if base == 'CA-HR':
                continue
            for k in METRICS:
                a, b = d['CA-HR'][k], d[base][k]
                tests[f'CA-HR vs {base} | {k}'] = {
                    'cahr': float(np.mean(a)), 'base': float(np.mean(b)),
                    'p_one_sided': wilcoxon_greater(a, b), 'd': cohend(a, b)}
        # does metadata help on the STRONG BGE backbone? (key reviewer question)
        from scipy import stats as sstats
        bge_tests = {}
        for other in ('SBERT-Dense', 'CA-HR', 'Neural-Hybrid'):
            x, y = d['BGE-Dense']['N@10'], d[other]['N@10']
            diff = x - y
            nz = diff != 0
            p = float(sstats.wilcoxon(x[nz], y[nz]).pvalue) if nz.sum() else 1.0
            bge_tests[f'BGE-Dense vs {other} | N@10'] = {
                'bge': float(x.mean()), 'other': float(y.mean()),
                'p_two_sided': p, 'd': cohend(x, y)}
        # BGE-CA-HR vs its metadata-free counterparts (one-sided "greater")
        for other in ('BGE-Dense', 'BGE-Hybrid'):
            a, b = d['BGE-CA-HR']['N@10'], d[other]['N@10']
            bge_tests[f'BGE-CA-HR vs {other} | N@10'] = {
                'cahr_bge': float(a.mean()), 'base': float(b.mean()),
                'p_one_sided': wilcoxon_greater(a, b), 'd': cohend(a, b)}
        # CA-HR vs BGE-CA-HR (backbone swap effect)
        a, b = d['BGE-CA-HR']['N@10'], d['CA-HR']['N@10']
        diff = a - b
        nz = diff != 0
        bge_tests['BGE-CA-HR vs CA-HR | N@10'] = {
            'bge_cahr': float(a.mean()), 'cahr': float(b.mean()),
            'p_two_sided': float(sstats.wilcoxon(a[nz], b[nz]).pvalue)
            if nz.sum() else 1.0, 'd': cohend(a, b)}
        routed = ['UMA-RAG', 'LP-RAG', 'CA-HR']
        oracle = {k: float(np.mean(np.max(
            np.stack([d[m][k] for m in routed]), axis=0))) for k in METRICS}
        best_single = max(METHODS10, key=lambda m: avg[m]['N@10'])
        out['main'][ds] = {'n_queries': len(d['BM25']['N@10']), 'avg': avg,
                           'tests_vs_cahr': tests, 'bge_tests': bge_tests,
                           'best_single_N@10': best_single}
        out['oracle'][ds] = {'routed_oracle': oracle,
                             'headroom_vs_best_routed': {
                                 k: oracle[k] - max(avg[m][k] for m in routed)
                                 for k in METRICS},
                             'headroom_vs_best_single': {
                                 'method': best_single,
                                 'N@10_gap': oracle['N@10'] - avg[best_single]['N@10']}}

    for ds in DS_META:
        if ds in ('scidocs', 'scifact'):
            out['ablation'][ds] = json.load(open(os.path.join(V2, f'{ds}_ablation.json')))
            out['robust'][ds] = json.load(open(os.path.join(V2, f'{ds}_robust.json')))
            out['router'][ds] = json.load(open(os.path.join(V2, f'{ds}_router.json')))
            out['latency'][ds] = json.load(open(os.path.join(V2, f'{ds}_latency.json')))
        else:
            out['ablation'][ds] = json.load(open(os.path.join(R3, f'{ds}_ablation.json')))
            out['robust'][ds] = json.load(open(os.path.join(R3, f'{ds}_robust.json')))
            out['router'][ds] = json.load(open(os.path.join(R3, f'{ds}_router.json')))

    out['generation'] = json.load(open(os.path.join(R3, 'gen_eval_summary.json')))

    # bibliographic metadata diagnostics (citation-relevance association etc.)
    out['diagnostics'] = json.load(
        open(os.path.join(R3, 'metadata_diagnostics.json')))

    # ---- primary comparison family with Holm-Bonferroni correction --------
    # Pre-specified primary tests (all NDCG@10):
    #   per dataset: CA-HR vs {BM25, SBERT-Dense, Neural-Hybrid}  (12 tests)
    #   per dataset: BGE-CA-HR vs BGE-Hybrid                      (4 tests)
    primary = []
    for ds in DS_META:
        for base in ('BM25', 'SBERT-Dense', 'Neural-Hybrid'):
            t = out['main'][ds]['tests_vs_cahr'][f'CA-HR vs {base} | N@10']
            primary.append({'dataset': ds, 'test': f'CA-HR vs {base} | N@10',
                            'p_raw': t['p_one_sided'], 'd': t['d']})
        t = out['main'][ds]['bge_tests']['BGE-CA-HR vs BGE-Hybrid | N@10']
        primary.append({'dataset': ds, 'test': 'BGE-CA-HR vs BGE-Hybrid | N@10',
                        'p_raw': t['p_one_sided'], 'd': t['d']})
    m = len(primary)
    order = sorted(range(m), key=lambda i: primary[i]['p_raw'])
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * primary[i]['p_raw'])
        prev = max(prev, adj)
        primary[i]['p_holm'] = prev
    out['primary_tests'] = primary

    fp = os.path.join(R3, 'eswa_tables.json')
    json.dump(out, open(fp, 'w'), indent=1)
    print('saved', fp)
    for t in primary:
        sig = '*' if t['p_holm'] < 0.05 else ' '
        print(f"  {sig} {t['dataset']:11s} {t['test']:34s} "
              f"raw={t['p_raw']:.4g} holm={t['p_holm']:.4g} d={t['d']:+.3f}")
    for ds in DS_META:
        a = out['main'][ds]['avg']
        print(f"{ds:11s} best={out['main'][ds]['best_single_N@10']:14s} "
              f"N@10: CA-HR={a['CA-HR']['N@10']:.4f} "
              f"BGE={a['BGE-Dense']['N@10']:.4f} "
              f"SBERT={a['SBERT-Dense']['N@10']:.4f} "
              f"NH={a['Neural-Hybrid']['N@10']:.4f}")


if __name__ == '__main__':
    main()

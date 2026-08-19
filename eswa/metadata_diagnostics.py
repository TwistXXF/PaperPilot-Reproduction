# -*- coding: utf-8 -*-
"""Bibliographic metadata diagnostics per dataset.

Quantifies citation-relevance association without treating it as a causal
explanation. Positive and background document IDs are deduplicated. Every
dataset uses the same seeded background-sampling rule, and the AUC diagnostic
is accompanied by an approximate 95% interval and explicit sample sizes.

Output: results/metadata_diagnostics.json
"""
import json
import os
from statistics import median

import numpy as np
from scipy.stats import mannwhitneyu

import _layout as L

BASE = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    ds: {
        'corpus': os.path.join(L.raw_ds(ds), 'corpus.jsonl'),
        'qrels': os.path.join(L.raw_ds(ds), 'qrels', 'test.tsv'),
        'meta': L.meta_file(ds),
    }
    for ds in ('scidocs', 'scifact', 'nfcorpus', 'trec-covid')
}

T_REF = 2024  # same reference year as the CA-HR recency prior
BACKGROUND_SEED = 7
BACKGROUND_RATIO = 10
BACKGROUND_CAP = 25_000
METADATA_SNAPSHOT = 'frozen repository snapshot collected during 2025-2026'


def load_qrels(path):
    rel, nonrel = set(), set()
    with open(path, encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(parts[2])
            if score >= 1:
                rel.add(did)
            else:
                nonrel.add(did)
    return rel, nonrel


def diagnose(ds, cfg):
    meta = {k: v for k, v in json.load(open(cfg['meta'])).items() if v}
    n_docs = sum(1 for _ in open(cfg['corpus'], encoding='utf-8'))
    cit_of = {k: (v.get('citations') if v.get('citations') is not None else 0)
              for k, v in meta.items()}
    matched = sum(1 for v in meta.values() if v.get('citations') is not None)
    cits = np.array(list(cit_of.values()), dtype=float)
    years = np.array([m.get('year') for m in meta.values()
                      if m.get('year')], dtype=float)

    rel, _ = load_qrels(cfg['qrels'])
    rel_c = np.array([cit_of[d] for d in rel if d in cit_of], dtype=float)
    corpus_ids = []
    with open(cfg['corpus'], encoding='utf-8') as f:
        for line in f:
            corpus_ids.append(str(json.loads(line)['_id']))
    pool = [d for d in corpus_ids if d not in rel and d in cit_of]
    rng = np.random.default_rng(BACKGROUND_SEED)
    sample_size = min(BACKGROUND_RATIO * len(rel_c), len(pool), BACKGROUND_CAP)
    take = rng.choice(len(pool), size=sample_size, replace=False)
    nonrel_c = np.array([cit_of[pool[i]] for i in take], dtype=float)
    nonrel_note = (
        'seeded background sample from documents without positive test qrels; '
        f'ratio<={BACKGROUND_RATIO}:1, cap={BACKGROUND_CAP}'
    )
    # rank-biserial AUC: P(citation_rel > citation_nonrel) + 0.5*P(tie)
    u, p = mannwhitneyu(rel_c, nonrel_c, alternative='two-sided')
    auc = u / (len(rel_c) * len(nonrel_c))
    # Hanley-McNeil large-sample approximation for an interpretable interval.
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    variance = (
        auc * (1.0 - auc)
        + (len(rel_c) - 1) * (q1 - auc * auc)
        + (len(nonrel_c) - 1) * (q2 - auc * auc)
    ) / (len(rel_c) * len(nonrel_c))
    standard_error = float(np.sqrt(max(variance, 0.0)))
    auc_ci = [max(0.0, auc - 1.96 * standard_error),
              min(1.0, auc + 1.96 * standard_error)]

    out = {
        'n_docs': n_docs,
        'coverage': matched / n_docs,
        'median_citations': float(median(cits)),
        'mean_citations': float(cits.mean()),
        'pct_zero_citation': float((cits == 0).mean()),
        'median_age': float(median(T_REF - years)),
        'rel_docs': len(rel_c),
        'nonrel_docs': len(nonrel_c),
        'rel_median_citations': float(median(rel_c)),
        'nonrel_median_citations': float(median(nonrel_c)),
        'rel_mean_citations': float(rel_c.mean()),
        'nonrel_mean_citations': float(nonrel_c.mean()),
        'cit_rel_auc': float(auc),
        'cit_rel_auc_95ci': [float(value) for value in auc_ci],
        'cit_rel_auc_ci_method': 'Hanley-McNeil large-sample approximation',
        'cit_rel_mwu_p': float(p),
        'nonrel_source': nonrel_note,
        'deduplication': 'unique document IDs in positive and background sets',
        'background_seed': BACKGROUND_SEED,
        'metadata_snapshot': METADATA_SNAPSHOT,
    }
    print(f"{ds}: cov={out['coverage']:.3f} medcit={out['median_citations']:.0f} "
          f"zero={out['pct_zero_citation']:.1%} age={out['median_age']:.0f}y "
          f"rel_med={out['rel_median_citations']:.0f} vs "
          f"nonrel_med={out['nonrel_median_citations']:.0f} "
          f"AUC={auc:.3f} p={p:.2e}")
    return out


def main():
    res = {ds: diagnose(ds, cfg) for ds, cfg in DATASETS.items()}
    out = os.path.join(BASE, 'results', 'metadata_diagnostics.json')
    json.dump(res, open(out, 'w'), indent=1)
    print('saved', out)


if __name__ == '__main__':
    main()

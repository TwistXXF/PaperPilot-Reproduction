# -*- coding: utf-8 -*-
"""Bibliographic metadata diagnostics per dataset.

Explains WHY citation priors help on some corpora and not others, beyond
raw coverage: citation distribution, zero-citation share, document age, and
crucially the citation-relevance association (do relevant documents carry
higher citation counts than judged non-relevant ones?).

Output: results/metadata_diagnostics.json
"""
import json
import os
from statistics import median

import numpy as np
from scipy.stats import mannwhitneyu

BASE = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    'scidocs': {
        'corpus': os.path.join(BASE, '..', 'exp_v2', 'scidocs', 'corpus.jsonl'),
        'qrels': os.path.join(BASE, '..', 'exp_v2', 'scidocs', 'qrels', 'test.tsv'),
        'meta': os.path.join(BASE, '..', 'exp_v2', 'scidocs_metadata.json'),
    },
    'scifact': {
        'corpus': os.path.join(BASE, '..', 'exp_v2', 'scifact', 'corpus.jsonl'),
        'qrels': os.path.join(BASE, '..', 'exp_v2', 'scifact', 'qrels', 'test.tsv'),
        'meta': os.path.join(BASE, '..', 'exp_v2', 'scifact_metadata.json'),
    },
    'nfcorpus': {
        'corpus': os.path.join(BASE, 'data', 'nfcorpus', 'corpus.jsonl'),
        'qrels': os.path.join(BASE, 'data', 'nfcorpus', 'qrels', 'test.tsv'),
        'meta': os.path.join(BASE, 'data', 'metadata', 'nfcorpus_metadata.json'),
    },
    'trec-covid': {
        'corpus': os.path.join(BASE, 'data', 'trec-covid', 'corpus.jsonl'),
        'qrels': os.path.join(BASE, 'data', 'trec-covid', 'qrels', 'test.tsv'),
        'meta': os.path.join(BASE, 'data', 'metadata', 'trec-covid_metadata.json'),
    },
}

T_REF = 2024  # same reference year as the CA-HR recency prior


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

    rel, nonrel = load_qrels(cfg['qrels'])
    rel_c = np.array([cit_of[d] for d in rel if d in cit_of], dtype=float)
    nonrel_note = 'judged non-relevant'
    if nonrel:
        nonrel_c = np.array([cit_of[d] for d in nonrel if d in cit_of],
                            dtype=float)
    else:
        # No explicit non-relevant judgments (SciFact, NFCorpus): use a
        # seeded random background sample of unjudged corpus documents.
        nonrel_note = 'seeded background sample (no explicit non-relevant qrels)'
        corpus_ids = []
        with open(cfg['corpus'], encoding='utf-8') as f:
            for line in f:
                corpus_ids.append(json.loads(line)['_id'])
        pool = [d for d in corpus_ids if d not in rel and d in cit_of]
        rng = np.random.default_rng(7)
        take = rng.choice(len(pool), size=min(10 * len(rel), len(pool)),
                          replace=False)
        nonrel_c = np.array([cit_of[pool[i]] for i in take], dtype=float)
    # rank-biserial AUC: P(citation_rel > citation_nonrel) + 0.5*P(tie)
    u, p = mannwhitneyu(rel_c, nonrel_c, alternative='greater')
    auc = u / (len(rel_c) * len(nonrel_c))

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
        'cit_rel_mwu_p': float(p),
        'nonrel_source': nonrel_note,
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

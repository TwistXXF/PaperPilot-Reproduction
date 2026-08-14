# -*- coding: utf-8 -*-
"""Audit the ESWA manuscript against results/eswa_tables.json.

Every number the manuscript is supposed to contain must appear in the
extracted docx text. Also runs staleness checks (no leftover IP&M /
two-dataset phrasing). Exit code 0 = all checks pass.
"""
import json
import os
import sys

from docx import Document

BASE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(BASE, 'results', 'eswa_tables.json')))
DOCX = os.path.normpath(os.path.join(BASE, '..', 'ESWA_submission',
                                      '01_Manuscript_ESWA.docx'))

doc = Document(DOCX)
parts = [p.text for p in doc.paragraphs]
for tbl in doc.tables:
    for row in tbl.rows:
        for cell in row.cells:
            parts.append(cell.text)
TEXT = '\n'.join(parts)

checks = []


def expect(label, value):
    checks.append((label, str(value)))


METHODS = ['BM25', 'LSA-Dense', 'SBERT-Dense', 'BGE-Dense', 'Neural-Hybrid',
           'UMA-RAG', 'LP-RAG', 'CA-HR', 'BGE-Hybrid', 'BGE-CA-HR']
DS = ['scidocs', 'scifact', 'nfcorpus', 'trec-covid']

# 1. main table values
for ds in DS:
    for m in METHODS:
        a = T['main'][ds]['avg'][m]
        expect(f'{ds}/{m} N@10', f"{a['N@10']:.4f}")
        expect(f'{ds}/{m} R@10', f"{a['R@10']:.4f}")

# 2. ablation values
ABL = ['full', '-citation', '-recency', '-dense (alpha=1)',
       '-sparse (alpha=0)', '-rerank (plain hybrid)']
for ds in DS:
    for v in ABL:
        expect(f'ablation {ds} {v}', f"{T['ablation'][ds][v]['N@10']:.4f}")

# 3. robustness quoted values (nested metric dicts; manuscript quotes N@10)
for ds, m in [('scifact', 'CA-HR'), ('scifact', 'BM25'),
              ('scifact', 'Neural-Hybrid'),
              ('trec-covid', 'CA-HR'), ('trec-covid', 'BM25'),
              ('trec-covid', 'Neural-Hybrid'),
              ('nfcorpus', 'CA-HR'), ('nfcorpus', 'Neural-Hybrid'),
              ('scidocs', 'BM25'), ('scidocs', 'CA-HR'),
              ('scidocs', 'Neural-Hybrid')]:
    expect(f'robust {ds} 0.4 {m}',
           f"{T['robust'][ds]['0.4'][m]['N@10']:.4f}")

# 3b. BGE backbone-transfer effect sizes quoted in Section 6.1
for ds in DS:
    d_val = T['main'][ds]['bge_tests']['BGE-CA-HR vs BGE-Hybrid | N@10']['d']
    expect(f'bge transfer d {ds}', f'd = {d_val:.3f}')

# 4. router / oracle
for ds in DS:
    rt = T['router'][ds]
    expect(f'router acc {ds}', f"{rt['cv_accuracy']:.3f}")
    expect(f'router kappa {ds}', f"{rt['kappa']:.3f}")
    expect(f'routed N@10 {ds}', f"{rt['routed_system']['N@10']:.4f}")
    expect(f'oracle N@10 {ds}',
           f"{T['oracle'][ds]['routed_oracle']['N@10']:.4f}")

# 5. generation summary (200 paired queries, four datasets)
g = T['generation']
for sysname in ('CA-HR', 'Neural-Hybrid'):
    s = g[sysname]
    expect(f'gen {sysname} relevance mean', f"{s['relevance']['mean']:.2f}")
    expect(f'gen {sysname} faith mean', f"{s['faithfulness']['mean']:.2f}")


def pstr(p):
    return 'p < 0.001' if p < 0.001 else 'p = %.3f' % p


for key in ['paired_relevance', 'paired_faithfulness',
            'paired_citation_precision', 'paired_n_rel_context']:
    expect(f'gen {key} p', pstr(g[key]['wilcoxon_p_two_sided']))
expect('gen paired citprec CA-HR', '%.3f' % g['paired_citation_precision']['mean_CA-HR'])
expect('gen paired citprec NH', '%.3f' % g['paired_citation_precision']['mean_Neural-Hybrid'])
expect('gen rel ctx CA-HR', '%.2f' % g['paired_n_rel_context']['mean_CA-HR'])
expect('gen rel ctx NH', '%.2f' % g['paired_n_rel_context']['mean_Neural-Hybrid'])
# by-dataset citation-precision contrast quoted in text
expect('gen citprec trec-covid', '0.87')
expect('gen citprec scidocs', '0.15')

# 6. dataset facts
for s in ['25,657', '1,000', '5,183', '300', '3,633', '323', '171,332',
          '69.8%', '96.4%', '92.5%', '99.7%', '94.1%', '94.0%']:
    expect(f'dataset fact {s}', s)

# 7. deployment facts
for s in ['10 July 2026', '2 vCPU', '1.6 GB', 'Node.js 20.20.2',
          'MySQL 8.0.46', 'Nginx 1.18', '26 chat conversations',
          '63 messages', '6 registered users']:
    expect(f'deployment {s}', s)

# 8. p-value strings quoted in text
for ds, key in [('scidocs', 'CA-HR vs BM25 | N@10'),
                ('trec-covid', 'CA-HR vs BM25 | N@10'),
                ('scifact', 'CA-HR vs SBERT-Dense | N@10')]:
    pv = T['main'][ds]['tests_vs_cahr'][key]['p_one_sided']
    if pv < 0.001:
        expect(f'pval {ds} {key}', 'p < 0.001')

fails = []
for label, value in checks:
    if value not in TEXT:
        fails.append((label, value))

# staleness checks
stale = []
for bad in ['Information Processing', 'IP&M', 'two benchmarks',
            'two datasets', 'PAV-Agent', 'eight retrieval configurations',
            'Eight retrieval configurations', 'eight methods',
            'eight configurations', '100 paired', '100 test queries',
            'all p > 0.17',
            'SCIDOCS (computer science; 25,657 documents, '
            '1,000 queries) and SciFact (biomedical;']:
    if bad in TEXT:
        stale.append(bad)

print(f'checks: {len(checks)}, failed: {len(fails)}, stale: {len(stale)}')
for label, value in fails:
    print('MISSING:', label, '->', value)
for s in stale:
    print('STALE:', s)
sys.exit(1 if (fails or stale) else 0)

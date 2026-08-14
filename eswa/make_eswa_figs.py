# -*- coding: utf-8 -*-
"""Generate all four-dataset figures for the ESWA manuscript from
results/eswa_tables.json. Outputs exp_v3/figures/Fig2..Fig5 (PDF+PNG)."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

setup_plot()

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'figures')
os.makedirs(OUT, exist_ok=True)
T = json.load(open(os.path.join(ROOT, 'results', 'eswa_tables.json')))

DS = ['scidocs', 'scifact', 'nfcorpus', 'trec-covid']
DS_LABEL = {'scidocs': 'SCIDOCS', 'scifact': 'SciFact',
            'nfcorpus': 'NFCorpus', 'trec-covid': 'TREC-COVID'}
METHODS = ['BM25', 'LSA-Dense', 'SBERT-Dense', 'BGE-Dense', 'Neural-Hybrid',
           'UMA-RAG', 'LP-RAG', 'CA-HR', 'BGE-Hybrid', 'BGE-CA-HR']
COLORS = {'BM25': '#94a3b8', 'LSA-Dense': '#cbd5e1', 'SBERT-Dense': '#60a5fa',
          'BGE-Dense': '#1d4ed8', 'Neural-Hybrid': '#f59e0b',
          'UMA-RAG': '#a78bfa', 'LP-RAG': '#c4b5fd', 'CA-HR': '#dc2626',
          'BGE-Hybrid': '#7c3aed', 'BGE-CA-HR': '#f43f5e'}
XLB = ['BM25', 'LSA', 'MiniLM', 'BGE', 'NH', 'UMA', 'LP', 'CA-HR',
       'BGE-H', 'BGE-CA']


def save(fig, name):
    fig.savefig(os.path.join(OUT, f'{name}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, f'{name}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---- Fig 2: main NDCG@10 grouped bars ------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.6), sharey=False)
for ax, ds in zip(axes, DS):
    avg = T['main'][ds]['avg']
    vals = [avg[m]['N@10'] for m in METHODS]
    bars = ax.bar(range(len(METHODS)), vals,
                  color=[COLORS[m] for m in METHODS], width=0.75)
    best = int(np.argmax(vals))
    bars[best].set_edgecolor('black')
    bars[best].set_linewidth(2.0)
    ax.set_title(DS_LABEL[ds], fontsize=11)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(XLB, rotation=45, ha='right', fontsize=7.5)
    ax.grid(axis='y', alpha=0.3)
    if ds == 'scidocs':
        ax.set_ylabel('NDCG@10')
fig.suptitle('Retrieval effectiveness (NDCG@10) across four domains '
             '(black edge = best single method)', fontsize=11, y=1.02)
fig.tight_layout()
save(fig, 'Fig2_main_results')

# ---- Fig 3: CA-HR ablation ------------------------------------------------
ABL = ['full', '-citation', '-recency', '-dense (alpha=1)',
       '-sparse (alpha=0)', '-rerank (plain hybrid)']
ABL_LABEL = ['Full CA-HR', '− citation', '− recency', '− dense (α=1)',
             '− sparse (α=0)', '− re-rank']
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4))
for ax, ds in zip(axes, DS):
    ab = T['ablation'][ds]
    vals = [ab[k]['N@10'] for k in ABL]
    colors = ['#dc2626'] + ['#94a3b8'] * 5
    ax.bar(range(len(ABL)), vals, color=colors, width=0.7)
    ax.axhline(vals[0], color='#dc2626', ls='--', lw=1, alpha=0.6)
    ax.set_title(DS_LABEL[ds], fontsize=11)
    ax.set_xticks(range(len(ABL)))
    ax.set_xticklabels(ABL_LABEL, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    if ds == 'scidocs':
        ax.set_ylabel('NDCG@10')
fig.suptitle('CA-HR ablation (NDCG@10; dashed line = full model)',
             fontsize=11, y=1.02)
fig.tight_layout()
save(fig, 'Fig3_ablation')

# ---- Fig 4: robustness under word-drop noise ------------------------------
# SCIDOCS noise study uses a fixed 300-query subsample (seed 7); use the
# subsample's clean scores as the noise=0 point so the curves are consistent.
import numpy as _np
_pq = _np.load(os.path.join(ROOT, os.pardir, 'exp_v2',
                            'scidocs_perquery.npz'), allow_pickle=True)
_qids = _pq['qids'].tolist()
_sub = sorted(_np.random.RandomState(7).choice(_qids, size=300,
                                               replace=False).tolist())
_idx = [_qids.index(q) for q in _sub]
CLEAN0 = {'scidocs': {m: float(_np.mean(_pq[f'{m}||N@10'][_idx]))
                      for m in ('BM25', 'Neural-Hybrid', 'CA-HR')}}

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4), sharey=False)
for ax, ds in zip(axes, DS):
    rob = T['robust'][ds]
    xs = [0.0, 0.1, 0.2, 0.3, 0.4]
    if ds == 'scidocs':
        base = CLEAN0['scidocs']['BM25']
        nh0 = CLEAN0['scidocs']['Neural-Hybrid']
        ca0 = CLEAN0['scidocs']['CA-HR']
    else:
        base = T['main'][ds]['avg']['BM25']['N@10']
        nh0 = T['main'][ds]['avg']['Neural-Hybrid']['N@10']
        ca0 = T['main'][ds]['avg']['CA-HR']['N@10']
    ax.plot(xs, [base] + [rob[str(k)]['BM25']['N@10'] for k in (0.1, 0.2, 0.3, 0.4)],
            'o-', color='#94a3b8', label='BM25')
    ax.plot(xs, [nh0] + [rob[str(k)]['Neural-Hybrid']['N@10']
                         for k in (0.1, 0.2, 0.3, 0.4)],
            's-', color='#f59e0b', label='Neural-Hybrid')
    ax.plot(xs, [ca0] + [rob[str(k)]['CA-HR']['N@10'] for k in (0.1, 0.2, 0.3, 0.4)],
            '^-', color='#dc2626', label='CA-HR')
    ax.set_title(DS_LABEL[ds], fontsize=11)
    ax.set_xlabel('word-drop ratio', fontsize=9)
    ax.grid(alpha=0.3)
    if ds == 'scidocs':
        ax.set_ylabel('NDCG@10')
        ax.legend(fontsize=8)
fig.suptitle('Robustness to query corruption (NDCG@10 vs. word-drop noise)',
             fontsize=11, y=1.02)
fig.tight_layout()
save(fig, 'Fig4_robustness')

# ---- Fig 5: oracle headroom vs routed system ------------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.6))
labels, oracle_v, routed_v, best_v, cahr_v = [], [], [], [], []
for ds in DS:
    labels.append(DS_LABEL[ds])
    oracle_v.append(T['oracle'][ds]['routed_oracle']['N@10'])
    routed_v.append(T['router'][ds]['routed_system']['N@10'])
    best = max(['UMA-RAG', 'LP-RAG', 'CA-HR'],
               key=lambda m: T['main'][ds]['avg'][m]['N@10'])
    best_v.append(T['main'][ds]['avg'][best]['N@10'])
    cahr_v.append(T['main'][ds]['avg']['CA-HR']['N@10'])
x = np.arange(len(DS))
w = 0.2
ax.bar(x - 1.5 * w, best_v, w, label='best single metadata-aware',
       color='#a78bfa')
ax.bar(x - 0.5 * w, cahr_v, w, label='CA-HR (always)', color='#dc2626')
ax.bar(x + 0.5 * w, routed_v, w, label='PAV-Agent routed', color='#60a5fa')
ax.bar(x + 1.5 * w, oracle_v, w, label='per-query oracle', color='#1e293b')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('NDCG@10')
ax.legend(fontsize=8, ncol=1, loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.set_title('Oracle headroom vs. learned routing (NDCG@10)', fontsize=11)
fig.tight_layout()
save(fig, 'Fig5_routing')

print('figures:', sorted(os.listdir(OUT)))

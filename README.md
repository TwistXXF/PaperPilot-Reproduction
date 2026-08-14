# PaperPilot-Reproduction

Reproduction package for the manuscript:

> **When does bibliographic metadata help scientific retrieval-augmented
> generation? A four-domain evaluation of metadata-aware hybrid retrieval
> with a deployed academic writing assistant**
> (submitted to *Expert Systems with Applications*)

This repository contains **all code, raw data, cached metadata, and final
result files** needed to reproduce every number, table, and figure in the
paper. Nothing is simulated: the corpora are the official BEIR benchmarks
(SCIDOCS, SciFact, NFCorpus, TREC-COVID), the citation metadata was
collected live from the Semantic Scholar / OpenAlex APIs, and the dense
retrievers are the public `all-MiniLM-L6-v2` and `bge-small-en-v1.5`
checkpoints.

The four-dataset ESWA study lives in [`eswa/`](#eswa-study-four-datasets).
The repository root keeps the original two-dataset (SCIDOCS + SciFact)
conference-version pipeline for reference.

---

## 1. Repository layout

```
PaperPilot-Reproduction/
├── eswa/                      # *** the ESWA study (start here) ***
│   ├── reproduce.py           # single staged entry point (layout-aware)
│   ├── _layout.py             # path resolver (repo vs. dev tree)
│   ├── bge_baseline.py        # BGE-small dense baseline, four datasets
│   ├── bge_hybrid.py          # BGE-Hybrid / BGE-CA-HR backbone transfer
│   ├── sensitivity_rrf.py     # beta x gamma grid + RRF(k=60) baselines
│   ├── metadata_diagnostics.py# citation-relevance association diagnostics
│   ├── gen_eval.py            # generation-side eval (needs DeepSeek key)
│   ├── make_eswa_tables.py    # consolidates results/eswa_tables.json
│   ├── make_eswa_figs.py      # manuscript Fig. 2-5 (PDF + PNG)
│   ├── build_eswa.py          # regenerates the submission package
│   ├── verify_eswa.py         # 208-check manuscript audit
│   ├── results/               # exact published result files
│   └── figures/               # manuscript figures
├── data/
│   ├── scidocs/  scifact/     # BEIR raw data (corpus/queries/qrels)
│   └── metadata/              # real citation metadata, all four datasets
│       ├── scidocs_metadata.json   # 25,582/25,657 matched (99.7%)
│       ├── scifact_metadata.json   #  4,879/5,183  matched (94.1%)
│       ├── nfcorpus_metadata.json  # 94.0% via Semantic Scholar batch API
│       └── trec-covid_metadata.json# 69.8% citations / 96.4% years /
│                                   # 92.5% venues (CORD-19 -> DOI -> OpenAlex)
├── artifacts/                 # cached MiniLM embeddings (scidocs/scifact)
├── results/                   # two-dataset per-query results (conference ver.)
├── manuscript/                # 01_Manuscript_ESWA.docx (audited by verify_eswa.py)
├── reproduce.py               # original two-dataset pipeline (conference ver.)
└── verify_paper_numbers.py    # conference-version audit
```

## 2. Installation

Python 3.10+ is required.

```bash
pip install -r requirements.txt
```

## 3. Models

Both encoders download automatically on first use (and are cached under
`eswa/models/`):

- `sentence-transformers/all-MiniLM-L6-v2` (main dense retriever)
- `BAAI/bge-small-en-v1.5` (stronger-backbone baseline)

(If huggingface.co is unreachable from your network, set
`HF_ENDPOINT=https://hf-mirror.com` before running.)

## 4. Reproducing the ESWA paper

The pipeline is staged; each stage is idempotent and skips work whose
outputs already exist, so it can be resumed at any point.

```bash
cd eswa
python reproduce.py all
```

This runs, in order:

| Stage | What it does | Fresh-clone runtime (CPU) |
|---|---|---|
| `download` | Fetch the four BEIR datasets (SCIDOCS/SciFact ship with the repo; NFCorpus/TREC-COVID download from the official BEIR zips) | ~10-30 min |
| `metadata` | Fetch citation metadata from Semantic Scholar (skipped: all four `*_metadata.json` ship with the repo) | 0 |
| `encode` | MiniLM embeddings for all corpora/queries (SCIDOCS/SciFact ship cached) | ~1 h |
| `retrieval` | BM25 / LSA / SBERT / Neural-Hybrid / UMA-RAG / LP-RAG / CA-HR, per-query metrics | ~15 min |
| `bge` | BGE-small dense baseline + BGE-Hybrid + BGE-CA-HR | ~2-4 h |
| `tables` | Aggregate tables + Wilcoxon / Cohen's d | < 1 min |
| `ablation` | CA-HR component ablation, four datasets | ~5 min |
| `robust` | Query word-drop noise robustness (10-40%) | ~10 min |
| `router` | PAV-Agent 5-fold CV routing analysis | < 1 min |
| `sensitivity` | BGE-CA-HR beta x gamma grid (30 combos x 4 datasets, Holm-corrected) + RRF(k=60) baselines on both encoders | ~15 min |
| `diagnostics` | Bibliographic-metadata diagnostics (citation-relevance AUC) | ~2 min |
| `eswa_tables` | Consolidate everything into `results/eswa_tables.json` | < 1 min |
| `eswa_figs` | Regenerate the manuscript figures into `figures/` | < 1 min |

Any stage can be run individually, optionally restricted to one dataset:

```bash
python reproduce.py sensitivity scidocs
```

The **generation-side evaluation is optional** and not part of `all`
(it calls the DeepSeek API; 200 paired generations + LLM judging):

```bash
echo "sk-..." > .deepseek_key        # git-ignored, never committed
python reproduce.py generation
```

All result files it would produce (`results/gen_eval_ckpt.jsonl`,
`results/gen_eval_summary.json`) ship with the repository.

## 5. Hyper-parameters (exactly as reported in the paper)

| Component | Setting |
|---|---|
| BM25 | rank_bm25 (k1=1.5, b=0.75, epsilon=0.25) |
| Dense retrievers | all-MiniLM-L6-v2; bge-small-en-v1.5 (official query instruction) |
| CA-HR | alpha=0.6 hybrid fusion; beta=0.15 citation, gamma=0.10 recency re-ranking of the top-100 |
| UMA-RAG | delta=0.10 venue, epsilon=0.10 citation |
| LP-RAG | length-prior scaling eta=0.2, mu=5000 |
| RRF baselines | k=60 |
| Sensitivity grid | beta in {0,.05,.10,.15,.20,.30}, gamma in {0,.05,.10,.15,.20} |
| Recency decay | exponential, lambda=0.1 per year, reference year 2024 |
| Robustness | query word-drop noise {10,20,30,40}% (seed 42); SCIDOCS on a fixed 300-query subsample (seed 7) |
| Router | logistic regression on 12 surface features, 5-fold stratified CV (seed 42) |
| Random seed | 42 everywhere unless stated |

## 6. Verifying the numbers

`eswa/verify_eswa.py` audits the manuscript itself: every number that
appears in `manuscript/01_Manuscript_ESWA.docx` (all table cells, all
in-text statistics, significance values, effect sizes, the sensitivity
grid, RRF baselines, dataset sizes, metadata coverage) is checked against
`results/eswa_tables.json`, plus staleness checks for leftover phrasing:

```bash
cd eswa
python verify_eswa.py
# -> "checks: 208, failed: 0, stale: 0"
```

`results/` contains the **exact files** from which every table was
typeset. Per-query scores in `*_perquery.npz` let you recompute any mean
and any significance test:

```python
import numpy as np
d = np.load("results/scidocs_bge_hybrid_perquery.npz", allow_pickle=True)
print(d["BGE-Hybrid||N@10"].mean())   # -> 0.1832 (Table 2)
```

## 7. Data sources and citations

- **SCIDOCS**: Cohan et al., *SPECTER: Document-level Representation
  Learning using Citation-informed Transformers*, ACL 2020.
- **SciFact**: Wadden et al., *Fact or Fiction: Verifying Scientific
  Claims*, EMNLP 2020.
- **NFCorpus / TREC-COVID / BEIR**: Thakur et al., *BEIR: A Heterogenous
  Benchmark for Zero-shot Evaluation of Information Retrieval Models*,
  NeurIPS 2021 (Datasets & Benchmarks).
- **Metadata**: [Semantic Scholar Academic Graph API](https://api.semanticscholar.org/)
  and [OpenAlex](https://openalex.org/) (collected 2025-2026).
- **Encoders**: Reimers & Gurevych, *Sentence-BERT*, EMNLP 2019
  (`all-MiniLM-L6-v2`); Xiao et al., *C-Pack*, 2023 (`bge-small-en-v1.5`).

## 8. Hardware / runtime notes

All experiments were run on a single CPU-only workstation (no GPU
required). With the shipped caches, re-running `retrieval` through
`eswa_figs` takes well under one hour; a from-scratch run including both
encoders takes roughly half a day, dominated by BGE encoding of
TREC-COVID (171k documents).

## 9. License

Code is released under the MIT License (see `LICENSE`). The datasets
remain under their original licenses (SCIDOCS: CC BY-NC-SA; SciFact:
CC BY-NC; NFCorpus / TREC-COVID: see the BEIR repository).

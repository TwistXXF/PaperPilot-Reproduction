# PaperPilot-Reproduction

Official reproduction package for the paper:

> **PaperPilot: Metadata-Aware Hybrid Retrieval and Personalized Routing for
> Academic Paper Recommendation** (submitted to *Information Processing &
> Management*)

This repository contains **all code, raw data, cached metadata, and final
result files** needed to reproduce every number, table, and figure in the
paper. Nothing is simulated: the corpora are the official BEIR/SciFact
benchmarks, the citation metadata was collected live from the Semantic
Scholar API, and the dense retriever is the public
`sentence-transformers/all-MiniLM-L6-v2` checkpoint.

---

## 1. Repository layout

```
PaperPilot-Reproduction/
├── reproduce.py               # single entry point, staged pipeline
├── requirements.txt
├── data/
│   ├── scidocs/               # BEIR SCIDOCS (25,657 docs / 1,000 test queries)
│   │   ├── corpus.jsonl
│   │   ├── queries.jsonl
│   │   └── qrels/test.tsv
│   ├── scifact/               # SciFact (5,183 docs / 300 test queries)
│   │   ├── corpus.jsonl
│   │   ├── queries.jsonl
│   │   └── qrels/{train,test}.tsv
│   └── metadata/              # real citation metadata from Semantic Scholar
│       ├── scidocs_metadata.json   # 25,582/25,657 matched (99.7%)
│       └── scifact_metadata.json   # 4,879/5,183 matched (94.1%)
├── artifacts/                 # cached embeddings (regenerable via `encode`)
│   ├── scidocs_emb/  scifact_emb/      # document embedding chunks
│   └── *_qemb.npy  *_qids.json         # query embeddings
├── results/                   # the exact result files reported in the paper
│   ├── tables.json                   # main results (Tables 2 & 3)
│   ├── {ds}_ablation.json            # component ablation (Table 4)
│   ├── {ds}_robust.json              # metadata-sparsity robustness (Table 5)
│   ├── {ds}_router.json              # routing analysis (Section 5.4)
│   ├── {ds}_latency.json             # per-query latency measurements
│   └── {ds}_perquery.npz             # per-query NDCG@10 for every method
├── figures/                   # Fig. 1–4 as they appear in the manuscript
└── models/minilm/             # local copy of all-MiniLM-L6-v2 (see §3)
```

## 2. Installation

Python 3.10+ is required.

```bash
pip install -r requirements.txt
```

## 3. Model

The dense retriever is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
A local copy is expected at `models/minilm/`. If it is missing, run:

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
m.save("models/minilm")
```

(If huggingface.co is unreachable from your network, set
`HF_ENDPOINT=https://hf-mirror.com` before running.)

## 4. Reproducing the paper

The pipeline is staged; each stage is idempotent and skips work whose outputs
already exist, so you can resume at any point.

```bash
python reproduce.py download     # fetch raw BEIR data (skipped — data/ ships with repo)
python reproduce.py metadata     # query Semantic Scholar (skipped — cached files ship with repo)
python reproduce.py encode       # embed all documents/queries (skipped — artifacts/ ship)
python reproduce.py retrieval    # BM25 + dense + all hybrid methods  (~3–5 min)
python reproduce.py tables       # aggregate main results -> results/tables.json
python reproduce.py ablation     # component ablation (Table 4)
python reproduce.py robust       # metadata-sparsity robustness (Table 5)
python reproduce.py router       # routing analysis (Section 5.4)
python reproduce.py figures      # regenerate Fig. 1–4 into figures/
```

Or run everything end-to-end:

```bash
python reproduce.py all
```

Because `data/`, `data/metadata/`, `artifacts/`, and `results/` all ship with
the repository, a fresh clone only needs `retrieval` → `figures` (a few
minutes on a CPU) to regenerate every number from the raw corpus upward.
Stages `download`, `metadata`, and `encode` are provided for full
transparency and can be re-run from scratch by deleting the corresponding
directories.

## 5. Hyper-parameters (exactly as reported in the paper)

| Component | Setting |
|---|---|
| BM25 | rank_bm25 default (k1=1.5, b=0.75), top-100 candidates |
| Dense retriever | all-MiniLM-L6-v2, cosine similarity |
| CA-HR weights | α=0.6 (dense), β=0.15 (BM25), γ=0.10 (citations), δ=0.10 (recency), ε=0.10 (venue) |
| LP-RAG | reciprocal-rank fusion (k=60) + paper-profile re-ranking, η=0.2 |
| Recency decay | exponential, reference year 2024 |
| Citation normalisation | log(1+c), clipped at μ=5000 |
| Router | λ=0.1, rule-based on query profile features |
| Robustness | query word-drop noise at {10,20,30,40}% (seed 42); SCIDOCS evaluated on a fixed 300-query subsample (seed 7) |
| Random seed | 42 everywhere unless stated |

## 6. Verifying the numbers

`verify_paper_numbers.py` audits the manuscript itself: it parses every table
in `01_Manuscript_IPM.docx` cell-by-cell and recomputes each value from the
raw per-query scores (including the Wilcoxon p-values and Cohen's d), then
checks every numeric claim in the running text (44.7% relative gain, oracle
gaps, router accuracy/kappa, robustness figures, metadata coverage, dataset
sizes). Run:

```bash
python verify_paper_numbers.py path/to/01_Manuscript_IPM.docx
# -> "142 passed, 0 failed"
```

`results/` contains the **exact files** from which every table in the paper
was typeset. `tables.json` is human-readable; per-query scores in
`*_perquery.npz` let you recompute any mean and any significance test:

```python
import numpy as np
d = np.load("results/scidocs_perquery.npz", allow_pickle=True)
print(d["SBERT-Dense"].mean())   # -> 0.2164 (Table 2)
```

## 7. Data sources and citations

- **SCIDOCS**: Cohan et al., *SPECTER: Document-level Representation Learning
  using Citation-informed Transformers*, ACL 2020. Distributed via
  [BEIR](https://github.com/beir-cellar/beir).
- **SciFact**: Wadden et al., *Fact or Fiction: Verifying Scientific Claims*,
  EMNLP 2020.
- **BEIR benchmark**: Thakur et al., *BEIR: A Heterogenous Benchmark for
  Zero-shot Evaluation of Information Retrieval Models*, NeurIPS 2021
  (Datasets & Benchmarks).
- **Metadata**: [Semantic Scholar Academic Graph API](https://api.semanticscholar.org/)
  (citation counts, years, venues; collected 2025).
- **Encoder**: Reimers & Gurevych, *Sentence-BERT*, EMNLP 2019;
  checkpoint `all-MiniLM-L6-v2`.

## 8. Hardware / runtime notes

All experiments were run on a single CPU-only workstation (no GPU required).
Full retrieval on SCIDOCS takes ≈2 minutes; the complete `all` pipeline from
raw data (including encoding) takes under one hour.

## 9. License

Code is released under the MIT License (see `LICENSE`). The datasets remain
under their original licenses (SCIDOCS: CC BY-NC-SA; SciFact: CC BY-NC).

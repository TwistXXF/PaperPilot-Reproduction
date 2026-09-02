# Locked protocol for the BiblioGuard re-evaluation

Protocol version: `3.2.0`

Frozen at: `2026-09-01T01:55:10Z`

v3.1 amended before locked-test metrics at: `2026-09-02T01:51:04Z`

v3.2 amended before locked-test metrics at: `2026-09-02T08:24:43Z`

The core v3.0 decisions were committed before the labelled
`allenai/scirepeval_test` RELISH configuration was downloaded.  Versions 3.1
and 3.2 record implementation and integrity corrections found before any
locked-test metric was computed.  By then, the public raw label container had
been downloaded and an initial development extractor had loaded the whole table
before filtering its output.  It persisted only training/calibration rows, and
no locked-label file or locked result was inspected, but this is not an
externally blinded study and we do not claim that locked label bytes were never
read.  The exact chronology and unchanged decision parameters are listed in
`ERRATA.md`.

The earlier BEIR results in this repository had already been inspected and tuned
against.  They are therefore retrospective evidence only and cannot be used as
a confirmatory test of the revised policy.

## Research question and stopping rule

The pre-committed locked question is whether a policy fitted on development queries
can use bibliographic metadata to improve SPECTER2 ranking on a locked RELISH
holdout without hiding query-level harm behind an overall mean.

The primary estimand is the paired mean change in NDCG@10 on every locked
holdout query, with abstentions assigned zero change.  The primary comparison
is BiblioGuard versus the unmodified SPECTER2 ranking.  We will not change the
method, action grid, split, metric, or operating-point rule after inspecting
the locked-holdout labels.  If a fixed action, a learned ranker, or a simpler
gain predictor matches or exceeds BiblioGuard, the paper will report that
outcome and will not claim superior safety or effectiveness.

## Dataset and split

- Unlabelled inputs: `allenai/scirepeval`, configuration `relish`.
- Labels: `allenai/scirepeval_test`, configuration `relish`.
- RELISH is used because its relevance judgements are supplied by scientists;
  unlike SCIDOCS, the target is not a future citation edge.
- Before label values are used for fitting or calibration, query titles are Unicode-NFKC normalised, lowercased,
  stripped of punctuation, and whitespace-collapsed.  Identical normalised
  titles form one group.  An empty title falls back to its Semantic Scholar
  CorpusId.
- Each group is assigned by
  `int(sha256("relish-v1|" + group_id).hexdigest(), 16) % 10`:
  buckets 0--1 are policy training, bucket 2 is calibration, and buckets 3--9
  are the locked test.  A content-only audit will report exact and near
  duplicates across splits; labels are never used to alter membership.
- Candidate order is not treated as a feature.  Score ties are broken by the
  dataset `doc_id`; metadata lookup separately uses exact CorpusId.
- The content-only near-duplicate audit uses character 3--5-gram TF--IDF cosine
  similarity over titles.  It does not alter the primary split.  A secondary
  sensitivity analysis excludes locked titles with similarity at least 0.80
  to a training or calibration title.

The raw public label Parquet contains all splits and may be present in the
evaluation environment.  Version 3.2 applies an Arrow-level `query_id` filter
before rows are converted to Python objects: `fit` materialises only training
rows, `calibrate` only calibration rows, `freeze` receives neither locked rows
nor locked metrics, and only the hash-gated evaluation phase materialises and
joins locked rows to frozen decisions.  This process isolation does not undo the
earlier whole-table read disclosed above and is not described as external
blinding.

## Inputs and provenance

For every query and candidate we retain CorpusId, title, abstract, and the raw
data row hash.  Bibliographic fields are obtained by exact CorpusId lookup from
the Semantic Scholar Graph API and stored as an immutable snapshot with source
URL, retrieval timestamp, HTTP status, requested fields, missingness, and
SHA-256.  The fields used by the policy are `citationCount` and `year` only.
They describe the current 2026 snapshot, not historical state; the paper must
not make a historical or causal claim from them.

Dataset revisions, model revisions, package versions, random seeds, hardware,
Git commit, and hashes of every decision-bearing input and output are written
to a machine-readable run manifest.

## Retrieval baselines

All models rank the same per-query candidate pool (53--60 documents, usually 60)
using title plus abstract.
The planned unsupervised baselines are BM25, BGE-small-en-v1.5, SciNCL, and
SPECTER2.  SPECTER2 is fixed a priori as the primary backbone; its choice is
not conditional on development performance.  A supervised LightGBM LambdaRank
model using the four retrieval scores plus the two metadata features is the
strong learned-fusion baseline.  Model identifiers and resolved Hugging Face
commit revisions are frozen in `config/models.json` and the run manifest.
SciNCL joins title and abstract with its official tokenizer separator; BGE uses
a space and SPECTER2 uses its base tokenizer separator.  LambdaRank is fitted
for the fixed 500 trees using training labels only and sets graded gains to
`[0,1,2]`; calibration labels are not used for early stopping.

## Metadata actions

Within each candidate pool, the SPECTER2 cosine score, `log1p(citationCount)`,
and year are min-max scaled to [0, 1].  A constant or fully missing field maps
to zero; missing individual values map to the observed median before scaling.
The fixed action family is:

- citation interpolation at weights 0.15 and 0.30;
- recency interpolation at weights 0.15 and 0.30;
- equal citation/recency interpolation at weights 0.15 and 0.30;
- reciprocal-rank fusion of SPECTER2 with citation rank at `k=60`;
- reciprocal-rank fusion of SPECTER2 with recency rank at `k=60`.

Equal metadata values receive midranks in reciprocal-rank fusion.  A constant
or fully missing metadata field therefore contributes a constant and cannot
create an identifier-driven ranking signal.

No new action or weight may be added after locked-test evaluation.  The
global-action baseline selects one of these actions by training mean NDCG@10
and applies it unchanged to the holdout.

## Selective policies

BiblioGuard uses word (1--2) and character (3--5) TF-IDF query-title features.
For a target query it takes `floor(sqrt(n_train))` nearest training queries,
clips negative cosine similarities to zero, adds `1e-8`, normalises the
weights, and estimates each action's weighted mean NDCG@10 change and standard
error.  The action predictor is fixed as the action with the largest estimated
mean; the gate score is the corresponding lower score `mean - 1.645 * SE`.
Action prediction is held fixed while a gate threshold is swept.

The operating threshold is chosen on calibration data as the lowest threshold
whose one-sided 95% Clopper--Pearson upper bound on the conditional harm rate
is at most 0.20, with at least 30 active calibration queries.  Among eligible
thresholds, choose the one with greatest coverage, breaking ties by higher
mean gain and then the stricter threshold.  If no threshold is eligible, the
policy abstains everywhere.  This is an empirical calibration rule, not a
formal per-query confidence guarantee.

Comparator selectors are: local kNN mean without the SE penalty; a
HistGradientBoosting multi-action gain predictor; and a global-action gain
regressor.  Each produces its own confidence score.  Curves are compared at
exact coverage budgets 10%, 25%, 50%, 75%, and 100% using deterministic top-k
selection, so methods are never compared only at their native coverage.

## Outcomes and inference

Primary metric: NDCG@10 with linear graded gains, matching
trec_eval/pytrec_eval.  Secondary retrieval metrics are NDCG@20, full-list
NDCG, MAP@10 (trec_eval `map_cut.10`, with all relevant candidates in the
denominator), Recall@50, and Precision@10.  The following policy outcomes are reported both
at the calibrated operating point and at matched coverage:

- overall mean gain, including zero for abstention;
- conditional mean gain among active queries;
- conditional harm probability `P(delta < 0 | active)`;
- severe-harm probability `P(delta <= -0.05 | active)`;
- mean negative shortfall `mean(max(0, -delta) | active)`;
- selective-risk curve and its trapezoidal area (AURC), where risk at coverage
  is conditional mean negative shortfall.

Uncertainty uses 10,000 query-group paired bootstrap replicates with seed
`20260901`.  The primary test is a two-sided paired randomisation test with
100,000 sign flips and the same seed.  Secondary method comparisons use Holm
adjustment.  Effect sizes and confidence intervals are reported even when a
p-value is not significant.  Missing metadata, zero-ideal-DCG queries, exact
duplicates, and all exclusions are counted explicitly.

## Retrospective BEIR audit

SCIDOCS, SciFact, NFCorpus, and TREC-COVID are retained only as an audit of the
previous paper.  Their old test-driven ablations cannot become confirmatory by
being rerun.  SCIDOCS citation-count interventions are excluded from efficacy
claims because current citation counts leak future/target information into a
citation-prediction task.  Any refreshed numbers are labelled exploratory and
are accompanied by duplicate-query and input-lineage checks.

# Revision audit against the KAIS editorial decision

This note maps the handling editor's two substantive criticisms to verifiable
changes.  It is an internal submission aid, not a claim that acceptance is assured.

## 1. “Comparative studies are insufficient”

- Every system now ranks the identical RELISH per-query pool (53--60 candidates)
  with the same labels, tie rule, and six retrieval metrics.
- Content baselines are BM25, BGE-small-en-v1.5, SciNCL, and the SPECTER2 proximity
  adapter at immutable model revisions.
- A training-only LightGBM LambdaRank fusion uses all four retrieval scores plus
  citation/year features; it is deliberately capable of disproving the need for the
  proposed selective layer.
- The selective policy is compared with a fixed global action, an ungated local kNN
  selector, a histogram-gradient-boosted multi-action selector, and a learned gate
  for the global action.
- Comparisons are reported at native operating points and at exact 10%, 25%, 50%,
  75%, and 100% coverage.  The manuscript includes a matched-coverage gain/harm
  table and a genuine conditional negative-shortfall risk--coverage curve.
- The stopping rule forbids a BiblioGuard superiority claim if a fixed action,
  LambdaRank, or a simpler selector matches or exceeds it.

## 2. “Recent references and top baselines are missing”

- The related-work section now includes directly relevant 2022--2025 work from
  KAIS, TKDE, KDD, SIGIR, ACL/NAACL/EMNLP, ECIR, Knowledge-Based Systems, and
  Scientometrics, with DOI-checked bibliography entries.
- Venue names were not added merely to satisfy a checklist.  Recent TKDD/ICDM work
  found in the audit did not provide a task-compatible RELISH seed-paper relatedness
  baseline; the manuscript instead explains why citation-context and academic-graph
  scores are not numerically comparable and adds the stronger task-aligned encoders
  and LambdaRank fusion.

## 3. Earlier claims that were withdrawn rather than polished

- The old SCIDOCS gain is retrospective because current citation counts are coupled
  to a citation-prediction target.
- SciFact/NFCorpus duplicate-query concerns and repeated official-test reuse are
  disclosed; those datasets are not reused as confirmatory tests.
- The unsupported “fuzzy gate” claim was removed because no implementation/result
  lineage exists.
- The old aggregate gain/coverage graphic is not called risk--coverage.  The new
  curve conditions negative shortfall on the selected queries.

## 4. Reproducibility and integrity corrections

- Inputs, raw metadata, model commits, environment versions, embeddings, aligned
  scores, decisions, and one-shot outputs are checksum-linked.
- Frozen decisions are committed and pushed before locked-test rows are materialised
  into an evaluation file.
- Publication refuses score/provenance files that differ from the public freeze.
  The verifier independently recomputes every `results.json` section and regenerates
  every numeric LaTeX table.
- The exact label-access limitation is disclosed: an early development extractor
  loaded the public all-split Parquet before filtering its output.  It persisted no
  locked-label file and computed no locked metric, but this prevents an external-
  blinding or “bytes were never read” claim.  Version 3.2 uses an Arrow-level query
  filter before Python materialisation.

## 5. Remaining limitations that cannot honestly be “fixed away”

- RELISH is biomedical and uses preassembled candidate pools; this is not a
  full-corpus retrieval or cross-domain evaluation.
- Citation/year metadata are a current snapshot, so the analysis is observational,
  not causal or historical.
- Calibration is an empirical split-specific screen, not a distribution-free safety
  guarantee.
- Offline NDCG and harm rates do not establish user satisfaction or discovery value.
- A null or negative locked result remains possible and will change the recommended
  target journal and paper framing.

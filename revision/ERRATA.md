# Pre-test implementation errata (v3.0 to v3.1)

These corrections were identified by a read-only code audit after the v3.0
protocol commit and before any locked RELISH labels were materialised.  They
are published before the one-shot test run so they cannot be chosen in response
to test performance.

1. **SciNCL preprocessing.** The generic sentence-transformer path joined title
   and abstract with a space.  The SciNCL model card requires `[SEP]`.  Version
   3.1 uses the tokenizer separator and records it in the embedding manifest.
   The interrupted space-joined SciNCL embedding is not used.
2. **RRF metadata ties.** Unique ID tie-breaking assigned artificial ranks to
   equal citation/year values.  Metadata now use midranks, so a fully missing
   or constant field contributes a constant and cannot rerank papers.
3. **Metric compatibility.** NDCG now uses linear gain and MAP@10 keeps the
   total relevant-document denominator, matching trec_eval/SciRepEval.
   Full-list NDCG is added.  LambdaRank uses `label_gain=[0,1,2]`.
4. **Calibration isolation.** LambdaRank no longer uses calibration labels for
   early stopping.  It fits the predeclared 500 trees on training labels only.
   Training and calibration labels and metric files are physically separate.
5. **Zero coverage.** Conditional gain and risk are `NA`, not zero, when a
   selector abstains on every query.
6. **Split audit.** The promised character-TFIDF near-duplicate report is now
   generated without labels.  Primary membership remains the v3.0 hash split;
   a predeclared 0.80-threshold sensitivity result is reported separately.
7. **Freeze integrity.** The freeze hashes every score and feature/layout input.
   Locked metrics require that manifest, refuse overwrite, and record its hash;
   final evaluation verifies the entire chain.
8. **Immutable model revisions.** Hugging Face commits are now inputs in
   `config/models.json`, not re-resolved from a moving repository head.

No action weight, coverage budget, primary cutoff, split bucket, harm threshold,
or stopping rule was changed.

# Pre-evaluation implementation errata (v3.0 to v3.2)

These corrections were identified after the v3.0 protocol commit and before any
locked RELISH metric was computed.  The public raw label container had already
been downloaded, and an early development extractor had read that full table
before filtering its output.  No locked-label file, locked metric, or locked result
was produced, but this history means the study is not described as externally
blinded or as one in which locked label bytes were never read.  The corrections
below are published before the one-shot test run and cannot be chosen in response
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
   Training and calibration output label and metric files are separate.
5. **Zero coverage.** Conditional gain and risk are `NA`, not zero, when a
   selector abstains on every query.
6. **Split audit.** The promised character-TFIDF near-duplicate report is now
   generated without labels.  Primary membership remains the v3.0 hash split;
   a predeclared 0.80-threshold sensitivity result is reported separately.
7. **Freeze and release integrity.** The freeze hashes every score,
   feature/layout input, score-provenance manifest, model configuration, and
   environment lock.  Locked metrics require that manifest and refuse overwrite.
   Publication now refuses any score or provenance file whose hash differs from
   the public freeze; the verifier independently recomputes every numeric section
   of `results.json` and the generated LaTeX tables.
8. **Immutable model revisions.** Hugging Face commits are now inputs in
   `config/models.json`, not re-resolved from a moving repository head.
9. **Label-container access.** The first development extraction converted the
   complete public label Parquet to Python rows and filtered afterwards.  It wrote
   only training/calibration rows and no locked outcome was inspected, but the old
   “never read” interpretation was too strong.  Version 3.2 applies an Arrow-level
   `query_id` filter before Python materialisation and records that the raw container
   itself contains all splits.  This limitation remains disclosed for the actual
   chronology rather than being erased by the code correction.
10. **Generated counts.** Dataset counts in the paper are now taken from the
    frozen prepared-data manifest instead of hard-coded LaTeX macros.

No action weight, coverage budget, primary cutoff, split bucket, harm threshold,
or stopping rule was changed.

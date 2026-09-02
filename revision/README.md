# BiblioGuard v3.2: leakage-aware RELISH reproduction

This directory is the clean experiment and paper lineage.  The files at the
repository root are retained only for forensic comparison with the rejected
manuscript; they are not presented as new confirmatory evidence.

The core v3.0 protocol was committed before the new RELISH label container was
downloaded.  Pre-evaluation corrections and the exact label-access chronology
are listed in `ERRATA.md`.  The pipeline keeps training, calibration, frozen
decisions, hash-gated label materialisation, and one-shot evaluation in separate
outputs and command phases.  It is not presented as externally blinded.

## Environment

Python 3.12 and an NVIDIA CUDA GPU were used for the released run.  Create an
isolated environment and install the exact lock file:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r revision\requirements-lock.txt
.venv\Scripts\python.exe -m unittest discover -s revision\tests -p "test_*.py" -v
```

The PyTorch lock targets CUDA 13.0.  CPU reproduction is possible by changing
the PyTorch build and passing `--device cpu`, but neural encoding will be much
slower and floating-point embedding hashes may differ.

## Full phased run

The commands below are ordered.  Do not run the locked-label commands until the
freeze files have been committed and pushed publicly.

```powershell
$PY = ".venv\Scripts\python.exe"
& $PY revision\run.py download-inputs
& $PY revision\run.py prepare
& $PY revision\run.py audit-content
& $PY revision\run.py download-labels
& $PY revision\run.py extract-development-labels
& $PY revision\run.py build-layout
& $PY revision\run.py bm25
& $PY revision\run.py metadata
& $PY revision\run.py encode --model bge --device cuda
& $PY revision\run.py score --model bge
& $PY revision\run.py encode --model scincl --device cuda
& $PY revision\run.py score --model scincl
& $PY revision\run.py encode --model specter2 --device cuda
& $PY revision\run.py score --model specter2
& $PY revision\run.py actions
& $PY revision\run.py lambdarank
& $PY revision\run.py development-metrics
& $PY revision\run.py freeze
& $PY revision\release.py freeze
```

Commit and push `revision/frozen/`.  Only then unlock and evaluate once:

```powershell
$MANIFEST = "revision\frozen\decision_manifest.json"
& $PY revision\run.py extract-locked-labels --decision-manifest $MANIFEST
& $PY revision\run.py locked-metrics --decision-manifest $MANIFEST
& $PY revision\run.py evaluate --decision-manifest $MANIFEST
& $PY revision\release.py publish
& $PY revision\verify_release.py
```

Locked label, metric, and final result files refuse overwrite.  The metrics
phase verifies every score-array hash against the public freeze; the evaluator
then verifies the locked-metric manifest and decision hash.  Publication also
checks every copied score/provenance hash against the public freeze, and the
final verifier independently recomputes retrieval means, policy outcomes,
matched coverage, risk curves/AURC, intervals, randomisation tests, sensitivity
analysis, multiplicity adjustment, and generated LaTeX numbers.

## Released artefacts

`published/` contains the machine-readable results, per-query outcomes, aligned
candidate score arrays, citation/year vectors, the complete compressed Semantic
Scholar snapshot, and all provenance manifests.  Large raw benchmark inputs and
document embeddings are regenerated from immutable dataset/model revisions.
`paper/generated/` is produced only from `published/results.json`.

The release preserves negative or null outcomes.  If a fixed action,
LambdaRank, or a simpler selector outperforms BiblioGuard, the paper reports that
fact and does not claim BiblioGuard superiority or a formal safety guarantee.

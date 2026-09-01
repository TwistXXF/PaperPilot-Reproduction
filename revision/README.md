# BiblioGuard v3: leakage-aware reproducibility package

This directory replaces the mixed-lineage ESWA scripts with a phased,
manifested experiment.  The first commit in this branch contains the locked
protocol and predates labelled RELISH evaluation.

Planned phases are `prepare`, `fit`, `calibrate`, `freeze`, `evaluate`, and
`paper`.  Each phase writes checksummed artefacts and refuses to consume files
from a later phase.  Exact commands and completed-run hashes will be added as
the implementation is verified.

The legacy files at repository root are retained for forensic comparison; they
are not silently presented as fresh results.

#!/usr/bin/env python
"""Audit the novelty-focused ESWA manuscript against released artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
PACKAGE = BASE.parent / "ESWA_submission_revised"
DATASETS = ("scidocs", "scifact", "nfcorpus", "trec-covid")
NAMES = {
    "scidocs": "SCIDOCS",
    "scifact": "SciFact",
    "nfcorpus": "NFCorpus",
    "trec-covid": "TREC-COVID",
}


def full_text(path: Path) -> str:
    doc = Document(path)
    parts = [paragraph.text for paragraph in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def contains(text: str, value: str, message: str) -> None:
    check(value in text, f"missing {message}: {value!r}")


def main() -> None:
    manuscript_path = PACKAGE / "01_Manuscript_ESWA.docx"
    highlights_path = PACKAGE / "02_Highlights.docx"
    cover_path = PACKAGE / "03_Cover_Letter.docx"
    declarations_path = PACKAGE / "04_Declarations.docx"
    for path in (manuscript_path, highlights_path, cover_path, declarations_path):
        check(path.exists(), f"missing package file {path.name}")

    manuscript = full_text(manuscript_path)
    highlights = full_text(highlights_path)
    cover = full_text(cover_path)
    declarations = full_text(declarations_path)
    bg = json.loads((RESULTS / "biblioguard_results.json").read_text())
    transfer = json.loads(
        (RESULTS / "biblioguard_transfer_results.json").read_text()
    )
    diagnostics = json.loads(
        (RESULTS / "metadata_diagnostics.json").read_text()
    )
    checks = 0

    contains(manuscript, "multi-domain scientific retrieval-augmented generation",
             "revised title")
    contains(manuscript, "operational pessimistic decision score",
             "score caveat")
    contains(manuscript, "not a formal confidence bound",
             "formal-bound disclaimer")
    contains(manuscript, "same underlying content retriever",
             "same-content action definition")
    contains(manuscript, "test qrels are opened only after decisions are frozen",
             "train-to-test isolation")
    checks += 5

    stale = (
        "Confidence-gated bibliographic metadata intervention",
        "cross-domain scientific retrieval",
        "BGE-Hybrid fallback on SciFact",
        "actions use CA-HR's 0.6/0.4 content mix",
        "BiblioGuard provides a distribution-free safety guarantee",
        "10.5281/zenodo.21930729",
        "SCIDOCS NDCG@10 rises by 0.0049",
    )
    for phrase in stale:
        check(phrase not in manuscript + highlights + declarations,
              f"stale or overclaimed phrase: {phrase}")
        checks += 1

    for dataset in DATASETS:
        row = bg["results"][dataset]
        contains(manuscript, f"{row['fallback_N@10']:.4f}",
                 f"{dataset} fallback")
        contains(manuscript, f"{row['biblioguard_N@10']:.4f}",
                 f"{dataset} BiblioGuard")
        contains(manuscript, f"{row['gain_N@10']:+.4f}",
                 f"{dataset} gain")
        ci = row["paired_bootstrap_95ci"]
        contains(manuscript, f"[{ci[0]:.4f}, {ci[1]:.4f}]",
                 f"{dataset} bootstrap interval")
        contains(manuscript,
                 f"{100 * row['selection_rate']:.1f}%",
                 f"{dataset} activation")
        contains(manuscript,
                 f"{row['repeated_seeds']['gain_mean']:+.4f} ± "
                 f"{row['repeated_seeds']['gain_std']:.4f}",
                 f"{dataset} repeated gain")
        for method in ("global_best", "local_mean", "uncorrected",
                       "empirical_bernstein", "biblioguard"):
            comparison = row["comparisons"][method]
            contains(manuscript, f"{comparison['N@10']:.4f}",
                     f"{dataset}/{method} score")
            contains(manuscript, str(comparison["outcomes_active"]["harmed"]),
                     f"{dataset}/{method} harmed count")
            checks += 2
        checks += 6

    for dataset in ("scifact", "nfcorpus"):
        row = transfer["results"][dataset]
        contains(manuscript, str(row["n_train"]), f"{dataset} transfer train n")
        contains(manuscript, str(row["n_test"]), f"{dataset} transfer test n")
        contains(manuscript, row["selected_content_base"],
                 f"{dataset} transfer base")
        contains(manuscript, f"{row['fallback_N@10']:.4f}",
                 f"{dataset} transfer fallback")
        contains(manuscript, f"{100 * row['selection_rate']:.1f}%",
                 f"{dataset} transfer activation")
        checks += 5

    for dataset in DATASETS:
        row = diagnostics[dataset]
        ci = row["cit_rel_auc_95ci"]
        contains(manuscript,
                 f"{row['cit_rel_auc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
                 f"{dataset} diagnostic AUC")
        contains(manuscript, str(row["rel_docs"]),
                 f"{dataset} positive diagnostic n")
        contains(manuscript, str(row["nonrel_docs"]),
                 f"{dataset} background diagnostic n")
        checks += 3

    # Highlights: five bullet lines, no jargon-heavy rejected wording, <=85 chars.
    highlight_doc = Document(highlights_path)
    bullets = [p.text for p in highlight_doc.paragraphs
               if p.style.name.startswith("List Bullet")]
    check(3 <= len(bullets) <= 5, "Elsevier highlight count")
    check(all(len(item) <= 85 for item in bullets), "Elsevier 85-character limit")
    check(all("qrels" not in item and "NDCG" not in item for item in bullets),
          "highlights contain specialist shorthand")
    checks += 3

    contains(cover, "ESWA-D-26-34170", "previous manuscript ID")
    contains(cover, "This is not a textual resubmission", "substantive revision")
    contains(cover, "same content score, fusion weights, and top-100",
             "cover confound fix")
    contains(cover, "official train-to-test", "cover independent validation")
    checks += 4

    print(f"ESWA manuscript verification passed: {checks} checks")


if __name__ == "__main__":
    main()

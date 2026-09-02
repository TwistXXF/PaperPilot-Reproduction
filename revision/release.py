from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / "artifacts" / ".matplotlib")
)
import matplotlib.pyplot as plt


REVISION = Path(__file__).resolve().parent
REPOSITORY = REVISION.parent
sys.path.insert(0, str(REVISION / "src"))

from biblioguard_v3.io import read_json, sha256_file, write_json  # noqa: E402


METHOD_LABELS = {
    "biblioguard": "BiblioGuard",
    "knn_mean": "kNN mean (ungated)",
    "hgb_gain": "HGB gain selector",
    "global_gain_gate": "Global action + learned gate",
    "global_all": "Global action (all queries)",
}

RETRIEVER_LABELS = {
    "bm25": "BM25",
    "bge": "BGE-small-en-v1.5",
    "scincl": "SciNCL",
    "specter2": "SPECTER2 proximity",
    "lambdarank": "LambdaRank fusion",
}

ACTION_LABELS = {
    "citation_015": r"Citation ($w=0.15$)",
    "citation_030": r"Citation ($w=0.30$)",
    "recency_015": r"Recency ($w=0.15$)",
    "recency_030": r"Recency ($w=0.30$)",
    "balanced_015": r"Citation + recency ($w=0.15$)",
    "balanced_030": r"Citation + recency ($w=0.30$)",
    "rrf_citation": "RRF with citation",
    "rrf_recency": "RRF with recency",
}

METRIC_ORDER = [
    "ndcg_at_10",
    "ndcg_at_20",
    "ndcg_full",
    "map_cut_10",
    "recall_at_50",
    "precision_at_10",
]


def git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPOSITORY, check=True, capture_output=True, text=True
    ).stdout.strip()


def copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256_file(source)
    if sha256_file(destination) != source_hash:
        raise RuntimeError(f"Copy verification failed: {source} -> {destination}")
    return {"path": destination.relative_to(REPOSITORY).as_posix(), "sha256": source_hash}


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: {actual} != {expected}")


def latex_number(value: float, digits: int = 4, signed: bool = False) -> str:
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value:.{digits}f}"


def latex_pvalue(value: float) -> str:
    if value < 0.0001:
        exponent = int(f"{value:.1e}".split("e")[1])
        mantissa = value / (10**exponent)
        return f"{mantissa:.2f}\\times 10^{{{exponent}}}"
    return f"{value:.4f}"


def optional_number(value: float | None, *, percent: bool = False, signed: bool = False) -> str:
    if value is None:
        return "--"
    numeric = 100 * value if percent else value
    suffix = r"\%" if percent else ""
    sign = "+" if signed and numeric > 0 else ""
    digits = 1 if percent else 4
    return f"{sign}{numeric:.{digits}f}{suffix}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def freeze_release(artifacts: Path) -> dict[str, Any]:
    source = artifacts / "frozen"
    destination = REVISION / "frozen"
    manifest = read_json(source / "decision_manifest.json")
    if manifest.get("test_labels_consumed") is not False:
        raise RuntimeError("Refusing to release decisions without test_labels_consumed=false")
    if manifest.get("test_labels_materialised_for_evaluation") is not False:
        raise RuntimeError("Refusing to release decisions after locked labels were materialised")
    copied = [
        copy_verified(source / "decision_manifest.json", destination / "decision_manifest.json"),
        copy_verified(source / manifest["decisions_file"], destination / manifest["decisions_file"]),
        copy_verified(source / manifest["calibration_file"], destination / manifest["calibration_file"]),
    ]
    report = {
        "phase": "public_freeze",
        "source_commit": git_value("rev-parse", "HEAD"),
        "decision_manifest_sha256": sha256_file(destination / "decision_manifest.json"),
        "test_labels_consumed": False,
        "test_labels_materialised_for_evaluation": False,
        "files": copied,
    }
    write_json(destination / "release.json", report)
    return report


def render_tables_and_figure(results: dict[str, Any], generated: Path) -> None:
    primary = results["primary"]
    biblioguard = results["operating_point"]["biblioguard"]
    dataset = results["dataset_summary"]
    low, high = primary["bootstrap_95_ci"]
    macros = "\n".join(
        [
            rf"\newcommand{{\TrainQueries}}{{{dataset['train_queries']}}}",
            rf"\newcommand{{\CalibrationQueries}}{{{dataset['calibration_queries']}}}",
            rf"\newcommand{{\TestQueries}}{{{dataset['locked_queries']}}}",
            rf"\newcommand{{\CandidatePairs}}{{{dataset['candidate_pairs']}}}",
            rf"\newcommand{{\UniqueDocuments}}{{{dataset['unique_documents']}}}",
            rf"\newcommand{{\PrimaryGain}}{{{latex_number(primary['mean_effect'], signed=True)}}}",
            rf"\newcommand{{\PrimaryCI}}{{[{latex_number(low, signed=True)}, {latex_number(high, signed=True)}]}}",
            rf"\newcommand{{\PrimaryPValue}}{{{latex_pvalue(primary['paired_randomisation_p'])}}}",
            rf"\newcommand{{\PrimaryCoverage}}{{{100 * biblioguard['coverage']:.1f}\%}}",
            "",
        ]
    )
    write_text(generated / "results_macros.tex", macros)

    retrieval_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Locked-test retrieval effectiveness on RELISH. All values are query means.}",
        r"\label{tab:retrievers}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"System & NDCG@10 & NDCG@20 & NDCG & MAP@10 & Recall@50 & P@10 \\",
        r"\midrule",
    ]
    for system in ["bm25", "bge", "scincl", "specter2", "lambdarank"]:
        values = results["retrieval_metrics"][system]
        cells = " & ".join(latex_number(values[name]) for name in METRIC_ORDER)
        retrieval_lines.append(f"{RETRIEVER_LABELS[system]} & {cells} \\\\")
    retrieval_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    write_text(generated / "table_retrievers.tex", "\n".join(retrieval_lines))

    policy_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Native locked-test operating points. Gain is measured against SPECTER2 NDCG@10 over all queries; the remaining risk columns condition on intervention.}",
        r"\label{tab:policy}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Selector & Coverage & Policy NDCG & Mean gain & Active gain & Harm & Shortfall \\",
        r"\midrule",
    ]
    for method in ["biblioguard", "global_gain_gate", "hgb_gain", "knn_mean", "global_all"]:
        row = results["operating_point"][method]
        policy_lines.append(
            f"{METHOD_LABELS[method]} & {100 * row['coverage']:.1f}\\% & "
            f"{row['policy_ndcg_at_10']:.4f} & {row['overall_mean_gain']:+.4f} & "
            f"{optional_number(row['conditional_mean_gain'], signed=True)} & "
            f"{optional_number(row['conditional_harm_probability'], percent=True)} & "
            f"{optional_number(row['mean_negative_shortfall'])} \\\\"
        )
    policy_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    write_text(generated / "table_policy.tex", "\n".join(policy_lines))

    matched_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Matched-coverage comparison. Each cell is overall mean NDCG@10 gain / conditional harm probability among active queries.}",
        r"\label{tab:matched}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Selector & 10\% & 25\% & 50\% & 75\% & 100\% \\",
        r"\midrule",
    ]
    for method in ["biblioguard", "global_gain_gate", "hgb_gain", "knn_mean"]:
        cells: list[str] = []
        for budget in ["0.10", "0.25", "0.50", "0.75", "1.00"]:
            row = results["matched_coverage"][method][budget]
            harm = row["conditional_harm_probability"]
            harm_text = "--" if harm is None else f"{100 * harm:.1f}\\%"
            cells.append(f"{row['overall_mean_gain']:+.4f} / {harm_text}")
        matched_lines.append(f"{METHOD_LABELS[method]} & " + " & ".join(cells) + r" \\")
    matched_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    write_text(generated / "table_matched.tex", "\n".join(matched_lines))

    action_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Fixed metadata actions on every locked query.}",
        r"\label{tab:actions}",
        r"\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Action & NDCG@10 & Gain vs. SPECTER2 \\",
        r"\midrule",
    ]
    fixed_actions = results["fixed_action_ndcg_at_10"]
    action_order = [name for name in ACTION_LABELS if name in fixed_actions]
    action_order.extend(sorted(set(fixed_actions).difference(action_order)))
    for action in action_order:
        ndcg = results["fixed_action_ndcg_at_10"][action]
        gain = results["fixed_action_gain_vs_specter2"][action]
        label = ACTION_LABELS.get(action, action.replace("_", r"\_"))
        action_lines.append(f"{label} & {ndcg:.4f} & {gain:+.4f} \\\\")
    action_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    write_text(generated / "table_actions.tex", "\n".join(action_lines))

    figure_path = generated / "figure_risk_coverage.pdf"
    plt.figure(figsize=(6.4, 4.0))
    for method in ["biblioguard", "global_gain_gate", "hgb_gain", "knn_mean"]:
        curve = results["risk_coverage"][method]
        plt.plot(curve["coverage"], curve["risk"], label=f"{METHOD_LABELS[method]} (AURC={curve['aurc']:.4f})")
    plt.xlabel("Coverage")
    plt.ylabel("Conditional negative shortfall")
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    generated.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()


def publish_release(artifacts: Path) -> dict[str, Any]:
    import torch

    git_dirty_before_publish = bool(git_value("status", "--porcelain"))
    source_results = artifacts / "results"
    source_frozen = artifacts / "frozen"
    source_metrics = artifacts / "metrics"
    published = REVISION / "published"
    generated = REVISION / "paper" / "generated"
    results = read_json(source_results / "results.json")
    frozen_manifest = read_json(source_frozen / "decision_manifest.json")
    if results["frozen_decisions_sha256"] != frozen_manifest["decisions_sha256"]:
        raise RuntimeError("Results do not descend from the frozen decision file")
    public_frozen_manifest_path = REVISION / "frozen" / "decision_manifest.json"
    require_hash(
        public_frozen_manifest_path,
        sha256_file(source_frozen / "decision_manifest.json"),
        "Public frozen decision manifest",
    )
    require_hash(
        REVISION / "frozen" / frozen_manifest["decisions_file"],
        frozen_manifest["decisions_sha256"],
        "Public frozen decisions",
    )
    if results.get("dataset_summary") != frozen_manifest.get("dataset_summary"):
        raise RuntimeError("Result dataset summary differs from the public freeze")

    action_names = sorted(results["fixed_action_ndcg_at_10"])
    expected_score_names = {
        "bm25",
        "bge",
        "scincl",
        "specter2",
        "lambdarank",
        *(f"action_{name}" for name in action_names),
    }
    frozen_score_hashes = frozen_manifest.get("score_hashes")
    if not isinstance(frozen_score_hashes, dict) or set(frozen_score_hashes) != expected_score_names:
        raise RuntimeError("Frozen score hash family is incomplete or unexpected")
    for name, expected in frozen_score_hashes.items():
        require_hash(artifacts / "scores" / f"{name}.npy", expected, f"Frozen score {name}")

    frozen_feature_hashes = frozen_manifest.get("feature_hashes")
    if not isinstance(frozen_feature_hashes, dict):
        raise RuntimeError("Frozen feature/layout hashes are missing")
    for name, expected in frozen_feature_hashes.items():
        require_hash(artifacts / "scores" / name, expected, f"Frozen feature {name}")
    require_hash(
        artifacts / "prepared" / "prepare_manifest.json",
        frozen_manifest["prepare_manifest_sha256"],
        "Prepared-data manifest",
    )
    prepared_manifest = read_json(artifacts / "prepared" / "prepare_manifest.json")
    raw_input_download = read_json(artifacts / "raw" / "relish_inputs.parquet.download.json")
    if prepared_manifest.get("source", {}).get("sha256") != raw_input_download.get("sha256"):
        raise RuntimeError("Prepared data do not descend from the pinned raw input file")
    require_hash(
        artifacts / "prepared" / "content_audit.json",
        frozen_manifest["content_audit_sha256"],
        "Content-only split audit",
    )
    require_hash(
        source_metrics / "train.jsonl.gz",
        frozen_manifest["training_metrics_sha256"],
        "Frozen training metrics",
    )
    require_hash(
        source_metrics / "calibration.jsonl.gz",
        frozen_manifest["calibration_metrics_sha256"],
        "Frozen calibration metrics",
    )
    require_hash(
        REVISION / "config" / "models.json",
        frozen_manifest["models_config_sha256"],
        "Model configuration",
    )
    require_hash(
        REVISION / "requirements-lock.txt",
        frozen_manifest["requirements_lock_sha256"],
        "Requirements lock",
    )
    prepared_documents_sha256 = prepared_manifest["outputs"]["documents"]["sha256"]
    for model in ("bge", "scincl", "specter2"):
        score_manifest = read_json(artifacts / "scores" / f"{model}.manifest.json")
        embedding_manifest_path = artifacts / "embeddings" / f"{model}.manifest.json"
        require_hash(
            embedding_manifest_path,
            score_manifest["model_manifest_sha256"],
            f"Embedding manifest {model}",
        )
        embedding_manifest = read_json(embedding_manifest_path)
        if embedding_manifest.get("documents_sha256") != prepared_documents_sha256:
            raise RuntimeError(f"{model} embeddings used a different prepared document file")
        if score_manifest.get("scores_sha256") != frozen_score_hashes[model]:
            raise RuntimeError(f"{model} score manifest differs from the frozen score array")

    bm25_manifest = read_json(artifacts / "scores" / "bm25.manifest.json")
    if bm25_manifest.get("scores_sha256") != frozen_score_hashes["bm25"]:
        raise RuntimeError("BM25 manifest differs from the frozen score array")

    actions_manifest = read_json(artifacts / "scores" / "actions.manifest.json")
    metadata_path = artifacts / "metadata" / "semantic_scholar.jsonl.gz"
    require_hash(metadata_path, actions_manifest["metadata_sha256"], "Metadata snapshot")
    metadata_manifest = read_json(metadata_path.with_suffix(metadata_path.suffix + ".manifest.json"))
    if metadata_manifest.get("snapshot_sha256") != sha256_file(metadata_path):
        raise RuntimeError("Metadata snapshot differs from its provenance manifest")
    for name, expected in actions_manifest.get("outputs", {}).items():
        path = artifacts / "scores" / f"{name}.npy"
        require_hash(path, expected, f"Action output {name}")

    lambdarank_manifest = read_json(artifacts / "scores" / "lambdarank.manifest.json")
    if lambdarank_manifest.get("scores_sha256") != frozen_score_hashes["lambdarank"]:
        raise RuntimeError("LambdaRank manifest differs from the frozen score array")
    require_hash(
        artifacts / "scores" / "lambdarank_model.txt",
        lambdarank_manifest["model_sha256"],
        "LambdaRank model",
    )

    raw_label_download = read_json(artifacts / "raw" / "relish_labels.parquet.download.json")
    split_files = {
        "train": ("train.jsonl.gz", "train.jsonl.gz"),
        "calibration": ("calibration.jsonl.gz", "calibration.jsonl.gz"),
        "locked_test": ("locked_test.jsonl.gz", "locked_test.jsonl.gz"),
    }
    for split, (label_name, metric_name) in split_files.items():
        label_path = artifacts / "labels" / label_name
        label_manifest = read_json(label_path.with_suffix(label_path.suffix + ".manifest.json"))
        require_hash(label_path, label_manifest["output_sha256"], f"{split} labels")
        if label_manifest.get("source_sha256") != raw_label_download.get("sha256"):
            raise RuntimeError(f"{split} labels do not descend from the pinned raw label file")
        metric_path = source_metrics / metric_name
        metric_manifest = read_json(metric_path.with_suffix(metric_path.suffix + ".manifest.json"))
        if metric_manifest.get("labels_sha256") != label_manifest["output_sha256"]:
            raise RuntimeError(f"{split} metrics and label manifests differ")

    locked_metrics_path = source_metrics / "locked_test.jsonl.gz"
    locked_metrics_manifest = read_json(
        locked_metrics_path.with_suffix(locked_metrics_path.suffix + ".manifest.json")
    )
    require_hash(locked_metrics_path, results["locked_metrics_sha256"], "Locked metrics")
    if locked_metrics_manifest.get("score_hashes") != frozen_score_hashes:
        raise RuntimeError("Locked metrics and frozen score hashes differ")
    if locked_metrics_manifest.get("frozen_decision_manifest_sha256") != sha256_file(
        public_frozen_manifest_path
    ):
        raise RuntimeError("Locked metrics do not descend from the public freeze")
    require_hash(
        source_results / "locked_per_query.jsonl.gz",
        results["per_query_sha256"],
        "Locked per-query outcomes",
    )

    copied = [
        copy_verified(source_results / "results.json", published / "results.json"),
        copy_verified(source_results / "locked_per_query.jsonl.gz", published / "locked_per_query.jsonl.gz"),
        copy_verified(source_metrics / "locked_test.jsonl.gz", published / "locked_metrics.jsonl.gz"),
        copy_verified(source_metrics / "train.jsonl.gz", published / "train_metrics.jsonl.gz"),
        copy_verified(
            source_metrics / "calibration.jsonl.gz", published / "calibration_metrics.jsonl.gz"
        ),
        copy_verified(
            metadata_path,
            published / "metadata" / "semantic_scholar.jsonl.gz",
        ),
    ]
    score_files = [
        "qids.json",
        "candidate_doc_ids.json",
        "offsets.npy",
        "citation_count.npy",
        "year.npy",
        "lambdarank_model.txt",
    ] + [f"{name}.npy" for name in sorted(frozen_score_hashes)]
    for name in score_files:
        copied.append(
            copy_verified(artifacts / "scores" / name, published / "scores" / name)
        )
    manifest_sources = [
        ("raw", artifacts / "raw" / "relish_inputs.parquet.download.json"),
        ("raw", artifacts / "raw" / "relish_labels.parquet.download.json"),
        ("prepared", artifacts / "prepared" / "prepare_manifest.json"),
        ("prepared", artifacts / "prepared" / "content_audit.json"),
        ("labels", artifacts / "labels" / "train.jsonl.gz.manifest.json"),
        ("labels", artifacts / "labels" / "calibration.jsonl.gz.manifest.json"),
        ("labels", artifacts / "labels" / "locked_test.jsonl.gz.manifest.json"),
        ("metrics", source_metrics / "train.jsonl.gz.manifest.json"),
        ("metrics", source_metrics / "calibration.jsonl.gz.manifest.json"),
        ("metrics", source_metrics / "locked_test.jsonl.gz.manifest.json"),
        ("metadata", artifacts / "metadata" / "semantic_scholar.jsonl.gz.manifest.json"),
        ("embeddings", artifacts / "embeddings" / "bge.manifest.json"),
        ("embeddings", artifacts / "embeddings" / "scincl.manifest.json"),
        ("embeddings", artifacts / "embeddings" / "specter2.manifest.json"),
        ("scores", artifacts / "scores" / "layout.manifest.json"),
        ("scores", artifacts / "scores" / "bm25.manifest.json"),
        ("scores", artifacts / "scores" / "bge.manifest.json"),
        ("scores", artifacts / "scores" / "scincl.manifest.json"),
        ("scores", artifacts / "scores" / "specter2.manifest.json"),
        ("scores", artifacts / "scores" / "actions.manifest.json"),
        ("scores", artifacts / "scores" / "lambdarank.manifest.json"),
        ("frozen", source_frozen / "decision_manifest.json"),
    ]
    for category, source in manifest_sources:
        copied.append(copy_verified(source, published / "manifests" / category / source.name))

    render_tables_and_figure(results, generated)
    generated_files = sorted(generated.glob("*"))
    report = {
        "phase": "publish",
        "source_commit": git_value("rev-parse", "HEAD"),
        "git_dirty_before_publish": git_dirty_before_publish,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "protocol_sha256": sha256_file(REVISION / "config" / "protocol.json"),
        "models_config_sha256": sha256_file(REVISION / "config" / "models.json"),
        "requirements_lock_sha256": sha256_file(REVISION / "requirements-lock.txt"),
        "results_sha256": sha256_file(published / "results.json"),
        "frozen_decisions_sha256": results["frozen_decisions_sha256"],
        "locked_metrics_sha256": results["locked_metrics_sha256"],
        "copied_files": copied,
        "generated_files": [
            {"path": path.relative_to(REPOSITORY).as_posix(), "sha256": sha256_file(path)}
            for path in generated_files
        ],
    }
    write_json(published / "release_manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Release frozen decisions or verified paper artefacts")
    parser.add_argument("phase", choices=["freeze", "publish"])
    parser.add_argument(
        "--artifacts", type=Path, default=REVISION / "artifacts" / "v3", help="Run artefact root"
    )
    arguments = parser.parse_args()
    report = (
        freeze_release(arguments.artifacts.resolve())
        if arguments.phase == "freeze"
        else publish_release(arguments.artifacts.resolve())
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

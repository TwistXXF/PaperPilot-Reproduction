from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REVISION = Path(__file__).resolve().parent
REPOSITORY = REVISION.parent
sys.path.insert(0, str(REVISION / "src"))

from biblioguard_v3.data import (  # noqa: E402
    download_file,
    extract_labels,
    fetch_semantic_scholar_metadata,
    prepare_unlabelled,
)
from biblioguard_v3.io import read_json, sha256_file, write_json  # noqa: E402
from biblioguard_v3.experiment import (  # noqa: E402
    build_metadata_actions,
    evaluate_frozen_decisions,
    evaluate_score_files,
    freeze_decisions,
)
from biblioguard_v3.protocol import load_protocol  # noqa: E402
from biblioguard_v3.retrieval import (  # noqa: E402
    build_candidate_layout,
    encode_sentence_transformer,
    encode_specter2,
    score_bm25,
    score_embedding_model,
)


def paths(root: Path) -> dict[str, Path]:
    return {
        "raw": root / "raw",
        "prepared": root / "prepared",
        "labels": root / "labels",
        "metadata": root / "metadata",
        "frozen": root / "frozen",
        "embeddings": root / "embeddings",
        "scores": root / "scores",
        "metrics": root / "metrics",
        "results": root / "results",
    }


def emit(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def download_named_source(name: str, destination: Path) -> dict[str, object]:
    source = read_json(REVISION / "config" / "sources.json")[name]
    if destination.exists():
        return {
            "status": "already_present",
            "path": destination.resolve().as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "pinned_revision": source["revision"],
        }
    report = download_file(source["url"], destination)
    report["pinned_revision"] = source["revision"]
    report["repository"] = source["repository"]
    write_json(destination.with_suffix(destination.suffix + ".download.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="BiblioGuard v3 phased reproduction")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=REVISION / "artifacts" / "v3",
        help="Generated artefact root (default: revision/artifacts/v3)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download-inputs", help="Download pinned unlabelled RELISH inputs")
    subparsers.add_parser("prepare", help="Normalise unlabelled inputs and freeze the split")
    subparsers.add_parser("download-labels", help="Download pinned RELISH labels")
    subparsers.add_parser(
        "extract-development-labels",
        help="Materialise training and calibration labels, never locked-test labels",
    )
    locked = subparsers.add_parser(
        "extract-locked-labels",
        help="Materialise locked labels only after frozen decisions exist",
    )
    locked.add_argument("--decision-manifest", type=Path, required=True)
    subparsers.add_parser("metadata", help="Fetch exact-CorpusId Semantic Scholar metadata")
    subparsers.add_parser("build-layout", help="Freeze the shared candidate score layout")
    subparsers.add_parser("bm25", help="Generate BM25 candidate scores")
    encode = subparsers.add_parser("encode", help="Encode all unique documents with one model")
    encode.add_argument("--model", choices=["bge", "scincl", "specter2"], required=True)
    encode.add_argument("--device", choices=["cpu", "cuda"], default=None)
    score = subparsers.add_parser("score", help="Generate candidate scores from embeddings")
    score.add_argument("--model", choices=["bge", "scincl", "specter2"], required=True)
    subparsers.add_parser("actions", help="Generate the frozen metadata action score family")
    subparsers.add_parser("development-metrics", help="Evaluate score files on development labels")
    subparsers.add_parser("freeze", help="Fit, calibrate, and freeze locked-test decisions")
    subparsers.add_parser("locked-metrics", help="Evaluate all score files on unlocked test labels")
    subparsers.add_parser("evaluate", help="Evaluate immutable frozen decisions")
    arguments = parser.parse_args()

    artefact_paths = paths(arguments.artifacts.resolve())
    protocol = load_protocol(REVISION / "config" / "protocol.json")
    sources = read_json(REVISION / "config" / "sources.json")
    models = read_json(REVISION / "config" / "models.json")
    input_parquet = artefact_paths["raw"] / "relish_inputs.parquet"
    label_parquet = artefact_paths["raw"] / "relish_labels.parquet"

    if arguments.command == "download-inputs":
        emit(download_named_source("relish_inputs", input_parquet))
    elif arguments.command == "prepare":
        if not input_parquet.exists():
            parser.error("Run download-inputs before prepare")
        emit(prepare_unlabelled(input_parquet, artefact_paths["prepared"], protocol["dataset"]))
    elif arguments.command == "download-labels":
        emit(download_named_source("relish_labels", label_parquet))
    elif arguments.command == "extract-development-labels":
        if not label_parquet.exists():
            parser.error("Run download-labels before extracting development labels")
        emit(
            extract_labels(
                label_parquet,
                artefact_paths["prepared"],
                artefact_paths["labels"] / "development.jsonl.gz",
                {"train", "calibration"},
            )
        )
    elif arguments.command == "extract-locked-labels":
        if not label_parquet.exists():
            parser.error("Run download-labels before extracting locked labels")
        emit(
            extract_labels(
                label_parquet,
                artefact_paths["prepared"],
                artefact_paths["labels"] / "locked_test.jsonl.gz",
                {"locked_test"},
                frozen_decisions_manifest=arguments.decision_manifest.resolve(),
            )
        )
    elif arguments.command == "metadata":
        emit(
            fetch_semantic_scholar_metadata(
                artefact_paths["prepared"] / "documents.jsonl.gz",
                artefact_paths["metadata"] / "semantic_scholar.jsonl.gz",
                sources["semantic_scholar"],
            )
        )
    elif arguments.command == "build-layout":
        emit(build_candidate_layout(artefact_paths["prepared"], artefact_paths["scores"]))
    elif arguments.command == "bm25":
        emit(score_bm25(artefact_paths["prepared"], artefact_paths["scores"]))
    elif arguments.command == "encode":
        config = models[arguments.model]
        if config["kind"] == "sentence_transformer":
            emit(
                encode_sentence_transformer(
                    artefact_paths["prepared"] / "documents.jsonl.gz",
                    artefact_paths["embeddings"],
                    arguments.model,
                    config,
                    device=arguments.device,
                )
            )
        elif config["kind"] == "adapter":
            emit(
                encode_specter2(
                    artefact_paths["prepared"] / "documents.jsonl.gz",
                    artefact_paths["embeddings"],
                    arguments.model,
                    config,
                    device=arguments.device,
                )
            )
        else:
            raise ValueError(f"Unknown model kind: {config['kind']}")
    elif arguments.command == "score":
        emit(
            score_embedding_model(
                artefact_paths["prepared"],
                artefact_paths["embeddings"],
                artefact_paths["scores"],
                arguments.model,
            )
        )
    elif arguments.command == "actions":
        emit(
            build_metadata_actions(
                artefact_paths["prepared"],
                artefact_paths["scores"],
                artefact_paths["metadata"] / "semantic_scholar.jsonl.gz",
                protocol,
            )
        )
    elif arguments.command == "development-metrics":
        artefact_paths["metrics"].mkdir(parents=True, exist_ok=True)
        emit(
            evaluate_score_files(
                artefact_paths["labels"] / "development.jsonl.gz",
                artefact_paths["scores"],
                artefact_paths["metrics"] / "development.jsonl.gz",
                [action["name"] for action in protocol["actions"]],
            )
        )
    elif arguments.command == "freeze":
        emit(
            freeze_decisions(
                artefact_paths["prepared"],
                artefact_paths["scores"],
                artefact_paths["metrics"] / "development.jsonl.gz",
                REVISION / "config" / "protocol.json",
                artefact_paths["frozen"],
            )
        )
    elif arguments.command == "locked-metrics":
        artefact_paths["metrics"].mkdir(parents=True, exist_ok=True)
        emit(
            evaluate_score_files(
                artefact_paths["labels"] / "locked_test.jsonl.gz",
                artefact_paths["scores"],
                artefact_paths["metrics"] / "locked_test.jsonl.gz",
                [action["name"] for action in protocol["actions"]],
            )
        )
    elif arguments.command == "evaluate":
        emit(
            evaluate_frozen_decisions(
                artefact_paths["frozen"],
                artefact_paths["metrics"] / "locked_test.jsonl.gz",
                REVISION / "config" / "protocol.json",
                artefact_paths["results"],
            )
        )
    else:
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()

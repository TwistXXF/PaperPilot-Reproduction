from __future__ import annotations

import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests

from .io import (
    read_json,
    read_jsonl_gz,
    sha256_file,
    write_json,
    write_jsonl_gz,
)
from .splits import assign_split, audit_splits, group_id, split_bucket


def download_file(url: str, destination: Path, timeout: int = 60) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    started = datetime.now(timezone.utc).isoformat()
    with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        headers = {
            key.lower(): value
            for key, value in response.headers.items()
            if key.lower() in {"content-length", "etag", "last-modified", "x-repo-commit"}
        }
        resolved_url = response.url
    os.replace(temporary, destination)
    return {
        "requested_url": url,
        "resolved_url": resolved_url,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "response_headers": headers,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "Reading the pinned Parquet files requires pyarrow; install the locked dependency set."
        ) from error
    return parquet.read_table(path).to_pylist()


def _clean_document(value: dict[str, Any]) -> dict[str, Any]:
    corpus_id = value.get("corpus_id")
    corpus_id = None if corpus_id is None or int(corpus_id) < 0 else int(corpus_id)
    doc_id = str(value.get("doc_id", ""))
    if not doc_id:
        raise ValueError("Every RELISH document must have doc_id")
    return {
        "doc_id": doc_id,
        "corpus_id": corpus_id,
        "title": str(value.get("title") or ""),
        "abstract": str(value.get("abstract") or ""),
    }


def prepare_unlabelled(
    parquet_path: Path,
    output_directory: Path,
    dataset_config: dict[str, Any],
) -> dict[str, Any]:
    raw_rows = _read_parquet_rows(parquet_path)
    queries: list[dict[str, Any]] = []
    documents: dict[str, dict[str, Any]] = {}
    candidate_counts: list[int] = []
    text_conflicts = 0
    for index, row in enumerate(raw_rows):
        if "query" not in row or "candidates" not in row:
            raise ValueError(f"RELISH row {index} lacks query/candidates")
        query = _clean_document(row["query"])
        candidates = [_clean_document(value) for value in row["candidates"]]
        if not candidates:
            raise ValueError(f"RELISH query {query['doc_id']} has no candidates")
        qid = query["doc_id"]
        if not qid:
            raise ValueError(f"RELISH row {index} has an empty query doc_id")
        candidate_counts.append(len(candidates))
        for document in [query, *candidates]:
            previous = documents.get(document["doc_id"])
            if previous is not None and previous != document:
                comparable_previous = {key: previous[key] for key in ("title", "abstract")}
                comparable_current = {key: document[key] for key in ("title", "abstract")}
                if comparable_previous != comparable_current:
                    text_conflicts += 1
                    candidates_for_canonical = [previous, document]
                    document = max(
                        candidates_for_canonical,
                        key=lambda item: (len(item["abstract"]), len(item["title"]), item["abstract"], item["title"]),
                    )
            documents[document["doc_id"]] = document
        split = assign_split(query, dataset_config)
        group = group_id(query)
        queries.append(
            {
                "qid": qid,
                "query_corpus_id": query["corpus_id"],
                "title": query["title"],
                "abstract": query["abstract"],
                "group_id": group,
                "bucket": split_bucket(group, str(dataset_config["split_salt"])),
                "split": split,
                "candidate_doc_ids": [value["doc_id"] for value in candidates],
                "candidate_corpus_ids": [value["corpus_id"] for value in candidates],
            }
        )
    if len({query["qid"] for query in queries}) != len(queries):
        raise ValueError("RELISH query doc_id values are not unique")
    audit = audit_splits([{"query": {"title": row["title"], "corpus_id": row["query_corpus_id"]}} for row in queries], dataset_config)
    output_directory.mkdir(parents=True, exist_ok=True)
    queries_path = output_directory / "queries.jsonl.gz"
    documents_path = output_directory / "documents.jsonl.gz"
    write_jsonl_gz(queries_path, sorted(queries, key=lambda row: row["qid"]))
    write_jsonl_gz(
        documents_path,
        (documents[key] for key in sorted(documents)),
    )
    report = {
        "source": {
            "path": parquet_path.resolve().as_posix(),
            "bytes": parquet_path.stat().st_size,
            "sha256": sha256_file(parquet_path),
        },
        "queries": len(queries),
        "documents": len(documents),
        "documents_missing_corpus_id": sum(
            1 for document in documents.values() if document["corpus_id"] is None
        ),
        "repeated_doc_id_text_conflicts": text_conflicts,
        "candidate_instances": int(sum(candidate_counts)),
        "candidate_count_min": min(candidate_counts),
        "candidate_count_max": max(candidate_counts),
        "candidate_count_mean": float(np.mean(candidate_counts)),
        "split_audit": audit,
        "outputs": {
            "queries": {"path": queries_path.name, "sha256": sha256_file(queries_path)},
            "documents": {"path": documents_path.name, "sha256": sha256_file(documents_path)},
        },
    }
    write_json(output_directory / "prepare_manifest.json", report)
    return report


def extract_labels(
    parquet_path: Path,
    prepared_directory: Path,
    output_path: Path,
    phases: set[str],
    frozen_decisions_manifest: Path | None = None,
) -> dict[str, Any]:
    allowed = {"train", "calibration", "locked_test"}
    if not phases or not phases.issubset(allowed):
        raise ValueError(f"phases must be a non-empty subset of {sorted(allowed)}")
    if "locked_test" in phases:
        if frozen_decisions_manifest is None or not frozen_decisions_manifest.exists():
            raise RuntimeError("Locked-test labels require a pre-existing frozen decision manifest")
        decision_manifest = read_json(frozen_decisions_manifest)
        decision_path = frozen_decisions_manifest.parent / decision_manifest["decisions_file"]
        if sha256_file(decision_path) != decision_manifest["decisions_sha256"]:
            raise RuntimeError("Frozen decisions do not match their committed manifest")
    queries = read_jsonl_gz(prepared_directory / "queries.jsonl.gz")
    query_by_id = {row["qid"]: row for row in queries}
    selected_ids = {qid for qid, row in query_by_id.items() if row["split"] in phases}
    labels_by_query: dict[str, dict[str, int]] = {qid: {} for qid in selected_ids}
    duplicate_pairs = 0
    for row in _read_parquet_rows(parquet_path):
        qid = str(row["query_id"])
        if qid not in selected_ids:
            continue
        candidate_id = str(row["cand_id"])
        if candidate_id in labels_by_query[qid]:
            duplicate_pairs += 1
            continue
        labels_by_query[qid][candidate_id] = int(row["score"])
    output_rows: list[dict[str, Any]] = []
    missing_pairs = 0
    extra_pairs = 0
    label_counts: Counter[int] = Counter()
    for qid in sorted(selected_ids):
        query = query_by_id[qid]
        candidate_ids = query["candidate_doc_ids"]
        mapping = labels_by_query[qid]
        expected = set(candidate_ids)
        missing = expected.difference(mapping)
        extra = set(mapping).difference(expected)
        missing_pairs += len(missing)
        extra_pairs += len(extra)
        if missing or extra:
            raise ValueError(
                f"Label alignment failed for {qid}: {len(missing)} missing, {len(extra)} extra"
            )
        labels = [mapping[candidate_id] for candidate_id in candidate_ids]
        label_counts.update(labels)
        output_rows.append({"qid": qid, "split": query["split"], "labels": labels})
    write_jsonl_gz(output_path, output_rows)
    report = {
        "phases": sorted(phases),
        "queries": len(output_rows),
        "pairs": int(sum(len(row["labels"]) for row in output_rows)),
        "label_counts": {str(key): value for key, value in sorted(label_counts.items())},
        "duplicate_pairs": duplicate_pairs,
        "missing_pairs": missing_pairs,
        "extra_pairs": extra_pairs,
        "source_sha256": sha256_file(parquet_path),
        "output_sha256": sha256_file(output_path),
    }
    write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), report)
    return report


def fetch_semantic_scholar_metadata(
    documents_path: Path,
    output_path: Path,
    source_config: dict[str, Any],
    batch_size: int = 500,
    pause_seconds: float = 1.05,
    maximum_attempts: int = 8,
) -> dict[str, Any]:
    documents = read_jsonl_gz(documents_path)
    corpus_ids = sorted(
        {int(row["corpus_id"]) for row in documents if row.get("corpus_id") is not None}
    )
    if output_path.exists():
        existing = read_jsonl_gz(output_path)
        metadata_by_id = {int(row["requested_corpus_id"]): row for row in existing}
    else:
        metadata_by_id = {}
    missing_ids = [value for value in corpus_ids if value not in metadata_by_id]
    endpoint = str(source_config["endpoint"])
    fields = list(source_config["fields"])
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    for start in range(0, len(missing_ids), batch_size):
        batch = missing_ids[start : start + batch_size]
        payload = {"ids": [f"CorpusId:{value}" for value in batch]}
        response = None
        for attempt in range(maximum_attempts):
            response = requests.post(
                endpoint,
                params={"fields": ",".join(fields)},
                headers=headers,
                json=payload,
                timeout=90,
            )
            if response.status_code == 200:
                break
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(60.0, 2.0**attempt)
            time.sleep(delay)
        if response is None or response.status_code != 200:
            raise RuntimeError(f"Semantic Scholar batch failed after {maximum_attempts} attempts")
        values = response.json()
        if len(values) != len(batch):
            raise ValueError("Semantic Scholar returned a misaligned batch")
        retrieved_at = datetime.now(timezone.utc).isoformat()
        for requested_id, value in zip(batch, values):
            metadata_by_id[requested_id] = {
                "requested_corpus_id": requested_id,
                "retrieved_at_utc": retrieved_at,
                "record": value,
            }
        write_jsonl_gz(
            output_path,
            (metadata_by_id[key] for key in sorted(metadata_by_id)),
        )
        if start + batch_size < len(missing_ids):
            time.sleep(pause_seconds)
    missing_records = sum(1 for value in metadata_by_id.values() if value["record"] is None)
    report = {
        "endpoint": endpoint,
        "fields": fields,
        "documents_sha256": sha256_file(documents_path),
        "records": len(metadata_by_id),
        "missing_records": missing_records,
        "snapshot_sha256": sha256_file(output_path),
        "uses_api_key": bool(api_key),
    }
    write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), report)
    return report

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .io import (
    iter_jsonl_gz,
    read_json,
    read_jsonl_gz,
    sha256_file,
    write_json,
)


_TOKEN = re.compile(r"[a-z0-9]+")


def document_text(document: dict[str, Any], separator: str = " ") -> str:
    title = str(document.get("title") or "").strip()
    abstract = str(document.get("abstract") or "").strip()
    return separator.join(value for value in (title, abstract) if value)


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return values / norms


def _write_embeddings(
    documents_path: Path,
    output_directory: Path,
    model_name: str,
    dimension: int,
    batch_size: int,
    encode: Callable[[list[dict[str, Any]]], np.ndarray],
    model_manifest: dict[str, Any],
) -> dict[str, Any]:
    prepare_manifest = read_json(documents_path.parent / "prepare_manifest.json")
    count = int(prepare_manifest["documents"])
    output_directory.mkdir(parents=True, exist_ok=True)
    embedding_path = output_directory / f"{model_name}.npy"
    temporary_path = output_directory / f".{model_name}.partial.npy"
    if temporary_path.exists():
        temporary_path.unlink()
    matrix = np.lib.format.open_memmap(
        temporary_path, mode="w+", dtype=np.float16, shape=(count, dimension)
    )
    doc_ids: list[str] = []
    batch: list[dict[str, Any]] = []
    offset = 0
    for document in iter_jsonl_gz(documents_path):
        batch.append(document)
        if len(batch) < batch_size:
            continue
        values = _normalise_rows(encode(batch))
        if values.shape != (len(batch), dimension):
            raise ValueError(f"Encoder returned {values.shape}, expected {(len(batch), dimension)}")
        matrix[offset : offset + len(batch)] = values.astype(np.float16)
        doc_ids.extend(str(item["doc_id"]) for item in batch)
        offset += len(batch)
        batch = []
    if batch:
        values = _normalise_rows(encode(batch))
        if values.shape != (len(batch), dimension):
            raise ValueError(f"Encoder returned {values.shape}, expected {(len(batch), dimension)}")
        matrix[offset : offset + len(batch)] = values.astype(np.float16)
        doc_ids.extend(str(item["doc_id"]) for item in batch)
        offset += len(batch)
    if offset != count or len(set(doc_ids)) != count:
        raise ValueError(f"Document count/identity mismatch: encoded={offset}, expected={count}")
    matrix.flush()
    del matrix
    os.replace(temporary_path, embedding_path)
    ids_path = output_directory / f"{model_name}_doc_ids.json"
    write_json(ids_path, doc_ids)
    report = {
        **model_manifest,
        "model_name": model_name,
        "documents": count,
        "dimension": dimension,
        "dtype": "float16",
        "normalised": True,
        "documents_sha256": sha256_file(documents_path),
        "embeddings_file": embedding_path.name,
        "embeddings_sha256": sha256_file(embedding_path),
        "doc_ids_file": ids_path.name,
        "doc_ids_sha256": sha256_file(ids_path),
    }
    write_json(output_directory / f"{model_name}.manifest.json", report)
    return report


def encode_sentence_transformer(
    documents_path: Path,
    output_directory: Path,
    model_name: str,
    config: dict[str, Any],
    device: str | None = None,
) -> dict[str, Any]:
    import sentence_transformers
    import torch
    from sentence_transformers import SentenceTransformer

    model_id = str(config["model_id"])
    revision = str(config["revision"])
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"Model revision must be an immutable 40-character commit: {revision}")
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_id, revision=revision, device=selected_device)
    model.max_seq_length = int(config["max_length"])
    dimension = int(model.get_sentence_embedding_dimension())
    separator_mode = str(config.get("separator", "space"))
    if separator_mode == "space":
        separator = " "
    elif separator_mode == "tokenizer_sep":
        separator = str(model.tokenizer.sep_token or "")
        if not separator:
            raise ValueError(f"{model_id} does not expose a tokenizer separator token")
    else:
        raise ValueError(f"Unknown document separator mode: {separator_mode}")

    def encode(batch: list[dict[str, Any]]) -> np.ndarray:
        texts = [document_text(document, separator=separator) for document in batch]
        return np.asarray(
            model.encode(
                texts,
                batch_size=len(texts),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ),
            dtype=np.float32,
        )

    return _write_embeddings(
        documents_path,
        output_directory,
        model_name,
        dimension,
        int(config["batch_size"]),
        encode,
        {
            "kind": "sentence_transformer",
            "model_id": model_id,
            "resolved_revision": revision,
            "document_separator_mode": separator_mode,
            "document_separator": separator,
            "max_length": int(config["max_length"]),
            "batch_size": int(config["batch_size"]),
            "device": selected_device,
            "torch_version": torch.__version__,
            "sentence_transformers_version": sentence_transformers.__version__,
        },
    )


def encode_specter2(
    documents_path: Path,
    output_directory: Path,
    model_name: str,
    config: dict[str, Any],
    device: str | None = None,
) -> dict[str, Any]:
    import adapters
    import torch
    import transformers
    from adapters import AutoAdapterModel
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    base_id = str(config["base_model_id"])
    adapter_id = str(config["adapter_model_id"])
    base_revision = str(config["base_revision"])
    adapter_revision = str(config["adapter_revision"])
    for revision in (base_revision, adapter_revision):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"Model revision must be an immutable 40-character commit: {revision}")
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_id, revision=base_revision)
    model = AutoAdapterModel.from_pretrained(base_id, revision=base_revision)
    adapter_path = snapshot_download(repo_id=adapter_id, revision=adapter_revision)
    model.load_adapter(adapter_path, load_as="specter2", set_active=True)
    model.to(selected_device)
    model.eval()
    dimension = int(model.config.hidden_size)

    def encode(batch: list[dict[str, Any]]) -> np.ndarray:
        texts = [document_text(document, separator=tokenizer.sep_token) for document in batch]
        tokens = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(config["max_length"]),
            return_tensors="pt",
        )
        tokens = {key: value.to(selected_device) for key, value in tokens.items()}
        with torch.inference_mode():
            values = model(**tokens).last_hidden_state[:, 0, :]
        return values.detach().cpu().float().numpy()

    return _write_embeddings(
        documents_path,
        output_directory,
        model_name,
        dimension,
        int(config["batch_size"]),
        encode,
        {
            "kind": "adapter",
            "base_model_id": base_id,
            "base_resolved_revision": base_revision,
            "adapter_model_id": adapter_id,
            "adapter_resolved_revision": adapter_revision,
            "document_separator_mode": "tokenizer_sep",
            "document_separator": tokenizer.sep_token,
            "max_length": int(config["max_length"]),
            "batch_size": int(config["batch_size"]),
            "device": selected_device,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "adapters_version": adapters.__version__,
        },
    )


def build_candidate_layout(prepared_directory: Path, scores_directory: Path) -> dict[str, Any]:
    queries_path = prepared_directory / "queries.jsonl.gz"
    queries = read_jsonl_gz(queries_path)
    scores_directory.mkdir(parents=True, exist_ok=True)
    qids = [str(query["qid"]) for query in queries]
    offsets = [0]
    candidate_doc_ids: list[str] = []
    for query in queries:
        candidate_doc_ids.extend(str(value) for value in query["candidate_doc_ids"])
        offsets.append(len(candidate_doc_ids))
    qids_path = scores_directory / "qids.json"
    candidate_ids_path = scores_directory / "candidate_doc_ids.json"
    offsets_path = scores_directory / "offsets.npy"
    write_json(qids_path, qids)
    write_json(candidate_ids_path, candidate_doc_ids)
    np.save(offsets_path, np.asarray(offsets, dtype=np.int64), allow_pickle=False)
    report = {
        "queries_sha256": sha256_file(queries_path),
        "queries": len(qids),
        "candidate_instances": len(candidate_doc_ids),
        "qids_sha256": sha256_file(qids_path),
        "candidate_doc_ids_sha256": sha256_file(candidate_ids_path),
        "offsets_sha256": sha256_file(offsets_path),
    }
    write_json(scores_directory / "layout.manifest.json", report)
    return report


def _load_layout(scores_directory: Path) -> tuple[list[str], list[str], np.ndarray]:
    qids = [str(value) for value in read_json(scores_directory / "qids.json")]
    candidate_ids = [str(value) for value in read_json(scores_directory / "candidate_doc_ids.json")]
    offsets = np.load(scores_directory / "offsets.npy", allow_pickle=False)
    if len(offsets) != len(qids) + 1 or int(offsets[-1]) != len(candidate_ids):
        raise ValueError("Candidate layout is inconsistent")
    return qids, candidate_ids, offsets


def score_embedding_model(
    prepared_directory: Path,
    embeddings_directory: Path,
    scores_directory: Path,
    model_name: str,
) -> dict[str, Any]:
    if not (scores_directory / "layout.manifest.json").exists():
        build_candidate_layout(prepared_directory, scores_directory)
    qids, candidate_ids, offsets = _load_layout(scores_directory)
    model_manifest = read_json(embeddings_directory / f"{model_name}.manifest.json")
    doc_ids = [str(value) for value in read_json(embeddings_directory / f"{model_name}_doc_ids.json")]
    embeddings = np.load(embeddings_directory / f"{model_name}.npy", mmap_mode="r")
    index = {doc_id: position for position, doc_id in enumerate(doc_ids)}
    queries = read_jsonl_gz(prepared_directory / "queries.jsonl.gz")
    if [str(query["qid"]) for query in queries] != qids:
        raise ValueError("Query order changed after layout creation")
    output = np.empty(len(candidate_ids), dtype=np.float32)
    for query_number, query in enumerate(queries):
        start, stop = int(offsets[query_number]), int(offsets[query_number + 1])
        query_index = index[str(query["qid"])]
        candidate_indices = [index[value] for value in candidate_ids[start:stop]]
        query_vector = np.asarray(embeddings[query_index], dtype=np.float32)
        candidate_matrix = np.asarray(embeddings[candidate_indices], dtype=np.float32)
        output[start:stop] = candidate_matrix @ query_vector
    output_path = scores_directory / f"{model_name}.npy"
    np.save(output_path, output, allow_pickle=False)
    report = {
        "model_name": model_name,
        "model_manifest_sha256": sha256_file(embeddings_directory / f"{model_name}.manifest.json"),
        "layout_manifest_sha256": sha256_file(scores_directory / "layout.manifest.json"),
        "scores": len(output),
        "scores_sha256": sha256_file(output_path),
        "minimum": float(np.min(output)),
        "maximum": float(np.max(output)),
    }
    write_json(scores_directory / f"{model_name}.manifest.json", report)
    return report


def score_bm25(prepared_directory: Path, scores_directory: Path) -> dict[str, Any]:
    from rank_bm25 import BM25Okapi

    if not (scores_directory / "layout.manifest.json").exists():
        build_candidate_layout(prepared_directory, scores_directory)
    qids, candidate_ids, offsets = _load_layout(scores_directory)
    documents = {
        str(document["doc_id"]): document_text(document)
        for document in iter_jsonl_gz(prepared_directory / "documents.jsonl.gz")
    }
    queries = read_jsonl_gz(prepared_directory / "queries.jsonl.gz")
    output = np.empty(len(candidate_ids), dtype=np.float32)
    for query_number, query in enumerate(queries):
        start, stop = int(offsets[query_number]), int(offsets[query_number + 1])
        ids = candidate_ids[start:stop]
        candidate_tokens = [_TOKEN.findall(documents[value].lower()) for value in ids]
        query_tokens = _TOKEN.findall(document_text(query).lower())
        if not query_tokens or not any(candidate_tokens):
            output[start:stop] = 0.0
        else:
            output[start:stop] = BM25Okapi(candidate_tokens).get_scores(query_tokens)
    output_path = scores_directory / "bm25.npy"
    np.save(output_path, output, allow_pickle=False)
    report = {
        "model_name": "bm25",
        "implementation": "rank_bm25.BM25Okapi",
        "layout_manifest_sha256": sha256_file(scores_directory / "layout.manifest.json"),
        "scores": len(output),
        "scores_sha256": sha256_file(output_path),
        "minimum": float(np.min(output)),
        "maximum": float(np.max(output)),
    }
    write_json(scores_directory / "bm25.manifest.json", report)
    return report

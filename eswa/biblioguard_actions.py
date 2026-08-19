#!/usr/bin/env python
"""Generate confound-free BiblioGuard action outcomes.

Each action preserves the content-only fallback's score, fusion weights, and
top-100 candidate set.  It changes only one bibliographic term: citation or
recency.  This removes the configuration-switch confound in the first
BiblioGuard revision.

Prerequisites are produced by ``reproduce.py download metadata encode
retrieval bge``.  Outputs are small per-query archives; large embeddings and
score matrices remain ignored intermediates.
"""
from __future__ import annotations

import json
import math
import argparse
from pathlib import Path

import numpy as np

import _layout as L
from reproduce import (
    LAMBDA_RECENCY,
    REF_YEAR,
    TOPK,
    build_eval_arrays,
    load_jsonl,
    load_qrels,
    minmax,
    per_query_metrics,
    tokenize,
)


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
DATASETS = ("scidocs", "scifact", "nfcorpus", "trec-covid")
CONTENT_BASES = ("SBERT-Dense", "BGE-Dense", "Neural-Hybrid", "BGE-Hybrid")
ACTIONS = tuple(
    [("citation", weight) for weight in (0.05, 0.10, 0.15, 0.20, 0.30)]
    + [("recency", weight) for weight in (0.05, 0.10, 0.15, 0.20)]
)


def action_label(signal: str, weight: float) -> str:
    return f"{signal}:{weight:.2f}"


def _minilm_archive(dataset: str) -> Path:
    if dataset in ("scidocs", "scifact"):
        return Path(L.v2_perquery(dataset))
    return RESULTS / f"{dataset}_perquery.npz"


def _load_bge(dataset: str, doc_ids: list[str], query_ids: list[str]):
    emb_dir = Path(L.emb_dir(dataset, bge=True))
    bge_ids = json.loads((emb_dir / "ids.json").read_text(encoding="utf-8"))
    starts = sorted(
        int(path.stem.split("_")[1]) for path in emb_dir.glob("chunk_*.npy")
    )
    embeddings = np.vstack(
        [np.load(emb_dir / f"chunk_{start}.npy") for start in starts]
    )
    bge_index = {doc_id: index for index, doc_id in enumerate(bge_ids)}
    embeddings = embeddings[
        np.asarray([bge_index[doc_id] for doc_id in doc_ids])
    ]
    all_query_embeddings = np.load(L.art_path(dataset, f"{dataset}_bge_qemb.npy"))
    all_query_ids = json.loads(
        Path(L.art_path(dataset, f"{dataset}_bge_qids.json")).read_text(
            encoding="utf-8"
        )
    )
    query_index = {
        query_id: index for index, query_id in enumerate(all_query_ids)
    }
    queries = np.stack(
        [all_query_embeddings[query_index[query_id]] for query_id in query_ids]
    )
    return embeddings, queries


def _metadata_vectors(dataset: str, doc_ids: list[str]):
    prep = Path(L.prep_dir(dataset))
    if (prep / "C.npy").exists() and (prep / "R.npy").exists():
        saved_ids = json.loads((prep / "doc_ids.json").read_text(encoding="utf-8"))
        if saved_ids != doc_ids:
            raise RuntimeError(f"{dataset}: metadata tensor/doc-id order mismatch")
        return np.load(prep / "C.npy"), np.load(prep / "R.npy")

    metadata = json.loads(Path(L.meta_file(dataset)).read_text(encoding="utf-8"))
    citations, years = [], []
    for doc_id in doc_ids:
        row = metadata.get(doc_id)
        citations.append(0 if row is None else (row.get("citations") or 0))
        years.append(None if row is None else row.get("year"))
    observed_years = [year for year in years if year]
    if not observed_years:
        raise RuntimeError(f"{dataset}: no publication years are available")
    median_year = int(np.median(observed_years))
    years = np.asarray(
        [median_year if year is None else year for year in years], dtype=float
    )
    citation = np.log1p(np.asarray(citations, dtype=float))
    citation /= citation.max() + 1e-9
    recency = np.exp(-LAMBDA_RECENCY * (REF_YEAR - years))
    return citation, recency


def _load_content_results(dataset: str, query_ids: list[str]) -> dict[str, np.ndarray]:
    minilm = np.load(_minilm_archive(dataset), allow_pickle=True)
    bge = np.load(RESULTS / f"{dataset}_bge_perquery.npz", allow_pickle=True)
    hybrid = np.load(
        RESULTS / f"{dataset}_bge_hybrid_perquery.npz", allow_pickle=True
    )
    expected = np.asarray(query_ids, dtype=str)
    for name, archive in (("MiniLM", minilm), ("BGE", bge), ("BGE hybrid", hybrid)):
        if not np.array_equal(archive["qids"].astype(str), expected):
            raise RuntimeError(f"{dataset}: {name} qid order mismatch")
    return {
        "SBERT-Dense": np.asarray(minilm["SBERT-Dense||N@10"], dtype=float),
        "BGE-Dense": np.asarray(bge["BGE-Dense||N@10"], dtype=float),
        "Neural-Hybrid": np.asarray(minilm["Neural-Hybrid||N@10"], dtype=float),
        "BGE-Hybrid": np.asarray(hybrid["BGE-Hybrid||N@10"], dtype=float),
    }


def generate(dataset: str, split: str = "test") -> Path:
    raw = Path(L.raw_ds(dataset))
    documents = load_jsonl(str(raw / "corpus.jsonl"))
    doc_ids = [str(document["_id"]) for document in documents]
    doc_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    qrels_path = raw / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        raise FileNotFoundError(f"{dataset}: missing official {split} qrels")
    qrels = load_qrels(str(qrels_path))
    query_ids = sorted(qrels)
    relevance, gains = build_eval_arrays(
        qrels, query_ids, doc_index, len(doc_ids)
    )
    citation, recency = _metadata_vectors(dataset, doc_ids)
    signal_vector = {"citation": citation, "recency": recency}

    scoremat_path = Path(L.scoremats(dataset))
    if split == "test" and scoremat_path.exists():
        scoremats = np.load(scoremat_path)
        bm25_scores = scoremats["S_bm"]
        sbert_scores = scoremats["S_sb"]
    elif dataset == "trec-covid":
        # The strong BGE-Dense fallback dominates the released hybrid on this
        # 171k-document corpus. Avoid materialising a very large BM25 index
        # solely for an action base that is not selected.
        bm25_scores = None
        sbert_scores = None
    else:
        from rank_bm25 import BM25Okapi

        tokenized_documents = [
            tokenize(
                (document.get("title") or "")
                + " "
                + (document.get("text") or "")
            )
            for document in documents
        ]
        bm25_model = BM25Okapi(
            tokenized_documents, k1=1.5, b=0.75, epsilon=0.25
        )
        query_text = {
            str(row["_id"]): row.get("text") or ""
            for row in load_jsonl(str(raw / "queries.jsonl"))
        }
        bm25_scores = np.stack(
            [
                bm25_model.get_scores(tokenize(query_text[query_id]))
                for query_id in query_ids
            ]
        )
        sbert_scores = None
    if bm25_scores is not None and bm25_scores.shape[0] != len(query_ids):
        raise RuntimeError(f"{dataset}: score-matrix/query mismatch")
    bge_emb_dir = Path(L.emb_dir(dataset, bge=True))
    has_bge = (bge_emb_dir / "ids.json").exists()
    bge_embeddings = bge_queries = None
    if has_bge:
        bge_embeddings, bge_queries = _load_bge(dataset, doc_ids, query_ids)
    released_baselines = (
        _load_content_results(dataset, query_ids) if split == "test" else {}
    )

    available_bases = ["BGE-Dense"] if has_bge else []
    if has_bge and bm25_scores is not None:
        available_bases.append("BGE-Hybrid")
    if sbert_scores is not None:
        available_bases.extend(["SBERT-Dense", "Neural-Hybrid"])
    available_bases = [name for name in CONTENT_BASES if name in available_bases]
    if not available_bases:
        raise RuntimeError(f"{dataset}: no document-level content scores available")
    baseline_orders = {name: [] for name in available_bases}
    action_orders = {
        (name, signal, weight): []
        for name in available_bases
        for signal, weight in ACTIONS
    }
    for query_index in range(len(query_ids)):
        bm25 = (
            minmax(bm25_scores[query_index].astype(np.float64))
            if bm25_scores is not None
            else None
        )
        content_scores = {}
        if has_bge:
            bge = minmax(
                (bge_embeddings @ bge_queries[query_index]).astype(np.float64)
            )
            content_scores["BGE-Dense"] = bge
            if bm25 is not None:
                content_scores["BGE-Hybrid"] = 0.5 * bm25 + 0.5 * bge
        if sbert_scores is not None:
            sbert = minmax(sbert_scores[query_index].astype(np.float64))
            content_scores["SBERT-Dense"] = sbert
            content_scores["Neural-Hybrid"] = 0.5 * bm25 + 0.5 * sbert
        for base_name, content_score in content_scores.items():
            candidate = np.argpartition(-content_score, TOPK)[:TOPK]
            baseline_orders[base_name].append(
                candidate[np.argsort(-content_score[candidate])]
            )
            for signal, weight in ACTIONS:
                scores = (
                    content_score[candidate]
                    + weight * signal_vector[signal][candidate]
                )
                action_orders[(base_name, signal, weight)].append(
                    candidate[np.argsort(-scores)]
                )
        if (query_index + 1) % 100 == 0:
            print(dataset, query_index + 1, "/", len(query_ids), flush=True)

    payload: dict[str, np.ndarray] = {"qids": np.asarray(query_ids)}
    for base_name, values in released_baselines.items():
        payload[f"base::{base_name}"] = values
    for base_name, orders in baseline_orders.items():
        recomputed = np.asarray(
            per_query_metrics(orders, relevance, gains)["N@10"], dtype=float
        )
        if base_name in released_baselines and not np.allclose(
            recomputed, released_baselines[base_name], atol=1e-12, rtol=1e-10
        ):
            maximum = float(np.max(np.abs(recomputed - released_baselines[base_name])))
            mismatch_count = int(
                np.sum(
                    ~np.isclose(
                        recomputed,
                        released_baselines[base_name],
                        atol=1e-12,
                        rtol=1e-10,
                    )
                )
            )
            mean_difference = float(
                abs(recomputed.mean() - released_baselines[base_name].mean())
            )
            if mean_difference > 1e-3 or mismatch_count > math.ceil(0.01 * len(query_ids)):
                raise RuntimeError(
                    f"{dataset}/{base_name}: reconstructed fallback differs "
                    f"materially (recomputed={recomputed.mean():.12f}, "
                    f"released={released_baselines[base_name].mean():.12f}, "
                    f"mismatches={mismatch_count}, "
                    f"max |difference|={maximum:.3g})"
                )
            print(
                f"warning: {dataset}/{base_name} has {mismatch_count} "
                f"tie-sensitive per-query differences; using the internally "
                f"consistent reconstruction (mean shift {mean_difference:.3g})",
                flush=True,
            )
        payload[f"base::{base_name}"] = recomputed
    for (base_name, signal, weight), orders in action_orders.items():
        payload[f"action::{base_name}::{action_label(signal, weight)}"] = np.asarray(
            per_query_metrics(orders, relevance, gains)["N@10"], dtype=float
        )
    split_suffix = "" if split == "test" else f"_{split}"
    output = RESULTS / f"{dataset}{split_suffix}_biblioguard_actions.npz"
    np.savez_compressed(output, **payload)
    print("saved", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", choices=DATASETS)
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    args = parser.parse_args()
    targets = args.datasets or list(DATASETS)
    for dataset in targets:
        generate(dataset, split=args.split)


if __name__ == "__main__":
    main()

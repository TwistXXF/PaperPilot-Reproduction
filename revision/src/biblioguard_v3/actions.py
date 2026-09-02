from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .metrics import stable_order


def minmax(values: Iterable[float | int | None]) -> np.ndarray:
    raw = list(values)
    present = np.asarray([float(value) for value in raw if value is not None], dtype=float)
    if len(present) == 0:
        return np.zeros(len(raw), dtype=float)
    if not np.isfinite(present).all():
        raise ValueError("Metadata contains a non-finite value")
    median = float(np.median(present))
    array = np.asarray([median if value is None else float(value) for value in raw], dtype=float)
    low = float(np.min(array))
    high = float(np.max(array))
    if high <= low:
        return np.zeros(len(array), dtype=float)
    return (array - low) / (high - low)


def reciprocal_rank_scores(
    semantic: np.ndarray, metadata: np.ndarray, corpus_ids: list[str | int], k: int
) -> np.ndarray:
    semantic_order = stable_order(semantic, corpus_ids)
    semantic_rank = np.empty(len(semantic_order), dtype=int)
    semantic_rank[semantic_order] = np.arange(1, len(semantic_order) + 1)
    metadata = np.asarray(metadata, dtype=float)
    if not np.isfinite(metadata).all():
        raise ValueError("Metadata rank input contains a non-finite value")
    metadata_order = np.argsort(-metadata, kind="stable")
    sorted_values = metadata[metadata_order]
    metadata_rank = np.empty(len(metadata_order), dtype=float)
    start = 0
    while start < len(metadata_order):
        stop = start + 1
        while stop < len(metadata_order) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        midrank = 0.5 * ((start + 1) + stop)
        metadata_rank[metadata_order[start:stop]] = midrank
        start = stop
    return 1.0 / (k + semantic_rank) + 1.0 / (k + metadata_rank)


def action_scores(
    semantic_scores: Iterable[float],
    citation_counts: Iterable[int | None],
    years: Iterable[int | None],
    corpus_ids: Iterable[str | int],
    actions: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    semantic = minmax(list(semantic_scores))
    citation_values: list[float | None] = []
    for value in citation_counts:
        if value is None:
            citation_values.append(None)
        elif value < 0:
            raise ValueError("citationCount must be non-negative")
        else:
            citation_values.append(float(np.log1p(value)))
    citation = minmax(citation_values)
    recency = minmax(list(years))
    balanced = 0.5 * (citation + recency)
    metadata_by_name = {"citation": citation, "recency": recency, "balanced": balanced}
    ids = list(corpus_ids)
    if not (len(semantic) == len(citation) == len(recency) == len(ids)):
        raise ValueError("Candidate arrays are not aligned")
    output: dict[str, np.ndarray] = {}
    for action in actions:
        metadata = metadata_by_name[action["metadata"]]
        if action["kind"] == "linear":
            weight = float(action["weight"])
            output[action["name"]] = (1.0 - weight) * semantic + weight * metadata
        elif action["kind"] == "rrf":
            output[action["name"]] = reciprocal_rank_scores(
                semantic, metadata, ids, int(action["rrf_k"])
            )
        else:
            raise ValueError(f"Unknown action kind: {action['kind']}")
    return output

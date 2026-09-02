from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def stable_order(scores: Iterable[float], corpus_ids: Iterable[str | int]) -> np.ndarray:
    score_array = np.asarray(list(scores), dtype=float)
    id_array = np.asarray([str(value) for value in corpus_ids], dtype=object)
    if score_array.ndim != 1 or len(score_array) != len(id_array):
        raise ValueError("scores and corpus_ids must be aligned one-dimensional arrays")
    if not np.isfinite(score_array).all():
        raise ValueError("scores contain a non-finite value")
    return np.lexsort((id_array, -score_array))


def dcg(relevance: np.ndarray, k: int) -> float:
    values = np.asarray(relevance, dtype=float)[:k]
    if len(values) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(values) + 2, dtype=float))
    # SciRepEval delegates NDCG to trec_eval/pytrec_eval, whose graded gain is
    # the relevance level itself (RELISH: 0, 1, 2), not 2**rel - 1.
    return float(np.sum(values / discounts))


def ndcg_at_k(relevance: Iterable[float], order: np.ndarray, k: int) -> float:
    relevance_array = np.asarray(list(relevance), dtype=float)
    actual = dcg(relevance_array[order], k)
    ideal = dcg(np.sort(relevance_array)[::-1], k)
    return actual / ideal if ideal > 0.0 else 0.0


def precision_at_k(relevance: Iterable[float], order: np.ndarray, k: int) -> float:
    relevance_array = np.asarray(list(relevance), dtype=float)
    selected = relevance_array[order[:k]] > 0
    return float(np.sum(selected) / k) if k > 0 else 0.0


def recall_at_k(relevance: Iterable[float], order: np.ndarray, k: int) -> float:
    relevance_array = np.asarray(list(relevance), dtype=float)
    positives = int(np.sum(relevance_array > 0))
    if positives == 0:
        return 0.0
    return float(np.sum(relevance_array[order[:k]] > 0) / positives)


def average_precision_at_k(relevance: Iterable[float], order: np.ndarray, k: int) -> float:
    relevance_array = np.asarray(list(relevance), dtype=float)
    binary = relevance_array[order[:k]] > 0
    positives = int(np.sum(relevance_array > 0))
    if positives == 0:
        return 0.0
    hits = 0
    total = 0.0
    for rank, is_relevant in enumerate(binary, start=1):
        if is_relevant:
            hits += 1
            total += hits / rank
    # trec_eval map_cut.k keeps the total number of relevant documents in the
    # denominator; it does not renormalise to one when more than k are relevant.
    return float(total / positives)


def evaluate_ranking(
    relevance: Iterable[float], scores: Iterable[float], corpus_ids: Iterable[str | int]
) -> dict[str, float]:
    relevance_array = np.asarray(list(relevance), dtype=float)
    order = stable_order(scores, corpus_ids)
    return {
        "ndcg_at_10": ndcg_at_k(relevance_array, order, 10),
        "ndcg_at_20": ndcg_at_k(relevance_array, order, 20),
        "ndcg_full": ndcg_at_k(relevance_array, order, len(relevance_array)),
        "map_cut_10": average_precision_at_k(relevance_array, order, 10),
        "recall_at_50": recall_at_k(relevance_array, order, 50),
        "precision_at_10": precision_at_k(relevance_array, order, 10),
    }


def validate_metric_range(metrics: dict[str, float]) -> None:
    for name, value in metrics.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Metric {name} is outside [0, 1]: {value}")

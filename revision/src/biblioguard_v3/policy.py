from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.stats import beta
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class PolicyPrediction:
    action_index: np.ndarray
    estimated_gain: np.ndarray
    standard_error: np.ndarray
    gate_score: np.ndarray


class LocalPolicy:
    def __init__(self, lcb_z: float = 1.645) -> None:
        self.lcb_z = float(lcb_z)
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
        self.matrix: sparse.csr_matrix | None = None
        self.effects: np.ndarray | None = None
        self.k: int | None = None

    def fit(self, titles: Iterable[str], action_effects: np.ndarray) -> "LocalPolicy":
        titles = list(titles)
        effects = np.asarray(action_effects, dtype=float)
        if effects.ndim != 2 or effects.shape[0] != len(titles):
            raise ValueError("action_effects must be n_query by n_action")
        if len(titles) < 2 or not np.isfinite(effects).all():
            raise ValueError("Policy training data are invalid")
        word_matrix = self.word.fit_transform(titles)
        char_matrix = self.char.fit_transform(titles)
        matrix = sparse.hstack([word_matrix, char_matrix], format="csr")
        row_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
        row_norms[row_norms == 0.0] = 1.0
        self.matrix = sparse.diags(1.0 / row_norms) @ matrix
        self.effects = effects
        self.k = max(1, int(math.floor(math.sqrt(len(titles)))))
        return self

    def _transform(self, titles: Iterable[str]) -> sparse.csr_matrix:
        word_matrix = self.word.transform(list(titles))
        char_matrix = self.char.transform(list(titles))
        matrix = sparse.hstack([word_matrix, char_matrix], format="csr")
        row_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
        row_norms[row_norms == 0.0] = 1.0
        return sparse.diags(1.0 / row_norms) @ matrix

    def predict(self, titles: Iterable[str]) -> PolicyPrediction:
        if self.matrix is None or self.effects is None or self.k is None:
            raise RuntimeError("Policy is not fitted")
        query_matrix = self._transform(list(titles))
        similarities = (query_matrix @ self.matrix.T).toarray()
        action_indices: list[int] = []
        estimated_gains: list[float] = []
        standard_errors: list[float] = []
        gate_scores: list[float] = []
        for row in similarities:
            nearest = np.argsort(-row, kind="stable")[: self.k]
            weights = np.maximum(row[nearest], 0.0) + 1e-8
            weights = weights / np.sum(weights)
            local = self.effects[nearest]
            means = weights @ local
            sum_square_weights = float(np.sum(weights**2))
            denominator = max(1e-12, 1.0 - sum_square_weights)
            variances = np.sum(weights[:, None] * (local - means) ** 2, axis=0) / denominator
            standard_error = np.sqrt(np.maximum(variances, 0.0) * sum_square_weights)
            action_index = int(np.argmax(means))
            mean = float(means[action_index])
            se = float(standard_error[action_index])
            action_indices.append(action_index)
            estimated_gains.append(mean)
            standard_errors.append(se)
            gate_scores.append(mean - self.lcb_z * se)
        return PolicyPrediction(
            action_index=np.asarray(action_indices, dtype=int),
            estimated_gain=np.asarray(estimated_gains, dtype=float),
            standard_error=np.asarray(standard_errors, dtype=float),
            gate_score=np.asarray(gate_scores, dtype=float),
        )


def clopper_pearson_upper(harms: int, active: int, confidence: float = 0.95) -> float:
    if active <= 0 or harms < 0 or harms > active:
        raise ValueError("Invalid binomial counts")
    if harms == active:
        return 1.0
    return float(beta.ppf(confidence, harms + 1, active - harms))


def calibrate_threshold(
    gate_scores: np.ndarray,
    realised_effects: np.ndarray,
    harm_upper_bound: float,
    confidence: float,
    minimum_active: int,
) -> dict[str, float | int | bool]:
    scores = np.asarray(gate_scores, dtype=float)
    effects = np.asarray(realised_effects, dtype=float)
    if scores.shape != effects.shape or scores.ndim != 1:
        raise ValueError("Calibration arrays must be aligned vectors")
    candidates = np.unique(scores)
    eligible: list[dict[str, float | int | bool]] = []
    for threshold in candidates:
        active_mask = scores >= threshold
        active = int(np.sum(active_mask))
        if active < minimum_active:
            continue
        harms = int(np.sum(effects[active_mask] < 0.0))
        upper = clopper_pearson_upper(harms, active, confidence)
        if upper <= harm_upper_bound:
            eligible.append(
                {
                    "threshold": float(threshold),
                    "active": active,
                    "coverage": active / len(scores),
                    "harms": harms,
                    "harm_upper": upper,
                    "mean_gain": float(np.mean(effects[active_mask])),
                    "eligible": True,
                }
            )
    if not eligible:
        return {
            "threshold": float("inf"),
            "active": 0,
            "coverage": 0.0,
            "harms": 0,
            "harm_upper": 0.0,
            "mean_gain": 0.0,
            "eligible": False,
        }
    eligible.sort(key=lambda row: (-float(row["coverage"]), -float(row["mean_gain"]), -float(row["threshold"])))
    return eligible[0]


from __future__ import annotations

from typing import Iterable

import numpy as np


def deterministic_top_k(scores: Iterable[float], coverage: float) -> np.ndarray:
    values = np.asarray(list(scores), dtype=float)
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage must be in [0, 1]")
    count = int(round(coverage * len(values)))
    order = np.argsort(-values, kind="stable")
    selected = np.zeros(len(values), dtype=bool)
    selected[order[:count]] = True
    return selected


def policy_outcomes(effects: Iterable[float], active: Iterable[bool]) -> dict[str, float | int]:
    delta = np.asarray(list(effects), dtype=float)
    mask = np.asarray(list(active), dtype=bool)
    if delta.shape != mask.shape:
        raise ValueError("effects and active must be aligned")
    selected = delta[mask]
    overall = np.where(mask, delta, 0.0)
    if len(selected) == 0:
        return {
            "queries": len(delta),
            "active": 0,
            "coverage": 0.0,
            "overall_mean_gain": 0.0,
            "conditional_mean_gain": 0.0,
            "conditional_harm_probability": 0.0,
            "severe_harm_probability": 0.0,
            "mean_negative_shortfall": 0.0,
        }
    return {
        "queries": len(delta),
        "active": len(selected),
        "coverage": len(selected) / len(delta),
        "overall_mean_gain": float(np.mean(overall)),
        "conditional_mean_gain": float(np.mean(selected)),
        "conditional_harm_probability": float(np.mean(selected < 0.0)),
        "severe_harm_probability": float(np.mean(selected <= -0.05)),
        "mean_negative_shortfall": float(np.mean(np.maximum(0.0, -selected))),
    }


def risk_coverage_curve(
    effects: Iterable[float], confidence_scores: Iterable[float]
) -> dict[str, list[float] | float]:
    delta = np.asarray(list(effects), dtype=float)
    scores = np.asarray(list(confidence_scores), dtype=float)
    if delta.shape != scores.shape or delta.ndim != 1:
        raise ValueError("effects and confidence_scores must be aligned vectors")
    order = np.argsort(-scores, kind="stable")
    ordered_loss = np.maximum(0.0, -delta[order])
    risk = np.cumsum(ordered_loss) / np.arange(1, len(delta) + 1)
    coverage = np.arange(1, len(delta) + 1) / len(delta)
    coverage_with_zero = np.concatenate([[0.0], coverage])
    risk_with_zero = np.concatenate([[0.0], risk])
    aurc = float(np.trapezoid(risk_with_zero, coverage_with_zero))
    return {
        "coverage": coverage.tolist(),
        "risk": risk.tolist(),
        "aurc": aurc,
    }


def paired_bootstrap_ci(
    effects: Iterable[float], replicates: int, seed: int, confidence: float = 0.95
) -> tuple[float, float]:
    values = np.asarray(list(effects), dtype=float)
    if len(values) == 0:
        raise ValueError("effects must not be empty")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 1000):
        size = min(1000, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[start : start + size] = np.mean(values[indices], axis=1)
    alpha = 1.0 - confidence
    return tuple(float(value) for value in np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0]))


def paired_randomisation_pvalue(effects: Iterable[float], replicates: int, seed: int) -> float:
    values = np.asarray(list(effects), dtype=float)
    observed = abs(float(np.mean(values)))
    rng = np.random.default_rng(seed)
    exceedances = 0
    generated = 0
    for start in range(0, replicates, 1000):
        size = min(1000, replicates - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, len(values)))
        randomised = np.abs(np.mean(signs * values, axis=1))
        exceedances += int(np.sum(randomised >= observed))
        generated += size
    return (exceedances + 1.0) / (generated + 1.0)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * float(value)))
        adjusted[name] = running
    return adjusted


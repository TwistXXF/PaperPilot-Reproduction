#!/usr/bin/env python
"""Official train-to-test evaluation for BiblioGuard.

The content fallback, query representation, neighbours, and action decisions
are determined from the official training split. Test qrels are used only
after all decisions have been frozen to compute NDCG@10.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

import _layout as L
from biblioguard import (
    ALPHA_FAMILY,
    BOOTSTRAP_REPS,
    EPS,
    _bootstrap_ci,
    _empirical_bernstein_score,
    _load_queries,
    _outcome_counts,
    _text_features,
    _weighted_effect,
)


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
DATASETS = ("scifact", "nfcorpus")


def _load_archive(dataset: str, split: str) -> dict:
    suffix = "" if split == "test" else f"_{split}"
    path = RESULTS / f"{dataset}{suffix}_biblioguard_actions.npz"
    archive = np.load(path, allow_pickle=True)
    qids = archive["qids"].astype(str).tolist()
    bases = {
        key.split("::", 1)[1]: np.asarray(archive[key], dtype=float)
        for key in archive.files
        if key.startswith("base::")
    }
    actions_by_base = {}
    for key in archive.files:
        if not key.startswith("action::"):
            continue
        _, base, action = key.split("::", 2)
        actions_by_base.setdefault(base, {})[action] = np.asarray(
            archive[key], dtype=float
        )
    actions = sorted(next(iter(actions_by_base.values())))
    return {
        "qids": qids,
        "texts": _load_queries(dataset, qids),
        "bases": bases,
        "action_bases": sorted(actions_by_base),
        "actions": actions,
        "outcomes": {
            base: np.column_stack([rows[action] for action in actions])
            for base, rows in actions_by_base.items()
        },
    }


def _activate(
    score: np.ndarray,
    outcomes: np.ndarray,
    query_index: int,
    base: str,
    actions: list[str],
) -> tuple[float, str]:
    action_index = int(np.argmax(score))
    if score[action_index] > 0.0:
        return (
            float(outcomes[query_index, action_index]),
            f"{base}::{actions[action_index]}",
        )
    return math.nan, "fallback"


def evaluate(dataset: str) -> tuple[dict, dict]:
    train = _load_archive(dataset, "train")
    test = _load_archive(dataset, "test")
    common_bases = sorted(set(train["action_bases"]) & set(test["action_bases"]))
    if train["actions"] != test["actions"]:
        raise RuntimeError(f"{dataset}: train/test action families differ")
    if set(train["qids"]) & set(test["qids"]):
        raise RuntimeError(f"{dataset}: official train and test queries overlap")
    base = max(common_bases, key=lambda name: float(train["bases"][name].mean()))
    train_base = train["bases"][base]
    test_base = test["bases"][base]
    train_outcomes = train["outcomes"][base]
    test_outcomes = test["outcomes"][base]
    train_effects = train_outcomes - train_base[:, None]

    train_x, test_x = _text_features(train["texts"], test["texts"], "word+char")
    similarity = (test_x @ train_x.T).toarray()
    k = max(2, int(math.ceil(math.sqrt(len(train["qids"])))))
    neighbours = np.argpartition(-similarity, k - 1, axis=1)[:, :k]

    methods = {
        name: {
            "outcome": test_base.copy(),
            "selected": np.full(len(test["qids"]), "fallback", dtype=object),
        }
        for name in (
            "global_best",
            "local_mean",
            "uncorrected",
            "empirical_bernstein",
            "biblioguard",
        )
    }
    global_mean = train_effects.mean(axis=0)
    for query_index in range(len(test["qids"])):
        local_neighbours = neighbours[query_index]
        (
            mean,
            standard_error,
            degrees_of_freedom,
            effective_n,
            variance,
        ) = _weighted_effect(
            train_effects[local_neighbours], similarity[query_index, local_neighbours]
        )
        bonferroni_critical = float(
            stats.t.ppf(
                1.0 - ALPHA_FAMILY / len(train["actions"]),
                degrees_of_freedom,
            )
        )
        uncorrected_critical = float(
            stats.t.ppf(1.0 - ALPHA_FAMILY, degrees_of_freedom)
        )
        scores = {
            "global_best": global_mean,
            "local_mean": mean,
            "uncorrected": mean - uncorrected_critical * standard_error,
            "empirical_bernstein": _empirical_bernstein_score(
                mean, variance, effective_n, len(train["actions"])
            ),
            "biblioguard": mean - bonferroni_critical * standard_error,
        }
        for name, score in scores.items():
            outcome, selected = _activate(
                score, test_outcomes, query_index, base, train["actions"]
            )
            if selected != "fallback":
                methods[name]["outcome"][query_index] = outcome
                methods[name]["selected"][query_index] = selected

    comparisons = {}
    for name, method in methods.items():
        outcome = method["outcome"]
        selected = method["selected"].astype(str)
        difference = outcome - test_base
        active = selected != "fallback"
        p_value = (
            float(stats.wilcoxon(outcome, test_base, alternative="two-sided").pvalue)
            if np.any(np.abs(difference) > EPS)
            else 1.0
        )
        comparisons[name] = {
            "N@10": float(outcome.mean()),
            "gain_N@10": float(difference.mean()),
            "selection_rate": float(active.mean()),
            "outcomes_all": _outcome_counts(difference),
            "outcomes_active": _outcome_counts(difference, active),
            "wilcoxon_p_two_sided": p_value,
            "selected_counts": dict(Counter(selected.tolist())),
        }
        if name == "biblioguard":
            comparisons[name]["paired_bootstrap_95ci"] = list(
                _bootstrap_ci(difference)
            )
    guarded = methods["biblioguard"]
    result = {
        "dataset": dataset,
        "protocol": "official train-to-test; test qrels used only for final evaluation",
        "n_train": len(train["qids"]),
        "n_test": len(test["qids"]),
        "selected_content_base": base,
        "neighbour_count": k,
        "n_actions": len(train["actions"]),
        "fallback_N@10": float(test_base.mean()),
        "biblioguard_N@10": comparisons["biblioguard"]["N@10"],
        "gain_N@10": comparisons["biblioguard"]["gain_N@10"],
        "selection_rate": comparisons["biblioguard"]["selection_rate"],
        "wilcoxon_p_two_sided": comparisons["biblioguard"][
            "wilcoxon_p_two_sided"
        ],
        "paired_bootstrap_95ci": comparisons["biblioguard"][
            "paired_bootstrap_95ci"
        ],
        "comparisons": comparisons,
    }
    arrays = {
        "qids": np.asarray(test["qids"]),
        "Fallback||N@10": test_base,
        "BiblioGuard||N@10": guarded["outcome"],
        "BiblioGuard||selected_action": guarded["selected"],
    }
    for name, method in methods.items():
        arrays[f"{name}||N@10"] = method["outcome"]
        arrays[f"{name}||selected_action"] = method["selected"]
    return result, arrays


def main() -> None:
    results = {}
    for dataset in DATASETS:
        result, arrays = evaluate(dataset)
        results[dataset] = result
        np.savez_compressed(
            RESULTS / f"{dataset}_biblioguard_transfer_perquery.npz", **arrays
        )
        print(
            f"{dataset:9s} {result['selected_content_base']:10s} "
            f"fallback={result['fallback_N@10']:.4f} "
            f"BiblioGuard={result['biblioguard_N@10']:.4f} "
            f"gain={result['gain_N@10']:+.4f} "
            f"select={result['selection_rate']:.1%}"
        )
    payload = {
        "method": "BiblioGuard",
        "protocol": "official train-to-test transfer",
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "results": results,
    }
    output = RESULTS / "biblioguard_transfer_results.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("saved", output)


if __name__ == "__main__":
    main()

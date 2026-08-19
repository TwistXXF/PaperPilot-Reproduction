#!/usr/bin/env python
"""BiblioGuard: confidence-gated bibliographic-metadata intervention.

The policy treats every non-zero citation or recency weight as an intervention
relative to BGE-Hybrid.  For a held-out query it retrieves lexically similar
training queries, estimates the paired NDCG@10 treatment effect of each atomic
metadata action, and activates an action only when its simultaneous one-sided
lower confidence bound is positive.  Otherwise it abstains to BGE-Hybrid.

Evaluation is five-fold cross-fitted: the held-out query's qrels never enter
its representation, neighbour set, effect estimate, or action decision.

Outputs
-------
results/biblioguard_results.json
results/{dataset}_biblioguard_perquery.npz
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize

import _layout as L


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
DATASETS = ("scidocs", "scifact", "nfcorpus", "trec-covid")
N_SPLITS = 5
ALPHA_FAMILY = 0.05
RANDOM_STATE = 42


def _load_queries(dataset: str, wanted: list[str]) -> list[str]:
    records = {}
    path = Path(L.raw_ds(dataset)) / "queries.jsonl"
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            records[str(row["_id"])] = row.get("text") or ""
    missing = [query_id for query_id in wanted if query_id not in records]
    if missing:
        raise RuntimeError(f"{dataset}: {len(missing)} query texts are missing")
    return [records[query_id] for query_id in wanted]


def _single_signal_actions(sensitivity: dict, dataset: str) -> list[str]:
    actions = sorted(
        sensitivity[dataset]["grid"],
        key=lambda key: tuple(float(part.split("=")[1]) for part in key.split("|")),
    )
    single_signal = []
    for action in actions:
        beta, gamma = (float(part.split("=")[1]) for part in action.split("|"))
        if (beta > 0 and gamma == 0) or (beta == 0 and gamma > 0):
            single_signal.append(action)
    if len(single_signal) != 9:
        raise RuntimeError(
            f"expected 9 single-signal actions, found {len(single_signal)}"
        )
    return single_signal


def _load_domain(dataset: str, sensitivity: dict) -> dict:
    archive = np.load(
        RESULTS / f"{dataset}_bge_hybrid_perquery.npz", allow_pickle=True
    )
    query_ids = [str(query_id) for query_id in archive["qids"]]
    baseline = np.asarray(archive["BGE-Hybrid||N@10"], dtype=float)
    actions = _single_signal_actions(sensitivity, dataset)
    outcomes = np.column_stack(
        [
            np.asarray(sensitivity[dataset]["grid"][action]["_pq"], dtype=float)
            for action in actions
        ]
    )
    if outcomes.shape != (len(query_ids), len(actions)):
        raise RuntimeError(f"{dataset}: action outcomes do not align with query ids")
    return {
        "query_ids": query_ids,
        "texts": _load_queries(dataset, query_ids),
        "baseline": baseline,
        "actions": actions,
        "outcomes": outcomes,
        "effects": outcomes - baseline[:, None],
    }


def _text_features(train_texts: list[str], test_texts: list[str]):
    word = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=12000,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=12000,
        sublinear_tf=True,
    )
    train = hstack([word.fit_transform(train_texts), char.fit_transform(train_texts)])
    test = hstack([word.transform(test_texts), char.transform(test_texts)])
    return normalize(train), normalize(test)


def _paired_effect_lcb(
    effects: np.ndarray,
    similarities: np.ndarray,
    family_alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    weights = np.maximum(similarities, 0.0) + 1e-3
    weights /= weights.sum()
    mean = np.sum(effects * weights[:, None], axis=0)
    variance = np.sum(weights[:, None] * (effects - mean[None, :]) ** 2, axis=0)
    effective_n = 1.0 / np.sum(weights**2)
    degrees_of_freedom = max(int(math.floor(effective_n)) - 1, 1)
    critical = float(
        stats.t.ppf(1.0 - family_alpha / effects.shape[1], degrees_of_freedom)
    )
    standard_error = np.sqrt(variance / max(effective_n, 1.0))
    return mean, mean - critical * standard_error, critical


def cross_fitted_policy(dataset: str, sensitivity: dict) -> dict:
    data = _load_domain(dataset, sensitivity)
    n_queries = len(data["query_ids"])
    routed = data["baseline"].copy()
    routed_unconstrained = data["baseline"].copy()
    selected = np.full(n_queries, "baseline", dtype=object)
    selected_unconstrained = np.full(n_queries, "baseline", dtype=object)
    lower_bound = np.zeros(n_queries, dtype=float)
    estimated_effect = np.zeros(n_queries, dtype=float)
    neighbour_count = np.zeros(n_queries, dtype=int)
    critical_values = np.zeros(n_queries, dtype=float)

    folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for train_index, test_index in folds.split(np.arange(n_queries)):
        train_texts = [data["texts"][index] for index in train_index]
        test_texts = [data["texts"][index] for index in test_index]
        train_x, test_x = _text_features(train_texts, test_texts)
        similarity = (test_x @ train_x.T).toarray()
        k = max(2, int(math.ceil(math.sqrt(len(train_index)))))
        neighbours = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        for local_index, global_index in enumerate(test_index):
            local_neighbours = neighbours[local_index]
            mean, lower, critical = _paired_effect_lcb(
                data["effects"][train_index[local_neighbours]],
                similarity[local_index, local_neighbours],
                ALPHA_FAMILY,
            )
            action_index = int(np.argmax(lower))
            lower_bound[global_index] = lower[action_index]
            estimated_effect[global_index] = mean[action_index]
            neighbour_count[global_index] = k
            critical_values[global_index] = critical
            if lower[action_index] > 0.0:
                routed[global_index] = data["outcomes"][global_index, action_index]
                selected[global_index] = data["actions"][action_index]
            unconstrained_index = int(np.argmax(mean))
            if mean[unconstrained_index] > 0.0:
                routed_unconstrained[global_index] = data["outcomes"][
                    global_index, unconstrained_index
                ]
                selected_unconstrained[global_index] = data["actions"][
                    unconstrained_index
                ]

    difference = routed - data["baseline"]
    nonzero = difference != 0
    p_greater = (
        float(stats.wilcoxon(routed, data["baseline"], alternative="greater").pvalue)
        if nonzero.any()
        else 1.0
    )
    p_two_sided = (
        float(stats.wilcoxon(routed, data["baseline"], alternative="two-sided").pvalue)
        if nonzero.any()
        else 1.0
    )
    sd = float(difference.std(ddof=1))
    effect_size = float(difference.mean() / sd) if sd > 1e-12 else 0.0
    result = {
        "dataset": dataset,
        "n_queries": n_queries,
        "n_splits": N_SPLITS,
        "family_alpha": ALPHA_FAMILY,
        "n_actions": len(data["actions"]),
        "actions": data["actions"],
        "neighbour_rule": "ceil(sqrt(n_training_queries))",
        "baseline_N@10": float(data["baseline"].mean()),
        "biblioguard_N@10": float(routed.mean()),
        "gain_N@10": float(difference.mean()),
        "selection_rate": float(np.mean(selected != "baseline")),
        "selected_counts": dict(Counter(selected.tolist())),
        "wilcoxon_p_greater": p_greater,
        "wilcoxon_p_two_sided": p_two_sided,
        "paired_cohen_d": effect_size,
        "ablation_unconstrained": {
            "N@10": float(routed_unconstrained.mean()),
            "gain_N@10": float((routed_unconstrained - data["baseline"]).mean()),
            "selection_rate": float(
                np.mean(selected_unconstrained != "baseline")
            ),
            "selected_counts": dict(Counter(selected_unconstrained.tolist())),
        },
    }
    np.savez_compressed(
        RESULTS / f"{dataset}_biblioguard_perquery.npz",
        qids=np.asarray(data["query_ids"]),
        **{
            "BGE-Hybrid||N@10": data["baseline"],
            "BiblioGuard||N@10": routed,
            "BiblioGuard-unconstrained||N@10": routed_unconstrained,
            "BiblioGuard||selected_action": selected,
            "BiblioGuard-unconstrained||selected_action": selected_unconstrained,
            "BiblioGuard||estimated_effect": estimated_effect,
            "BiblioGuard||lower_bound": lower_bound,
            "BiblioGuard||neighbour_count": neighbour_count,
            "BiblioGuard||critical_value": critical_values,
        },
    )
    return result


def _holm(results: dict[str, dict]) -> None:
    ordered = sorted(DATASETS, key=lambda dataset: results[dataset]["wilcoxon_p_greater"])
    running = 0.0
    for rank, dataset in enumerate(ordered):
        adjusted = min(
            1.0,
            (len(ordered) - rank) * results[dataset]["wilcoxon_p_greater"],
        )
        running = max(running, adjusted)
        results[dataset]["wilcoxon_p_holm"] = running


def main() -> None:
    sensitivity = json.load((RESULTS / "bge_sensitivity.json").open(encoding="utf-8"))
    results = {dataset: cross_fitted_policy(dataset, sensitivity) for dataset in DATASETS}
    _holm(results)
    payload = {
        "method": "BiblioGuard",
        "protocol": "five-fold cross-fitted paired treatment-effect routing",
        "results": results,
        "macro": {
            "baseline_N@10": float(np.mean([row["baseline_N@10"] for row in results.values()])),
            "biblioguard_N@10": float(np.mean([row["biblioguard_N@10"] for row in results.values()])),
            "gain_N@10": float(np.mean([row["gain_N@10"] for row in results.values()])),
        },
    }
    with (RESULTS / "biblioguard_results.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    for dataset, row in results.items():
        print(
            f"{dataset:11s} baseline={row['baseline_N@10']:.4f} "
            f"BiblioGuard={row['biblioguard_N@10']:.4f} "
            f"gain={row['gain_N@10']:+.4f} select={row['selection_rate']:.1%} "
            f"Holm p={row['wilcoxon_p_holm']:.4g}"
        )
    print("saved", RESULTS / "biblioguard_results.json")


if __name__ == "__main__":
    main()

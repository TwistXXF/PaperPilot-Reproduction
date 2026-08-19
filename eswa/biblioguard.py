#!/usr/bin/env python
"""Evaluate BiblioGuard with confound-free bibliographic interventions.

Within every outer training fold, the strongest available content-only
retriever is selected using training labels. Every candidate action preserves
that retriever's content score, fusion rule, and top-100 candidate set and
changes exactly one bibliographic term. A held-out query is routed only when a
Bonferroni-corrected pessimistic decision score is positive.

The score is an operational, variance-penalised decision rule. It is not
claimed to be a formal confidence bound because neighbouring queries are not
independent draws. All query representations and decisions are cross-fitted.
"""
from __future__ import annotations

from collections import Counter
import argparse
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
PRIMARY_SEED = 42
REPEAT_SEEDS = tuple(range(10))
BOOTSTRAP_SEED = 2026
BOOTSTRAP_REPS = 10_000
RISK_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
EPS = 1e-12


def _load_queries(dataset: str, wanted: list[str]) -> list[str]:
    records: dict[str, str] = {}
    path = Path(L.raw_ds(dataset)) / "queries.jsonl"
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            records[str(row["_id"])] = row.get("text") or ""
    missing = [query_id for query_id in wanted if query_id not in records]
    if missing:
        raise RuntimeError(f"{dataset}: {len(missing)} query texts are missing")
    return [records[query_id] for query_id in wanted]


def _load_domain(dataset: str) -> dict:
    path = RESULTS / f"{dataset}_biblioguard_actions.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path.name}; run biblioguard_actions.py {dataset} first"
        )
    archive = np.load(path, allow_pickle=True)
    query_ids = archive["qids"].astype(str).tolist()
    bases = {
        key.split("::", 1)[1]: np.asarray(archive[key], dtype=float)
        for key in archive.files
        if key.startswith("base::")
    }
    action_bases = sorted(
        {
            key.split("::", 2)[1]
            for key in archive.files
            if key.startswith("action::")
        }
    )
    actions_by_base: dict[str, dict[str, np.ndarray]] = {}
    for base in action_bases:
        prefix = f"action::{base}::"
        actions_by_base[base] = {
            key[len(prefix) :]: np.asarray(archive[key], dtype=float)
            for key in archive.files
            if key.startswith(prefix)
        }
    action_sets = {tuple(sorted(rows)) for rows in actions_by_base.values()}
    if len(action_sets) != 1:
        raise RuntimeError(f"{dataset}: action families differ between bases")
    actions = list(next(iter(action_sets)))
    if len(actions) != 9:
        raise RuntimeError(f"{dataset}: expected 9 actions, found {len(actions)}")
    for base in action_bases:
        if base not in bases:
            raise RuntimeError(f"{dataset}: missing content fallback {base}")
        if any(
            len(values) != len(query_ids)
            for values in actions_by_base[base].values()
        ):
            raise RuntimeError(f"{dataset}/{base}: outcome length mismatch")
    return {
        "query_ids": query_ids,
        "texts": _load_queries(dataset, query_ids),
        "bases": bases,
        "action_bases": action_bases,
        "actions": actions,
        "outcomes": {
            base: np.column_stack(
                [actions_by_base[base][action] for action in actions]
            )
            for base in action_bases
        },
    }


def _text_features(
    train_texts: list[str], test_texts: list[str], mode: str
):
    matrices_train = []
    matrices_test = []
    if mode in ("word", "word+char"):
        word = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=12_000,
            sublinear_tf=True,
        )
        matrices_train.append(word.fit_transform(train_texts))
        matrices_test.append(word.transform(test_texts))
    if mode in ("char", "word+char"):
        char = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=12_000,
            sublinear_tf=True,
        )
        matrices_train.append(char.fit_transform(train_texts))
        matrices_test.append(char.transform(test_texts))
    if not matrices_train:
        raise ValueError(f"unknown feature mode: {mode}")
    train = (
        matrices_train[0]
        if len(matrices_train) == 1
        else hstack(matrices_train)
    )
    test = (
        matrices_test[0]
        if len(matrices_test) == 1
        else hstack(matrices_test)
    )
    return normalize(train), normalize(test)


def _weighted_effect(
    effects: np.ndarray, similarities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Return weighted mean, standard error, and effective degrees of freedom."""
    weights = np.maximum(similarities, 0.0) + 1e-3
    weights /= weights.sum()
    mean = np.sum(effects * weights[:, None], axis=0)
    sum_w2 = float(np.sum(weights**2))
    denominator = max(1.0 - sum_w2, EPS)
    variance = np.sum(
        weights[:, None] * (effects - mean[None, :]) ** 2, axis=0
    ) / denominator
    standard_error = np.sqrt(np.maximum(variance, 0.0) * sum_w2)
    effective_n = 1.0 / sum_w2
    degrees_of_freedom = max(int(math.floor(effective_n)) - 1, 1)
    return (
        mean,
        standard_error,
        float(degrees_of_freedom),
        float(effective_n),
        variance,
    )


def _empirical_bernstein_score(
    mean: np.ndarray,
    variance: np.ndarray,
    effective_n: float,
    n_actions: int,
) -> np.ndarray:
    """Conservative bounded-effect comparator (operational, not a guarantee)."""
    log_term = math.log(3.0 * n_actions / ALPHA_FAMILY)
    # Paired NDCG effects lie in [-1, 1], hence range=2.
    penalty = np.sqrt(
        2.0 * np.maximum(variance, 0.0) * log_term / effective_n
    ) + 6.0 * log_term / effective_n
    return mean - penalty


def _empty_routes(n_queries: int, fallback: np.ndarray) -> dict:
    return {
        name: {
            "outcome": fallback.copy(),
            "selected": np.full(n_queries, "fallback", dtype=object),
        }
        for name in (
            "global_best",
            "local_mean",
            "uncorrected",
            "empirical_bernstein",
            "biblioguard",
        )
    }


def _activate(
    route: dict,
    index: int,
    score: np.ndarray,
    outcome_row: np.ndarray,
    base: str,
    actions: list[str],
) -> None:
    action_index = int(np.argmax(score))
    if score[action_index] > 0.0:
        route["outcome"][index] = outcome_row[action_index]
        route["selected"][index] = f"{base}::{actions[action_index]}"


def run_cross_fitted(
    data: dict,
    seed: int,
    feature_mode: str = "word+char",
    k_multiplier: float = 1.0,
    keep_diagnostics: bool = False,
) -> dict:
    n_queries = len(data["query_ids"])
    fallback = np.full(n_queries, np.nan, dtype=float)
    fallback_base = np.full(n_queries, "", dtype=object)
    routes = _empty_routes(n_queries, fallback)
    risk_routes = {
        scale: {
            "outcome": fallback.copy(),
            "selected": np.full(n_queries, "fallback", dtype=object),
        }
        for scale in RISK_SCALES
    }
    estimated_effect = np.zeros(n_queries, dtype=float)
    penalty = np.zeros(n_queries, dtype=float)
    decision_score = np.zeros(n_queries, dtype=float)
    neighbour_count = np.zeros(n_queries, dtype=int)

    folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for train_index, test_index in folds.split(np.arange(n_queries)):
        base = max(
            data["action_bases"],
            key=lambda name: float(data["bases"][name][train_index].mean()),
        )
        base_values = data["bases"][base]
        outcomes = data["outcomes"][base]
        effects = outcomes - base_values[:, None]
        fallback[test_index] = base_values[test_index]
        fallback_base[test_index] = base
        for route in list(routes.values()) + list(risk_routes.values()):
            route["outcome"][test_index] = base_values[test_index]

        train_texts = [data["texts"][index] for index in train_index]
        test_texts = [data["texts"][index] for index in test_index]
        train_x, test_x = _text_features(train_texts, test_texts, feature_mode)
        similarity = (test_x @ train_x.T).toarray()
        k = int(math.ceil(math.sqrt(len(train_index)) * k_multiplier))
        k = min(len(train_index), max(2, k))
        neighbours = np.argpartition(-similarity, k - 1, axis=1)[:, :k]

        global_mean = effects[train_index].mean(axis=0)
        global_index = int(np.argmax(global_mean))
        for index in test_index:
            if global_mean[global_index] > 0.0:
                routes["global_best"]["outcome"][index] = outcomes[
                    index, global_index
                ]
                routes["global_best"]["selected"][index] = (
                    f"{base}::{data['actions'][global_index]}"
                )

        for local_index, query_index in enumerate(test_index):
            local_neighbours = neighbours[local_index]
            local_effects = effects[train_index[local_neighbours]]
            local_similarity = similarity[local_index, local_neighbours]
            (
                mean,
                standard_error,
                degrees_of_freedom,
                effective_n,
                variance,
            ) = _weighted_effect(
                local_effects, local_similarity
            )
            critical_bonferroni = float(
                stats.t.ppf(
                    1.0 - ALPHA_FAMILY / len(data["actions"]),
                    degrees_of_freedom,
                )
            )
            critical_uncorrected = float(
                stats.t.ppf(1.0 - ALPHA_FAMILY, degrees_of_freedom)
            )
            bonferroni_penalty = critical_bonferroni * standard_error
            score = mean - bonferroni_penalty
            _activate(
                routes["local_mean"],
                query_index,
                mean,
                outcomes[query_index],
                base,
                data["actions"],
            )
            _activate(
                routes["uncorrected"],
                query_index,
                mean - critical_uncorrected * standard_error,
                outcomes[query_index],
                base,
                data["actions"],
            )
            _activate(
                routes["empirical_bernstein"],
                query_index,
                _empirical_bernstein_score(
                    mean, variance, effective_n, len(data["actions"])
                ),
                outcomes[query_index],
                base,
                data["actions"],
            )
            _activate(
                routes["biblioguard"],
                query_index,
                score,
                outcomes[query_index],
                base,
                data["actions"],
            )
            for scale, route in risk_routes.items():
                _activate(
                    route,
                    query_index,
                    mean - scale * bonferroni_penalty,
                    outcomes[query_index],
                    base,
                    data["actions"],
                )
            chosen = int(np.argmax(score))
            estimated_effect[query_index] = mean[chosen]
            penalty[query_index] = bonferroni_penalty[chosen]
            decision_score[query_index] = score[chosen]
            neighbour_count[query_index] = k

    if not np.isfinite(fallback).all():
        raise RuntimeError("cross-fitting left missing fallback outcomes")
    payload = {
        "fallback": fallback,
        "fallback_base": fallback_base,
        "routes": routes,
        "risk_routes": risk_routes,
    }
    if keep_diagnostics:
        payload.update(
            {
                "estimated_effect": estimated_effect,
                "penalty": penalty,
                "decision_score": decision_score,
                "neighbour_count": neighbour_count,
            }
        )
    return payload


def _outcome_counts(
    difference: np.ndarray, active: np.ndarray | None = None
) -> dict:
    values = difference if active is None else difference[active]
    return {
        "improved": int(np.sum(values > EPS)),
        "unchanged": int(np.sum(np.abs(values) <= EPS)),
        "harmed": int(np.sum(values < -EPS)),
    }


def _bootstrap_ci(difference: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPS, dtype=float)
    for start in range(0, BOOTSTRAP_REPS, 500):
        stop = min(start + 500, BOOTSTRAP_REPS)
        indices = rng.integers(
            0, len(difference), size=(stop - start, len(difference))
        )
        estimates[start:stop] = difference[indices].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def _summarise_route(
    route: dict, fallback: np.ndarray, bootstrap: bool = False
) -> dict:
    outcome = np.asarray(route["outcome"], dtype=float)
    selected = np.asarray(route["selected"], dtype=str)
    difference = outcome - fallback
    active = selected != "fallback"
    if np.any(np.abs(difference) > EPS):
        p_two_sided = float(
            stats.wilcoxon(outcome, fallback, alternative="two-sided").pvalue
        )
    else:
        p_two_sided = 1.0
    summary = {
        "N@10": float(outcome.mean()),
        "gain_N@10": float(difference.mean()),
        "selection_rate": float(active.mean()),
        "selected_counts": dict(Counter(selected.tolist())),
        "outcomes_all": _outcome_counts(difference),
        "outcomes_active": _outcome_counts(difference, active),
        "wilcoxon_p_two_sided": p_two_sided,
    }
    if bootstrap:
        low, high = _bootstrap_ci(difference)
        summary["paired_bootstrap_95ci"] = [low, high]
    return summary


def evaluate_dataset(dataset: str) -> tuple[dict, dict]:
    data = _load_domain(dataset)
    primary = run_cross_fitted(data, PRIMARY_SEED, keep_diagnostics=True)
    fallback = primary["fallback"]
    comparisons = {
        name: _summarise_route(
            route, fallback, bootstrap=(name == "biblioguard")
        )
        for name, route in primary["routes"].items()
    }
    repeats = []
    for seed in REPEAT_SEEDS:
        repeated = run_cross_fitted(data, seed)
        row = _summarise_route(
            repeated["routes"]["biblioguard"], repeated["fallback"]
        )
        repeats.append(
            {
                "seed": seed,
                "fallback_N@10": float(repeated["fallback"].mean()),
                "biblioguard_N@10": row["N@10"],
                "gain_N@10": row["gain_N@10"],
                "selection_rate": row["selection_rate"],
            }
        )
    repeated_gains = np.asarray([row["gain_N@10"] for row in repeats])
    repeated_scores = np.asarray(
        [row["biblioguard_N@10"] for row in repeats]
    )
    repeated_selection = np.asarray(
        [row["selection_rate"] for row in repeats]
    )

    risk_coverage = []
    for scale in RISK_SCALES:
        route = primary["risk_routes"][scale]
        summary = _summarise_route(route, fallback)
        risk_coverage.append(
            {
                "penalty_scale": scale,
                "coverage": summary["selection_rate"],
                "gain_N@10": summary["gain_N@10"],
                "harmed_active": summary["outcomes_active"]["harmed"],
            }
        )

    ablations = {}
    for label, feature_mode, k_multiplier in (
        ("word_only", "word", 1.0),
        ("char_only", "char", 1.0),
        ("k_half", "word+char", 0.5),
        ("k_double", "word+char", 2.0),
    ):
        run = run_cross_fitted(
            data,
            PRIMARY_SEED,
            feature_mode=feature_mode,
            k_multiplier=k_multiplier,
        )
        ablations[label] = _summarise_route(
            run["routes"]["biblioguard"], run["fallback"]
        )

    guarded = np.asarray(
        primary["routes"]["biblioguard"]["outcome"], dtype=float
    )
    selected = np.asarray(
        primary["routes"]["biblioguard"]["selected"], dtype=object
    )
    result = {
        "dataset": dataset,
        "n_queries": len(data["query_ids"]),
        "n_splits": N_SPLITS,
        "family_alpha": ALPHA_FAMILY,
        "n_actions": len(data["actions"]),
        "actions": data["actions"],
        "available_content_bases": data["action_bases"],
        "fallback_selection": "training-fold best content-only retriever",
        "fallback_counts": dict(Counter(primary["fallback_base"].tolist())),
        "feature_mode": "word+char",
        "neighbour_rule": "ceil(sqrt(n_training_queries))",
        "decision_rule": (
            "mean effect minus Bonferroni t penalty; activate iff positive"
        ),
        "decision_score_caveat": (
            "operational pessimistic score, not a formal confidence bound"
        ),
        "fallback_N@10": float(fallback.mean()),
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
        "repeated_seeds": {
            "seeds": list(REPEAT_SEEDS),
            "runs": repeats,
            "gain_mean": float(repeated_gains.mean()),
            "gain_std": float(repeated_gains.std(ddof=1)),
            "score_mean": float(repeated_scores.mean()),
            "score_std": float(repeated_scores.std(ddof=1)),
            "selection_mean": float(repeated_selection.mean()),
            "selection_std": float(repeated_selection.std(ddof=1)),
        },
        "risk_coverage": risk_coverage,
        "ablations": ablations,
    }
    arrays = {
        "qids": np.asarray(data["query_ids"]),
        "Fallback||N@10": fallback,
        "Fallback||content_base": primary["fallback_base"],
        "BiblioGuard||N@10": guarded,
        "BiblioGuard||selected_action": selected,
        "BiblioGuard||estimated_effect": primary["estimated_effect"],
        "BiblioGuard||penalty": primary["penalty"],
        "BiblioGuard||decision_score": primary["decision_score"],
        "BiblioGuard||neighbour_count": primary["neighbour_count"],
    }
    for name, route in primary["routes"].items():
        arrays[f"{name}||N@10"] = route["outcome"]
        arrays[f"{name}||selected_action"] = route["selected"]
    return result, arrays


def _holm_two_sided(results: dict[str, dict]) -> None:
    ordered = sorted(
        results,
        key=lambda dataset: results[dataset]["wilcoxon_p_two_sided"],
    )
    running = 0.0
    for rank, dataset in enumerate(ordered):
        adjusted = min(
            1.0,
            (len(ordered) - rank)
            * results[dataset]["wilcoxon_p_two_sided"],
        )
        running = max(running, adjusted)
        results[dataset]["wilcoxon_p_holm_two_sided"] = running


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", choices=DATASETS)
    args = parser.parse_args(argv)
    datasets = tuple(args.datasets) or DATASETS
    results = {}
    archives = {}
    for dataset in datasets:
        results[dataset], archives[dataset] = evaluate_dataset(dataset)
        print(
            f"{dataset:11s} fallback={results[dataset]['fallback_N@10']:.4f} "
            f"BiblioGuard={results[dataset]['biblioguard_N@10']:.4f} "
            f"gain={results[dataset]['gain_N@10']:+.4f} "
            f"select={results[dataset]['selection_rate']:.1%}"
        )
    _holm_two_sided(results)
    payload = {
        "method": "BiblioGuard",
        "protocol": "five-fold cross-fitted selective metadata intervention",
        "primary_seed": PRIMARY_SEED,
        "repeat_seeds": list(REPEAT_SEEDS),
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "results": results,
        "macro": {
            "fallback_N@10": float(
                np.mean([row["fallback_N@10"] for row in results.values()])
            ),
            "biblioguard_N@10": float(
                np.mean([row["biblioguard_N@10"] for row in results.values()])
            ),
            "gain_N@10": float(
                np.mean([row["gain_N@10"] for row in results.values()])
            ),
        },
    }
    with (RESULTS / "biblioguard_results.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2)
    for dataset, arrays in archives.items():
        np.savez_compressed(
            RESULTS / f"{dataset}_biblioguard_perquery.npz", **arrays
        )
    print("saved", RESULTS / "biblioguard_results.json")


if __name__ == "__main__":
    main()

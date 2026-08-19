#!/usr/bin/env python
"""Integrity audit for the revised BiblioGuard artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.model_selection import KFold

from biblioguard import (
    ALPHA_FAMILY,
    DATASETS,
    N_SPLITS,
    PRIMARY_SEED,
    REPEAT_SEEDS,
    _bootstrap_ci,
    _load_domain,
    run_cross_fitted,
)


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
EXPECTED_N = {"scidocs": 1000, "scifact": 300, "nfcorpus": 323,
              "trec-covid": 50}
EXPECTED_ACTIONS = {
    "citation:0.05", "citation:0.10", "citation:0.15",
    "citation:0.20", "citation:0.30", "recency:0.05",
    "recency:0.10", "recency:0.15", "recency:0.20",
}
EPS = 1e-12


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, message: str, atol=1e-12) -> None:
    check(bool(np.isclose(actual, expected, atol=atol, rtol=1e-10)),
          f"{message}: {actual} != {expected}")


def _verify_route_alignment(
    dataset: str,
    method: str,
    archive,
    action_archive,
    fallback: np.ndarray,
) -> int:
    outcome = archive[f"{method}||N@10"].astype(float)
    selected = archive[f"{method}||selected_action"].astype(str)
    check(len(outcome) == len(fallback), f"{dataset}/{method}: length")
    inactive = selected == "fallback"
    check(np.all(outcome[inactive] == fallback[inactive]),
          f"{dataset}/{method}: fallback outcomes changed")
    checks = 2
    for label in set(selected) - {"fallback"}:
        base, action = label.split("::", 1)
        check(action in EXPECTED_ACTIONS,
              f"{dataset}/{method}: unknown action {action}")
        key = f"action::{base}::{action}"
        check(key in action_archive.files,
              f"{dataset}/{method}: missing action source {key}")
        mask = selected == label
        check(np.all(outcome[mask] == action_archive[key][mask]),
              f"{dataset}/{method}: outcome/source mismatch for {label}")
        checks += 3
    return checks


def verify_cross_fitted(payload: dict) -> int:
    check(payload["method"] == "BiblioGuard", "method name")
    check(set(payload["results"]) == set(DATASETS), "four-domain result set")
    check(payload["primary_seed"] == PRIMARY_SEED, "primary seed")
    check(payload["repeat_seeds"] == list(REPEAT_SEEDS), "repeat seeds")
    checks = 4
    raw_p = {}
    for dataset in DATASETS:
        row = payload["results"][dataset]
        archive = np.load(
            RESULTS / f"{dataset}_biblioguard_perquery.npz", allow_pickle=True
        )
        actions = np.load(
            RESULTS / f"{dataset}_biblioguard_actions.npz", allow_pickle=True
        )
        qids = archive["qids"].astype(str)
        check(len(qids) == EXPECTED_N[dataset], f"{dataset}: query count")
        check(len(set(qids)) == len(qids), f"{dataset}: unique qids")
        check(np.array_equal(qids, actions["qids"].astype(str)),
              f"{dataset}: action qid alignment")
        check(set(row["actions"]) == EXPECTED_ACTIONS,
              f"{dataset}: action family")
        check(row["n_actions"] == 9, f"{dataset}: action count")
        check(row["n_splits"] == N_SPLITS, f"{dataset}: fold count")
        check(row["family_alpha"] == ALPHA_FAMILY,
              f"{dataset}: family alpha")
        check("not a formal confidence bound" in row["decision_score_caveat"],
              f"{dataset}: caveat missing")
        checks += 8

        fallback = archive["Fallback||N@10"].astype(float)
        fallback_base = archive["Fallback||content_base"].astype(str)
        guarded = archive["BiblioGuard||N@10"].astype(float)
        selected = archive["BiblioGuard||selected_action"].astype(str)
        score = archive["BiblioGuard||decision_score"].astype(float)
        check(np.array_equal(selected != "fallback", score > 0.0),
              f"{dataset}: gate/selection mismatch")
        for base in set(fallback_base):
            key = f"base::{base}"
            check(key in actions.files, f"{dataset}: missing fallback {base}")
            mask = fallback_base == base
            check(np.all(fallback[mask] == actions[key][mask]),
                  f"{dataset}: fallback source mismatch for {base}")
            checks += 2

        # Reconstruct training-fold content-base selection independently.
        data = _load_domain(dataset)
        expected_base = np.full(len(qids), "", dtype=object)
        folds = KFold(n_splits=N_SPLITS, shuffle=True,
                      random_state=PRIMARY_SEED)
        for train_index, test_index in folds.split(np.arange(len(qids))):
            base = max(
                data["action_bases"],
                key=lambda name: float(data["bases"][name][train_index].mean()),
            )
            expected_base[test_index] = base
        check(np.array_equal(fallback_base, expected_base),
              f"{dataset}: training-fold best fallback not reproduced")
        checks += 1

        for method in ("global_best", "local_mean", "uncorrected",
                       "empirical_bernstein", "biblioguard"):
            checks += _verify_route_alignment(
                dataset, method, archive, actions, fallback
            )

        difference = guarded - fallback
        close(fallback.mean(), row["fallback_N@10"],
              f"{dataset}: fallback mean")
        close(guarded.mean(), row["biblioguard_N@10"],
              f"{dataset}: guarded mean")
        close(difference.mean(), row["gain_N@10"],
              f"{dataset}: gain")
        close(np.mean(selected != "fallback"), row["selection_rate"],
              f"{dataset}: selection rate")
        p_value = (
            float(stats.wilcoxon(guarded, fallback,
                                 alternative="two-sided").pvalue)
            if np.any(np.abs(difference) > EPS) else 1.0
        )
        close(p_value, row["wilcoxon_p_two_sided"],
              f"{dataset}: two-sided Wilcoxon")
        bootstrap_ci = _bootstrap_ci(difference)
        close(bootstrap_ci[0], row["paired_bootstrap_95ci"][0],
              f"{dataset}: bootstrap lower")
        close(bootstrap_ci[1], row["paired_bootstrap_95ci"][1],
              f"{dataset}: bootstrap upper")
        check(row["gain_N@10"] >= -EPS,
              f"{dataset}: guarded negative mean transfer")
        raw_p[dataset] = p_value
        checks += 8

        # Recompute every repeated seed from action archives.
        gains = []
        scores = []
        selections = []
        runs = row["repeated_seeds"]["runs"]
        check([item["seed"] for item in runs] == list(REPEAT_SEEDS),
              f"{dataset}: repeated seed list")
        checks += 1
        for expected, seed in zip(runs, REPEAT_SEEDS):
            rerun = run_cross_fitted(data, seed)
            result = rerun["routes"]["biblioguard"]
            outcome = result["outcome"]
            base = rerun["fallback"]
            gain = float((outcome - base).mean())
            selection = float(np.mean(result["selected"] != "fallback"))
            close(base.mean(), expected["fallback_N@10"],
                  f"{dataset}/seed{seed}: fallback")
            close(outcome.mean(), expected["biblioguard_N@10"],
                  f"{dataset}/seed{seed}: score")
            close(gain, expected["gain_N@10"],
                  f"{dataset}/seed{seed}: gain")
            close(selection, expected["selection_rate"],
                  f"{dataset}/seed{seed}: selection")
            gains.append(gain)
            scores.append(float(outcome.mean()))
            selections.append(selection)
            checks += 4
        repeated = row["repeated_seeds"]
        close(np.mean(gains), repeated["gain_mean"],
              f"{dataset}: repeated gain mean")
        close(np.std(gains, ddof=1), repeated["gain_std"],
              f"{dataset}: repeated gain sd")
        close(np.mean(scores), repeated["score_mean"],
              f"{dataset}: repeated score mean")
        close(np.std(scores, ddof=1), repeated["score_std"],
              f"{dataset}: repeated score sd")
        close(np.mean(selections), repeated["selection_mean"],
              f"{dataset}: repeated selection mean")
        close(np.std(selections, ddof=1), repeated["selection_std"],
              f"{dataset}: repeated selection sd")
        checks += 6

    # Independent Holm reconstruction.
    ordered = sorted(DATASETS, key=lambda name: raw_p[name])
    running = 0.0
    for rank, dataset in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * raw_p[dataset]))
        close(running, payload["results"][dataset]["wilcoxon_p_holm_two_sided"],
              f"{dataset}: Holm p")
        checks += 1
    return checks


def verify_transfer(payload: dict) -> int:
    check(set(payload["results"]) == {"scifact", "nfcorpus"},
          "transfer dataset set")
    checks = 1
    for dataset in ("scifact", "nfcorpus"):
        row = payload["results"][dataset]
        train = np.load(
            RESULTS / f"{dataset}_train_biblioguard_actions.npz",
            allow_pickle=True,
        )
        test_actions = np.load(
            RESULTS / f"{dataset}_biblioguard_actions.npz", allow_pickle=True
        )
        test = np.load(
            RESULTS / f"{dataset}_biblioguard_transfer_perquery.npz",
            allow_pickle=True,
        )
        check(not (set(train["qids"].astype(str)) &
                   set(test["qids"].astype(str))),
              f"{dataset}: train/test query overlap")
        check(row["n_train"] == len(train["qids"]),
              f"{dataset}: train count")
        check(row["n_test"] == len(test["qids"]),
              f"{dataset}: test count")
        common = sorted(
            {key.split("::", 2)[1] for key in train.files
             if key.startswith("action::")}
            & {key.split("::", 2)[1] for key in test_actions.files
               if key.startswith("action::")}
        )
        expected_base = max(
            common, key=lambda name: float(train[f"base::{name}"].mean())
        )
        check(expected_base == row["selected_content_base"],
              f"{dataset}: train-only content base")
        fallback = test["Fallback||N@10"].astype(float)
        guarded = test["BiblioGuard||N@10"].astype(float)
        selected = test["BiblioGuard||selected_action"].astype(str)
        check(np.array_equal(fallback,
                             test_actions[f"base::{expected_base}"]),
              f"{dataset}: transfer fallback source")
        inactive = selected == "fallback"
        check(np.all(guarded[inactive] == fallback[inactive]),
              f"{dataset}: transfer abstention changed fallback")
        for label in set(selected) - {"fallback"}:
            base, action = label.split("::", 1)
            source = test_actions[f"action::{base}::{action}"]
            mask = selected == label
            check(np.all(guarded[mask] == source[mask]),
                  f"{dataset}: transfer action mismatch")
            checks += 1
        close(fallback.mean(), row["fallback_N@10"],
              f"{dataset}: transfer fallback mean")
        close(guarded.mean(), row["biblioguard_N@10"],
              f"{dataset}: transfer guarded mean")
        close((guarded - fallback).mean(), row["gain_N@10"],
              f"{dataset}: transfer gain")
        close(np.mean(selected != "fallback"), row["selection_rate"],
              f"{dataset}: transfer selection")
        checks += 10
    return checks


def main() -> None:
    payload = json.loads(
        (RESULTS / "biblioguard_results.json").read_text(encoding="utf-8")
    )
    transfer = json.loads(
        (RESULTS / "biblioguard_transfer_results.json").read_text(
            encoding="utf-8"
        )
    )
    checks = verify_cross_fitted(payload) + verify_transfer(transfer)
    print(f"BiblioGuard verification passed: {checks} checks, 4 domains")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Independent integrity checks for the released BiblioGuard artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
DATASETS = ("scidocs", "scifact", "nfcorpus", "trec-covid")
EXPECTED_N = {"scidocs": 1000, "scifact": 300, "nfcorpus": 323,
              "trec-covid": 50}
EXPECTED_ACTIONS = {
    "beta=0.0|gamma=0.05", "beta=0.0|gamma=0.1",
    "beta=0.0|gamma=0.15", "beta=0.0|gamma=0.2",
    "beta=0.05|gamma=0.0", "beta=0.1|gamma=0.0",
    "beta=0.15|gamma=0.0", "beta=0.2|gamma=0.0",
    "beta=0.3|gamma=0.0",
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, message: str) -> None:
    check(bool(np.isclose(actual, expected, atol=1e-12, rtol=1e-10)),
          f"{message}: {actual} != {expected}")


def main() -> None:
    payload = json.loads((RESULTS / "biblioguard_results.json").read_text(
        encoding="utf-8"))
    sensitivity = json.loads((RESULTS / "bge_sensitivity.json").read_text(
        encoding="utf-8"))
    check(payload["method"] == "BiblioGuard", "unexpected method name")
    check(set(payload["results"]) == set(DATASETS), "dataset set changed")

    checks = 0
    domain_gains = []
    for dataset in DATASETS:
        row = payload["results"][dataset]
        archive = np.load(RESULTS / f"{dataset}_biblioguard_perquery.npz",
                          allow_pickle=True)
        qids = archive["qids"].astype(str)
        baseline = archive["BGE-Hybrid||N@10"].astype(float)
        guarded = archive["BiblioGuard||N@10"].astype(float)
        unconstrained = archive["BiblioGuard-unconstrained||N@10"].astype(float)
        selected = archive["BiblioGuard||selected_action"].astype(str)
        selected_unconstrained = archive[
            "BiblioGuard-unconstrained||selected_action"].astype(str)
        lower = archive["BiblioGuard||lower_bound"].astype(float)

        check(len(qids) == EXPECTED_N[dataset], f"{dataset}: query count")
        check(len(set(qids)) == len(qids), f"{dataset}: duplicate query ids")
        check(set(row["actions"]) == EXPECTED_ACTIONS,
              f"{dataset}: action family changed")
        check(row["n_actions"] == 9, f"{dataset}: action count")
        check(row["n_splits"] == 5, f"{dataset}: fold count")
        check(row["family_alpha"] == 0.05, f"{dataset}: family alpha")
        check(set(selected) <= EXPECTED_ACTIONS | {"baseline"},
              f"{dataset}: invalid guarded action")
        check(set(selected_unconstrained) <= EXPECTED_ACTIONS | {"baseline"},
              f"{dataset}: invalid unconstrained action")
        check(np.array_equal(selected != "baseline", lower > 0.0),
              f"{dataset}: confidence gate and actions disagree")
        check(np.all(guarded[selected == "baseline"] ==
                     baseline[selected == "baseline"]),
              f"{dataset}: fallback does not reproduce baseline")
        check(np.all(unconstrained[selected_unconstrained == "baseline"] ==
                     baseline[selected_unconstrained == "baseline"]),
              f"{dataset}: unconstrained fallback changed baseline")

        for action in EXPECTED_ACTIONS:
            outcome = np.asarray(
                sensitivity[dataset]["grid"][action]["_pq"], dtype=float)
            check(len(outcome) == len(qids), f"{dataset}/{action}: alignment")
            mask = selected == action
            check(np.all(guarded[mask] == outcome[mask]),
                  f"{dataset}/{action}: guarded outcome mismatch")
            mask_u = selected_unconstrained == action
            check(np.all(unconstrained[mask_u] == outcome[mask_u]),
                  f"{dataset}/{action}: unconstrained outcome mismatch")

        close(baseline.mean(), row["baseline_N@10"],
              f"{dataset}: baseline mean")
        close(guarded.mean(), row["biblioguard_N@10"],
              f"{dataset}: guarded mean")
        close((guarded - baseline).mean(), row["gain_N@10"],
              f"{dataset}: guarded gain")
        close(np.mean(selected != "baseline"), row["selection_rate"],
              f"{dataset}: selection rate")
        close(unconstrained.mean(), row["ablation_unconstrained"]["N@10"],
              f"{dataset}: unconstrained mean")
        close((unconstrained - baseline).mean(),
              row["ablation_unconstrained"]["gain_N@10"],
              f"{dataset}: unconstrained gain")

        if np.any(guarded != baseline):
            p_value = stats.wilcoxon(
                guarded, baseline, alternative="greater").pvalue
            close(float(p_value), row["wilcoxon_p_greater"],
                  f"{dataset}: Wilcoxon p")
        else:
            close(row["wilcoxon_p_greater"], 1.0,
                  f"{dataset}: null Wilcoxon p")

        domain_gains.append(row["gain_N@10"])
        checks += 33

    close(np.mean(domain_gains), payload["macro"]["gain_N@10"],
          "macro gain")
    check(all(gain >= -1e-15 for gain in domain_gains),
          "negative mean transfer detected in guarded results")
    check(sum(payload["results"][d]["ablation_unconstrained"]["gain_N@10"] < 0
              for d in DATASETS) == 3,
          "unconstrained ablation no longer exhibits three-domain transfer harm")
    print(f"BiblioGuard verification passed: {checks + 3} checks, 4 domains")


if __name__ == "__main__":
    main()

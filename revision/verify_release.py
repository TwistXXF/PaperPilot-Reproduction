from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


REVISION = Path(__file__).resolve().parent
REPOSITORY = REVISION.parent
sys.path.insert(0, str(REVISION))
sys.path.insert(0, str(REVISION / "src"))

from release import render_tables_and_figure  # noqa: E402
from biblioguard_v3.io import read_json, read_jsonl_gz, sha256_file  # noqa: E402
from biblioguard_v3.statistics import (  # noqa: E402
    deterministic_top_k,
    holm_adjust,
    paired_bootstrap_ci,
    paired_randomisation_pvalue,
    policy_outcomes,
    risk_coverage_curve,
)


RETRIEVAL_SYSTEMS = ["bm25", "bge", "scincl", "specter2", "lambdarank"]


def assert_close(actual: float | None, expected: float | None, label: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        return
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"{label}: {actual} != {expected}")


def assert_nested(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            actual_keys = sorted(actual) if isinstance(actual, dict) else type(actual).__name__
            raise AssertionError(f"{label} keys/type: {actual_keys!r} != {sorted(expected)!r}")
        for key, value in expected.items():
            assert_nested(actual[key], value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{label} list length/type differs")
        for index, value in enumerate(expected):
            assert_nested(actual[index], value, f"{label}[{index}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, int):
        if not isinstance(actual, int) or isinstance(actual, bool) or actual != expected:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, (float, np.floating)):
        assert_close(actual, float(expected), label)
        return
    if actual != expected:
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def main() -> None:
    published = REVISION / "published"
    release = read_json(published / "release_manifest.json")
    for section in ("copied_files", "generated_files"):
        for item in release[section]:
            path = REPOSITORY / item["path"]
            if sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"Hash mismatch: {path}")

    results_path = published / "results.json"
    per_query_path = published / "locked_per_query.jsonl.gz"
    locked_metrics_path = published / "locked_metrics.jsonl.gz"
    results = read_json(results_path)
    per_query_rows = read_jsonl_gz(per_query_path)
    metric_rows = read_jsonl_gz(locked_metrics_path)

    frozen_manifest_path = REVISION / "frozen" / "decision_manifest.json"
    frozen = read_json(frozen_manifest_path)
    decisions_path = REVISION / "frozen" / frozen["decisions_file"]
    decisions = {str(row["qid"]): row for row in read_jsonl_gz(decisions_path)}
    if sha256_file(decisions_path) != frozen["decisions_sha256"]:
        raise RuntimeError("Public frozen decisions do not match their manifest")
    if frozen.get("test_labels_consumed") is not False:
        raise RuntimeError("Public freeze does not predate locked evaluation")
    if results.get("dataset_summary") != frozen.get("dataset_summary"):
        raise RuntimeError("Published dataset summary differs from the public freeze")

    frozen_scores = frozen.get("score_hashes")
    if not isinstance(frozen_scores, dict):
        raise RuntimeError("Public freeze has no score hash family")
    for name, expected in frozen_scores.items():
        path = published / "scores" / f"{name}.npy"
        if sha256_file(path) != expected:
            raise RuntimeError(f"Published score differs from public freeze: {name}")

    frozen_features = frozen.get("feature_hashes")
    if not isinstance(frozen_features, dict):
        raise RuntimeError("Public freeze has no feature/layout hash family")
    for name, expected in frozen_features.items():
        path = (
            published / "manifests" / "scores" / name
            if name.endswith(".manifest.json")
            else published / "scores" / name
        )
        if sha256_file(path) != expected:
            raise RuntimeError(f"Published feature differs from public freeze: {name}")

    if sha256_file(REVISION / "config" / "models.json") != frozen["models_config_sha256"]:
        raise RuntimeError("Model configuration changed after the public freeze")
    if sha256_file(REVISION / "requirements-lock.txt") != frozen["requirements_lock_sha256"]:
        raise RuntimeError("Requirements lock changed after the public freeze")
    for model in ("bge", "scincl", "specter2"):
        score_manifest = read_json(published / "manifests" / "scores" / f"{model}.manifest.json")
        embedding_manifest = published / "manifests" / "embeddings" / f"{model}.manifest.json"
        if sha256_file(embedding_manifest) != score_manifest["model_manifest_sha256"]:
            raise RuntimeError(f"Published embedding provenance differs for {model}")

    locked_manifest = read_json(
        published / "manifests" / "metrics" / "locked_test.jsonl.gz.manifest.json"
    )
    if locked_manifest.get("frozen_decision_manifest_sha256") != sha256_file(
        frozen_manifest_path
    ):
        raise RuntimeError("Published locked metrics do not descend from the public freeze")
    if locked_manifest.get("score_hashes") != frozen_scores:
        raise RuntimeError("Published locked metrics used a different score family")
    if locked_manifest.get("output_sha256") != sha256_file(locked_metrics_path):
        raise RuntimeError("Published locked metrics differ from their manifest")

    if len(metric_rows) != results["locked_queries"] or len(per_query_rows) != len(metric_rows):
        raise AssertionError("Locked metric/per-query/result row counts differ")
    metric_by_qid = {str(row["qid"]): row for row in metric_rows}
    reported_by_qid = {str(row["qid"]): row for row in per_query_rows}
    if len(metric_by_qid) != len(metric_rows) or len(reported_by_qid) != len(per_query_rows):
        raise AssertionError("Duplicate query identifier in published locked data")
    if set(metric_by_qid) != set(reported_by_qid) or set(metric_by_qid) != set(decisions):
        raise AssertionError("Metric, per-query, and frozen decision query sets differ")

    method_effects: dict[str, list[float]] = {}
    method_active: dict[str, list[bool]] = {}
    method_confidence: dict[str, list[float]] = {}
    retrieval_metrics: dict[str, dict[str, list[float]]] = {}
    action_metrics: dict[str, list[float]] = {}
    independently_rebuilt_rows: list[dict[str, Any]] = []
    for metric_row in metric_rows:
        qid = str(metric_row["qid"])
        metrics = metric_row["metrics"]
        baseline = float(metrics["specter2"]["ndcg_at_10"])
        for system in RETRIEVAL_SYSTEMS:
            for metric_name, value in metrics[system].items():
                retrieval_metrics.setdefault(system, {}).setdefault(metric_name, []).append(
                    float(value)
                )
        for system, values in metrics.items():
            if system.startswith("action_"):
                action_metrics.setdefault(system.removeprefix("action_"), []).append(
                    float(values["ndcg_at_10"])
                )
        rebuilt: dict[str, Any] = {
            "qid": qid,
            "content_audit": decisions[qid]["content_audit"],
            "methods": {},
        }
        for method, decision in decisions[qid]["methods"].items():
            action = str(decision["action"])
            effect = float(metrics[f"action_{action}"]["ndcg_at_10"] - baseline)
            active = bool(decision["active"])
            confidence = float(decision["confidence"])
            method_effects.setdefault(method, []).append(effect)
            method_active.setdefault(method, []).append(active)
            method_confidence.setdefault(method, []).append(confidence)
            rebuilt["methods"][method] = {
                "action": action,
                "effect": effect,
                "active": active,
                "confidence": confidence,
            }
        assert_nested(reported_by_qid[qid], rebuilt, f"per_query.{qid}")
        independently_rebuilt_rows.append(rebuilt)

    specter2_mean = float(np.mean(retrieval_metrics["specter2"]["ndcg_at_10"]))
    protocol = read_json(REVISION / "config" / "protocol.json")
    budgets = [float(value) for value in protocol["evaluation"]["coverage_budgets"]]
    operating: dict[str, Any] = {}
    matched: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    for method in sorted(method_effects):
        effects = np.asarray(method_effects[method], dtype=float)
        active = np.asarray(method_active[method], dtype=bool)
        confidence = np.asarray(method_confidence[method], dtype=float)
        operating[method] = policy_outcomes(effects, active)
        operating[method]["policy_ndcg_at_10"] = (
            specter2_mean + float(operating[method]["overall_mean_gain"])
        )
        operating[method]["active_action_counts"] = {
            action: sum(
                1
                for row in independently_rebuilt_rows
                if row["methods"][method]["active"]
                and row["methods"][method]["action"] == action
            )
            for action in sorted(
                {
                    row["methods"][method]["action"]
                    for row in independently_rebuilt_rows
                    if row["methods"][method]["active"]
                }
            )
        }
        matched[method] = {
            f"{budget:.2f}": policy_outcomes(
                effects, deterministic_top_k(confidence, budget)
            )
            for budget in budgets
        }
        curves[method] = risk_coverage_curve(effects, confidence)

    primary_effects = np.where(
        np.asarray(method_active["biblioguard"], dtype=bool),
        np.asarray(method_effects["biblioguard"], dtype=float),
        0.0,
    )
    seed = int(protocol["seed"])
    bootstrap_replicates = int(protocol["evaluation"]["bootstrap_replicates"])
    randomisation_replicates = int(protocol["evaluation"]["randomisation_replicates"])
    sensitivity_keep = np.asarray(
        [
            not bool(row["content_audit"]["exclude_from_near_duplicate_sensitivity"])
            for row in independently_rebuilt_rows
        ],
        dtype=bool,
    )
    sensitivity_effects = primary_effects[sensitivity_keep]
    comparator_pvalues: dict[str, float] = {}
    for method in sorted(method_effects):
        if method == "biblioguard":
            continue
        difference = primary_effects - np.where(
            np.asarray(method_active[method], dtype=bool),
            np.asarray(method_effects[method], dtype=float),
            0.0,
        )
        comparator_pvalues[method] = paired_randomisation_pvalue(
            difference, randomisation_replicates, seed
        )

    expected_results = {
        "phase": "evaluate",
        "frozen_decisions_sha256": sha256_file(decisions_path),
        "locked_metrics_sha256": sha256_file(locked_metrics_path),
        "locked_queries": len(metric_rows),
        "dataset_summary": frozen["dataset_summary"],
        "retrieval_metrics": {
            system: {metric: float(np.mean(values)) for metric, values in metrics.items()}
            for system, metrics in retrieval_metrics.items()
        },
        "fixed_action_ndcg_at_10": {
            action: float(np.mean(values)) for action, values in action_metrics.items()
        },
        "fixed_action_gain_vs_specter2": {
            action: float(np.mean(values) - specter2_mean)
            for action, values in action_metrics.items()
        },
        "operating_point": operating,
        "matched_coverage": matched,
        "risk_coverage": curves,
        "primary": {
            "estimand": "BiblioGuard policy mean NDCG@10 change versus SPECTER2",
            "mean_effect": float(np.mean(primary_effects)),
            "bootstrap_95_ci": list(
                paired_bootstrap_ci(primary_effects, bootstrap_replicates, seed)
            ),
            "paired_randomisation_p": paired_randomisation_pvalue(
                primary_effects, randomisation_replicates, seed
            ),
        },
        "near_duplicate_sensitivity": {
            "threshold": float(protocol["evaluation"]["near_duplicate_sensitivity_threshold"]),
            "included_queries": int(np.sum(sensitivity_keep)),
            "excluded_queries": int(np.sum(~sensitivity_keep)),
            "mean_effect": float(np.mean(sensitivity_effects)),
            "bootstrap_95_ci": list(
                paired_bootstrap_ci(sensitivity_effects, bootstrap_replicates, seed + 1)
            ),
            "paired_randomisation_p": paired_randomisation_pvalue(
                sensitivity_effects, randomisation_replicates, seed + 1
            ),
        },
        "comparisons_vs_biblioguard_holm_p": holm_adjust(comparator_pvalues),
        "per_query_sha256": sha256_file(per_query_path),
    }
    assert_nested(results, expected_results, "results")

    with tempfile.TemporaryDirectory() as directory:
        regenerated = Path(directory)
        render_tables_and_figure(results, regenerated)
        for name in (
            "results_macros.tex",
            "table_retrievers.tex",
            "table_policy.tex",
            "table_actions.tex",
        ):
            if (regenerated / name).read_bytes() != (REVISION / "paper" / "generated" / name).read_bytes():
                raise RuntimeError(f"Generated LaTeX is not a faithful rendering of results: {name}")

    print(
        f"OK: independently recomputed every results.json section for {len(metric_rows)} "
        f"locked queries and verified {len(frozen_scores)} frozen score arrays; "
        f"results SHA-256 {sha256_file(results_path)}"
    )


if __name__ == "__main__":
    main()

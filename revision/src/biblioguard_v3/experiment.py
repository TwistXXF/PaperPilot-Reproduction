from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from .actions import action_scores
from .io import (
    read_json,
    read_jsonl_gz,
    sha256_file,
    write_json,
    write_jsonl_gz,
)
from .metrics import evaluate_ranking
from .policy import LocalPolicy, calibrate_threshold
from .splits import normalise_title
from .statistics import (
    deterministic_top_k,
    holm_adjust,
    paired_bootstrap_ci,
    paired_randomisation_pvalue,
    policy_outcomes,
    risk_coverage_curve,
)


RETRIEVAL_SYSTEMS = ["bm25", "bge", "scincl", "specter2", "lambdarank"]


def _layout(scores_directory: Path) -> tuple[list[str], list[str], np.ndarray]:
    qids = [str(value) for value in read_json(scores_directory / "qids.json")]
    candidate_ids = [str(value) for value in read_json(scores_directory / "candidate_doc_ids.json")]
    offsets = np.load(scores_directory / "offsets.npy", allow_pickle=False)
    if len(offsets) != len(qids) + 1 or int(offsets[-1]) != len(candidate_ids):
        raise ValueError("Invalid score layout")
    return qids, candidate_ids, offsets


def _system_score_paths(scores_directory: Path, action_names: list[str]) -> dict[str, Path]:
    paths = {name: scores_directory / f"{name}.npy" for name in RETRIEVAL_SYSTEMS}
    paths.update(
        {f"action_{name}": scores_directory / f"action_{name}.npy" for name in action_names}
    )
    return paths


def _verify_frozen_scores(
    scores_directory: Path, action_names: list[str], frozen_manifest: dict[str, Any]
) -> dict[str, str]:
    paths = _system_score_paths(scores_directory, action_names)
    expected = frozen_manifest.get("score_hashes")
    if not isinstance(expected, dict) or set(expected) != set(paths):
        raise RuntimeError("Frozen manifest does not cover the complete score family")
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != expected:
        changed = sorted(name for name in paths if actual.get(name) != expected.get(name))
        raise RuntimeError(f"Score files changed after decision freeze: {changed}")
    return actual


def build_metadata_actions(
    prepared_directory: Path,
    scores_directory: Path,
    metadata_path: Path,
    protocol: dict[str, Any],
    protocol_path: Path,
) -> dict[str, Any]:
    qids, candidate_doc_ids, offsets = _layout(scores_directory)
    queries = read_jsonl_gz(prepared_directory / "queries.jsonl.gz")
    documents = {
        str(row["doc_id"]): row for row in read_jsonl_gz(prepared_directory / "documents.jsonl.gz")
    }
    if [str(row["qid"]) for row in queries] != qids:
        raise ValueError("Prepared query order does not match score layout")
    metadata_rows = read_jsonl_gz(metadata_path)
    metadata = {int(row["requested_corpus_id"]): row["record"] for row in metadata_rows}
    semantic = np.load(scores_directory / "specter2.npy", mmap_mode="r")
    if len(semantic) != len(candidate_doc_ids):
        raise ValueError("SPECTER2 score count does not match layout")
    citation_flat = np.full(len(candidate_doc_ids), np.nan, dtype=np.float64)
    year_flat = np.full(len(candidate_doc_ids), np.nan, dtype=np.float64)
    action_flat = {
        action["name"]: np.empty(len(candidate_doc_ids), dtype=np.float32)
        for action in protocol["actions"]
    }
    title_compared = 0
    title_exact = 0
    record_missing = 0
    corpus_id_missing = 0
    corpus_id_mismatch = 0
    for query_number, query in enumerate(queries):
        start, stop = int(offsets[query_number]), int(offsets[query_number + 1])
        corpus_ids = query["candidate_corpus_ids"]
        if len(corpus_ids) != stop - start:
            raise ValueError(f"Candidate CorpusId count mismatch for {query['qid']}")
        citations: list[int | None] = []
        years: list[int | None] = []
        for relative, corpus_id in enumerate(corpus_ids):
            if corpus_id is None:
                corpus_id_missing += 1
                citations.append(None)
                years.append(None)
                continue
            record = metadata.get(int(corpus_id))
            if record is None:
                record_missing += 1
                citations.append(None)
                years.append(None)
                continue
            returned_corpus_id = record.get("corpusId")
            if returned_corpus_id is not None and int(returned_corpus_id) != int(corpus_id):
                corpus_id_mismatch += 1
                raise ValueError(f"Semantic Scholar CorpusId mismatch for requested {corpus_id}")
            citation = record.get("citationCount")
            year = record.get("year")
            citations.append(None if citation is None else int(citation))
            years.append(None if year is None else int(year))
            candidate_id = candidate_doc_ids[start + relative]
            source_title = documents[candidate_id]["title"]
            returned_title = str(record.get("title") or "")
            if source_title and returned_title:
                title_compared += 1
                if normalise_title(source_title) == normalise_title(returned_title):
                    title_exact += 1
        values = action_scores(
            semantic_scores=np.asarray(semantic[start:stop], dtype=float),
            citation_counts=citations,
            years=years,
            corpus_ids=candidate_doc_ids[start:stop],
            actions=protocol["actions"],
        )
        for relative, value in enumerate(citations):
            if value is not None:
                citation_flat[start + relative] = value
        for relative, value in enumerate(years):
            if value is not None:
                year_flat[start + relative] = value
        for name, scores in values.items():
            action_flat[name][start:stop] = scores
    citation_path = scores_directory / "citation_count.npy"
    year_path = scores_directory / "year.npy"
    np.save(citation_path, citation_flat, allow_pickle=False)
    np.save(year_path, year_flat, allow_pickle=False)
    output_hashes: dict[str, str] = {
        "citation_count": sha256_file(citation_path),
        "year": sha256_file(year_path),
    }
    for name, values in action_flat.items():
        path = scores_directory / f"action_{name}.npy"
        np.save(path, values, allow_pickle=False)
        output_hashes[f"action_{name}"] = sha256_file(path)
    report = {
        "protocol_sha256": sha256_file(protocol_path),
        "metadata_sha256": sha256_file(metadata_path),
        "specter2_scores_sha256": sha256_file(scores_directory / "specter2.npy"),
        "candidate_instances": len(candidate_doc_ids),
        "candidate_instances_missing_corpus_id": corpus_id_missing,
        "candidate_instances_missing_metadata_record": record_missing,
        "corpus_id_mismatches": corpus_id_mismatch,
        "title_pairs_compared": title_compared,
        "normalised_title_exact_matches": title_exact,
        "normalised_title_exact_rate": title_exact / title_compared if title_compared else 0.0,
        "citation_missing": int(np.sum(np.isnan(citation_flat))),
        "year_missing": int(np.sum(np.isnan(year_flat))),
        "outputs": output_hashes,
    }
    write_json(scores_directory / "actions.manifest.json", report)
    return report


def evaluate_score_files(
    labels_path: Path,
    scores_directory: Path,
    output_path: Path,
    action_names: list[str],
    frozen_manifest_path: Path | None = None,
    refuse_overwrite: bool = False,
) -> dict[str, Any]:
    if refuse_overwrite and output_path.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot metrics: {output_path}")
    qids, candidate_ids, offsets = _layout(scores_directory)
    qid_to_index = {qid: index for index, qid in enumerate(qids)}
    system_paths = _system_score_paths(scores_directory, action_names)
    missing = [path for path in system_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing score files: {missing}")
    systems = {name: np.load(path, mmap_mode="r") for name, path in system_paths.items()}
    rows: list[dict[str, Any]] = []
    zero_ideal_queries = 0
    for label_row in read_jsonl_gz(labels_path):
        qid = str(label_row["qid"])
        query_number = qid_to_index[qid]
        start, stop = int(offsets[query_number]), int(offsets[query_number + 1])
        labels = np.asarray(label_row["labels"], dtype=float)
        if len(labels) != stop - start:
            raise ValueError(f"Label count mismatch for {qid}")
        if not np.any(labels > 0):
            zero_ideal_queries += 1
        ids = candidate_ids[start:stop]
        metrics = {
            name: evaluate_ranking(labels, np.asarray(values[start:stop], dtype=float), ids)
            for name, values in systems.items()
        }
        rows.append({"qid": qid, "split": label_row["split"], "metrics": metrics})
    write_jsonl_gz(output_path, rows)
    report: dict[str, Any] = {
        "labels_sha256": sha256_file(labels_path),
        "queries": len(rows),
        "systems": sorted(systems),
        "zero_ideal_queries": zero_ideal_queries,
        "output_sha256": sha256_file(output_path),
        "score_hashes": {name: sha256_file(path) for name, path in system_paths.items()},
    }
    if frozen_manifest_path is not None:
        frozen_manifest = read_json(frozen_manifest_path)
        actual = _verify_frozen_scores(scores_directory, action_names, frozen_manifest)
        if actual != report["score_hashes"]:
            raise AssertionError("Score hashes changed during locked evaluation")
        report["frozen_decision_manifest_sha256"] = sha256_file(frozen_manifest_path)
        report["frozen_decisions_sha256"] = frozen_manifest["decisions_sha256"]
    write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), report)
    return report


def query_features(prepared_directory: Path, scores_directory: Path) -> tuple[list[str], np.ndarray]:
    qids, _, offsets = _layout(scores_directory)
    queries = read_jsonl_gz(prepared_directory / "queries.jsonl.gz")
    arrays: dict[str, np.ndarray] = {}
    for name in ["bm25", "bge", "scincl", "specter2"]:
        arrays[name] = np.load(scores_directory / f"{name}.npy", mmap_mode="r")
    citation = np.load(scores_directory / "citation_count.npy", mmap_mode="r")
    year = np.load(scores_directory / "year.npy", mmap_mode="r")
    rows: list[list[float]] = []
    for index, query in enumerate(queries):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        features = [
            float(len(str(query["title"]).split())),
            float(len(str(query["abstract"]).split())),
            float(stop - start),
        ]
        for name in ["bm25", "bge", "scincl", "specter2"]:
            values = np.asarray(arrays[name][start:stop], dtype=float)
            ordered = np.sort(values)[::-1]
            features.extend(
                [
                    float(np.mean(values)),
                    float(np.std(values)),
                    float(ordered[0]),
                    float(ordered[0] - ordered[1]) if len(ordered) > 1 else 0.0,
                    float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                ]
            )
        for values, transform in [
            (np.asarray(citation[start:stop], dtype=float), np.log1p),
            (np.asarray(year[start:stop], dtype=float), lambda item: item),
        ]:
            missing = np.isnan(values)
            present = values[~missing]
            transformed = transform(present) if len(present) else np.asarray([0.0])
            features.extend(
                [
                    float(np.mean(missing)),
                    float(np.mean(transformed)),
                    float(np.std(transformed)),
                    float(np.max(transformed)),
                ]
            )
        rows.append(features)
    if [str(query["qid"]) for query in queries] != qids:
        raise ValueError("Feature query order does not match layout")
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Query features contain non-finite values")
    return qids, matrix


def _within_query_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    present = values[~np.isnan(values)]
    if len(present) == 0:
        return np.zeros(len(values), dtype=np.float32)
    median = float(np.median(present))
    filled = np.where(np.isnan(values), median, values)
    low = float(np.min(filled))
    high = float(np.max(filled))
    if high <= low:
        return np.zeros(len(values), dtype=np.float32)
    return np.asarray((filled - low) / (high - low), dtype=np.float32)


def candidate_features(scores_directory: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    qids, _, offsets = _layout(scores_directory)
    retrieval = {
        name: np.load(scores_directory / f"{name}.npy", mmap_mode="r")
        for name in ["bm25", "bge", "scincl", "specter2"]
    }
    citation = np.load(scores_directory / "citation_count.npy", mmap_mode="r")
    year = np.load(scores_directory / "year.npy", mmap_mode="r")
    features = np.empty((int(offsets[-1]), 6), dtype=np.float32)
    for query_number in range(len(qids)):
        start, stop = int(offsets[query_number]), int(offsets[query_number + 1])
        columns = [
            _within_query_minmax(np.asarray(retrieval[name][start:stop], dtype=float))
            for name in ["bm25", "bge", "scincl", "specter2"]
        ]
        citation_values = np.asarray(citation[start:stop], dtype=float)
        citation_values = np.where(np.isnan(citation_values), np.nan, np.log1p(citation_values))
        columns.append(_within_query_minmax(citation_values))
        columns.append(_within_query_minmax(np.asarray(year[start:stop], dtype=float)))
        features[start:stop] = np.column_stack(columns)
    if not np.isfinite(features).all():
        raise ValueError("Candidate features contain non-finite values")
    return qids, offsets, features


def fit_lambdarank(
    labels_path: Path,
    scores_directory: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    import lightgbm as lgb

    protocol = read_json(protocol_path)
    qids, offsets, features = candidate_features(scores_directory)
    qid_to_index = {qid: index for index, qid in enumerate(qids)}
    labels = read_jsonl_gz(labels_path)
    if not labels or {row["split"] for row in labels} != {"train"}:
        raise ValueError("LambdaRank fit accepts training labels only")

    def materialise(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
        feature_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        groups: list[int] = []
        for row in rows:
            index = qid_to_index[str(row["qid"])]
            start, stop = int(offsets[index]), int(offsets[index + 1])
            relevance = np.asarray(row["labels"], dtype=np.int32)
            if len(relevance) != stop - start:
                raise ValueError(f"LambdaRank label count mismatch for {row['qid']}")
            feature_parts.append(features[start:stop])
            label_parts.append(relevance)
            groups.append(stop - start)
        return np.vstack(feature_parts), np.concatenate(label_parts), groups

    train_x, train_y, train_group = materialise(labels)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "label_gain": [0, 1, 2],
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
        "random_state": int(protocol["seed"]),
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    ranker = lgb.LGBMRanker(**params)
    ranker.fit(train_x, train_y, group=train_group)
    predictions = np.asarray(ranker.predict(features), dtype=np.float32)
    output_path = scores_directory / "lambdarank.npy"
    np.save(output_path, predictions, allow_pickle=False)
    model_path = scores_directory / "lambdarank_model.txt"
    ranker.booster_.save_model(str(model_path))
    report = {
        "implementation": "lightgbm.LGBMRanker",
        "lightgbm_version": lgb.__version__,
        "feature_names": ["bm25", "bge", "scincl", "specter2", "log_citation", "year"],
        "parameters": params,
        "iterations": int(params["n_estimators"]),
        "training_queries": len(labels),
        "calibration_labels_consumed": False,
        "training_labels_sha256": sha256_file(labels_path),
        "actions_manifest_sha256": sha256_file(scores_directory / "actions.manifest.json"),
        "scores_sha256": sha256_file(output_path),
        "model_sha256": sha256_file(model_path),
    }
    write_json(scores_directory / "lambdarank.manifest.json", report)
    return report


def _selected_effects(effects: np.ndarray, action_index: np.ndarray) -> np.ndarray:
    return effects[np.arange(len(effects)), action_index]


def _safe_calibration(
    scores: np.ndarray,
    effects: np.ndarray,
    policy_config: dict[str, Any],
) -> dict[str, Any]:
    result = calibrate_threshold(
        scores,
        effects,
        float(policy_config["calibration_harm_upper_bound"]),
        float(policy_config["calibration_confidence"]),
        int(policy_config["minimum_active_calibration_queries"]),
    )
    if not bool(result["eligible"]):
        result["threshold"] = None
    return result


def freeze_decisions(
    prepared_directory: Path,
    scores_directory: Path,
    training_metrics_path: Path,
    calibration_metrics_path: Path,
    content_audit_path: Path,
    protocol_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    manifest_path = output_directory / "decision_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen decisions: {manifest_path}")
    protocol = read_json(protocol_path)
    action_names = [action["name"] for action in protocol["actions"]]
    queries = read_jsonl_gz(prepared_directory / "queries.jsonl.gz")
    query_by_id = {str(query["qid"]): query for query in queries}
    train_rows = read_jsonl_gz(training_metrics_path)
    calibration_rows = read_jsonl_gz(calibration_metrics_path)
    if not train_rows or {row["split"] for row in train_rows} != {"train"}:
        raise ValueError("Training metrics file must contain training rows only")
    if not calibration_rows or {row["split"] for row in calibration_rows} != {"calibration"}:
        raise ValueError("Calibration metrics file must contain calibration rows only")
    content_audit = read_json(content_audit_path)
    locked_similarity = content_audit["locked_max_similarity_to_development"]
    prepare_manifest_path = prepared_directory / "prepare_manifest.json"
    prepare_manifest = read_json(prepare_manifest_path)
    split_counts = prepare_manifest["split_audit"]["split_counts"]
    if int(split_counts["train"]) != len(train_rows):
        raise ValueError("Prepared/train metric query counts differ")
    if int(split_counts["calibration"]) != len(calibration_rows):
        raise ValueError("Prepared/calibration metric query counts differ")

    def effect_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            [
                [
                    row["metrics"][f"action_{name}"]["ndcg_at_10"]
                    - row["metrics"]["specter2"]["ndcg_at_10"]
                    for name in action_names
                ]
                for row in rows
            ],
            dtype=float,
        )

    train_effects = effect_matrix(train_rows)
    calibration_effects = effect_matrix(calibration_rows)
    train_titles = [query_by_id[str(row["qid"])]["title"] for row in train_rows]
    calibration_titles = [query_by_id[str(row["qid"])]["title"] for row in calibration_rows]
    locked_queries = [query for query in queries if query["split"] == "locked_test"]
    locked_titles = [query["title"] for query in locked_queries]

    local_policy = LocalPolicy(float(protocol["policy"]["lcb_z"])).fit(train_titles, train_effects)
    local_calibration = local_policy.predict(calibration_titles)
    local_locked = local_policy.predict(locked_titles)
    local_calibration_effect = _selected_effects(calibration_effects, local_calibration.action_index)
    local_rule = _safe_calibration(
        local_calibration.gate_score, local_calibration_effect, protocol["policy"]
    )

    all_qids, features = query_features(prepared_directory, scores_directory)
    feature_index = {qid: index for index, qid in enumerate(all_qids)}
    train_features = features[[feature_index[str(row["qid"])] for row in train_rows]]
    calibration_features = features[[feature_index[str(row["qid"])] for row in calibration_rows]]
    locked_features = features[[feature_index[str(query["qid"])] for query in locked_queries]]
    base_regressor = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, l2_regularization=1.0, random_state=int(protocol["seed"])
    )
    multi = MultiOutputRegressor(base_regressor).fit(train_features, train_effects)
    hgb_calibration_matrix = np.asarray(multi.predict(calibration_features), dtype=float)
    hgb_locked_matrix = np.asarray(multi.predict(locked_features), dtype=float)
    hgb_calibration_action = np.argmax(hgb_calibration_matrix, axis=1)
    hgb_locked_action = np.argmax(hgb_locked_matrix, axis=1)
    hgb_calibration_score = np.max(hgb_calibration_matrix, axis=1)
    hgb_locked_score = np.max(hgb_locked_matrix, axis=1)
    hgb_calibration_effect = _selected_effects(calibration_effects, hgb_calibration_action)
    hgb_rule = _safe_calibration(hgb_calibration_score, hgb_calibration_effect, protocol["policy"])

    action_train_means = np.mean(train_effects, axis=0)
    global_action = int(np.argmax(action_train_means))
    global_train_effect = train_effects[:, global_action]
    global_regressor = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, l2_regularization=1.0, random_state=int(protocol["seed"])
    ).fit(train_features, global_train_effect)
    global_calibration_score = np.asarray(global_regressor.predict(calibration_features), dtype=float)
    global_locked_score = np.asarray(global_regressor.predict(locked_features), dtype=float)
    global_rule = _safe_calibration(
        global_calibration_score, calibration_effects[:, global_action], protocol["policy"]
    )

    def active(score: float, rule: dict[str, Any]) -> bool:
        return bool(rule["eligible"] and score >= float(rule["threshold"]))

    decision_rows: list[dict[str, Any]] = []
    for index, query in enumerate(locked_queries):
        decision_rows.append(
            {
                "qid": str(query["qid"]),
                "content_audit": {
                    "max_similarity_to_development": float(
                        locked_similarity[str(query["qid"])]["similarity"]
                    ),
                    "nearest_development_qid": locked_similarity[str(query["qid"])]["qid"],
                    "exclude_from_near_duplicate_sensitivity": bool(
                        locked_similarity[str(query["qid"])]["similarity"]
                        >= float(protocol["evaluation"]["near_duplicate_sensitivity_threshold"])
                    ),
                },
                "methods": {
                    "biblioguard": {
                        "action": action_names[int(local_locked.action_index[index])],
                        "confidence": float(local_locked.gate_score[index]),
                        "estimated_gain": float(local_locked.estimated_gain[index]),
                        "standard_error": float(local_locked.standard_error[index]),
                        "active": active(float(local_locked.gate_score[index]), local_rule),
                    },
                    "knn_mean": {
                        "action": action_names[int(local_locked.action_index[index])],
                        "confidence": float(local_locked.estimated_gain[index]),
                        "active": True,
                    },
                    "hgb_gain": {
                        "action": action_names[int(hgb_locked_action[index])],
                        "confidence": float(hgb_locked_score[index]),
                        "active": active(float(hgb_locked_score[index]), hgb_rule),
                    },
                    "global_gain_gate": {
                        "action": action_names[global_action],
                        "confidence": float(global_locked_score[index]),
                        "active": active(float(global_locked_score[index]), global_rule),
                    },
                    "global_all": {
                        "action": action_names[global_action],
                        "confidence": float(global_locked_score[index]),
                        "active": True,
                    },
                },
            }
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    decisions_path = output_directory / "locked_decisions.jsonl.gz"
    write_jsonl_gz(decisions_path, decision_rows)
    calibration_path = output_directory / "calibration.json"
    calibration_report = {
        "biblioguard": local_rule,
        "hgb_gain": hgb_rule,
        "global_gain_gate": global_rule,
        "global_action": action_names[global_action],
        "global_action_train_mean_effect": float(action_train_means[global_action]),
        "action_train_mean_effects": {
            name: float(value) for name, value in zip(action_names, action_train_means)
        },
        "train_queries": len(train_rows),
        "calibration_queries": len(calibration_rows),
        "locked_queries": len(locked_queries),
    }
    write_json(calibration_path, calibration_report)
    lambdarank_manifest_path = scores_directory / "lambdarank.manifest.json"
    if not lambdarank_manifest_path.exists():
        raise FileNotFoundError("LambdaRank must be fitted before freezing decisions")
    score_paths = _system_score_paths(scores_directory, action_names)
    score_hashes = {name: sha256_file(path) for name, path in score_paths.items()}
    feature_files = [
        "qids.json",
        "candidate_doc_ids.json",
        "offsets.npy",
        "citation_count.npy",
        "year.npy",
        "layout.manifest.json",
        "bm25.manifest.json",
        "bge.manifest.json",
        "scincl.manifest.json",
        "specter2.manifest.json",
        "actions.manifest.json",
        "lambdarank.manifest.json",
    ]
    models_config_path = protocol_path.parent / "models.json"
    requirements_lock_path = protocol_path.parent.parent / "requirements-lock.txt"
    manifest = {
        "phase": "freeze",
        "test_labels_consumed": False,
        "test_labels_materialised_for_evaluation": False,
        "raw_label_container_may_contain_all_splits": True,
        "protocol_sha256": sha256_file(protocol_path),
        "models_config_sha256": sha256_file(models_config_path),
        "requirements_lock_sha256": sha256_file(requirements_lock_path),
        "prepare_manifest_sha256": sha256_file(prepare_manifest_path),
        "training_metrics_sha256": sha256_file(training_metrics_path),
        "calibration_metrics_sha256": sha256_file(calibration_metrics_path),
        "content_audit_sha256": sha256_file(content_audit_path),
        "queries_sha256": sha256_file(prepared_directory / "queries.jsonl.gz"),
        "actions_manifest_sha256": sha256_file(scores_directory / "actions.manifest.json"),
        "lambdarank_manifest_sha256": sha256_file(lambdarank_manifest_path),
        "score_hashes": score_hashes,
        "feature_hashes": {
            name: sha256_file(scores_directory / name) for name in feature_files
        },
        "decisions_file": decisions_path.name,
        "decisions_sha256": sha256_file(decisions_path),
        "calibration_file": calibration_path.name,
        "calibration_sha256": sha256_file(calibration_path),
        "locked_queries": len(decision_rows),
        "dataset_summary": {
            "queries": int(prepare_manifest["queries"]),
            "train_queries": int(split_counts["train"]),
            "calibration_queries": int(split_counts["calibration"]),
            "locked_queries": int(split_counts["locked_test"]),
            "candidate_pairs": int(prepare_manifest["candidate_instances"]),
            "unique_documents": int(prepare_manifest["documents"]),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def evaluate_frozen_decisions(
    frozen_manifest_path: Path,
    locked_metrics_path: Path,
    protocol_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    results_path = output_directory / "results.json"
    if results_path.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot locked results: {results_path}")
    protocol = read_json(protocol_path)
    manifest = read_json(frozen_manifest_path)
    if manifest.get("test_labels_consumed") is not False:
        raise RuntimeError("Invalid frozen decision manifest")
    if manifest["protocol_sha256"] != sha256_file(protocol_path):
        raise RuntimeError("Protocol changed after decision freeze")
    decisions_path = frozen_manifest_path.parent / manifest["decisions_file"]
    if sha256_file(decisions_path) != manifest["decisions_sha256"]:
        raise RuntimeError("Frozen decisions changed after the freeze phase")
    locked_metrics_manifest_path = locked_metrics_path.with_suffix(
        locked_metrics_path.suffix + ".manifest.json"
    )
    locked_metrics_manifest = read_json(locked_metrics_manifest_path)
    if locked_metrics_manifest.get("frozen_decision_manifest_sha256") != sha256_file(
        frozen_manifest_path
    ):
        raise RuntimeError("Locked metrics were not generated from this frozen manifest")
    if locked_metrics_manifest.get("score_hashes") != manifest.get("score_hashes"):
        raise RuntimeError("Locked metric score hashes differ from the frozen score family")
    if locked_metrics_manifest.get("output_sha256") != sha256_file(locked_metrics_path):
        raise RuntimeError("Locked metrics changed after generation")
    decisions = {str(row["qid"]): row for row in read_jsonl_gz(decisions_path)}
    metric_rows = read_jsonl_gz(locked_metrics_path)
    if set(decisions) != {str(row["qid"]) for row in metric_rows}:
        raise ValueError("Frozen decision and locked metric query sets differ")
    method_effects: dict[str, list[float]] = {}
    method_active: dict[str, list[bool]] = {}
    method_confidence: dict[str, list[float]] = {}
    retrieval_metrics: dict[str, dict[str, list[float]]] = {}
    action_metrics: dict[str, list[float]] = {}
    retrieval_systems = RETRIEVAL_SYSTEMS
    per_query: list[dict[str, Any]] = []
    for metric_row in metric_rows:
        qid = str(metric_row["qid"])
        metrics = metric_row["metrics"]
        baseline = float(metrics["specter2"]["ndcg_at_10"])
        for system in retrieval_systems:
            for metric_name, value in metrics[system].items():
                retrieval_metrics.setdefault(system, {}).setdefault(metric_name, []).append(float(value))
        for system, values in metrics.items():
            if system.startswith("action_"):
                action_metrics.setdefault(system.removeprefix("action_"), []).append(
                    float(values["ndcg_at_10"])
                )
        query_result: dict[str, Any] = {
            "qid": qid,
            "content_audit": decisions[qid]["content_audit"],
            "methods": {},
        }
        for method, decision in decisions[qid]["methods"].items():
            action = str(decision["action"])
            effect = float(metrics[f"action_{action}"]["ndcg_at_10"] - baseline)
            method_effects.setdefault(method, []).append(effect)
            method_active.setdefault(method, []).append(bool(decision["active"]))
            method_confidence.setdefault(method, []).append(float(decision["confidence"]))
            query_result["methods"][method] = {
                "action": action,
                "effect": effect,
                "active": bool(decision["active"]),
                "confidence": float(decision["confidence"]),
            }
        per_query.append(query_result)
    operating: dict[str, Any] = {}
    matched: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    specter2_mean = float(np.mean(retrieval_metrics["specter2"]["ndcg_at_10"]))
    budgets = [float(value) for value in protocol["evaluation"]["coverage_budgets"]]
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
                for row in per_query
                if row["methods"][method]["active"] and row["methods"][method]["action"] == action
            )
            for action in sorted(
                {
                    row["methods"][method]["action"]
                    for row in per_query
                    if row["methods"][method]["active"]
                }
            )
        }
        matched[method] = {
            f"{budget:.2f}": policy_outcomes(effects, deterministic_top_k(confidence, budget))
            for budget in budgets
        }
        curves[method] = risk_coverage_curve(effects, confidence)
    primary_effects = np.where(
        np.asarray(method_active["biblioguard"], dtype=bool),
        np.asarray(method_effects["biblioguard"], dtype=float),
        0.0,
    )
    seed = int(protocol["seed"])
    replicates = int(protocol["evaluation"]["bootstrap_replicates"])
    randomisation_replicates = int(protocol["evaluation"]["randomisation_replicates"])
    primary_ci = paired_bootstrap_ci(primary_effects, replicates, seed)
    primary_p = paired_randomisation_pvalue(primary_effects, randomisation_replicates, seed)
    sensitivity_keep = np.asarray(
        [
            not bool(row["content_audit"]["exclude_from_near_duplicate_sensitivity"])
            for row in per_query
        ],
        dtype=bool,
    )
    sensitivity_effects = primary_effects[sensitivity_keep]
    comparator_pvalues = {}
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
    report = {
        "phase": "evaluate",
        "frozen_decisions_sha256": sha256_file(decisions_path),
        "locked_metrics_sha256": sha256_file(locked_metrics_path),
        "locked_queries": len(metric_rows),
        "dataset_summary": manifest["dataset_summary"],
        "retrieval_metrics": {
            system: {metric: float(np.mean(values)) for metric, values in metrics.items()}
            for system, metrics in retrieval_metrics.items()
        },
        "fixed_action_ndcg_at_10": {
            action: float(np.mean(values)) for action, values in action_metrics.items()
        },
        "fixed_action_gain_vs_specter2": {
            action: float(np.mean(values) - np.mean(retrieval_metrics["specter2"]["ndcg_at_10"]))
            for action, values in action_metrics.items()
        },
        "operating_point": operating,
        "matched_coverage": matched,
        "risk_coverage": curves,
        "primary": {
            "estimand": "BiblioGuard policy mean NDCG@10 change versus SPECTER2",
            "mean_effect": float(np.mean(primary_effects)),
            "bootstrap_95_ci": list(primary_ci),
            "paired_randomisation_p": primary_p,
        },
        "near_duplicate_sensitivity": {
            "threshold": float(protocol["evaluation"]["near_duplicate_sensitivity_threshold"]),
            "included_queries": int(np.sum(sensitivity_keep)),
            "excluded_queries": int(np.sum(~sensitivity_keep)),
            "mean_effect": float(np.mean(sensitivity_effects)),
            "bootstrap_95_ci": list(
                paired_bootstrap_ci(sensitivity_effects, replicates, seed + 1)
            ),
            "paired_randomisation_p": paired_randomisation_pvalue(
                sensitivity_effects, randomisation_replicates, seed + 1
            ),
        },
        "comparisons_vs_biblioguard_holm_p": holm_adjust(comparator_pvalues),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    per_query_path = output_directory / "locked_per_query.jsonl.gz"
    write_jsonl_gz(per_query_path, per_query)
    report["per_query_sha256"] = sha256_file(per_query_path)
    write_json(results_path, report)
    return report

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REVISION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION / "src"))

from biblioguard_v3.actions import action_scores, minmax, reciprocal_rank_scores  # noqa: E402
from biblioguard_v3 import data as data_module  # noqa: E402
from biblioguard_v3.data import extract_labels  # noqa: E402
from biblioguard_v3.experiment import _verify_frozen_scores, freeze_decisions  # noqa: E402
from biblioguard_v3.io import (  # noqa: E402
    read_json,
    read_jsonl_gz,
    sha256_file,
    write_json,
    write_jsonl_gz,
)
from biblioguard_v3.metrics import evaluate_ranking, stable_order  # noqa: E402
from biblioguard_v3.policy import (  # noqa: E402
    LocalPolicy,
    calibrate_threshold,
    clopper_pearson_upper,
)
from biblioguard_v3.protocol import load_protocol  # noqa: E402
from biblioguard_v3.retrieval import document_text  # noqa: E402
from biblioguard_v3.splits import (  # noqa: E402
    assign_split,
    audit_splits,
    normalise_title,
    split_bucket,
)
from biblioguard_v3.statistics import (  # noqa: E402
    deterministic_top_k,
    holm_adjust,
    paired_bootstrap_ci,
    paired_randomisation_pvalue,
    policy_outcomes,
    risk_coverage_curve,
)


class ProtocolTests(unittest.TestCase):
    def test_protocol_is_complete_and_hashable(self) -> None:
        path = REVISION / "config" / "protocol.json"
        protocol = load_protocol(path)
        self.assertEqual(protocol["protocol_version"], "3.1.0")
        self.assertEqual(len(sha256_file(path)), 64)

    def test_json_write_is_stable_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            write_json(path, {"z": 1, "a": "测试"})
            self.assertEqual(read_json(path), {"a": "测试", "z": 1})
            self.assertTrue(path.read_bytes().endswith(b"\n"))


class SplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_protocol(REVISION / "config" / "protocol.json")["dataset"]

    def test_normalisation(self) -> None:
        self.assertEqual(normalise_title("  A—B:  C! "), "a b c")
        self.assertEqual(normalise_title("ＡＢＣ"), "abc")

    def test_split_is_deterministic(self) -> None:
        group = "same title"
        self.assertEqual(split_bucket(group, "relish-v1"), split_bucket(group, "relish-v1"))
        query = {"title": "Same title", "corpus_id": 1}
        self.assertEqual(assign_split(query, self.config), assign_split(query, self.config))

    def test_duplicate_titles_cannot_cross_splits(self) -> None:
        rows = [
            {"query": {"title": "Same Title", "corpus_id": 1}},
            {"query": {"title": "same-title", "corpus_id": 2}},
        ]
        audit = audit_splits(rows, self.config)
        self.assertEqual(audit["duplicate_query_titles"], 1)
        self.assertEqual(audit["cross_split_groups"], 0)


class MetricTests(unittest.TestCase):
    def test_stable_tie_break(self) -> None:
        order = stable_order([1.0, 1.0, 0.0], ["b", "a", "c"])
        np.testing.assert_array_equal(order, [1, 0, 2])

    def test_document_text_respects_model_separator(self) -> None:
        document = {"title": "A title", "abstract": "An abstract"}
        self.assertEqual(document_text(document), "A title An abstract")
        self.assertEqual(document_text(document, separator="[SEP]"), "A title[SEP]An abstract")

    def test_ideal_ranking_scores_one(self) -> None:
        metrics = evaluate_ranking([3, 2, 0], [3.0, 2.0, 0.0], [1, 2, 3])
        self.assertAlmostEqual(metrics["ndcg_at_10"], 1.0)
        self.assertAlmostEqual(metrics["precision_at_10"], 0.2)
        self.assertAlmostEqual(metrics["recall_at_50"], 1.0)
        self.assertAlmostEqual(metrics["map_cut_10"], 1.0)

    def test_map_cut_uses_all_relevant_documents_in_denominator(self) -> None:
        relevance = [1] * 20
        scores = list(reversed(range(20)))
        metrics = evaluate_ranking(relevance, scores, list(range(20)))
        self.assertAlmostEqual(metrics["map_cut_10"], 0.5)

    def test_no_relevant_documents_is_zero(self) -> None:
        metrics = evaluate_ranking([0, 0], [0.2, 0.1], [1, 2])
        self.assertTrue(all(value == 0.0 for value in metrics.values()))


class ActionTests(unittest.TestCase):
    def test_minmax_imputes_observed_median(self) -> None:
        scaled = minmax([0.0, None, 10.0])
        np.testing.assert_allclose(scaled, [0.0, 0.5, 1.0])

    def test_constant_rrf_metadata_preserves_semantic_order(self) -> None:
        semantic = np.asarray([0.2, 0.9, 0.5])
        fused = reciprocal_rank_scores(semantic, np.zeros(3), ["a", "b", "c"], 60)
        self.assertEqual(np.argsort(-fused, kind="stable").tolist(), [1, 2, 0])

    def test_action_family_is_aligned_and_finite(self) -> None:
        protocol = load_protocol(REVISION / "config" / "protocol.json")
        scores = action_scores(
            semantic_scores=[0.2, 0.1, 0.0],
            citation_counts=[0, None, 100],
            years=[2020, 2022, None],
            corpus_ids=["a", "b", "c"],
            actions=protocol["actions"],
        )
        self.assertEqual(set(scores), {item["name"] for item in protocol["actions"]})
        self.assertTrue(all(value.shape == (3,) for value in scores.values()))
        self.assertTrue(all(np.isfinite(value).all() for value in scores.values()))


class PhaseIsolationTests(unittest.TestCase):
    def test_locked_labels_require_hashed_frozen_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            prepared.mkdir()
            write_jsonl_gz(
                prepared / "queries.jsonl.gz",
                [
                    {
                        "qid": "q-train",
                        "split": "train",
                        "candidate_doc_ids": ["a", "b"],
                    },
                    {
                        "qid": "q-test",
                        "split": "locked_test",
                        "candidate_doc_ids": ["c", "d"],
                    },
                ],
            )
            fake_parquet = root / "labels.parquet"
            fake_parquet.write_bytes(b"fixture")
            rows = [
                {"query_id": "q-train", "cand_id": "a", "score": 2},
                {"query_id": "q-train", "cand_id": "b", "score": 0},
                {"query_id": "q-test", "cand_id": "c", "score": 1},
                {"query_id": "q-test", "cand_id": "d", "score": 0},
            ]
            original = data_module._read_parquet_rows
            data_module._read_parquet_rows = lambda _: rows
            try:
                development_path = root / "development.jsonl.gz"
                extract_labels(fake_parquet, prepared, development_path, {"train"})
                self.assertEqual([row["qid"] for row in read_jsonl_gz(development_path)], ["q-train"])
                with self.assertRaises(RuntimeError):
                    extract_labels(
                        fake_parquet,
                        prepared,
                        root / "locked.jsonl.gz",
                        {"locked_test"},
                    )
                frozen = root / "frozen"
                frozen.mkdir()
                decisions = frozen / "decisions.jsonl.gz"
                write_jsonl_gz(decisions, [{"qid": "q-test", "methods": {}}])
                manifest = frozen / "decision_manifest.json"
                write_json(
                    manifest,
                    {
                        "phase": "freeze",
                        "test_labels_consumed": False,
                        "score_hashes": {"fixture": "0" * 64},
                        "decisions_file": decisions.name,
                        "decisions_sha256": sha256_file(decisions),
                    },
                )
                locked_path = root / "locked.jsonl.gz"
                extract_labels(
                    fake_parquet,
                    prepared,
                    locked_path,
                    {"locked_test"},
                    frozen_decisions_manifest=manifest,
                )
                self.assertEqual([row["qid"] for row in read_jsonl_gz(locked_path)], ["q-test"])
            finally:
                data_module._read_parquet_rows = original

    def test_freeze_is_invariant_to_unpassed_locked_labels(self) -> None:
        protocol_path = REVISION / "config" / "protocol.json"
        protocol = load_protocol(protocol_path)
        action_names = [item["name"] for item in protocol["actions"]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            scores = root / "scores"
            prepared.mkdir()
            scores.mkdir()
            queries = []
            metric_rows = []
            qids = []
            candidate_ids = []
            offsets = [0]
            for index in range(12):
                qid = f"q{index:02d}"
                split = "train" if index < 4 else "calibration" if index < 8 else "locked_test"
                candidates = [f"d{index:02d}a", f"d{index:02d}b"]
                queries.append(
                    {
                        "qid": qid,
                        "title": f"query title {index}",
                        "abstract": f"abstract text {index}",
                        "split": split,
                        "candidate_doc_ids": candidates,
                        "candidate_corpus_ids": [index * 2 + 1, index * 2 + 2],
                    }
                )
                qids.append(qid)
                candidate_ids.extend(candidates)
                offsets.append(len(candidate_ids))
                if split != "locked_test":
                    systems = {
                        "specter2": {"ndcg_at_10": 0.5},
                        "bm25": {"ndcg_at_10": 0.4},
                        "bge": {"ndcg_at_10": 0.45},
                        "scincl": {"ndcg_at_10": 0.46},
                        "lambdarank": {"ndcg_at_10": 0.51},
                    }
                    for action_index, action_name in enumerate(action_names):
                        systems[f"action_{action_name}"] = {
                            "ndcg_at_10": 0.5 + 0.002 * action_index - 0.001 * (index % 2)
                        }
                    metric_rows.append({"qid": qid, "split": split, "metrics": systems})
            write_jsonl_gz(prepared / "queries.jsonl.gz", queries)
            write_json(scores / "qids.json", qids)
            write_json(scores / "candidate_doc_ids.json", candidate_ids)
            np.save(scores / "offsets.npy", np.asarray(offsets, dtype=np.int64), allow_pickle=False)
            write_json(scores / "layout.manifest.json", {"fixture": True})
            rng = np.random.default_rng(11)
            for name in ["bm25", "bge", "scincl", "specter2", "lambdarank"]:
                np.save(scores / f"{name}.npy", rng.normal(size=len(candidate_ids)), allow_pickle=False)
            np.save(scores / "citation_count.npy", np.arange(len(candidate_ids)), allow_pickle=False)
            np.save(scores / "year.npy", 2000 + np.arange(len(candidate_ids)), allow_pickle=False)
            write_json(scores / "actions.manifest.json", {"fixture": True})
            write_json(scores / "lambdarank.manifest.json", {"fixture": True})
            for action_name in action_names:
                np.save(
                    scores / f"action_{action_name}.npy",
                    rng.normal(size=len(candidate_ids)),
                    allow_pickle=False,
                )
            training = root / "training.jsonl.gz"
            calibration = root / "calibration.jsonl.gz"
            write_jsonl_gz(training, [row for row in metric_rows if row["split"] == "train"])
            write_jsonl_gz(
                calibration, [row for row in metric_rows if row["split"] == "calibration"]
            )
            content_audit = root / "content_audit.json"
            write_json(
                content_audit,
                {
                    "locked_max_similarity_to_development": {
                        f"q{index:02d}": {"qid": "q00", "similarity": 0.1}
                        for index in range(8, 12)
                    }
                },
            )
            unrelated_locked_labels = root / "unrelated_locked.jsonl.gz"
            write_jsonl_gz(unrelated_locked_labels, [{"secret": "version-one"}])
            first = freeze_decisions(
                prepared,
                scores,
                training,
                calibration,
                content_audit,
                protocol_path,
                root / "frozen-one",
            )
            write_jsonl_gz(unrelated_locked_labels, [{"secret": "changed-labels"}])
            second = freeze_decisions(
                prepared,
                scores,
                training,
                calibration,
                content_audit,
                protocol_path,
                root / "frozen-two",
            )
            self.assertFalse(first["test_labels_consumed"])
            self.assertFalse(second["test_labels_consumed"])
            self.assertEqual(first["decisions_sha256"], second["decisions_sha256"])
            _verify_frozen_scores(scores, action_names, first)
            np.save(scores / "specter2.npy", np.zeros(len(candidate_ids)), allow_pickle=False)
            with self.assertRaises(RuntimeError):
                _verify_frozen_scores(scores, action_names, first)


class PolicyTests(unittest.TestCase):
    def test_local_policy_fixes_action_before_gate(self) -> None:
        titles = ["graph retrieval", "neural retrieval", "medical retrieval", "citation graph"]
        effects = np.asarray([[0.2, -0.1], [0.1, -0.2], [0.3, -0.1], [0.1, -0.3]])
        policy = LocalPolicy(lcb_z=1.645).fit(titles, effects)
        prediction = policy.predict(["graph retrieval", "medical search"])
        np.testing.assert_array_equal(prediction.action_index, [0, 0])
        self.assertTrue(np.all(prediction.gate_score <= prediction.estimated_gain))

    def test_calibration_abstains_when_risk_rule_fails(self) -> None:
        result = calibrate_threshold(
            gate_scores=np.asarray([0.3, 0.2, 0.1]),
            realised_effects=np.asarray([-0.1, -0.1, -0.1]),
            harm_upper_bound=0.2,
            confidence=0.95,
            minimum_active=2,
        )
        self.assertFalse(result["eligible"])
        self.assertEqual(result["active"], 0)
        self.assertIsNone(result["harm_upper"])
        self.assertIsNone(result["mean_gain"])
        self.assertTrue(math.isinf(float(result["threshold"])))

    def test_clopper_pearson_is_bounded(self) -> None:
        self.assertGreater(clopper_pearson_upper(0, 30), 0.0)
        self.assertLess(clopper_pearson_upper(0, 30), 0.2)
        self.assertEqual(clopper_pearson_upper(30, 30), 1.0)


class StatisticsTests(unittest.TestCase):
    def test_zero_coverage_conditional_outcomes_are_undefined(self) -> None:
        result = policy_outcomes([0.1, -0.2], [False, False])
        self.assertEqual(result["overall_mean_gain"], 0.0)
        self.assertIsNone(result["conditional_mean_gain"])
        self.assertIsNone(result["conditional_harm_probability"])

    def test_exact_coverage_and_policy_outcomes(self) -> None:
        mask = deterministic_top_k([0.4, 0.1, 0.3, 0.2], 0.5)
        np.testing.assert_array_equal(mask, [True, False, True, False])
        outcome = policy_outcomes([0.1, -0.1, -0.2, 0.0], mask)
        self.assertEqual(outcome["active"], 2)
        self.assertAlmostEqual(outcome["coverage"], 0.5)
        self.assertAlmostEqual(outcome["conditional_harm_probability"], 0.5)
        self.assertAlmostEqual(outcome["mean_negative_shortfall"], 0.1)

    def test_risk_curve_uses_conditional_negative_shortfall(self) -> None:
        curve = risk_coverage_curve([0.2, -0.4], [2.0, 1.0])
        np.testing.assert_allclose(curve["coverage"], [0.5, 1.0])
        np.testing.assert_allclose(curve["risk"], [0.0, 0.2])
        self.assertAlmostEqual(curve["aurc"], 0.05)

    def test_resampling_is_reproducible(self) -> None:
        values = [0.1, 0.2, 0.3, -0.1]
        self.assertEqual(
            paired_bootstrap_ci(values, 200, 7), paired_bootstrap_ci(values, 200, 7)
        )
        self.assertEqual(
            paired_randomisation_pvalue(values, 500, 7),
            paired_randomisation_pvalue(values, 500, 7),
        )

    def test_holm_adjustment_is_monotone(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
        self.assertLessEqual(adjusted["a"], adjusted["b"])
        self.assertLessEqual(adjusted["b"], adjusted["c"])


if __name__ == "__main__":
    unittest.main()

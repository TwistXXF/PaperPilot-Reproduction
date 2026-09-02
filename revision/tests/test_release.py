from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REVISION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION))

from release import METRIC_ORDER, render_tables_and_figure  # noqa: E402


class ReleaseRenderingTests(unittest.TestCase):
    def test_generated_latex_and_figure_are_complete(self) -> None:
        retrieval = {
            name: {metric: 0.5 for metric in METRIC_ORDER}
            for name in ("bm25", "bge", "scincl", "specter2", "lambdarank")
        }
        operating = {}
        curves = {}
        matched = {}
        for name in ("biblioguard", "knn_mean", "hgb_gain", "global_gain_gate", "global_all"):
            operating[name] = {
                "coverage": 0.0 if name == "biblioguard" else 1.0,
                "policy_ndcg_at_10": 0.5,
                "overall_mean_gain": 0.0,
                "conditional_mean_gain": None if name == "biblioguard" else 0.0,
                "conditional_harm_probability": None if name == "biblioguard" else 0.0,
                "mean_negative_shortfall": None if name == "biblioguard" else 0.0,
            }
            curves[name] = {"coverage": [0.5, 1.0], "risk": [0.0, 0.1], "aurc": 0.025}
            matched[name] = {
                budget: {
                    "overall_mean_gain": 0.0,
                    "conditional_harm_probability": 0.0,
                }
                for budget in ("0.10", "0.25", "0.50", "0.75", "1.00")
            }
        results = {
            "dataset_summary": {
                "queries": 12,
                "train_queries": 4,
                "calibration_queries": 4,
                "locked_queries": 4,
                "candidate_pairs": 24,
                "unique_documents": 24,
            },
            "primary": {
                "mean_effect": 0.0,
                "bootstrap_95_ci": [-0.01, 0.01],
                "paired_randomisation_p": 1.0,
            },
            "operating_point": operating,
            "matched_coverage": matched,
            "retrieval_metrics": retrieval,
            "fixed_action_ndcg_at_10": {"citation_015": 0.51},
            "fixed_action_gain_vs_specter2": {"citation_015": 0.01},
            "risk_coverage": curves,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            render_tables_and_figure(results, output)
            self.assertTrue((output / "figure_risk_coverage.pdf").exists())
            self.assertTrue((output / "table_matched.tex").exists())
            policy = (output / "table_policy.tex").read_text(encoding="utf-8")
            self.assertIn("BiblioGuard", policy)
            self.assertIn("--", policy)
            retriever_rows = [
                line for line in (output / "table_retrievers.tex").read_text(encoding="utf-8").splitlines()
                if line.startswith("BM25")
            ]
            self.assertEqual(len(retriever_rows), 1)
            self.assertTrue(retriever_rows[0].endswith("\\\\"))
            matched = (output / "table_matched.tex").read_text(encoding="utf-8")
            self.assertIn("Matched-coverage comparison", matched)
            self.assertIn("BiblioGuard", matched)


if __name__ == "__main__":
    unittest.main()

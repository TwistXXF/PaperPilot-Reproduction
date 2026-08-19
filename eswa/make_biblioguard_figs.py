#!/usr/bin/env python
"""Figures for the revised BiblioGuard evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"
FIGURES.mkdir(exist_ok=True)
DATASETS = ("scidocs", "scifact", "nfcorpus", "trec-covid")
LABELS = ("SCIDOCS", "SciFact", "NFCorpus", "TREC-COVID")
COLORS = {
    "fallback": "#9E9E9E",
    "global_best": "#E69F00",
    "local_mean": "#56B4E9",
    "biblioguard": "#0072B2",
}


def save(fig, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    payload = json.loads((RESULTS / "biblioguard_results.json").read_text())
    results = payload["results"]
    if set(results) != set(DATASETS):
        raise RuntimeError("run biblioguard.py on all four datasets first")

    x = np.arange(len(DATASETS))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    series = (
        ("fallback", "Strong content fallback"),
        ("global_best", "Global-best action"),
        ("local_mean", "Local kNN mean"),
        ("biblioguard", "BiblioGuard"),
    )
    for offset, (key, label) in enumerate(series):
        values = []
        for dataset in DATASETS:
            row = results[dataset]
            values.append(
                row["fallback_N@10"]
                if key == "fallback"
                else row["comparisons"][key]["N@10"]
            )
        bars = ax.bar(
            x + (offset - 1.5) * width,
            values,
            width,
            label=label,
            color=COLORS[key],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
            )
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("NDCG@10")
    ax.set_title("Same-content metadata interventions against strong fallbacks")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_ylim(0, max(ax.get_ylim()[1], 0.85))
    fig.tight_layout()
    save(fig, "Fig2_biblioguard_main")

    scidocs = results["scidocs"]
    risk = scidocs["risk_coverage"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    coverage = [row["coverage"] * 100 for row in risk]
    gain = [row["gain_N@10"] for row in risk]
    harmed = [row["harmed_active"] for row in risk]
    scatter = ax.scatter(
        coverage,
        gain,
        c=harmed,
        cmap="viridis_r",
        s=65,
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
    )
    ax.plot(coverage, gain, color="#0072B2", linewidth=1.5)
    offsets = {
        0.0: (-28, 10),
        0.25: (-42, -18),
        0.5: (5, 8),
        0.75: (5, 8),
        1.0: (5, 8),
        1.25: (5, 8),
        1.5: (5, 8),
    }
    for row, x_value, y_value in zip(risk, coverage, gain):
        ax.annotate(
            f"{row['penalty_scale']:g}×",
            (x_value, y_value),
            xytext=offsets[row["penalty_scale"]],
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Intervention coverage (%)")
    ax.set_ylabel("Mean NDCG@10 gain")
    ax.set_title("SCIDOCS risk–coverage trade-off")
    ax.grid(alpha=0.25)
    ax.margins(x=0.05, y=0.09)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Harmed active queries")
    fig.tight_layout()
    save(fig, "Fig3_risk_coverage")

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    improved, unchanged, harmed_values = [], [], []
    for dataset in DATASETS:
        counts = results[dataset]["comparisons"]["biblioguard"][
            "outcomes_active"
        ]
        improved.append(counts["improved"])
        unchanged.append(counts["unchanged"])
        harmed_values.append(counts["harmed"])
    ax.bar(x, improved, label="Improved", color="#009E73")
    ax.bar(x, unchanged, bottom=improved, label="Unchanged", color="#BDBDBD")
    bottoms = np.asarray(improved) + np.asarray(unchanged)
    ax.bar(x, harmed_values, bottom=bottoms, label="Harmed", color="#D55E00")
    for index, (good, same, bad) in enumerate(
        zip(improved, unchanged, harmed_values)
    ):
        total = good + same + bad
        if total == 0:
            ax.text(index, 5, "No intervention", ha="center", va="bottom",
                    fontsize=8, color="#555555")
        else:
            ax.text(index, good / 2, str(good), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
            ax.text(index, good + same / 2, str(same), ha="center", va="center",
                    fontsize=8, color="#333333", fontweight="bold")
            ax.text(index, total - bad / 2, str(bad), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("Active queries")
    ax.set_title("Per-query outcomes when BiblioGuard intervenes")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, "Fig4_active_outcomes")
    print("saved revised BiblioGuard figures to", FIGURES)


if __name__ == "__main__":
    main()

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REVISION = Path(__file__).resolve().parent
REPOSITORY = REVISION.parent
sys.path.insert(0, str(REVISION / "src"))

from biblioguard_v3.io import read_json, read_jsonl_gz, sha256_file  # noqa: E402
from biblioguard_v3.statistics import policy_outcomes  # noqa: E402


def assert_close(actual: float | None, expected: float | None, label: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        return
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"{label}: {actual} != {expected}")


def main() -> None:
    published = REVISION / "published"
    release = read_json(published / "release_manifest.json")
    for section in ("copied_files", "generated_files"):
        for item in release[section]:
            path = REPOSITORY / item["path"]
            if sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"Hash mismatch: {path}")

    results = read_json(published / "results.json")
    rows = read_jsonl_gz(published / "locked_per_query.jsonl.gz")
    if len(rows) != results["locked_queries"]:
        raise AssertionError("Per-query row count differs from results.json")
    methods = sorted(rows[0]["methods"])
    for method in methods:
        effects = np.asarray([row["methods"][method]["effect"] for row in rows], dtype=float)
        active = np.asarray([row["methods"][method]["active"] for row in rows], dtype=bool)
        recomputed = policy_outcomes(effects, active)
        reported: dict[str, Any] = results["operating_point"][method]
        for key, value in recomputed.items():
            if isinstance(value, int):
                if value != reported[key]:
                    raise AssertionError(f"{method}.{key}: {value} != {reported[key]}")
            else:
                assert_close(value, reported[key], f"{method}.{key}")

    primary = np.asarray(
        [
            row["methods"]["biblioguard"]["effect"]
            if row["methods"]["biblioguard"]["active"]
            else 0.0
            for row in rows
        ],
        dtype=float,
    )
    assert_close(float(np.mean(primary)), results["primary"]["mean_effect"], "primary.mean_effect")
    frozen = read_json(REVISION / "frozen" / "decision_manifest.json")
    decisions = REVISION / "frozen" / frozen["decisions_file"]
    if sha256_file(decisions) != results["frozen_decisions_sha256"]:
        raise RuntimeError("Published results do not descend from revision/frozen decisions")
    print(
        f"OK: {len(rows)} locked queries, {len(methods)} policies, "
        f"results SHA-256 {sha256_file(published / 'results.json')}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable


_SPACE = re.compile(r"\s+")


def normalise_title(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower()
    value = "".join(character if character.isalnum() else " " for character in value)
    return _SPACE.sub(" ", value).strip()


def group_id(query: dict[str, Any]) -> str:
    normalised = normalise_title(str(query.get("title", "")))
    if normalised:
        return normalised
    corpus_id = query.get("corpus_id")
    if corpus_id is None or int(corpus_id) < 0:
        raise ValueError("A query with an empty title must have corpus_id")
    return f"corpus:{corpus_id}"


def split_bucket(group: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}|{group}".encode("utf-8")).hexdigest()
    return int(digest, 16) % 10


def assign_split(query: dict[str, Any], dataset_config: dict[str, Any]) -> str:
    bucket = split_bucket(group_id(query), str(dataset_config["split_salt"]))
    if bucket in dataset_config["train_buckets"]:
        return "train"
    if bucket in dataset_config["calibration_buckets"]:
        return "calibration"
    if bucket in dataset_config["locked_test_buckets"]:
        return "locked_test"
    raise AssertionError(f"Unassigned bucket {bucket}")


def audit_splits(rows: Iterable[dict[str, Any]], dataset_config: dict[str, Any]) -> dict[str, Any]:
    rows = list(rows)
    groups: dict[str, set[str]] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        query = row["query"] if "query" in row else row
        split = assign_split(query, dataset_config)
        counts[split] += 1
        groups.setdefault(group_id(query), set()).add(split)
    cross_split_groups = {key: sorted(value) for key, value in groups.items() if len(value) > 1}
    if cross_split_groups:
        raise RuntimeError(f"Normalised-title groups crossed splits: {cross_split_groups}")
    return {
        "queries": len(rows),
        "groups": len(groups),
        "split_counts": dict(sorted(counts.items())),
        "duplicate_query_titles": len(rows) - len(groups),
        "cross_split_groups": 0,
    }

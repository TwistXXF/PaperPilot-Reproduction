from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json, sha256_file


REQUIRED_TOP_LEVEL = {
    "protocol_version",
    "frozen_at_utc",
    "seed",
    "dataset",
    "primary_backbone",
    "retrievers",
    "actions",
    "policy",
    "evaluation",
}


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    missing = REQUIRED_TOP_LEVEL.difference(protocol)
    if missing:
        raise ValueError(f"Protocol is missing keys: {sorted(missing)}")
    action_names = [action["name"] for action in protocol["actions"]]
    if len(action_names) != len(set(action_names)):
        raise ValueError("Action names must be unique")
    buckets = (
        protocol["dataset"]["train_buckets"]
        + protocol["dataset"]["calibration_buckets"]
        + protocol["dataset"]["locked_test_buckets"]
    )
    if sorted(buckets) != list(range(10)) or len(set(buckets)) != 10:
        raise ValueError("The split buckets must partition 0..9 exactly once")
    return protocol


def protocol_identity(path: Path) -> dict[str, str]:
    protocol = load_protocol(path)
    return {
        "version": str(protocol["protocol_version"]),
        "sha256": sha256_file(path),
    }


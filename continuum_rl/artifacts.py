"""Model artifact metadata and compatibility helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ARTIFACT_VERSION = "v4"


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def metadata_path_for(artifact_path: Path | str) -> Path:
    path = Path(artifact_path)
    return path.with_suffix(path.suffix + ".metadata.json")


def write_metadata(artifact_path: Path | str, metadata: dict[str, Any]) -> Path:
    meta_path = metadata_path_for(artifact_path)
    merged = {"artifact_version": ARTIFACT_VERSION, **metadata}
    meta_path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    return meta_path


def read_metadata(artifact_path: Path | str) -> dict[str, Any]:
    meta_path = metadata_path_for(artifact_path)
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def validate_metadata(
    artifact_path: Path | str,
    expected: dict[str, Any],
    *,
    strict_keys: tuple[str, ...] = (
        "state_dim",
        "obs_schema",
        "model_arch",
        "obstacle_count",
        "goal_type",
        "reward_function",
    ),
) -> dict[str, Any]:
    metadata = read_metadata(artifact_path)
    if not metadata:
        raise ValueError(
            f"Missing metadata sidecar for artifact '{artifact_path}'. "
            "Please use a checkpoint exported with v4 metadata (retrain required after observation-schema and architecture changes)."
        )

    mismatches: list[str] = []
    for key in strict_keys:
        if key not in expected:
            continue
        actual_val = metadata.get(key)
        expected_val = expected.get(key)
        if actual_val != expected_val:
            mismatches.append(f"{key}: expected={expected_val}, actual={actual_val}")

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            f"Artifact compatibility check failed for '{artifact_path}'. {mismatch_text}. "
            "Use a matching canonical-mode v4 checkpoint (retrain required after observation-schema and architecture changes)."
        )

    return metadata

"""Registry for reusable prepared Training Data artifacts.

The registry is intentionally separate from offline data preparation pipelines:
pipelines generate source runs, while Training Data artifacts are the stable
datasets consumed by Featurewise Ridge/MLP model training.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bimba3d_backend.app.schemas.workflow_data import TrainingDataManifest, TrainingDataRow
from bimba3d_backend.app.services.workflow_paths import DEFAULT_WORKFLOW_PATHS, WorkflowPaths

_WRITE_LOCK = threading.Lock()
MANIFEST_FILE = "manifest.json"
ROWS_FILE = "rows.json"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def build_training_data_id(name: str) -> str:
    token = _sanitize_token(name)
    return f"training_data_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{token}"


def create_manifest(
    *,
    name: str,
    source_pipeline_id: str,
    feature_schema: str,
    training_data_id: str | None = None,
    paths: WorkflowPaths = DEFAULT_WORKFLOW_PATHS,
) -> TrainingDataManifest:
    """Create an empty reusable Training Data manifest."""
    paths.ensure_roots()
    artifact_id = training_data_id or build_training_data_id(name)
    artifact_dir = paths.training_data_dir(artifact_id)
    artifact_dir.mkdir(parents=True, exist_ok=False)

    now = utc_now_iso()
    manifest = TrainingDataManifest(
        training_data_id=artifact_id,
        name=name,
        status="empty",
        source_pipeline_id=source_pipeline_id,
        dataset_version=uuid.uuid4().hex,
        feature_schema=feature_schema,
        row_count=0,
        schema_valid=False,
        rows_path=str(artifact_dir / ROWS_FILE),
        manifest_path=str(artifact_dir / MANIFEST_FILE),
        created_at=now,
        last_built_at=None,
        errors=[],
    )
    _write_json_atomic(artifact_dir / ROWS_FILE, [])
    _write_manifest(manifest)
    return manifest


def list_manifests(paths: WorkflowPaths = DEFAULT_WORKFLOW_PATHS) -> list[TrainingDataManifest]:
    """List Training Data manifests, newest first."""
    root = paths.training_data_root
    if not root.exists():
        return []

    manifests: list[TrainingDataManifest] = []
    for manifest_path in root.glob(f"*/{MANIFEST_FILE}"):
        manifest = read_manifest(manifest_path.parent.name, paths=paths)
        if manifest is not None:
            manifests.append(manifest)

    manifests.sort(key=lambda item: item.last_built_at or item.created_at, reverse=True)
    return manifests


def list_usable_manifests(paths: WorkflowPaths = DEFAULT_WORKFLOW_PATHS) -> list[TrainingDataManifest]:
    """List Training Data manifests that can be used for model training."""
    return [manifest for manifest in list_manifests(paths=paths) if is_usable_manifest(manifest)]


def list_manifests_for_source_pipeline(
    source_pipeline_id: str,
    *,
    usable_only: bool = False,
    paths: WorkflowPaths = DEFAULT_WORKFLOW_PATHS,
) -> list[TrainingDataManifest]:
    items = [
        manifest
        for manifest in list_manifests(paths=paths)
        if manifest.source_pipeline_id == source_pipeline_id
    ]
    if usable_only:
        items = [manifest for manifest in items if is_usable_manifest(manifest)]
    return items


def is_usable_manifest(manifest: TrainingDataManifest) -> bool:
    return manifest.status == "ready" and manifest.schema_valid and manifest.row_count > 0


def read_manifest(training_data_id: str, paths: WorkflowPaths = DEFAULT_WORKFLOW_PATHS) -> TrainingDataManifest | None:
    manifest_path = paths.training_data_dir(training_data_id) / MANIFEST_FILE
    payload = _read_json(manifest_path)
    if not isinstance(payload, dict):
        return None
    return TrainingDataManifest.model_validate(payload)


def read_rows(training_data_id: str, paths: WorkflowPaths = DEFAULT_WORKFLOW_PATHS) -> list[TrainingDataRow]:
    manifest = read_manifest(training_data_id, paths=paths)
    if manifest is None:
        raise FileNotFoundError(f"Training Data manifest not found: {training_data_id}")

    payload = _read_json(Path(manifest.rows_path))
    if not isinstance(payload, list):
        raise ValueError(f"Training Data rows file is invalid: {manifest.rows_path}")
    return [TrainingDataRow.model_validate(row) for row in payload if isinstance(row, dict)]


def replace_rows(
    training_data_id: str,
    rows: list[TrainingDataRow | dict[str, Any]],
    *,
    build_options: dict[str, Any] | None = None,
    build_summary: dict[str, Any] | None = None,
    paths: WorkflowPaths = DEFAULT_WORKFLOW_PATHS,
) -> TrainingDataManifest:
    """Replace a dataset's row file and update its manifest metadata."""
    manifest = read_manifest(training_data_id, paths=paths)
    if manifest is None:
        raise FileNotFoundError(f"Training Data manifest not found: {training_data_id}")

    validated_rows = [
        row if isinstance(row, TrainingDataRow) else TrainingDataRow.model_validate(row)
        for row in rows
    ]
    row_payload = [row.model_dump(mode="json") for row in validated_rows]
    _write_json_atomic(Path(manifest.rows_path), row_payload)

    now = utc_now_iso()
    updated = manifest.model_copy(
        update={
            "status": "ready" if validated_rows else "empty",
            "dataset_version": uuid.uuid4().hex,
            "row_count": len(validated_rows),
            "schema_valid": bool(validated_rows),
            "last_built_at": now,
            "errors": [],
            "build_options": dict(build_options or {}),
            "build_summary": dict(build_summary or {}),
        }
    )
    _write_manifest(updated)
    return updated


def mark_failed(
    training_data_id: str,
    errors: list[str],
    *,
    paths: WorkflowPaths = DEFAULT_WORKFLOW_PATHS,
) -> TrainingDataManifest:
    manifest = read_manifest(training_data_id, paths=paths)
    if manifest is None:
        raise FileNotFoundError(f"Training Data manifest not found: {training_data_id}")

    updated = manifest.model_copy(
        update={
            "status": "failed",
            "schema_valid": False,
            "last_built_at": utc_now_iso(),
            "errors": errors,
        }
    )
    _write_manifest(updated)
    return updated


def _write_manifest(manifest: TrainingDataManifest) -> None:
    _write_json_atomic(Path(manifest.manifest_path), manifest.model_dump(mode="json"))


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    with _WRITE_LOCK:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)


def _sanitize_token(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "").strip().lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-_")
    return cleaned or "training-data"

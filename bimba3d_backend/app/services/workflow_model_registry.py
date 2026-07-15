"""Shared registry for thesis workflow model artifacts."""
from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from bimba3d_backend.app.schemas.workflow_data import ModelFamily, WorkflowModelManifest
from bimba3d_backend.app.services.training_data_registry import utc_now_iso
from bimba3d_backend.app.services.workflow_paths import DEFAULT_WORKFLOW_PATHS, WorkflowPaths

_WRITE_LOCK = threading.Lock()
INDEX_FILE = "models_index.json"
MANIFEST_FILE = "model.json"


def register_model(
    *,
    model_id: str,
    model_name: str,
    model_family: ModelFamily,
    source_training_data_id: str,
    artifact_path: str | Path,
    metadata_path: str | Path | None = None,
    training_report_path: str | Path | None = None,
    source_pipeline_id: str | None = None,
    training_samples: int | None = None,
    model_evaluation_step: int | None = None,
    metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    paths: WorkflowPaths = DEFAULT_WORKFLOW_PATHS,
) -> WorkflowModelManifest:
    """Register a trained Ridge/MLP artifact in the shared workflow registry."""
    paths.ensure_roots()
    _validate_model_id(model_id)
    artifact = Path(artifact_path).expanduser()
    if not artifact.exists() or not artifact.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {artifact}")

    if metadata_path is not None:
        metadata = Path(metadata_path).expanduser()
        if not metadata.exists() or not metadata.is_file():
            raise FileNotFoundError(f"Model metadata file does not exist: {metadata}")
        metadata_value = str(metadata)
    else:
        metadata_value = None

    if training_report_path is not None:
        report = Path(training_report_path).expanduser()
        if not report.exists() or not report.is_file():
            raise FileNotFoundError(f"Model training report does not exist: {report}")
        report_value = str(report)
    else:
        report_value = None

    model_dir = paths.model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = WorkflowModelManifest(
        model_id=model_id,
        model_name=model_name,
        model_family=model_family,
        source_training_data_id=source_training_data_id,
        source_pipeline_id=source_pipeline_id,
        artifact_path=str(artifact),
        metadata_path=metadata_value,
        training_report_path=report_value,
        trained_at=utc_now_iso(),
        training_samples=training_samples,
        model_evaluation_step=model_evaluation_step,
        metrics=metrics or {},
        config=config or {},
    )

    _write_manifest(paths, manifest)
    _upsert_index(paths, manifest)
    return manifest


def list_models(paths: WorkflowPaths = DEFAULT_WORKFLOW_PATHS) -> list[WorkflowModelManifest]:
    """List shared workflow models, newest first."""
    index_path = paths.models_root / INDEX_FILE
    payload = _read_json(index_path)
    if not isinstance(payload, list):
        return []

    models: list[WorkflowModelManifest] = []
    for item in payload:
        if isinstance(item, dict):
            models.append(WorkflowModelManifest.model_validate(item))
    models.sort(key=lambda item: item.trained_at, reverse=True)
    return models


def list_models_for_source(
    *,
    source_pipeline_id: str | None = None,
    source_training_data_id: str | None = None,
    paths: WorkflowPaths = DEFAULT_WORKFLOW_PATHS,
) -> list[WorkflowModelManifest]:
    models = list_models(paths=paths)
    if source_pipeline_id:
        models = [model for model in models if model.source_pipeline_id == source_pipeline_id]
    if source_training_data_id:
        models = [model for model in models if model.source_training_data_id == source_training_data_id]
    return models


def read_model(model_id: str, paths: WorkflowPaths = DEFAULT_WORKFLOW_PATHS) -> WorkflowModelManifest | None:
    manifest_path = paths.model_dir(model_id) / MANIFEST_FILE
    payload = _read_json(manifest_path)
    if not isinstance(payload, dict):
        return None
    return WorkflowModelManifest.model_validate(payload)


def rename_model(
    model_id: str,
    model_name: str,
    paths: WorkflowPaths = DEFAULT_WORKFLOW_PATHS,
) -> WorkflowModelManifest | None:
    """Rename one registered workflow model without changing its artifact id."""
    name = str(model_name or "").strip()
    if not name:
        raise ValueError("Model name is required.")

    model = read_model(model_id, paths=paths)
    if model is None:
        return None

    updated = model.model_copy(update={"model_name": name})
    _write_manifest(paths, updated)
    _upsert_index(paths, updated)
    return updated


def delete_model(model_id: str, paths: WorkflowPaths = DEFAULT_WORKFLOW_PATHS) -> bool:
    """Delete one registered workflow model and its owned registry folder."""
    model = read_model(model_id, paths=paths)
    if model is None:
        return False

    _validate_model_id(model_id)
    model_dir = paths.model_dir(model_id)
    owned_root = paths.models_root.resolve()
    resolved_dir = model_dir.resolve()
    if owned_root not in resolved_dir.parents and resolved_dir != owned_root:
        raise ValueError(f"Refusing to delete model outside workflow registry: {model_dir}")

    if model_dir.exists():
        shutil.rmtree(model_dir)

    index_path = paths.models_root / INDEX_FILE
    existing = _read_json(index_path)
    items = [item for item in existing if isinstance(item, dict) and item.get("model_id") != model_id] if isinstance(existing, list) else []
    _write_json_atomic(index_path, items)
    return True


def _write_manifest(paths: WorkflowPaths, manifest: WorkflowModelManifest) -> None:
    _write_json_atomic(paths.model_dir(manifest.model_id) / MANIFEST_FILE, manifest.model_dump(mode="json"))


def _upsert_index(paths: WorkflowPaths, manifest: WorkflowModelManifest) -> None:
    index_path = paths.models_root / INDEX_FILE
    existing = _read_json(index_path)
    items = [item for item in existing if isinstance(item, dict)] if isinstance(existing, list) else []
    payload = manifest.model_dump(mode="json")
    updated = False
    for index, item in enumerate(items):
        if item.get("model_id") == manifest.model_id:
            items[index] = payload
            updated = True
            break
    if not updated:
        items.append(payload)
    items.sort(key=lambda item: str(item.get("trained_at") or ""), reverse=True)
    _write_json_atomic(index_path, items)


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


def _validate_model_id(model_id: str) -> None:
    if not model_id or any(part in model_id for part in ("/", "\\", "..")):
        raise ValueError(f"Invalid workflow model id: {model_id!r}")

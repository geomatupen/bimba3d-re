"""Seed workflow-trained quality models into project-local runtime folders."""
from __future__ import annotations

import shutil
from pathlib import Path

from bimba3d_backend.app.schemas.workflow_data import WorkflowModelManifest
from bimba3d_backend.app.services import workflow_model_registry


def read_workflow_model(model_id: str) -> WorkflowModelManifest | None:
    model_key = str(model_id or "").strip()
    if not model_key:
        return None
    return workflow_model_registry.read_model(model_key)


def seed_workflow_model_into_project(model: WorkflowModelManifest, project_dir: Path) -> Path:
    """Copy a workflow model to the project path used by the AI selector runtime."""
    source_path = Path(model.artifact_path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"Workflow model artifact was not found: {source_path}")

    if model.model_family == "featurewise_mlp":
        target_dir = project_dir / "models" / "featurewise_mlp"
        target_name = "featurewise.pt"
        stale_pattern = "*.pt"
    elif model.model_family == "compact_featurewise_mlp":
        target_dir = project_dir / "models" / "compact_featurewise_mlp"
        target_name = "compact_featurewise.pt"
        stale_pattern = "*.pt"
    elif model.model_family == "featurewise_ridge_regression":
        target_dir = project_dir / "models" / "featurewise_ridge_regression"
        target_name = "exif_compact_featurewise.json"
        stale_pattern = "*.json"
    elif model.model_family == "compact_featurewise_ridge_regression":
        target_dir = project_dir / "models" / "compact_featurewise_ridge_regression"
        target_name = "exif_compact_featurewise.json"
        stale_pattern = "*.json"
    else:
        raise ValueError(f"Unsupported workflow model family: {model.model_family}")

    target_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in target_dir.glob(stale_pattern):
        if stale_path.is_file():
            stale_path.unlink()
    target_path = target_dir / target_name
    shutil.copy2(source_path, target_path)
    return target_path


def model_ai_profile(model: WorkflowModelManifest) -> dict[str, str]:
    return {
        "ai_input_mode": "exif_compact_featurewise",
        "ai_selector_strategy": model.model_family,
    }


def model_evaluation_step(model: WorkflowModelManifest) -> int | None:
    if isinstance(model.model_evaluation_step, int) and model.model_evaluation_step > 0:
        return model.model_evaluation_step
    for key in ("model_evaluation_step", "score_reference_step"):
        value = model.config.get(key) if isinstance(model.config, dict) else None
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and int(value) > 0:
            return int(value)
    return None

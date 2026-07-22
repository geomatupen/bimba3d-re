"""API routes for report-aligned model training."""
from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from bimba3d_backend.app.schemas.workflow_data import ModelFamily, TrainingDataManifest, WorkflowModelManifest
from bimba3d_backend.app.services import training_data_registry, workflow_model_registry
from bimba3d_backend.app.services.model_training import ModelTrainingOptions, train_model_from_training_data
from bimba3d_backend.app.services.workflow_summaries import build_model_training_summary
from bimba3d_backend.app.services.workflow_paths import DEFAULT_WORKFLOW_PATHS, WorkflowPaths

router = APIRouter()

# Tests can replace this with a temp-root path object.
WORKFLOW_PATHS: WorkflowPaths = DEFAULT_WORKFLOW_PATHS


class TrainWorkflowModelRequest(BaseModel):
    model_family: ModelFamily
    model_name: str = Field(..., min_length=1)
    source_training_data_id: str = Field(..., min_length=1)
    source_pipeline_id: str | None = None
    lambda_ridge: float | None = Field(None, gt=0)
    # Stored with the model as a prediction fallback. Test pipelines with an explicit candidate grid override it.
    candidate_points: int = Field(30, ge=5, le=101)
    regularize_intercept: bool = False
    include_phases: list[int] | None = None
    include_run_ids: list[str] | None = None


class ModelTrainingSourcesResponse(BaseModel):
    items: list[TrainingDataManifest]
    total: int


@router.get("/training-data-sources", response_model=ModelTrainingSourcesResponse)
def list_model_training_sources(usable_only: bool = True) -> ModelTrainingSourcesResponse:
    items = (
        training_data_registry.list_usable_manifests(paths=WORKFLOW_PATHS)
        if usable_only
        else training_data_registry.list_manifests(paths=WORKFLOW_PATHS)
    )
    return ModelTrainingSourcesResponse(items=items, total=len(items))


@router.get("/summary")
def get_model_training_summary() -> dict:
    return build_model_training_summary(paths=WORKFLOW_PATHS)


@router.post("/train", response_model=WorkflowModelManifest)
def train_workflow_model(request: TrainWorkflowModelRequest) -> WorkflowModelManifest:
    try:
        return train_model_from_training_data(
            ModelTrainingOptions(
                model_family=request.model_family,
                model_name=request.model_name.strip(),
                source_training_data_id=request.source_training_data_id.strip(),
                source_pipeline_id=request.source_pipeline_id.strip() if request.source_pipeline_id else None,
                lambda_ridge=request.lambda_ridge,
                candidate_points=request.candidate_points,
                regularize_intercept=request.regularize_intercept,
                include_phases=request.include_phases,
                include_run_ids=request.include_run_ids,
            ),
            paths=WORKFLOW_PATHS,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "training_data_not_found",
                "message": "Training Data artifact was not found.",
                "details": str(exc),
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "model_training_invalid_input",
                "message": "Model training request cannot be completed.",
                "details": str(exc),
            },
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "model_training_failed",
                "message": "Model training failed.",
                "details": str(exc),
            },
        ) from exc


@router.post("/upload", response_model=WorkflowModelManifest)
async def upload_workflow_model(
    model_family: ModelFamily = Form(...),
    model_name: str = Form(...),
    source_training_data_id: str = Form(...),
    source_pipeline_id: str | None = Form(None),
    artifact_file: UploadFile = File(...),
    metadata_file: UploadFile | None = File(None),
) -> WorkflowModelManifest:
    """Register an externally trained compact model as a workflow model artifact."""
    clean_name = model_name.strip()
    clean_training_data_id = source_training_data_id.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Model name is required.")
    if not clean_training_data_id:
        raise HTTPException(status_code=400, detail="Training Data source is required.")
    if model_family not in {
        "compact_featurewise_ridge_regression",
        "compact_featurewise_mlp",
        "compact_descriptor_mlp",
    }:
        raise HTTPException(
            status_code=400,
            detail="Upload currently supports compact Ridge JSON variants and compact MLP .pt models.",
        )

    manifest = training_data_registry.read_manifest(clean_training_data_id, paths=WORKFLOW_PATHS)
    if manifest is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "training_data_not_found",
                "message": "Training Data artifact was not found.",
                "training_data_id": clean_training_data_id,
            },
        )

    model_id = _build_uploaded_model_id(clean_name, model_family)
    model_dir = WORKFLOW_PATHS.model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata_payload: dict[str, Any] | None = None
    metadata_path: Path | None = None

    try:
        if metadata_file is not None and metadata_file.filename:
            metadata_payload = _read_uploaded_json(metadata_file, "metadata JSON")
            metadata_path = model_dir / _upload_metadata_filename(model_family)
            _write_json(metadata_path, metadata_payload)

        if model_family == "compact_featurewise_ridge_regression":
            artifact_payload = _read_uploaded_json(artifact_file, "Ridge model JSON")
            model_payload, metrics, config = _validate_compact_ridge_payload(artifact_payload)
            artifact_path = model_dir / "compact_featurewise_ridge_model.json"
            _write_json(artifact_path, artifact_payload)
            training_samples = _int_or_none(model_payload.get("runs") or model_payload.get("n"))
        else:
            artifact_path = model_dir / ("compact_descriptor.pt" if model_family == "compact_descriptor_mlp" else "compact_featurewise.pt")
            await _copy_upload_to_path(artifact_file, artifact_path)
            checkpoint = _validate_compact_mlp_checkpoint(artifact_path, expected_type=str(model_family))
            metrics, config = _compact_mlp_registry_fields(checkpoint, metadata_payload)
            training_samples = _int_or_none(checkpoint.get("training_samples") or (metadata_payload or {}).get("training_samples"))

        model_evaluation_step = _source_model_evaluation_step(clean_training_data_id)
        resolved_source_pipeline_id = (source_pipeline_id or "").strip() or manifest.source_pipeline_id
        upload_log = [
            {
                "level": "success",
                "message": f"Uploaded external {model_family.replace('_', ' ')} artifact and registered it for testing.",
            }
        ]
        metrics = {**metrics, "training_log": upload_log + list(metrics.get("training_log") or [])}
        config = {
            **config,
            "uploaded_model": True,
            "source_filename": artifact_file.filename,
            "metadata_filename": metadata_file.filename if metadata_file is not None else None,
            "model_evaluation_step": model_evaluation_step,
            "score_reference_step": model_evaluation_step,
        }

        return workflow_model_registry.register_model(
            model_id=model_id,
            model_name=clean_name,
            model_family=model_family,
            source_training_data_id=clean_training_data_id,
            source_pipeline_id=resolved_source_pipeline_id,
            artifact_path=artifact_path,
            metadata_path=metadata_path,
            training_samples=training_samples,
            model_evaluation_step=model_evaluation_step,
            metrics=metrics,
            config=config,
            paths=WORKFLOW_PATHS,
        )
    except HTTPException:
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
        raise
    except Exception as exc:
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail={
                "code": "model_upload_invalid",
                "message": "Uploaded model could not be registered.",
                "details": str(exc),
            },
        ) from exc


def _build_uploaded_model_id(model_name: str, model_family: ModelFamily) -> str:
    token = re.sub(r"[^a-zA-Z0-9_-]+", "-", model_name.strip().lower())
    token = re.sub(r"-+", "-", token).strip("-_") or "uploaded-model"
    family = {
        "compact_featurewise_ridge_regression": "compact-ridge",
        "compact_featurewise_mlp": "compact-mlp",
        "compact_descriptor_mlp": "compact-descriptor-mlp",
    }.get(model_family, "model")
    return f"model_{family}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{token}"


def _upload_metadata_filename(model_family: ModelFamily) -> str:
    if model_family == "compact_featurewise_mlp":
        return "compact_featurewise_mlp_metadata.json"
    if model_family == "compact_descriptor_mlp":
        return "compact_descriptor_mlp_metadata.json"
    return "compact_featurewise_ridge_metadata.json"


def _read_uploaded_json(upload: UploadFile, label: str) -> dict[str, Any]:
    raw = upload.file.read()
    upload.file.seek(0)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"{label} must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return payload


async def _copy_upload_to_path(upload: UploadFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    await upload.seek(0)


def _validate_compact_ridge_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model_payload = payload.get("model") if isinstance(payload.get("model"), dict) else payload
    family = str(payload.get("model_family") or model_payload.get("model_family") or "")
    schema = str(payload.get("schema") or "")
    if family != "compact_featurewise_ridge_regression" and schema != "compact_featurewise_ridge_regression_v1":
        raise ValueError("Ridge JSON must be a compact_featurewise_ridge_regression model.")
    if not isinstance(model_payload.get("A"), list) or not isinstance(model_payload.get("b"), list):
        raise ValueError("Ridge JSON is missing normal-equation arrays A and b.")
    if not isinstance(model_payload.get("feature_scaler"), dict):
        raise ValueError("Ridge JSON is missing feature_scaler.")
    if "candidate_points" not in model_payload:
        raise ValueError("Ridge JSON is missing candidate_points.")

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    config = {
        "candidate_points": model_payload.get("candidate_points"),
        "lambda_selected": payload.get("lambda_ridge") or model_payload.get("lambda_ridge"),
        "log_multiplier_bounds": model_payload.get("log_multiplier_bounds"),
        "regularize_intercept": model_payload.get("regularize_intercept"),
        "regularization": model_payload.get("regularization") or (payload.get("metrics") or {}).get("regularization"),
        "artifact_schema": schema or None,
    }
    return model_payload, metrics, config


def _validate_compact_mlp_checkpoint(path: Path, *, expected_type: str = "compact_featurewise_mlp") -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("PyTorch is required on the backend to validate uploaded MLP checkpoints.") from exc

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("MLP checkpoint must contain a dictionary.")
    if str(checkpoint.get("model_type") or "") != expected_type:
        raise ValueError(f"MLP checkpoint must have model_type {expected_type}.")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("MLP checkpoint is missing state_dict.")
    if not isinstance(checkpoint.get("input_dim"), int):
        raise ValueError("MLP checkpoint is missing integer input_dim.")
    if not isinstance(checkpoint.get("feature_scaler"), dict):
        raise ValueError("MLP checkpoint is missing feature_scaler. Retrain or export the compact MLP with standardized descriptors.")
    return checkpoint


def _compact_mlp_registry_fields(
    checkpoint: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = metadata or {}
    metrics = {
        "best_val_loss": meta.get("best_val_loss"),
        "final_train_loss": meta.get("final_train_loss"),
        "final_val_loss": meta.get("final_val_loss"),
        "epochs_trained": meta.get("epochs_trained"),
        "max_epochs": meta.get("max_epochs") or checkpoint.get("max_epochs"),
        "best_epoch": meta.get("best_epoch"),
        "early_stopping_patience": meta.get("early_stopping_patience") or checkpoint.get("early_stopping_patience"),
        "total_parameters": meta.get("total_parameters"),
        "learning_rate": meta.get("learning_rate") or checkpoint.get("learning_rate"),
        "weight_decay": meta.get("weight_decay") or checkpoint.get("weight_decay"),
        "hidden": checkpoint.get("hidden"),
        "dropout": checkpoint.get("dropout"),
        "seed": checkpoint.get("seed") or meta.get("seed"),
        "feature_standardization": checkpoint.get("feature_standardization") or meta.get("feature_standardization"),
    }
    config = {
        "candidate_points": checkpoint.get("candidate_points"),
        "hidden": checkpoint.get("hidden"),
        "dropout": checkpoint.get("dropout"),
        "learning_rate": checkpoint.get("learning_rate"),
        "weight_decay": checkpoint.get("weight_decay"),
        "max_epochs": checkpoint.get("max_epochs"),
        "early_stopping_patience": checkpoint.get("early_stopping_patience"),
        "seed": checkpoint.get("seed") or meta.get("seed"),
        "log_multiplier_bounds": checkpoint.get("log_multiplier_bounds"),
        "feature_standardization": checkpoint.get("feature_standardization") or meta.get("feature_standardization"),
    }
    return metrics, config


def _source_model_evaluation_step(training_data_id: str) -> int:
    rows = training_data_registry.read_rows(training_data_id, paths=WORKFLOW_PATHS)
    steps: set[int] = set()
    missing = 0
    for row in rows:
        if row.score_reference_step is None:
            missing += 1
            continue
        steps.add(int(row.score_reference_step))
    if missing:
        raise ValueError("Training Data rows must include score_reference_step before registering an uploaded model.")
    if len(steps) != 1:
        raise ValueError("Training Data rows must contain exactly one model evaluation step.")
    return next(iter(steps))


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

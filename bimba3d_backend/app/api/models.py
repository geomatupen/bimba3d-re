"""API routes for shared thesis workflow models."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from bimba3d_backend.app.schemas.workflow_data import WorkflowModelManifest
from bimba3d_backend.app.services import workflow_model_registry
from bimba3d_backend.app.services.workflow_paths import DEFAULT_WORKFLOW_PATHS, WorkflowPaths

router = APIRouter()

# Test code can replace this with a temp-root path object.
WORKFLOW_PATHS: WorkflowPaths = DEFAULT_WORKFLOW_PATHS


class ModelsListResponse(BaseModel):
    items: list[WorkflowModelManifest]
    total: int


class RenameModelRequest(BaseModel):
    model_name: str


@router.get("", response_model=ModelsListResponse)
def list_models(
    source_pipeline_id: str | None = None,
    source_training_data_id: str | None = None,
) -> ModelsListResponse:
    models = workflow_model_registry.list_models_for_source(
        source_pipeline_id=source_pipeline_id,
        source_training_data_id=source_training_data_id,
        paths=WORKFLOW_PATHS,
    )
    return ModelsListResponse(items=models, total=len(models))


@router.get("/{model_id}", response_model=WorkflowModelManifest)
def get_model(model_id: str) -> WorkflowModelManifest:
    model = workflow_model_registry.read_model(model_id, paths=WORKFLOW_PATHS)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workflow_model_not_found",
                "message": "Workflow model artifact was not found.",
                "model_id": model_id,
            },
        )
    return model


@router.patch("/{model_id}", response_model=WorkflowModelManifest)
def rename_model(model_id: str, payload: RenameModelRequest) -> WorkflowModelManifest:
    try:
        model = workflow_model_registry.rename_model(model_id, payload.model_name, paths=WORKFLOW_PATHS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if model is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workflow_model_not_found",
                "message": "Workflow model artifact was not found.",
                "model_id": model_id,
            },
        )
    return model


@router.delete("/{model_id}")
def delete_model(model_id: str) -> dict[str, str]:
    try:
        deleted = workflow_model_registry.delete_model(model_id, paths=WORKFLOW_PATHS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workflow_model_not_found",
                "message": "Workflow model artifact was not found.",
                "model_id": model_id,
            },
        )
    return {"status": "deleted", "model_id": model_id}

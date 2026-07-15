"""API routes for reusable prepared Training Data artifacts."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from bimba3d_backend.app.schemas.workflow_data import TrainingDataManifest, TrainingDataRow
from bimba3d_backend.app.services import pipeline_learning_rows
from bimba3d_backend.app.services import training_data_builder
from bimba3d_backend.app.services import training_data_registry
from bimba3d_backend.app.services.workflow_paths import DEFAULT_WORKFLOW_PATHS, WorkflowPaths

router = APIRouter()

# Kept as a module variable so tests can point the router at a temp data root.
WORKFLOW_PATHS: WorkflowPaths = DEFAULT_WORKFLOW_PATHS


class CreateTrainingDataRequest(BaseModel):
    name: str = Field(..., min_length=1)
    source_pipeline_id: str = Field(..., min_length=1)
    feature_schema: str = Field(default="mode3_exif_flight_scene_v1", min_length=1)
    training_data_id: str | None = None


class TrainingDataListResponse(BaseModel):
    items: list[TrainingDataManifest]
    total: int


class TrainingDataRowsResponse(BaseModel):
    training_data_id: str
    rows: list[TrainingDataRow]
    total: int


class TrainingDataValidityResponse(BaseModel):
    training_data_id: str
    usable_for_model_training: bool
    status: str
    row_count: int
    schema_valid: bool
    feature_schema: str
    last_built_at: str | None
    errors: list[str]


class TrainingDataByPipelineResponse(BaseModel):
    source_pipeline_id: str
    items: list[TrainingDataManifest]
    total: int


class BuildTrainingDataRowsRequest(BaseModel):
    source_pipeline_id: str = Field(..., min_length=1)
    rows: list[dict] = Field(default_factory=list)
    include_hard_cap_penalty_rows: bool = False


class BuildTrainingDataFromPipelineRequest(BaseModel):
    include_hard_cap_penalty_rows: bool = False


class BuildTrainingDataRowsResponse(BaseModel):
    training_data_id: str
    imported_rows: int
    skipped_rows: int
    hard_cap_penalty_rows: int = 0
    hard_cap_penalty: float | None = None
    errors: list[str]
    manifest: TrainingDataManifest


@router.get("", response_model=TrainingDataListResponse)
def list_training_data(usable_only: bool = Query(default=False)) -> TrainingDataListResponse:
    items = (
        training_data_registry.list_usable_manifests(paths=WORKFLOW_PATHS)
        if usable_only
        else training_data_registry.list_manifests(paths=WORKFLOW_PATHS)
    )
    return TrainingDataListResponse(items=items, total=len(items))


@router.post("", response_model=TrainingDataManifest)
def create_training_data(request: CreateTrainingDataRequest) -> TrainingDataManifest:
    try:
        return training_data_registry.create_manifest(
            name=request.name.strip(),
            source_pipeline_id=request.source_pipeline_id.strip(),
            feature_schema=request.feature_schema.strip(),
            training_data_id=request.training_data_id,
            paths=WORKFLOW_PATHS,
        )
    except FileExistsError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "training_data_already_exists",
                "message": "Training Data id already exists.",
                "details": str(exc),
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_training_data_request",
                "message": "Training Data request is invalid.",
                "details": str(exc),
            },
        ) from exc


@router.get("/by-source-pipeline/{source_pipeline_id}", response_model=TrainingDataByPipelineResponse)
def list_training_data_by_source_pipeline(
    source_pipeline_id: str,
    usable_only: bool = Query(default=False),
) -> TrainingDataByPipelineResponse:
    items = training_data_registry.list_manifests_for_source_pipeline(
        source_pipeline_id,
        usable_only=usable_only,
        paths=WORKFLOW_PATHS,
    )
    return TrainingDataByPipelineResponse(
        source_pipeline_id=source_pipeline_id,
        items=items,
        total=len(items),
    )


@router.get("/{training_data_id}", response_model=TrainingDataManifest)
def get_training_data(training_data_id: str) -> TrainingDataManifest:
    manifest = training_data_registry.read_manifest(training_data_id, paths=WORKFLOW_PATHS)
    if manifest is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "training_data_not_found",
                "message": "Training Data artifact was not found.",
                "training_data_id": training_data_id,
            },
        )
    return manifest


@router.get("/{training_data_id}/validity", response_model=TrainingDataValidityResponse)
def get_training_data_validity(training_data_id: str) -> TrainingDataValidityResponse:
    manifest = training_data_registry.read_manifest(training_data_id, paths=WORKFLOW_PATHS)
    if manifest is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "training_data_not_found",
                "message": "Training Data artifact was not found.",
                "training_data_id": training_data_id,
            },
        )
    return TrainingDataValidityResponse(
        training_data_id=manifest.training_data_id,
        usable_for_model_training=training_data_registry.is_usable_manifest(manifest),
        status=manifest.status,
        row_count=manifest.row_count,
        schema_valid=manifest.schema_valid,
        feature_schema=manifest.feature_schema,
        last_built_at=manifest.last_built_at,
        errors=manifest.errors,
    )


@router.get("/{training_data_id}/rows", response_model=TrainingDataRowsResponse)
def get_training_data_rows(training_data_id: str) -> TrainingDataRowsResponse:
    try:
        rows = training_data_registry.read_rows(training_data_id, paths=WORKFLOW_PATHS)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "training_data_not_found",
                "message": "Training Data artifact was not found.",
                "training_data_id": training_data_id,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "training_data_rows_invalid",
                "message": "Training Data rows file is invalid.",
                "details": str(exc),
            },
        ) from exc

    return TrainingDataRowsResponse(training_data_id=training_data_id, rows=rows, total=len(rows))


@router.post("/{training_data_id}/build-from-learning-rows", response_model=BuildTrainingDataRowsResponse)
def build_from_learning_rows(training_data_id: str, request: BuildTrainingDataRowsRequest) -> BuildTrainingDataRowsResponse:
    try:
        config_snapshot = pipeline_learning_rows.training_data_config_snapshot(request.source_pipeline_id.strip())
        result = training_data_builder.build_from_learning_rows(
            training_data_id=training_data_id,
            source_pipeline_id=request.source_pipeline_id.strip(),
            rows=request.rows,
            include_hard_cap_penalty_rows=request.include_hard_cap_penalty_rows,
            training_data_config_snapshot=config_snapshot,
            paths=WORKFLOW_PATHS,
        )
        manifest = training_data_registry.read_manifest(training_data_id, paths=WORKFLOW_PATHS)
        if manifest is None:
            raise FileNotFoundError(f"Training Data manifest not found: {training_data_id}")
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "training_data_not_found",
                "message": "Training Data artifact was not found.",
                "training_data_id": training_data_id,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "training_data_build_invalid_input",
                "message": "Training Data rows could not be built.",
                "details": str(exc),
            },
        ) from exc

    # The builder marks the manifest failed when row conversion finds errors.
    # Returning 200 lets the UI show row-level errors and the updated manifest.
    return BuildTrainingDataRowsResponse(
        training_data_id=training_data_id,
        imported_rows=result.imported_rows,
        skipped_rows=result.skipped_rows,
        hard_cap_penalty_rows=result.hard_cap_penalty_rows,
        hard_cap_penalty=result.hard_cap_penalty,
        errors=result.errors,
        manifest=manifest,
    )


@router.post("/{training_data_id}/build-from-pipeline/{pipeline_id}", response_model=BuildTrainingDataRowsResponse)
async def build_from_pipeline(
    training_data_id: str,
    pipeline_id: str,
    request: BuildTrainingDataFromPipelineRequest | None = None,
) -> BuildTrainingDataRowsResponse:
    try:
        build_request = request or BuildTrainingDataFromPipelineRequest()
        source = pipeline_learning_rows.collect_pipeline_learning_rows(
            pipeline_id,
            include_hard_cap=build_request.include_hard_cap_penalty_rows,
        )
        rows = source.get("rows") if isinstance(source, dict) else None
        if not isinstance(rows, list):
            raise ValueError("Pipeline learning-row response did not contain a row list.")

        result = training_data_builder.build_from_learning_rows(
            training_data_id=training_data_id,
            source_pipeline_id=pipeline_id,
            rows=rows,
            include_hard_cap_penalty_rows=build_request.include_hard_cap_penalty_rows,
            training_data_config_snapshot=source.get("training_data_config_snapshot") if isinstance(source, dict) else None,
            paths=WORKFLOW_PATHS,
        )
        manifest = training_data_registry.read_manifest(training_data_id, paths=WORKFLOW_PATHS)
        if manifest is None:
            raise FileNotFoundError(f"Training Data manifest not found: {training_data_id}")
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "training_data_not_found",
                "message": "Training Data artifact was not found.",
                "training_data_id": training_data_id,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "training_data_build_invalid_input",
                "message": "Training Data rows could not be built from the selected pipeline.",
                "details": str(exc),
            },
        ) from exc
    except HTTPException:
        raise

    return BuildTrainingDataRowsResponse(
        training_data_id=training_data_id,
        imported_rows=result.imported_rows,
        skipped_rows=result.skipped_rows,
        hard_cap_penalty_rows=result.hard_cap_penalty_rows,
        hard_cap_penalty=result.hard_cap_penalty,
        errors=result.errors,
        manifest=manifest,
    )

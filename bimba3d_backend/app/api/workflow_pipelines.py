from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from bimba3d_backend.app.services import workflow_pipeline_service
from bimba3d_backend.app.services.workflow_summaries import (
    build_fixed_log_space_schedule,
    build_offline_data_preparation_summary,
    build_testing_candidate_curves,
    build_testing_pipeline_summary,
)

router = APIRouter()


class WorkflowPipelineListResponse(BaseModel):
    items: list[dict[str, Any]]
    total: int


class CreateWorkflowPipelineRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    workflow_stage: str | None = None
    pipeline_type: str | None = None
    base_directory: str | None = None
    pipeline_directory: str | None = None
    source_model_id: str | None = None
    source_model_ids: list[str] | None = None
    contribute_to_training: bool = False
    projects: list[dict[str, Any]] = Field(default_factory=list)
    shared_config: dict[str, Any] = Field(default_factory=dict)
    phases: list[dict[str, Any]] = Field(default_factory=list)
    thermal_management: dict[str, Any] = Field(default_factory=dict)
    failure_handling: dict[str, Any] = Field(
        default_factory=lambda: {
            "continue_on_failure": True,
            "max_retries_per_run": 1,
            "skip_project_after_failures": 3,
        }
    )


class ScanDatasetDirectoryRequest(BaseModel):
    base_directory: str


class BatchCreateWorkflowProjectsRequest(BaseModel):
    datasets: list[dict[str, Any]] = Field(default_factory=list)
    shared_config: dict[str, Any] = Field(default_factory=dict)


class PredictWorkflowMultipliersRequest(BaseModel):
    model_id: str | None = None
    model_ids: list[str] | None = None


class SaveFixedLogSpacePreviewRequest(BaseModel):
    schedule: dict[str, Any] = Field(default_factory=dict)


@router.get("", response_model=WorkflowPipelineListResponse)
def list_workflow_pipelines(
    limit: int = Query(default=100, ge=1, le=1000),
    stage: str | None = Query(default=None),
) -> WorkflowPipelineListResponse:
    try:
        items = workflow_pipeline_service.list_workflow_pipelines(limit=limit, stage=stage)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "workflow_pipeline_list_invalid",
                "message": "Workflow pipeline list request is invalid.",
                "details": str(exc),
            },
        ) from exc
    return WorkflowPipelineListResponse(items=items, total=len(items))


@router.post("/scan-directory")
def scan_workflow_dataset_directory(request: ScanDatasetDirectoryRequest) -> dict[str, Any]:
    try:
        return workflow_pipeline_service.scan_dataset_directory(request.base_directory)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workflow_scan_directory_not_found",
                "message": str(exc),
                "base_directory": request.base_directory,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "workflow_scan_directory_invalid",
                "message": str(exc),
                "base_directory": request.base_directory,
            },
        ) from exc


@router.post("/batch-create-projects")
def batch_create_workflow_projects(request: BatchCreateWorkflowProjectsRequest) -> dict[str, Any]:
    return workflow_pipeline_service.batch_create_projects(
        datasets=request.datasets,
        shared_config=request.shared_config,
    )


@router.post("")
def create_workflow_pipeline(request: CreateWorkflowPipelineRequest) -> dict[str, Any]:
    try:
        return workflow_pipeline_service.create_workflow_pipeline(request.model_dump())
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "workflow_pipeline_create_invalid",
                "message": str(exc),
            },
        ) from exc


@router.get("/{pipeline_id}")
def get_workflow_pipeline(pipeline_id: str) -> dict[str, Any]:
    try:
        return workflow_pipeline_service.get_workflow_pipeline(pipeline_id)
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc


@router.put("/{pipeline_id}/config")
def update_workflow_pipeline_config(
    pipeline_id: str,
    request: CreateWorkflowPipelineRequest,
) -> dict[str, Any]:
    try:
        return workflow_pipeline_service.update_workflow_pipeline_config(pipeline_id, request.model_dump())
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc
    except ValueError as exc:
        raise _invalid_action(pipeline_id, exc) from exc


@router.get("/{pipeline_id}/learning-rows")
def get_workflow_pipeline_learning_rows(
    pipeline_id: str,
    include_hard_cap: bool = Query(default=False),
) -> dict[str, Any]:
    try:
        return workflow_pipeline_service.get_learning_rows(
            pipeline_id,
            include_hard_cap=include_hard_cap,
        )
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc


@router.get("/{pipeline_id}/worker-logs")
def get_workflow_pipeline_worker_logs(pipeline_id: str) -> dict[str, Any]:
    try:
        return workflow_pipeline_service.get_worker_logs(pipeline_id)
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc


@router.get("/{pipeline_id}/fixed-log-space-schedule")
def get_workflow_pipeline_fixed_log_space_schedule(pipeline_id: str) -> dict[str, Any]:
    try:
        return build_fixed_log_space_schedule(pipeline_id)
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc


@router.post("/{pipeline_id}/fixed-log-space-schedule/preview")
def preview_workflow_pipeline_fixed_log_space_schedule(
    pipeline_id: str,
    group: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return workflow_pipeline_service.preview_fixed_log_space_schedule(pipeline_id, group=group)
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc
    except ValueError as exc:
        raise _invalid_action(pipeline_id, exc) from exc


@router.post("/{pipeline_id}/fixed-log-space-schedule/save-preview")
def save_workflow_pipeline_fixed_log_space_schedule_preview(
    pipeline_id: str,
    request: SaveFixedLogSpacePreviewRequest,
) -> dict[str, Any]:
    try:
        return workflow_pipeline_service.save_fixed_log_space_schedule_preview(pipeline_id, request.schedule)
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc
    except ValueError as exc:
        raise _invalid_action(pipeline_id, exc) from exc


@router.get("/{pipeline_id}/offline-data-summary")
def get_workflow_pipeline_offline_data_summary(pipeline_id: str) -> dict[str, Any]:
    try:
        return build_offline_data_preparation_summary(pipeline_id)
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc
    except ValueError as exc:
        raise _invalid_action(pipeline_id, exc) from exc


@router.get("/{pipeline_id}/testing-summary")
def get_workflow_pipeline_testing_summary(pipeline_id: str) -> dict[str, Any]:
    try:
        return build_testing_pipeline_summary(pipeline_id)
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc
    except ValueError as exc:
        raise _invalid_action(pipeline_id, exc) from exc


@router.get("/{pipeline_id}/prediction-preview")
def get_workflow_pipeline_prediction_preview(
    pipeline_id: str,
    preview_key: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return workflow_pipeline_service.get_prediction_preview(pipeline_id, preview_key=preview_key)
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc
    except ValueError as exc:
        raise _invalid_action(pipeline_id, exc) from exc


@router.get("/{pipeline_id}/testing-candidate-curves")
def get_workflow_pipeline_testing_candidate_curves(
    pipeline_id: str,
    preview_key: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return build_testing_candidate_curves(pipeline_id, preview_key=preview_key)
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc
    except ValueError as exc:
        raise _invalid_action(pipeline_id, exc) from exc


@router.post("/{pipeline_id}/predict-multipliers")
async def predict_workflow_pipeline_multipliers(
    pipeline_id: str,
    request: PredictWorkflowMultipliersRequest,
) -> dict[str, Any]:
    try:
        return await workflow_pipeline_service.predict_multipliers(pipeline_id, request.model_dump())
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc
    except ValueError as exc:
        raise _invalid_action(pipeline_id, exc) from exc


@router.get("/{pipeline_id}/export-current-test")
async def export_workflow_pipeline_current_test(pipeline_id: str):
    try:
        return await workflow_pipeline_service.export_current_test(pipeline_id)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc
    except ValueError as exc:
        raise _invalid_action(pipeline_id, exc) from exc


@router.post("/{pipeline_id}/start")
def start_workflow_pipeline(pipeline_id: str) -> dict[str, Any]:
    try:
        return workflow_pipeline_service.start_pipeline(pipeline_id)
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc
    except ValueError as exc:
        raise _invalid_action(pipeline_id, exc) from exc


@router.post("/{pipeline_id}/pause")
def pause_workflow_pipeline(pipeline_id: str) -> dict[str, Any]:
    try:
        return workflow_pipeline_service.pause_pipeline(pipeline_id)
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc
    except ValueError as exc:
        raise _invalid_action(pipeline_id, exc) from exc


@router.post("/{pipeline_id}/resume")
def resume_workflow_pipeline(pipeline_id: str) -> dict[str, Any]:
    try:
        return workflow_pipeline_service.resume_pipeline(pipeline_id)
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc
    except ValueError as exc:
        raise _invalid_action(pipeline_id, exc) from exc


@router.post("/{pipeline_id}/stop")
def stop_workflow_pipeline(pipeline_id: str) -> dict[str, Any]:
    try:
        return workflow_pipeline_service.stop_pipeline(pipeline_id)
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc
    except ValueError as exc:
        raise _invalid_action(pipeline_id, exc) from exc


@router.post("/{pipeline_id}/restart")
async def restart_workflow_pipeline(
    pipeline_id: str,
    keep_baseline: bool = Query(default=False),
    keep_log_space_schedule: bool = Query(default=True),
) -> dict[str, Any]:
    try:
        return await workflow_pipeline_service.restart_pipeline(
            pipeline_id,
            keep_baseline=keep_baseline,
            keep_log_space_schedule=keep_log_space_schedule,
        )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc
    except ValueError as exc:
        raise _invalid_action(pipeline_id, exc) from exc


@router.post("/{pipeline_id}/retry-failed")
async def retry_failed_workflow_pipeline(
    pipeline_id: str,
    auto_start: bool = Query(default=True),
    include_hard_cap: bool = Query(default=False),
) -> dict[str, Any]:
    try:
        return await workflow_pipeline_service.retry_failed_runs(
            pipeline_id,
            auto_start=auto_start,
            include_hard_cap=include_hard_cap,
        )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc
    except ValueError as exc:
        raise _invalid_action(pipeline_id, exc) from exc


@router.delete("/{pipeline_id}/runs/{run_id}")
def delete_workflow_pipeline_run(pipeline_id: str, run_id: str) -> dict[str, Any]:
    try:
        return workflow_pipeline_service.delete_pipeline_run(pipeline_id, run_id)
    except FileNotFoundError as exc:
        raise _not_found(pipeline_id, exc) from exc
    except ValueError as exc:
        raise _invalid_action(pipeline_id, exc) from exc


def _not_found(pipeline_id: str, exc: FileNotFoundError) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={
            "code": "workflow_pipeline_not_found",
            "message": str(exc),
            "pipeline_id": pipeline_id,
        },
    )


def _invalid_action(pipeline_id: str, exc: ValueError) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "code": "workflow_pipeline_action_invalid",
            "message": str(exc),
            "pipeline_id": pipeline_id,
        },
    )

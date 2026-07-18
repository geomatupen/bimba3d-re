"""Workflow-facing helpers for existing pipeline records."""
from __future__ import annotations

import logging
import shutil
import time
import uuid
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from bimba3d_backend.app.config import DATA_DIR
from bimba3d_backend.app.services.fixed_log_space_schedule import build_fixed_log_space_config
from bimba3d_backend.app.services import pipeline_learning_rows
from bimba3d_backend.app.services import workflow_test_export
from bimba3d_backend.app.services import workflow_prediction_preview
from bimba3d_backend.app.services import workflow_pipeline_actions
from bimba3d_backend.app.services import training_pipeline_orchestrator
from bimba3d_backend.app.services import training_pipeline_storage
from bimba3d_backend.app.services.training_pipeline_storage import normalise_phase_run_keys, phase_run_count

logger = logging.getLogger(__name__)


def list_workflow_pipelines(*, limit: int = 100, stage: str | None = None) -> list[dict[str, Any]]:
    pipelines = [
        normalise_pipeline_summary(pipeline)
        for pipeline in training_pipeline_storage.list_pipelines(limit=limit)
    ]
    if stage:
        wanted = stage.strip().lower()
        if wanted in {"offline_data", "training_data"}:
            wanted = "offline_data_preparation"
        elif wanted in {"test", "testing"}:
            wanted = "testing_pipeline"
        pipelines = [
            pipeline
            for pipeline in pipelines
            if pipeline.get("workflow_stage") == wanted or pipeline.get("pipeline_type") == stage.strip().lower()
        ]
    return pipelines


def get_workflow_pipeline(pipeline_id: str) -> dict[str, Any]:
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
    if not pipeline:
        raise FileNotFoundError("Pipeline not found")
    return normalise_pipeline_detail(pipeline)


def get_learning_rows(pipeline_id: str) -> dict[str, Any]:
    return pipeline_learning_rows.collect_pipeline_learning_rows(pipeline_id)


def get_worker_logs(pipeline_id: str) -> dict[str, Any]:
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
    if not pipeline:
        raise FileNotFoundError("Pipeline not found")

    config = pipeline.get("config", {})
    pipeline_folder_value = config.get("pipeline_folder")
    if not pipeline_folder_value:
        raise FileNotFoundError("Pipeline folder not found")

    pipeline_folder = Path(pipeline_folder_value)
    if not pipeline_folder.exists():
        raise FileNotFoundError("Pipeline folder not found")

    known_run_ids = {
        str(run.get("run_id"))
        for run in pipeline.get("runs", [])
        if isinstance(run, dict) and run.get("run_id")
    }
    active_run = pipeline.get("active_run") if isinstance(pipeline.get("active_run"), dict) else {}
    active_project = str(active_run.get("project_name") or "")
    active_run_id = str(active_run.get("run_id") or "")
    if active_run_id:
        known_run_ids.add(active_run_id)
    should_filter_to_pipeline_state = bool(known_run_ids)

    logs: list[dict[str, Any]] = []
    for project_dir in pipeline_folder.iterdir():
        if not _is_project_folder(project_dir, pipeline_folder):
            continue

        for log_entry in _iter_processing_logs(project_dir):
            run_id = log_entry.get("run_id")
            if should_filter_to_pipeline_state and run_id and run_id not in known_run_ids:
                continue
            log_file = log_entry["path"]
            try:
                log_lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            if not log_lines:
                continue

            clipped_lines = log_lines[-1000:]
            logs.append(
                {
                    "id": log_entry["id"],
                    "project": log_entry["label"],
                    "project_name": log_entry["project_name"],
                    "run_id": run_id,
                    "log_path": str(log_file),
                    "logs": "\n".join(clipped_lines),
                    "lines": len(log_lines),
                    "modified_at": datetime.fromtimestamp(log_file.stat().st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
                }
            )

    logs.sort(
        key=lambda item: (
            bool(active_run_id and item.get("run_id") == active_run_id and item.get("project_name") == active_project),
            item.get("modified_at") or "",
        ),
        reverse=True,
    )

    return {
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline.get("name"),
        "logs": logs,
        "total_projects": len(logs),
    }


def scan_dataset_directory(base_directory: str) -> dict[str, Any]:
    base_path = Path(base_directory)
    if not base_path.exists():
        raise FileNotFoundError("Directory not found")
    if not base_path.is_dir():
        raise ValueError("Path is not a directory")

    datasets = []
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
        dataset = _scan_dataset_folder(item)
        if dataset is not None:
            datasets.append(dataset)

    datasets.sort(key=lambda item: item["name"])
    return {"datasets": datasets, "total": len(datasets)}


def batch_create_projects(datasets: list[dict[str, Any]], shared_config: dict[str, Any] | None = None) -> dict[str, Any]:
    shared_config = shared_config or {}
    created: list[dict[str, Any]] = []
    existing: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    for dataset in datasets:
        dataset_name = str(dataset.get("name") or "").strip()
        try:
            if not dataset_name:
                raise ValueError("Dataset name is required")

            dataset_path = str(dataset.get("path") or "").strip()
            image_count = int(dataset.get("image_count") or 0)
            project_dir = DATA_DIR / dataset_name

            if project_dir.exists():
                existing_project = _read_existing_project(project_dir, dataset_name, dataset_path, image_count)
                if existing_project is not None:
                    existing.append(existing_project)
                continue

            project_id = str(uuid.uuid4())
            project_dir.mkdir(parents=True, exist_ok=True)
            config = {
                "id": project_id,
                "name": dataset_name,
                "source_dir": dataset_path,
                "created_at": _utc_now(),
                **shared_config,
            }
            (project_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            created.append(
                {
                    "project_id": project_id,
                    "name": dataset_name,
                    "dataset_path": dataset_path,
                    "image_count": image_count,
                    "created": True,
                }
            )
        except Exception as exc:
            logger.error("Failed to create project for %s: %s", dataset_name or "<missing>", exc)
            failed.append({"dataset_name": dataset_name or "<missing>", "error": str(exc)})

    return {"created": created, "existing": existing, "failed": failed}


def create_workflow_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    prepared_config = prepare_pipeline_config(config)
    pipeline = training_pipeline_storage.create_pipeline(prepared_config)
    return normalise_pipeline_detail(pipeline)


def update_workflow_pipeline_config(pipeline_id: str, config: dict[str, Any]) -> dict[str, Any]:
    pipeline = _require_pipeline(pipeline_id)
    if pipeline.get("status") == "running":
        raise ValueError("Cannot update configuration while pipeline is running. Please stop the pipeline first.")

    old_config = pipeline.get("config", {}) if isinstance(pipeline.get("config"), dict) else {}
    new_config = _prepare_pipeline_config(config, preserve_existing_schedule=False)
    _preserve_existing_pipeline_state(new_config, old_config)

    total_runs = _calculate_total_runs(new_config)
    updates = {
        "config": new_config,
        "name": str(new_config.get("name") or pipeline.get("name") or pipeline_id),
        "total_runs": total_runs,
        "current_test_model_id": None,
    }
    old_total_runs = int(pipeline.get("total_runs") or 0)
    updated = training_pipeline_storage.update_pipeline(pipeline_id, updates)
    if not updated:
        raise FileNotFoundError("Pipeline not found")
    updated = training_pipeline_storage.refresh_pipeline_counters(pipeline_id) or updated

    resumable_changes = False
    terminal_statuses = {"completed", "completed_with_failures", "completed_with_hard_caps"}
    if pipeline.get("status") in terminal_statuses:
        resumable_changes = (
            total_runs > old_total_runs
            or _runs_increased(old_config.get("phases") or [], new_config.get("phases") or [])
            or _model_slots_increased(old_config, new_config)
        )
        has_pending_slots = int(updated.get("pending_runs") or 0) > 0
        if resumable_changes and has_pending_slots:
            updated = training_pipeline_storage.update_pipeline(
                pipeline_id,
                {
                    "status": "stopped",
                    "completed_at": None,
                },
            ) or updated

    logger.info("Updated workflow pipeline configuration for %s", pipeline_id)
    return {
        "success": True,
        "message": (
            "New test run slots were added - pipeline is now stopped. Click Resume to run only the pending runs."
            if resumable_changes and int(updated.get("pending_runs") or 0) > 0
            else "Configuration updated. Changes to projects, phases structure, or other settings require a Restart to take effect."
        ),
        "pipeline_id": pipeline_id,
        "total_runs": total_runs,
        "resumable": bool(resumable_changes and int(updated.get("pending_runs") or 0) > 0),
        "applied_source_model_id": new_config.get("source_model_id"),
        "applied_source_model_ids": new_config.get("source_model_ids") or [],
        "pipeline": normalise_pipeline_detail(updated),
    }


def preview_fixed_log_space_schedule(pipeline_id: str, *, group: str | None = None) -> dict[str, Any]:
    pipeline = _require_pipeline(pipeline_id)
    config = pipeline.get("config", {}) if isinstance(pipeline.get("config"), dict) else {}
    preview = build_fixed_log_space_config(dict(config))
    group_key = _normalise_schedule_group_key(group)
    return {
        "success": True,
        "pipeline_id": pipeline_id,
        "message": "Temporary preview generated. It has not been saved for processing.",
        "group": group_key,
        "schedule": preview,
        "editable": _fixed_schedule_is_editable(pipeline),
    }


def save_fixed_log_space_schedule_preview(pipeline_id: str, schedule: dict[str, Any]) -> dict[str, Any]:
    pipeline = _require_pipeline(pipeline_id)
    if pipeline.get("status") == "running" and _pipeline_has_active_non_baseline_run(pipeline):
        raise ValueError("Cannot save preview values while prediction/test runs are running. Stop the pipeline first.")
    _assert_fixed_schedule_editable(pipeline)

    config = dict(pipeline.get("config", {}) if isinstance(pipeline.get("config"), dict) else {})
    validated = _validate_fixed_log_space_schedule(schedule)
    config.update(validated)
    config["multiplier_current_index"] = 0

    updated = training_pipeline_storage.update_pipeline(pipeline_id, {"config": config})
    if not updated:
        raise FileNotFoundError("Pipeline not found")

    return {
        "success": True,
        "pipeline_id": pipeline_id,
        "message": "Preview values saved for processing.",
        "pipeline": normalise_pipeline_detail(updated),
        "schedule": validated,
    }


def prepare_pipeline_config(config: dict[str, Any]) -> dict[str, Any]:
    return _prepare_pipeline_config(config, preserve_existing_schedule=True)


def _prepare_pipeline_config(config: dict[str, Any], *, preserve_existing_schedule: bool) -> dict[str, Any]:
    prepared = normalise_phase_run_keys(dict(config))
    workflow_stage = str(prepared.get("workflow_stage") or "").strip().lower()
    pipeline_type = str(prepared.get("pipeline_type") or "").strip().lower()

    if workflow_stage:
        if workflow_stage == "offline_data_preparation":
            pipeline_type = "offline_data"
        elif workflow_stage == "testing_pipeline":
            pipeline_type = "test"
        else:
            raise ValueError(f"Unsupported workflow stage for pipeline creation: {workflow_stage}")

    if not pipeline_type:
        pipeline_type = "offline_data"
    if pipeline_type == "train":
        pipeline_type = "offline_data"
    if pipeline_type not in {"offline_data", "test"}:
        raise ValueError(f"Unsupported pipeline type: {pipeline_type}")

    prepared["pipeline_type"] = pipeline_type
    _canonicalise_model_selection(prepared)
    _set_restart_metadata(prepared)
    if preserve_existing_schedule and prepared.get("pre_generated_log_multipliers"):
        prepared["multiplier_current_index"] = int(prepared.get("multiplier_current_index") or 0)
    else:
        _set_fixed_multiplier_schedule(prepared)
    return prepared


def start_pipeline(pipeline_id: str) -> dict[str, Any]:
    pipeline = _require_pipeline(pipeline_id)
    if pipeline.get("status") == "running":
        raise ValueError("Pipeline is already running")

    killed_workers = _drain_pipeline_runtime_before_start(pipeline, reason="start")
    updated = training_pipeline_storage.update_pipeline(
        pipeline_id,
        {
            "status": "running",
            "started_at": _utc_now(),
            "completed_at": None,
            "last_error": None,
            "active_run": None,
            "cooldown_active": False,
            "next_run_scheduled_at": None,
            "cooldown_session_id": None,
            "current_phase": 1,
            "current_run": 1,
            "current_project_index": 0,
            "current_test_model_id": None,
        },
    )
    if not updated:
        raise FileNotFoundError("Pipeline not found")

    training_pipeline_orchestrator.start_pipeline_orchestrator(pipeline_id)
    return {
        "status": "running",
        "message": "Pipeline started",
        "killed_workers": killed_workers,
        "pipeline": normalise_pipeline_summary(updated),
    }


def pause_pipeline(pipeline_id: str) -> dict[str, Any]:
    pipeline = _require_pipeline(pipeline_id)
    if pipeline.get("status") != "running":
        raise ValueError("Pipeline is not running")

    updated = training_pipeline_storage.update_pipeline(pipeline_id, {"status": "paused"})
    if not updated:
        raise FileNotFoundError("Pipeline not found")

    training_pipeline_orchestrator.stop_pipeline_orchestrator(pipeline_id)
    killed_workers = _stop_active_pipeline_worker(pipeline)
    updated = training_pipeline_storage.update_pipeline(pipeline_id, {"active_run": None}) or updated
    return {
        "status": "paused",
        "message": "Pipeline paused",
        "killed_workers": killed_workers,
        "pipeline": normalise_pipeline_summary(updated),
    }


def resume_pipeline(pipeline_id: str) -> dict[str, Any]:
    pipeline = _require_pipeline(pipeline_id)
    status = str(pipeline.get("status") or "")
    allowed = {"paused", "stopped", "failed", "completed_with_failures"}
    if status not in allowed:
        raise ValueError(f"Pipeline cannot be resumed from status '{status}'. Must be paused, stopped, failed, or completed_with_failures.")

    config = dict(pipeline.get("config") or {})
    # Do NOT clear retry_mode_active or retry_target_slots here.
    # The orchestrator clears retry mode slot-by-slot as each targeted run completes,
    # and disables retry_mode_active automatically when all slots are consumed.
    # Clearing it here would break mid-session retry resumption for both offline-data
    # and test pipelines.

    killed_workers = _drain_pipeline_runtime_before_start(pipeline, reason="resume")
    updated = training_pipeline_storage.update_pipeline(
        pipeline_id,
        {
            "status": "running",
            "completed_at": None,
            "last_error": None,
            "active_run": None,
            "cooldown_active": False,
            "next_run_scheduled_at": None,
            "cooldown_session_id": None,
            "current_test_model_id": None,
        },
    )
    if not updated:
        raise FileNotFoundError("Pipeline not found")

    training_pipeline_orchestrator.start_pipeline_orchestrator(pipeline_id)
    return {
        "status": "running",
        "message": "Pipeline resumed",
        "killed_workers": killed_workers,
        "pipeline": normalise_pipeline_summary(updated),
    }


def stop_pipeline(pipeline_id: str) -> dict[str, Any]:
    _require_pipeline(pipeline_id)
    updated = training_pipeline_storage.update_pipeline(
        pipeline_id,
        {
            "status": "stopped",
            "completed_at": _utc_now(),
        },
    )
    if not updated:
        raise FileNotFoundError("Pipeline not found")

    training_pipeline_orchestrator.stop_pipeline_orchestrator(pipeline_id)

    from bimba3d_backend.app.services.colmap import stop_all_local_workers

    killed = stop_all_local_workers()
    if killed:
        logger.info("Killed %d active worker process(es) for pipeline stop", killed)

    return {
        "status": "stopped",
        "message": "Pipeline stopped",
        "killed_workers": killed,
        "pipeline": normalise_pipeline_summary(updated),
    }


def _stop_active_pipeline_worker(pipeline: dict[str, Any]) -> int:
    active_run = pipeline.get("active_run") if isinstance(pipeline.get("active_run"), dict) else {}
    config = pipeline.get("config") if isinstance(pipeline.get("config"), dict) else {}
    pipeline_folder = Path(str(config.get("pipeline_folder") or pipeline.get("pipeline_folder") or ""))
    project_name = str(active_run.get("project_name") or "").strip()
    if pipeline_folder and project_name:
        project_dir = pipeline_folder / project_name.replace(" ", "_")
        if project_dir.exists():
            try:
                (project_dir / "stop_requested").write_text("backend", encoding="utf-8")
            except Exception as exc:
                logger.warning("Could not write stop flag for paused pipeline project %s: %s", project_name, exc)

    from bimba3d_backend.app.services.colmap import stop_all_local_workers

    killed = stop_all_local_workers()
    if killed:
        logger.info("Killed %d active worker process(es) for pipeline pause", killed)
    return killed


def _drain_pipeline_runtime_before_start(pipeline: dict[str, Any], *, reason: str) -> int:
    """Best-effort cleanup before start/resume to avoid overlapping local workers.

    The in-memory local-worker registry disappears across backend restarts, so killing
    registered workers is not enough. Writing stop flags to every configured project
    gives any orphaned worker in the pipeline folder a chance to stop before a new
    orchestrator starts.
    """
    pipeline_id = str(pipeline.get("id") or "").strip()
    if pipeline_id:
        try:
            training_pipeline_orchestrator.stop_pipeline_orchestrator(pipeline_id)
        except Exception:
            logger.debug("Failed to stop existing orchestrator before pipeline %s", reason, exc_info=True)

    stop_flags = _write_pipeline_project_stop_flags(pipeline, reason=reason)

    from bimba3d_backend.app.services.colmap import stop_all_local_workers

    killed = stop_all_local_workers()
    if killed:
        logger.info("Killed %d registered local worker process(es) before pipeline %s", killed, reason)
    if stop_flags:
        logger.info("Wrote %d project stop flag(s) before pipeline %s", stop_flags, reason)
        time.sleep(5)
    elif killed:
        time.sleep(2)
    return killed


def _write_pipeline_project_stop_flags(pipeline: dict[str, Any], *, reason: str) -> int:
    config = pipeline.get("config") if isinstance(pipeline.get("config"), dict) else {}
    pipeline_folder_raw = str(config.get("pipeline_folder") or pipeline.get("pipeline_folder") or "").strip()
    if not pipeline_folder_raw:
        return 0
    pipeline_folder = Path(pipeline_folder_raw)
    if not pipeline_folder.exists():
        return 0

    project_names: set[str] = set()
    projects = config.get("projects") if isinstance(config.get("projects"), list) else []
    for project in projects:
        if isinstance(project, dict) and str(project.get("name") or "").strip():
            project_names.add(str(project.get("name")).strip())

    active_run = pipeline.get("active_run") if isinstance(pipeline.get("active_run"), dict) else {}
    if str(active_run.get("project_name") or "").strip():
        project_names.add(str(active_run.get("project_name")).strip())

    written = 0
    for project_name in sorted(project_names):
        project_dir = pipeline_folder / project_name.replace(" ", "_")
        if not project_dir.exists():
            continue
        try:
            (project_dir / "stop_requested").write_text(f"pipeline_{reason}", encoding="utf-8")
            written += 1
        except Exception as exc:
            logger.warning("Could not write stop flag for project %s before pipeline %s: %s", project_name, reason, exc)
    return written


async def restart_pipeline(
    pipeline_id: str,
    *,
    keep_baseline: bool = False,
    keep_log_space_schedule: bool = True,
) -> dict[str, Any]:
    return await workflow_pipeline_actions.restart_pipeline(
        pipeline_id,
        keep_baseline=keep_baseline,
        keep_log_space_schedule=keep_log_space_schedule,
    )


async def retry_failed_runs(
    pipeline_id: str,
    *,
    auto_start: bool = True,
    include_hard_cap: bool = False,
) -> dict[str, Any]:
    return await workflow_pipeline_actions.retry_failed_runs(
        pipeline_id,
        auto_start=auto_start,
        include_hard_cap=include_hard_cap,
    )


def get_prediction_preview(pipeline_id: str, *, preview_key: str | None = None) -> dict[str, Any]:
    pipeline = _require_pipeline(pipeline_id)
    config = pipeline.get("config", {}) if isinstance(pipeline.get("config"), dict) else {}
    pipeline_type = str(config.get("pipeline_type") or pipeline.get("pipeline_type") or "offline_data").strip().lower()
    if pipeline_type != "test":
        raise ValueError("Prediction previews are only available for testing pipelines.")

    previews = pipeline.get("prediction_previews") if isinstance(pipeline.get("prediction_previews"), dict) else {}
    artifacts = pipeline.get("prediction_preview_artifacts") if isinstance(pipeline.get("prediction_preview_artifacts"), dict) else {}

    selected_key = str(preview_key or pipeline.get("latest_prediction_preview_key") or "").strip()
    preview = previews.get(selected_key) if selected_key else None
    if preview is None and previews:
        selected_key, preview = sorted(
            previews.items(),
            key=lambda item: str((item[1] or {}).get("generated_at") or ""),
            reverse=True,
        )[0]

    rows = []
    if isinstance(preview, dict):
        raw_rows = preview.get("results") or preview.get("rows") or []
        rows = raw_rows if isinstance(raw_rows, list) else []

    return {
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline.get("name"),
        "preview_key": selected_key or None,
        "preview": preview,
        "rows": rows,
        "total_rows": len(rows),
        "artifact": artifacts.get(selected_key) if selected_key else None,
        "available_preview_keys": list(previews.keys()),
    }


async def predict_multipliers(pipeline_id: str, request_payload: dict[str, Any]) -> dict[str, Any]:
    return await workflow_prediction_preview.predict_multipliers(pipeline_id, request_payload)


async def export_current_test(pipeline_id: str):
    return await workflow_test_export.export_current_test(pipeline_id)


def delete_pipeline_run(pipeline_id: str, run_id: str) -> dict[str, Any]:
    pipeline = _require_pipeline(pipeline_id)
    if str(pipeline.get("status") or "").lower() == "running":
        raise ValueError("Stop the pipeline before deleting a run.")

    run_id = str(run_id or "").strip()
    if not run_id:
        raise ValueError("Run ID is required.")

    runs = [run for run in pipeline.get("runs", []) if isinstance(run, dict)]
    target = next((run for run in runs if str(run.get("run_id") or "") == run_id), None)
    if target is None:
        raise FileNotFoundError("Run not found in this pipeline.")

    active_run = pipeline.get("active_run") if isinstance(pipeline.get("active_run"), dict) else {}
    if str(active_run.get("run_id") or "") == run_id:
        raise ValueError("Cannot delete the active pipeline run.")

    try:
        target_phase = int(target.get("phase") or 0)
    except (TypeError, ValueError):
        target_phase = 0
    is_baseline = target_phase == 1 or str(target.get("run_name") or "").lower().find("baseline") >= 0
    if is_baseline:
        raise ValueError("Baseline runs cannot be deleted from the pipeline runs table.")

    config = pipeline.get("config", {}) if isinstance(pipeline.get("config"), dict) else {}
    pipeline_folder_value = str(config.get("pipeline_folder") or "").strip()
    if not pipeline_folder_value:
        raise FileNotFoundError("Pipeline folder not found.")
    pipeline_folder = Path(pipeline_folder_value).resolve()
    if not pipeline_folder.exists():
        raise FileNotFoundError("Pipeline folder not found.")

    run_dir = _pipeline_run_dir(pipeline_folder, target)
    if run_dir is None:
        raise FileNotFoundError("Run folder not found.")

    try:
        run_dir.relative_to(pipeline_folder)
    except ValueError as exc:
        raise ValueError("Resolved run folder is outside the pipeline folder.") from exc

    shutil.rmtree(run_dir)

    remaining_runs = [run for run in runs if str(run.get("run_id") or "") != run_id]
    updates: dict[str, Any] = {
        "runs": remaining_runs,
        "total_runs": len(remaining_runs),
    }

    updated = training_pipeline_storage.update_pipeline(pipeline_id, updates)
    if not updated:
        raise FileNotFoundError("Pipeline not found after run deletion.")
    updated = training_pipeline_storage.refresh_pipeline_counters(pipeline_id) or updated

    _repair_project_status_after_run_delete(pipeline_folder, target, remaining_runs)

    logger.info("Deleted pipeline run %s from %s", run_id, pipeline_id)
    return {
        "success": True,
        "message": "Run deleted.",
        "pipeline_id": pipeline_id,
        "run_id": run_id,
        "deleted_run_dir": str(run_dir),
        "pipeline": normalise_pipeline_detail(updated),
    }


def normalise_pipeline_summary(pipeline: dict[str, Any]) -> dict[str, Any]:
    config = pipeline.get("config", {}) if isinstance(pipeline.get("config"), dict) else {}
    pipeline_type = str(config.get("pipeline_type") or pipeline.get("pipeline_type") or "offline_data").strip().lower()
    workflow_stage = "testing_pipeline" if pipeline_type == "test" else "offline_data_preparation"
    pending_runs = pipeline.get("pending_runs", 0)
    # For offline_data pipelines in retry mode, only the explicitly targeted slots
    # will run, so show that count as pending.  For test pipelines the orchestrator
    # also executes all remaining non-targeted pending runs, so keep the real count.
    if bool(config.get("retry_mode_active")) and pipeline_type != "test":
        retry_targets = config.get("retry_target_slots")
        retry_fixed = config.get("retry_fixed_params")
        if isinstance(retry_targets, list):
            pending_runs = len(retry_targets)
        elif isinstance(retry_fixed, dict):
            pending_runs = len(retry_fixed)

    return {
        "id": pipeline.get("id"),
        "name": pipeline.get("name"),
        "workflow_stage": workflow_stage,
        "pipeline_type": pipeline_type,
        "status": pipeline.get("status"),
        "created_at": pipeline.get("created_at"),
        "started_at": pipeline.get("started_at"),
        "completed_at": pipeline.get("completed_at"),
        "updated_at": pipeline.get("updated_at"),
        "total_runs": pipeline.get("total_runs", 0),
        "completed_runs": pipeline.get("completed_runs", 0),
        "failed_runs": pipeline.get("failed_runs", 0),
        "hard_cap_runs": pipeline.get("hard_cap_runs", 0),
        "pending_runs": pending_runs,
        "current_phase": pipeline.get("current_phase", 1),
        "current_run": pipeline.get("current_run", 1),
        "current_project_index": pipeline.get("current_project_index", 0),
        "mean_relative_score": pipeline.get("mean_relative_score"),
        "best_relative_score": pipeline.get("best_relative_score"),
        "success_rate": pipeline.get("success_rate"),
        "last_error": pipeline.get("last_error"),
        "cooldown_active": pipeline.get("cooldown_active", False),
        "next_run_scheduled_at": pipeline.get("next_run_scheduled_at"),
        "source_model_ids": config.get("source_model_ids") or ([config.get("source_model_id")] if config.get("source_model_id") else []),
        "source_training_data_id": config.get("source_training_data_id") or config.get("training_data_target_id"),
        "pipeline_folder": config.get("pipeline_folder"),
    }


def normalise_pipeline_detail(pipeline: dict[str, Any]) -> dict[str, Any]:
    detail = normalise_pipeline_summary(pipeline)
    config = pipeline.get("config", {}) if isinstance(pipeline.get("config"), dict) else {}
    pipeline_folder = str(pipeline.get("pipeline_folder") or "").strip()
    if pipeline_folder and not config.get("pipeline_folder"):
        config = {**config, "pipeline_folder": pipeline_folder}
    config = _with_project_ids_from_pipeline_folder(config)
    active_run = _with_active_run_selection(pipeline, config)
    runs = pipeline.get("runs", [])
    if str(config.get("pipeline_type") or "").strip().lower() == "test":
        runs = _with_run_selection_snapshots(runs, config)
    detail.update(
        {
            "config": config,
            "runs": runs,
            "active_run": active_run,
            "fixed_log_space_schedule": {
                "pre_generated_log_multipliers": config.get("pre_generated_log_multipliers", {}),
                "test_candidate_log_multipliers": config.get("test_candidate_log_multipliers", {}),
                "multiplier_current_index": config.get("multiplier_current_index", 0),
                "fixed_log_space_seed": config.get("fixed_log_space_seed"),
                "test_candidate_seed": config.get("test_candidate_seed"),
                "test_candidate_count": config.get("test_candidate_count"),
                "test_candidate_generation": config.get("test_candidate_generation"),
                "fixed_log_space_bounds": config.get("fixed_log_space_bounds"),
                "fixed_log_space_bounds_source": config.get("fixed_log_space_bounds_source"),
                "restart_version": int(config.get("restart_version") or 0),
                "restart_token": str(config.get("restart_token") or ""),
                "last_restart_at": config.get("last_restart_at"),
            },
        }
    )
    return detail


def _with_run_selection_snapshots(runs: Any, config: dict[str, Any]) -> list[Any]:
    if not isinstance(runs, list):
        return []

    enriched_runs: list[Any] = []
    for run in runs:
        if not isinstance(run, dict):
            enriched_runs.append(run)
            continue
        phase = _safe_int(run.get("phase") or run.get("phase_number"), 1)
        if phase <= 1:
            enriched_runs.append(run)
            continue
        enriched_runs.append(_with_selection_snapshot(run, config))
    return enriched_runs


def _with_active_run_selection(pipeline: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    active_run = pipeline.get("active_run") if isinstance(pipeline.get("active_run"), dict) else None
    if not active_run:
        return None
    return _with_selection_snapshot(active_run, config)


def _with_selection_snapshot(run: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    run_id = str(run.get("run_id") or "").strip()
    project_name = str(run.get("project_name") or run.get("project") or "").strip()
    pipeline_folder_value = str(config.get("pipeline_folder") or "").strip()
    if not run_id or not project_name or not pipeline_folder_value:
        return dict(run)

    snapshot_path = Path(pipeline_folder_value) / project_name.replace(" ", "_") / "runs" / run_id / "retry_snapshot.json"
    snapshot = _read_json_if_exists(snapshot_path)
    if not isinstance(snapshot, dict):
        return dict(run)

    enriched = dict(run)
    for key in (
        "selected_preset",
        "selected_multipliers",
        "selected_multipliers_raw",
        "selected_log_multipliers",
        "candidate_score_checks",
        "initial_params",
    ):
        value = snapshot.get(key)
        if value is not None:
            enriched[key] = value
    return enriched


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _with_project_ids_from_pipeline_folder(config: dict[str, Any]) -> dict[str, Any]:
    projects = config.get("projects") if isinstance(config.get("projects"), list) else []
    pipeline_folder_value = str(config.get("pipeline_folder") or "").strip()
    if not projects or not pipeline_folder_value:
        return config

    pipeline_folder = Path(pipeline_folder_value)
    if not pipeline_folder.exists():
        return config

    changed = False
    next_config = dict(config)
    next_projects: list[Any] = []
    for project in projects:
        if not isinstance(project, dict):
            next_projects.append(project)
            continue
        next_project = dict(project)
        if not next_project.get("project_id"):
            project_name = str(next_project.get("name") or "").strip()
            project_config = _read_project_config(pipeline_folder, project_name)
            project_id = project_config.get("id") or project_config.get("project_id") if isinstance(project_config, dict) else None
            if project_id:
                next_project["project_id"] = str(project_id)
                changed = True
        next_projects.append(next_project)

    if changed:
        next_config["projects"] = next_projects
        return next_config
    return config


def _read_project_config(pipeline_folder: Path, project_name: str) -> dict[str, Any] | None:
    if not project_name:
        return None
    project_dir = pipeline_folder / project_name.replace(" ", "_")
    config_path = project_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _assert_fixed_schedule_editable(pipeline: dict[str, Any]) -> None:
    runs = pipeline.get("runs") if isinstance(pipeline.get("runs"), list) else []
    has_non_baseline_runs = any(
        isinstance(run, dict)
        and int(run.get("phase") or 0) > 1
        and str(run.get("status") or "").lower() in {"running", "success", "completed", "partial_completed", "failed"}
        for run in runs
    )
    if has_non_baseline_runs:
        raise ValueError(
            "Cannot save preview values because exploration/test runs already exist. "
            "Only pipelines with sparse/COLMAP work and baseline runs may update the fixed log-space schedule."
        )


def _pipeline_has_active_non_baseline_run(pipeline: dict[str, Any]) -> bool:
    active_run = pipeline.get("active_run") if isinstance(pipeline.get("active_run"), dict) else {}
    try:
        active_phase = int(active_run.get("phase") or pipeline.get("current_phase") or 1)
    except (TypeError, ValueError):
        active_phase = 1
    return active_phase > 1


def _fixed_schedule_is_editable(pipeline: dict[str, Any]) -> bool:
    try:
        _assert_fixed_schedule_editable(pipeline)
    except ValueError:
        return False
    return True


def _validate_fixed_log_space_schedule(schedule: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schedule, dict):
        raise ValueError("Preview schedule is invalid.")

    multipliers = schedule.get("pre_generated_log_multipliers")
    if not isinstance(multipliers, dict) or not multipliers:
        raise ValueError("Preview schedule has no generated multiplier values.")

    validated_multipliers: dict[str, list[float]] = {}
    for group in ("geometry_lr", "appearance_lr", "scale_lr"):
        raw_values = multipliers.get(group)
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"Preview schedule is missing values for {group}.")
        values: list[float] = []
        for value in raw_values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Preview schedule has an invalid value for {group}.")
            parsed = float(value)
            if parsed <= 0:
                raise ValueError(f"Preview schedule values must be positive for {group}.")
            values.append(parsed)
        validated_multipliers[group] = values

    validated = {
        "pre_generated_log_multipliers": validated_multipliers,
        "test_candidate_log_multipliers": schedule.get("test_candidate_log_multipliers") if isinstance(schedule.get("test_candidate_log_multipliers"), dict) else {},
        "fixed_log_space_seed": schedule.get("fixed_log_space_seed"),
        "fixed_log_space_group_seeds": schedule.get("fixed_log_space_group_seeds") if isinstance(schedule.get("fixed_log_space_group_seeds"), dict) else None,
        "test_candidate_seed": schedule.get("test_candidate_seed"),
        "test_candidate_count": schedule.get("test_candidate_count"),
        "test_candidate_generation": schedule.get("test_candidate_generation"),
        "fixed_log_space_generated_at": schedule.get("fixed_log_space_generated_at"),
        "fixed_log_space_mode": schedule.get("fixed_log_space_mode"),
        "fixed_log_space_method": schedule.get("fixed_log_space_method"),
        "fixed_log_space_interval_count": schedule.get("fixed_log_space_interval_count"),
        "fixed_log_space_bounds": schedule.get("fixed_log_space_bounds"),
        "fixed_log_space_bounds_source": schedule.get("fixed_log_space_bounds_source"),
        "fixed_log_space_phase_number": schedule.get("fixed_log_space_phase_number"),
    }
    return validated


def _normalise_schedule_group_key(group: str | None) -> str | None:
    if not group:
        return None
    clean = str(group).strip().lower()
    aliases = {
        "geometry": "geometry_lr",
        "geometry_lr": "geometry_lr",
        "geometry_lr_mult": "geometry_lr",
        "appearance": "appearance_lr",
        "appearance_lr": "appearance_lr",
        "appearance_lr_mult": "appearance_lr",
        "densification": "scale_lr",
        "scale": "scale_lr",
        "scale_lr": "scale_lr",
        "densification_mult": "scale_lr",
    }
    if clean not in aliases:
        raise ValueError(f"Unsupported schedule group: {group}")
    return aliases[clean]


def _canonicalise_model_selection(config: dict[str, Any]) -> None:
    pipeline_type = str(config.get("pipeline_type") or "offline_data").strip().lower()
    if pipeline_type != "test":
        config["source_model_id"] = None
        config["source_model_ids"] = None
        return

    raw_ids = config.get("source_model_ids") or []
    normalised_ids: list[str] = []
    seen: set[str] = set()
    for model_id in raw_ids:
        clean_id = str(model_id or "").strip()
        if clean_id and clean_id not in seen:
            normalised_ids.append(clean_id)
            seen.add(clean_id)

    if not normalised_ids:
        single_id = str(config.get("source_model_id") or "").strip()
        if single_id:
            normalised_ids = [single_id]

    config["source_model_ids"] = normalised_ids if normalised_ids else None
    config["source_model_id"] = normalised_ids[0] if normalised_ids else None


def _set_restart_metadata(config: dict[str, Any]) -> None:
    config["restart_version"] = int(config.get("restart_version") or 0)
    config["restart_token"] = str(config.get("restart_token") or uuid.uuid4().hex)
    config["last_restart_at"] = config.get("last_restart_at")


def _set_fixed_multiplier_schedule(config: dict[str, Any]) -> None:
    config.update(build_fixed_log_space_config(config))


def _preserve_existing_pipeline_state(new_config: dict[str, Any], old_config: dict[str, Any]) -> None:
    if old_config.get("pipeline_folder") and not new_config.get("pipeline_folder"):
        new_config["pipeline_folder"] = old_config.get("pipeline_folder")

    new_config["restart_version"] = int(old_config.get("restart_version") or 0)
    if old_config.get("restart_token"):
        new_config["restart_token"] = old_config.get("restart_token")
    if old_config.get("last_restart_at"):
        new_config["last_restart_at"] = old_config.get("last_restart_at")

    # Config save must not advance or regenerate the fixed log-space schedule.
    if old_config.get("pre_generated_log_multipliers"):
        new_config["pre_generated_log_multipliers"] = old_config.get("pre_generated_log_multipliers")
    new_config["multiplier_current_index"] = int(old_config.get("multiplier_current_index") or 0)


def _calculate_total_runs(config: dict[str, Any]) -> int:
    phases = config.get("phases") or []
    projects = config.get("projects") or []
    pipeline_type = str(config.get("pipeline_type") or "offline_data").strip().lower()
    model_count = len(config.get("source_model_ids") or []) or (1 if config.get("source_model_id") else 1)

    total_runs = 0
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        runs_per_project = phase_run_count(phase)
        phase_runs = runs_per_project * len(projects)
        if pipeline_type == "test" and int(phase.get("phase_number", 1) or 1) > 1 and model_count > 1:
            phase_runs *= model_count
        total_runs += phase_runs
    return total_runs


def _runs_increased(old_phases: list[Any], new_phases: list[Any]) -> bool:
    old_by_phase = {
        phase.get("phase_number"): phase_run_count(phase)
        for phase in old_phases
        if isinstance(phase, dict)
    }
    for phase in new_phases:
        if not isinstance(phase, dict):
            continue
        phase_number = phase.get("phase_number")
        new_runs = phase_run_count(phase)
        old_runs = old_by_phase.get(phase_number, 1)
        if new_runs > old_runs:
            return True
    return False


def _model_slots_increased(old_config: dict[str, Any], new_config: dict[str, Any]) -> bool:
    old_type = str(old_config.get("pipeline_type") or "offline_data").strip().lower()
    new_type = str(new_config.get("pipeline_type") or "offline_data").strip().lower()
    if old_type != "test" and new_type != "test":
        return False
    old_models = set(_selected_model_ids(old_config))
    new_models = set(_selected_model_ids(new_config))
    return bool(new_models - old_models)


def _selected_model_ids(config: dict[str, Any]) -> list[str]:
    raw = config.get("source_model_ids")
    values = raw if isinstance(raw, list) else []
    if not values and config.get("source_model_id"):
        values = [config.get("source_model_id")]
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        model_id = str(value or "").strip()
        if model_id and model_id not in seen:
            out.append(model_id)
            seen.add(model_id)
    return out


def _scan_dataset_folder(folder_path: Path) -> dict[str, Any] | None:
    if not folder_path.is_dir():
        return None

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".dng"}
    image_files = [path for path in folder_path.glob("*") if path.suffix.lower() in image_exts]
    if not image_files:
        return None

    total_size = sum(path.stat().st_size for path in image_files)
    return {
        "name": folder_path.name,
        "path": str(folder_path.absolute()),
        "image_count": len(image_files),
        "size_mb": round(total_size / (1024 * 1024), 2),
        "has_images": True,
    }


def _read_existing_project(project_dir: Path, dataset_name: str, dataset_path: str, image_count: int) -> dict[str, Any] | None:
    config_path = project_dir / "config.json"
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    return {
        "project_id": config.get("id"),
        "name": dataset_name,
        "dataset_path": dataset_path,
        "image_count": image_count,
        "created": False,
    }


def _require_pipeline(pipeline_id: str) -> dict[str, Any]:
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
    if not pipeline:
        raise FileNotFoundError("Pipeline not found")
    return pipeline


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_project_folder(project_dir: Path, pipeline_folder: Path) -> bool:
    if not project_dir.is_dir():
        return False
    if project_dir.name in {"shared_models", "training_pipelines"}:
        return False
    if (project_dir / "pipeline.json").exists() and project_dir == pipeline_folder:
        return False
    return True


def _pipeline_run_dir(pipeline_folder: Path, run: dict[str, Any]) -> Path | None:
    run_id = str(run.get("run_id") or "").strip()
    project_name = str(run.get("project_name") or run.get("project") or "").strip()
    if not run_id:
        return None

    candidate_dirs: list[Path] = []
    if project_name:
        candidate_dirs.append(pipeline_folder / project_name.replace(" ", "_") / "runs" / run_id)
        candidate_dirs.append(pipeline_folder / project_name / "runs" / run_id)

    for candidate in candidate_dirs:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    matches = [path.resolve() for path in pipeline_folder.glob(f"*/runs/{run_id}") if path.is_dir()]
    return matches[0] if len(matches) == 1 else None


def _repair_project_status_after_run_delete(
    pipeline_folder: Path,
    deleted_run: dict[str, Any],
    remaining_runs: list[dict[str, Any]],
) -> None:
    project_name = str(deleted_run.get("project_name") or deleted_run.get("project") or "").strip()
    deleted_run_id = str(deleted_run.get("run_id") or "").strip()
    if not project_name or not deleted_run_id:
        return

    status_path = pipeline_folder / project_name.replace(" ", "_") / "status.json"
    if not status_path.exists():
        return

    try:
        status = json.loads(status_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return
    if not isinstance(status, dict) or str(status.get("current_run_id") or "") != deleted_run_id:
        return

    candidates = [
        run
        for run in remaining_runs
        if str(run.get("project_name") or run.get("project") or "").strip() == project_name
        and str(run.get("run_id") or "").strip()
    ]
    if not candidates:
        status["current_run_id"] = None
        if isinstance(status.get("live_metrics"), dict):
            status["live_metrics"]["run_id"] = None
    else:
        latest = max(candidates, key=lambda run: str(run.get("completed_at") or run.get("timestamp") or ""))
        latest_run_id = str(latest.get("run_id") or "")
        status["current_run_id"] = latest_run_id
        if isinstance(status.get("live_metrics"), dict):
            status["live_metrics"]["run_id"] = latest_run_id
    status["updated_after_run_delete_at"] = _utc_now()
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def _iter_processing_logs(project_dir: Path):
    runs_dir = project_dir / "runs"
    if not runs_dir.exists():
        root_log = project_dir / "processing.log"
        if root_log.exists():
            yield {
                "id": f"{project_dir.name}:project",
                "label": project_dir.name,
                "project_name": project_dir.name,
                "run_id": None,
                "path": root_log,
            }
        return

    run_dirs = sorted(
        (item for item in runs_dir.iterdir() if item.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        run_log = run_dir / "processing.log"
        if run_log.exists():
            yield {
                "id": f"{project_dir.name}:{run_dir.name}",
                "label": f"{project_dir.name} / {run_dir.name}",
                "project_name": project_dir.name,
                "run_id": run_dir.name,
                "path": run_log,
            }

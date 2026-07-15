"""Training pipeline storage service using file-based JSON storage."""
from __future__ import annotations

import json
import os
import time
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from bimba3d_backend.app.config import DATA_DIR

PIPELINES_DIR = DATA_DIR / "training_pipelines"
PIPELINES_DIR.mkdir(parents=True, exist_ok=True)
_WRITE_LOCK = threading.RLock()


def _pipeline_path(pipeline_id: str) -> Path:
    """Get path to pipeline JSON file."""
    return PIPELINES_DIR / f"{pipeline_id}.json"


def _pipeline_state_path(pipeline: dict[str, Any]) -> Path | None:
    config = pipeline.get("config") if isinstance(pipeline.get("config"), dict) else {}
    pipeline_folder = str(config.get("pipeline_folder") or "").strip()
    if not pipeline_folder:
        return None
    return Path(pipeline_folder) / "pipeline_state.json"


def _pipeline_registry_payload(pipeline: dict[str, Any], state_path: Path | None) -> dict[str, Any]:
    config = pipeline.get("config") if isinstance(pipeline.get("config"), dict) else {}
    return {
        "id": pipeline.get("id"),
        "name": pipeline.get("name"),
        "pipeline_type": pipeline.get("pipeline_type"),
        "workflow_stage": pipeline.get("workflow_stage"),
        "status": pipeline.get("status"),
        "created_at": pipeline.get("created_at"),
        "updated_at": pipeline.get("updated_at"),
        "pipeline_folder": config.get("pipeline_folder"),
        "pipeline_state_path": str(state_path) if state_path else None,
        "registry_only": bool(state_path),
    }


def _write_pipeline_record(pipeline_id: str, pipeline: dict[str, Any]) -> None:
    state_path = _pipeline_state_path(pipeline)
    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(state_path, pipeline)
        _write_json_atomic(_pipeline_path(pipeline_id), _pipeline_registry_payload(pipeline, state_path))
        return

    _write_json_atomic(_pipeline_path(pipeline_id), pipeline)


def _load_pipeline_record(path: Path) -> Optional[dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None

    state_path_value = payload.get("pipeline_state_path")
    if state_path_value:
        state_path = Path(str(state_path_value))
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8-sig") as f:
                state_payload = json.load(f)
            return state_payload if isinstance(state_payload, dict) else None
        if payload.get("registry_only"):
            return None

    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so readers never observe a truncated file."""
    # Use a unique temp file to avoid writer collisions across threads/processes.
    temp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")

    # Serialize writers in-process to reduce rename races on Windows.
    with _WRITE_LOCK:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Windows may briefly lock either source/target file. Retry a few times.
        attempts = 6
        for i in range(attempts):
            try:
                temp_path.replace(path)
                return
            except PermissionError:
                if i == attempts - 1:
                    raise
                time.sleep(0.05 * (i + 1))

        # Defensive cleanup when replace loop exits unexpectedly.
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def _timestamp_now() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def phase_run_count(phase: dict[str, Any]) -> int:
    """Return the number of project runs configured for a phase."""
    value = phase.get("exploration_runs_per_project", 1)
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


def normalise_phase_run_keys(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize run-based phase keys for active pipeline JSON."""
    phases = config.get("phases")
    if not isinstance(phases, list):
        return config
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        phase["exploration_runs_per_project"] = phase_run_count(phase)
    return config


def create_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Create a new training pipeline.

    Args:
        config: Pipeline configuration including projects, phases, thermal settings

    Returns:
        Complete pipeline state with metadata
    """
    pipeline_id = f"pipeline_{uuid.uuid4().hex[:12]}"

    # Determine where to create pipeline folder
    pipeline_directory = config.get("pipeline_directory")
    if pipeline_directory:
        # User specified custom location
        pipeline_root = Path(pipeline_directory)
    else:
        # Default: same location as DATA_DIR
        pipeline_root = DATA_DIR

    # Create pipeline folder with sanitized name (replace spaces with underscores)
    pipeline_name = config.get("name", f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    sanitized_name = pipeline_name.replace(" ", "_")
    pipeline_folder = pipeline_root / sanitized_name
    pipeline_folder.mkdir(parents=True, exist_ok=True)

    # Store pipeline folder path in config for orchestrator
    config["pipeline_folder"] = str(pipeline_folder)
    config = normalise_phase_run_keys(config)

    # Calculate total runs.
    total_runs = 0
    pipeline_type = config.get("pipeline_type", "offline_data")
    model_count = len(config.get("source_model_ids") or []) or (1 if config.get("source_model_id") else 1)
    for phase in config.get("phases", []):
        runs_per_project = phase_run_count(phase)
        project_count = len(config.get("projects", []))
        phase_runs = runs_per_project * project_count
        # For test pipelines, Phase 2+ runs once per model
        if pipeline_type == "test" and phase.get("phase_number", 1) > 1 and model_count > 1:
            phase_runs *= model_count
        total_runs += phase_runs

    pipeline = {
        "id": pipeline_id,
        "name": config.get("name", f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        "status": "pending",
        "pipeline_type": config.get("pipeline_type", "offline_data"),
        "created_at": _timestamp_now(),
        "started_at": None,
        "completed_at": None,

        # Configuration
        "config": config,

        # Progress tracking
        "current_phase": 1,
        "current_run": 1,
        "current_project_index": 0,
        "current_test_model_id": None,
        "total_runs": total_runs,
        "completed_runs": 0,
        "failed_runs": 0,
        "hard_cap_runs": 0,
        "pending_runs": total_runs,

        # Statistics
        "mean_relative_score": None,
        "success_rate": None,
        "best_relative_score": None,

        # Thermal management
        "last_run_ended_at": None,
        "next_run_scheduled_at": None,
        "cooldown_active": False,

        # Error handling
        "last_error": None,
        "retry_count": 0,

        # Run history
        "runs": [],
    }

    # Save full state beside the selected pipeline folder; keep only a small
    # registry pointer under DATA_DIR so pipelines remain discoverable.
    _write_pipeline_record(pipeline_id, pipeline)

    # Also save pipeline.json marker in the pipeline folder so it's not listed as a project
    pipeline_marker = pipeline_folder / "pipeline.json"
    with open(pipeline_marker, "w") as f:
        json.dump({
            "pipeline_id": pipeline_id,
            "pipeline_name": pipeline["name"],
            "created_at": pipeline["created_at"],
        }, f, indent=2)

    return pipeline


def get_pipeline(pipeline_id: str) -> Optional[dict[str, Any]]:
    """Load pipeline by ID."""
    path = _pipeline_path(pipeline_id)
    if not path.exists():
        return None

    last_error: Exception | None = None
    with _WRITE_LOCK:
        for _ in range(3):
            try:
                pipeline = _load_pipeline_record(path)
                if pipeline and isinstance(pipeline.get("config"), dict):
                    pipeline["config"] = normalise_phase_run_keys(pipeline["config"])
                return pipeline
            except json.JSONDecodeError as exc:
                last_error = exc
                time.sleep(0.05)

    if last_error is not None:
        raise last_error
    return None


def update_pipeline(pipeline_id: str, updates: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Update pipeline with new data."""
    with _WRITE_LOCK:
        pipeline = get_pipeline(pipeline_id)
        if not pipeline:
            return None

        if isinstance(updates.get("config"), dict):
            updates = dict(updates)
            updates["config"] = normalise_phase_run_keys(updates["config"])

        pipeline.update(updates)

        pipeline["updated_at"] = _timestamp_now()
        _write_pipeline_record(pipeline_id, pipeline)

        return pipeline


def refresh_pipeline_counters(pipeline_id: str) -> Optional[dict[str, Any]]:
    """Recalculate run counters after metadata changes such as total_runs updates."""
    with _WRITE_LOCK:
        pipeline = get_pipeline(pipeline_id)
        if not pipeline:
            return None
        _refresh_run_counters(pipeline)
        _refresh_run_statistics(pipeline)
        pipeline["updated_at"] = _timestamp_now()
        _write_pipeline_record(pipeline_id, pipeline)
        return pipeline


def list_pipelines(limit: int = 50) -> list[dict[str, Any]]:
    """List all pipelines, most recent first."""
    pipelines = []

    for path in PIPELINES_DIR.glob("pipeline_*.json"):
        try:
            pipeline_id = path.stem
            pipeline = get_pipeline(pipeline_id)
            if pipeline:
                pipelines.append(pipeline)
        except Exception:
            continue

    # Sort by created_at descending
    pipelines.sort(key=lambda p: p.get("created_at", ""), reverse=True)

    return pipelines[:limit]


def add_run_result(pipeline_id: str, run_result: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Add a run result to pipeline history and update statistics."""
    with _WRITE_LOCK:
        pipeline = get_pipeline(pipeline_id)
        if not pipeline:
            return None

        runs = pipeline.setdefault("runs", [])
        if not isinstance(runs, list):
            runs = []
            pipeline["runs"] = runs

        run_id = str(run_result.get("run_id") or "").strip()
        existing_index = next(
            (
                index
                for index, existing in enumerate(runs)
                if isinstance(existing, dict)
                and run_id
                and str(existing.get("run_id") or "").strip() == run_id
            ),
            None,
        )
        if existing_index is None:
            runs.append(run_result)
        else:
            runs[existing_index] = {**runs[existing_index], **run_result}

        _refresh_run_counters(pipeline)
        _refresh_run_statistics(pipeline)

        pipeline["updated_at"] = _timestamp_now()
        _write_pipeline_record(pipeline_id, pipeline)

        return pipeline


def _refresh_run_counters(pipeline: dict[str, Any]) -> None:
    runs = [run for run in pipeline.get("runs", []) if isinstance(run, dict)]
    completed_slots: set[tuple[Any, int, int, str | None]] = set()
    failed_slots: set[tuple[Any, int, int, str | None]] = set()
    hard_cap_slots: set[tuple[Any, int, int, str | None]] = set()

    for run in runs:
        slot_key = _run_slot_key(run)
        status = str(run.get("status") or "").lower()
        if status in {"success", "partial_completed"} and not _is_hard_cap_run(run):
            completed_slots.add(slot_key)
        elif _is_hard_cap_run(run):
            hard_cap_slots.add(slot_key)
        elif status == "failed":
            failed_slots.add(slot_key)

    # A successful retry supersedes older failed/hard-cap attempts for that slot.
    hard_cap_slots -= completed_slots
    failed_slots -= completed_slots
    failed_slots -= hard_cap_slots

    total_runs = int(pipeline.get("total_runs") or 0)
    pipeline["completed_runs"] = len(completed_slots)
    pipeline["failed_runs"] = len(failed_slots)
    pipeline["hard_cap_runs"] = len(hard_cap_slots)
    pipeline["pending_runs"] = max(0, total_runs - len(completed_slots) - len(failed_slots) - len(hard_cap_slots))


def _run_slot_key(run: dict[str, Any]) -> tuple[Any, int, int, str | None]:
    project = run.get("project_name") or run.get("project") or run.get("project_id")
    try:
        phase = int(run.get("phase") or 0)
    except (TypeError, ValueError):
        phase = 0
    try:
        phase_run = int(run.get("run") or run.get("phase_run") or 0)
    except (TypeError, ValueError):
        phase_run = 0
    model_id = run.get("test_model_id") or run.get("source_model_id")
    return project, phase, phase_run, str(model_id) if model_id else None


def _is_hard_cap_run(run: dict[str, Any]) -> bool:
    status = str(run.get("status") or "").lower()
    if status == "hard_cap_reached":
        return True
    if run.get("gaussian_cap_reached") is True:
        return True
    reason = str(run.get("reason") or run.get("partial_reason") or "").lower()
    if reason == "gaussian_hard_cap_reached":
        return True
    text = " ".join(str(run.get(key) or "") for key in ("error", "message", "remarks")).lower()
    return "gaussian" in text and "hard cap" in text


def _refresh_run_statistics(pipeline: dict[str, Any]) -> None:
    runs = [run for run in pipeline.get("runs", []) if isinstance(run, dict)]
    scores = [run["score"] for run in runs if isinstance(run.get("score"), (int, float))]
    pipeline["mean_relative_score"] = sum(scores) / len(scores) if scores else None
    pipeline["best_relative_score"] = max(scores) if scores else None

    if runs:
        successful_runs = [run for run in runs if run.get("status") in {"success", "partial_completed"}]
        pipeline["success_rate"] = (len(successful_runs) / len(runs)) * 100
    else:
        pipeline["success_rate"] = None


def delete_pipeline(pipeline_id: str) -> bool:
    """Delete a pipeline and its folder."""
    import shutil
    import logging

    logger = logging.getLogger(__name__)
    path = _pipeline_path(pipeline_id)
    pipeline = get_pipeline(pipeline_id)
    if not pipeline:
        return False

    # Read pipeline config to get folder path
    try:
        pipeline_folder = pipeline.get("config", {}).get("pipeline_folder")

        # Delete the pipeline folder if it exists
        if pipeline_folder:
            folder_path = Path(pipeline_folder)
            if folder_path.exists():
                try:
                    shutil.rmtree(folder_path)
                    logger.info(f"Deleted pipeline folder: {folder_path}")
                except Exception as e:
                    logger.error(f"Failed to delete pipeline folder {folder_path}: {e}")

        # Also clean up any symlinks in DATA_DIR for pipeline projects
        if "projects" in pipeline.get("config", {}):
            for project in pipeline["config"]["projects"]:
                project_name = project.get("name")
                if project_name:
                    project_dir = Path(pipeline_folder) / project_name if pipeline_folder else None
                    if project_dir and project_dir.exists():
                        # Read project config to get UUID
                        config_file = project_dir / "config.json"
                        if config_file.exists():
                            try:
                                with open(config_file, "r") as f:
                                    proj_config = json.load(f)
                                project_id = proj_config.get("id")
                                if project_id:
                                    # Remove symlink in DATA_DIR
                                    symlink = DATA_DIR / project_id
                                    if symlink.exists():
                                        try:
                                            symlink.unlink()
                                            logger.info(f"Deleted project symlink: {symlink}")
                                        except Exception as e:
                                            logger.warning(f"Failed to delete symlink {symlink}: {e}")
                            except Exception as e:
                                logger.warning(f"Failed to clean up symlinks for project {project_name}: {e}")

    except Exception as e:
        logger.error(f"Failed to read pipeline config during deletion: {e}")

    # Delete the pipeline metadata JSON
    if path.exists():
        path.unlink()
    return True


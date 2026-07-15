import logging
import os
import threading
import json
from datetime import datetime
from pathlib import Path
from bimba3d_backend.app.services import colmap, storage, status
from bimba3d_backend.app.services.worker_mode import resolve_worker_mode

logger = logging.getLogger(__name__)

_ACTIVE_LOCAL_PROJECTS: set[str] = set()
_ACTIVE_LOCAL_LOCK = threading.Lock()


def _attach_scoped_file_handler(log_path: Path) -> logging.FileHandler:
    """Attach a file handler for one pipeline invocation."""
    resolved = log_path.resolve()
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(resolved, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(file_handler)
    return file_handler


def _detach_scoped_file_handlers(handlers: list[logging.FileHandler]) -> None:
    root_logger = logging.getLogger()
    for handler in handlers:
        try:
            root_logger.removeHandler(handler)
        finally:
            handler.close()


def _mark_active_sparse_shared_version_if_ready(project_id: str, params: dict, final_status: dict) -> None:
    """Advance active sparse shared version only after successful base COLMAP runs."""
    try:
        if not isinstance(final_status, dict):
            return
        if str(final_status.get("status")) not in {"completed", "done"}:
            return

        stage = str(params.get("stage") or "full")
        if stage not in {"full", "colmap_only"}:
            return

        run_id = str(params.get("run_id") or "").strip()
        status_info = status.get_status(project_id)
        base_session_id = str(status_info.get("base_session_id") or "").strip()
        if not run_id or not base_session_id or run_id != base_session_id:
            return

        project_dir = storage.get_project_dir(project_id)
        shared_path = project_dir / "shared_config.json"
        if not shared_path.exists():
            return

        with open(shared_path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
        if not isinstance(doc, dict):
            return

        current_version = doc.get("version")
        if not isinstance(current_version, int) or current_version < 1:
            current_version = 1

        shared_payload = doc.get("shared") if isinstance(doc.get("shared"), dict) else {}

        doc["active_sparse_version"] = current_version
        doc["active_sparse_updated_at"] = datetime.utcnow().isoformat() + "Z"
        doc["active_shared"] = json.loads(json.dumps(shared_payload))

        tmp_path = shared_path.with_suffix(shared_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, indent=2)
        tmp_path.replace(shared_path)
    except Exception as exc:
        logger.warning("Failed to mark active sparse shared version for %s: %s", project_id, exc)


def is_local_project_active(project_id: str) -> bool:
    with _ACTIVE_LOCAL_LOCK:
        return project_id in _ACTIVE_LOCAL_PROJECTS


def _set_local_project_active(project_id: str, active: bool) -> None:
    with _ACTIVE_LOCAL_LOCK:
        if active:
            _ACTIVE_LOCAL_PROJECTS.add(project_id)
        else:
            _ACTIVE_LOCAL_PROJECTS.discard(project_id)


def run_full_pipeline(project_id: str, params: dict | None = None):
    """
    Run the full Gaussian Splatting pipeline.
    Includes error handling and status tracking.
    
    Runtime mode resolution order:
    1) params["worker_mode"] ("docker" | "local")
    2) env WORKER_MODE
    3) legacy env USE_DOCKER_WORKER
    4) default "docker"
    """
    params = params or {}
    # Extract mode from params (default to baseline if not specified)
    mode = params.get("mode", "baseline")
    max_steps = params.get("max_steps")
    stage = params.get("stage", "full")
    worker_mode = resolve_worker_mode(params.get("worker_mode"))
    use_docker = worker_mode == "docker"
    params["worker_mode"] = worker_mode

    logger.info(
        "Pipeline runtime selected: worker_mode=%s (legacy USE_DOCKER_WORKER=%s)",
        worker_mode,
        os.getenv("USE_DOCKER_WORKER"),
    )
    scoped_log_handlers: list[logging.FileHandler] = []
    
    # Configure per-project file logging for local mode only.
    # In Docker mode, the worker writes processing.log directly.
    if not use_docker:
        try:
            from bimba3d_backend.app.services import storage as _storage
            from pathlib import Path
            # Check if pipeline specified a custom project directory
            _project_dir_override = params.get("project_dir_override")
            if _project_dir_override:
                _project_dir = Path(_project_dir_override)
                logger.info(f"Using user-specified project directory: {_project_dir}")
            else:
                _project_dir = _storage.get_project_dir(project_id)
            _log_file = _project_dir / "processing.log"
            _log_file.parent.mkdir(parents=True, exist_ok=True)
            scoped_log_handlers.append(_attach_scoped_file_handler(_log_file))
            logger.info(f"Initialized project log file: {_log_file}")

            _run_id = str(params.get("run_id") or "").strip()
            if _run_id:
                _run_log_file = _project_dir / "runs" / _run_id / "processing.log"
                _run_log_file.parent.mkdir(parents=True, exist_ok=True)
                scoped_log_handlers.append(_attach_scoped_file_handler(_run_log_file))
                logger.info(f"Initialized run log file: {_run_log_file}")
        except Exception as e:
            logger.warning(f"Failed to initialize project log file: {e}")

    # Use Docker worker for processing
    if use_docker:
        logger.info(f"Using DOCKER WORKER for project {project_id} (mode: {mode})")
        try:
            # Pre-Docker: provide useful, user-visible substep messages
            logger.info("DOCKER: Preparing Docker worker (check image, GPU, volumes)...")
            status.update_status(
                project_id,
                "processing",
                progress=5,
                mode=mode,
                maxSteps=max_steps,
                stage="docker",
                stage_progress=10,
                message="Preparing Docker worker (checking image, GPU, and volumes)...",
            )

            # Hint that image may be pulled/build required
            try:
                import subprocess as _sp
                _sp.run(["docker", "image", "inspect", "bimba3d-worker:latest"], check=True, capture_output=True)
                status.update_status(
                    project_id,
                    "processing",
                    progress=7,
                    stage="docker",
                    stage_progress=20,
                    message="Docker image found locally. Starting worker container...",
                )
                logger.info("DOCKER: Image present locally")
            except Exception:
                status.update_status(
                    project_id,
                    "processing",
                    progress=7,
                    stage="docker",
                    stage_progress=20,
                    message="Docker image not found locally. Pulling or using local build when starting...",
                )
                logger.info("DOCKER: Image not found locally; docker will pull if needed")

            # Launch the worker container (container will continue updating status.json)
            logger.info("DOCKER: Starting worker container")
            status.update_status(
                project_id,
                "processing",
                progress=9,
                stage="docker",
                stage_progress=50,
                message="Starting worker container...",
            )

            colmap.run_colmap_docker(project_id, params)

            # When docker finishes, the worker should have set final status.
            # Add a final docker-stage marker in case the worker couldn't start.
            logger.info("DOCKER: Worker container finished")
            status.update_status(
                project_id,
                "processing",
                stage="docker",
                stage_progress=100,
                message="Worker container finished. Finalizing...",
            )
            logger.info(f"Docker worker completed for project {project_id}")

            # Docker container writes the final status; respect stop state if present
            final_status = status.get_status(project_id)
            # Project dir and stop flag fallback
            try:
                project_dir = storage.get_project_dir(project_id)
                stop_flag = (project_dir / "stop_requested")
            except Exception:
                stop_flag = None

            if final_status.get("status") in {"stopped", "failed"}:
                logger.info(f"Pipeline ended with status {final_status.get('status')} for project {project_id}")
            elif final_status.get("stop_requested", False) or (stop_flag is not None and stop_flag.exists()):
                # If worker failed to persist a final stopped status, fall back to stop flag file
                s = status.get_status(project_id)
                pct = s.get("stopped_percentage") or s.get("percentage") or s.get("progress") or 0
                try:
                    pct_int = int(round(float(pct)))
                except Exception:
                    pct_int = int(pct) if isinstance(pct, int) else 0
                # Respect stopped_stage if worker recorded it, else fallback to stage
                stopped_stage = s.get("stopped_stage") or s.get("stage") or "colmap"
                status.update_status(project_id, "stopped", progress=pct_int, stop_requested=True, stage=stopped_stage, message="Processing stopped by user.")
                logger.info(f"Pipeline stopped by user (detected stop_requested) for project {project_id}; stage={stopped_stage}")
            else:
                status.update_status(project_id, "completed", progress=100, stop_requested=False, stage="export", message="Finished")
                logger.info(f"Pipeline completed for project {project_id}")

            _mark_active_sparse_shared_version_if_ready(project_id, params, status.get_status(project_id))
        except Exception as e:
            logger.error(f"Docker worker failed: {e}", exc_info=True)
            status.update_status(project_id, "failed", error=str(e))
            raise
        return
    
    try:
        _set_local_project_active(project_id, True)
        logger.info(f"Using LOCAL WORKER for project {project_id} (mode: {mode})")

        status.update_status(
            project_id,
            "processing",
            progress=5,
            mode=mode,
            maxSteps=max_steps,
            stage="worker",
            stage_progress=10,
            message="Starting local worker...",
        )

        colmap.run_worker_local(project_id, params)

        final_status = status.get_status(project_id)
        try:
            project_dir = storage.get_project_dir(project_id)
            stop_flag = (project_dir / "stop_requested")
        except Exception:
            stop_flag = None

        if final_status.get("status") in {"stopped", "failed", "completed", "done", "partial_completed"}:
            logger.info(f"Local worker finished with status {final_status.get('status')} for project {project_id}")
        elif final_status.get("gaussian_cap_reached"):
            status.update_status(
                project_id,
                "partial_completed",
                progress=int(final_status.get("progress") or 60),
                stop_requested=False,
                stage=final_status.get("stage") or "training",
                message=final_status.get("message") or "Gaussian hard cap reached; run marked partially completed.",
                gaussian_cap_reached=True,
                gaussian_cap_step=final_status.get("gaussian_cap_step"),
                gaussian_cap_count=final_status.get("gaussian_cap_count"),
                gaussian_hard_cap=final_status.get("gaussian_hard_cap"),
                quality_score_source=final_status.get("quality_score_source"),
                relative_quality_score=final_status.get("relative_quality_score"),
                exclude_from_normalization=final_status.get("exclude_from_normalization"),
                valid_completed_quality=final_status.get("valid_completed_quality"),
            )
            logger.info(f"Local worker reached gaussian hard cap for project {project_id}")
        elif final_status.get("stop_requested", False) or (stop_flag is not None and stop_flag.exists()):
            s = status.get_status(project_id)
            pct = s.get("stopped_percentage") or s.get("percentage") or s.get("progress") or 0
            try:
                pct_int = int(round(float(pct)))
            except Exception:
                pct_int = int(pct) if isinstance(pct, int) else 0
            stopped_stage = s.get("stopped_stage") or s.get("stage") or "training"
            status.update_status(project_id, "stopped", progress=pct_int, stop_requested=True, stage=stopped_stage, message="Processing stopped by user.")
            logger.info(f"Local worker stopped by user for project {project_id}; stage={stopped_stage}")
        else:
            status.update_status(project_id, "completed", progress=100, stop_requested=False, stage="export", message="Finished")
            logger.info(f"Local worker completed for project {project_id}")

        _mark_active_sparse_shared_version_if_ready(project_id, params, status.get_status(project_id))

        return None

    except Exception as e:
        logger.error(f"Local worker failed for project {project_id}: {str(e)}", exc_info=True)
        status.update_status(project_id, "failed", error=str(e))
        raise
    finally:
        _set_local_project_active(project_id, False)
        if scoped_log_handlers:
            _detach_scoped_file_handlers(scoped_log_handlers)

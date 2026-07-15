"""Operational actions for workflow pipelines."""
from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bimba3d_backend.app.services.fixed_log_space_schedule import build_fixed_log_space_config
from bimba3d_backend.app.services import training_pipeline_orchestrator
from bimba3d_backend.app.services import training_pipeline_storage

logger = logging.getLogger(__name__)


SELECTOR_OWNED_KEYS = {
    "feature_lr",
    "position_lr_init",
    "scaling_lr",
    "opacity_lr",
    "rotation_lr",
    "densify_grad_threshold",
    "opacity_threshold",
    "lambda_dssim",
}


async def restart_pipeline(
    pipeline_id: str,
    *,
    keep_baseline: bool = False,
    keep_log_space_schedule: bool = True,
) -> dict[str, Any]:
    pipeline = _require_pipeline(pipeline_id)
    config = pipeline.get("config", {}) if isinstance(pipeline.get("config"), dict) else {}
    pipeline_folder = _require_pipeline_folder(config)

    _stop_runtime_before_destructive_action(pipeline_id)
    deleted_summary = _clean_pipeline_projects(
        pipeline=pipeline,
        config=config,
        pipeline_folder=pipeline_folder,
        keep_baseline=keep_baseline,
    )

    _rmtree(pipeline_folder / "shared_models")

    projects = config.get("projects") if isinstance(config.get("projects"), list) else []
    config["projects"] = _sync_project_ids_after_restart(projects, deleted_summary, keep_baseline=keep_baseline)
    config["restart_version"] = int(config.get("restart_version") or 0) + 1
    config["restart_token"] = uuid.uuid4().hex
    config["last_restart_at"] = _utc_now()
    if not keep_log_space_schedule or not config.get("pre_generated_log_multipliers"):
        _regenerate_fixed_log_space_schedule(config)

    kept_runs = _baseline_runs_to_keep(pipeline.get("runs", []), keep_baseline=keep_baseline)
    training_pipeline_storage.update_pipeline(
        pipeline_id,
        {
            "status": "pending",
            "current_phase": 0,
            "current_run": 0,
            "current_project_index": 0,
            "completed_runs": len(kept_runs),
            "failed_runs": 0,
            "mean_relative_score": None,
            "best_relative_score": None,
            "success_rate": None,
            "last_error": None,
            "active_run": None,
            "cooldown_active": False,
            "next_run_scheduled_at": None,
            "cooldown_session_id": None,
            "current_test_model_id": None,
            "started_at": None,
            "completed_at": None,
            "runs": kept_runs,
            "config": config,
        },
    )

    training_pipeline_storage.update_pipeline(
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
    training_pipeline_orchestrator.start_pipeline_orchestrator(pipeline_id)

    logger.info("Pipeline %s restarted: %d projects cleaned", pipeline_id, len(deleted_summary))
    return {
        "status": "restarted_and_running",
        "pipeline_id": pipeline_id,
        "restart_version": config.get("restart_version"),
        "restart_token": config.get("restart_token"),
        "last_restart_at": config.get("last_restart_at"),
        "log_space_schedule_kept": bool(keep_log_space_schedule and config.get("pre_generated_log_multipliers")),
        "projects_cleaned": len(deleted_summary),
        "details": deleted_summary,
        "message": "Pipeline restarted and started from the current workflow configuration.",
    }


async def retry_failed_runs(
    pipeline_id: str,
    *,
    auto_start: bool = True,
    include_hard_cap: bool = False,
) -> dict[str, Any]:
    pipeline = _require_pipeline(pipeline_id)
    if pipeline.get("status") == "running":
        raise ValueError("Pause or stop the pipeline before retrying failed runs.")

    config = pipeline.get("config", {}) if isinstance(pipeline.get("config"), dict) else {}
    pipeline_type = str(config.get("pipeline_type") or pipeline.get("pipeline_type") or "offline_data").strip().lower()
    pipeline_folder = _require_pipeline_folder(config)
    _stop_runtime_before_destructive_action(pipeline_id)

    runs_all = list(pipeline.get("runs", []))
    failed_runs = _retryable_failed_runs(runs_all, include_hard_cap=include_hard_cap)
    if not failed_runs:
        return {
            "status": "noop",
            "message": "No retryable runs found.",
            "failed_count": 0,
            "prepared_fixed_slots": 0,
        }

    retry_fixed_params = dict(config.get("retry_fixed_params") or {})
    prepared_fixed_slots = 0
    deleted_failed_run_dirs: list[str] = []
    for failed_run in failed_runs:
        if pipeline_type == "offline_data":
            fixed_params = _read_retry_fixed_params(pipeline_folder, failed_run)
            if fixed_params:
                slot_key = _retry_slot_key(
                    project_name=str(failed_run.get("project_name") or ""),
                    phase_num=failed_run.get("phase"),
                    run_num=failed_run.get("run"),
                    model_id=failed_run.get("test_model_id") or failed_run.get("source_model_id"),
                )
                retry_fixed_params[slot_key] = fixed_params
                prepared_fixed_slots += 1

        deleted_dir = _delete_failed_run_dir(pipeline_folder, failed_run)
        if deleted_dir:
            deleted_failed_run_dirs.append(deleted_dir)

    failed_run_ids = {str(run.get("run_id") or "") for run in failed_runs if run.get("run_id")}
    failed_slot_keys = {
        _retry_slot_key(
            project_name=str(run.get("project_name") or ""),
            phase_num=run.get("phase"),
            run_num=run.get("run"),
            model_id=run.get("test_model_id") or run.get("source_model_id"),
        )
        for run in failed_runs
    }
    failed_slot_keys_legacy = {
        _retry_slot_key(
            project_name=str(run.get("project_name") or ""),
            phase_num=run.get("phase"),
            run_num=run.get("run"),
            model_id=None,
        )
        for run in failed_runs
    }
    kept_runs = [
        run
        for run in runs_all
        if not (
            str(run.get("status") or "").lower() in ({"failed", "hard_cap_reached"} if include_hard_cap else {"failed"})
            and (
                str(run.get("run_id") or "") in failed_run_ids
                or _retry_slot_key(
                    project_name=str(run.get("project_name") or ""),
                    phase_num=run.get("phase"),
                    run_num=run.get("run"),
                    model_id=run.get("test_model_id") or run.get("source_model_id"),
                )
                in failed_slot_keys
                or _retry_slot_key(
                    project_name=str(run.get("project_name") or ""),
                    phase_num=run.get("phase"),
                    run_num=run.get("run"),
                    model_id=None,
                )
                in failed_slot_keys_legacy
            )
        )
    ]
    success_count = sum(1 for run in kept_runs if str(run.get("status") or "").lower() in {"success", "partial_completed"})
    hard_cap_count = sum(1 for run in kept_runs if _is_hard_cap_retry_record(run))
    total_runs = int(pipeline.get("total_runs") or 0)
    pending_count = max(0, total_runs - success_count - hard_cap_count)
    scores = [float(run["score"]) for run in kept_runs if isinstance(run.get("score"), (int, float))]

    config["retry_fixed_params"] = retry_fixed_params
    config["retry_mode_active"] = bool(auto_start)
    config["retry_target_slots"] = sorted(failed_slot_keys)
    config["retry_total_slots"] = len(failed_slot_keys)
    config["retry_include_hard_cap"] = bool(include_hard_cap)
    training_pipeline_storage.update_pipeline(
        pipeline_id,
        {
            "config": config,
            "runs": kept_runs,
            "completed_runs": success_count,
            "failed_runs": 0,
            "hard_cap_runs": hard_cap_count,
            "pending_runs": pending_count,
            "mean_relative_score": (sum(scores) / len(scores)) if scores else None,
            "best_relative_score": max(scores) if scores else None,
            "success_rate": (success_count / len(kept_runs) * 100.0) if kept_runs else None,
            "last_error": None,
            "active_run": None,
            "status": "stopped",
            "completed_at": None,
        },
    )

    if auto_start:
        training_pipeline_storage.update_pipeline(
            pipeline_id,
            {
                "status": "running",
                "started_at": pipeline.get("started_at") or _utc_now(),
                "completed_at": None,
                "last_error": None,
                "cooldown_active": False,
                "next_run_scheduled_at": None,
                "cooldown_session_id": None,
            },
        )
        training_pipeline_orchestrator.start_pipeline_orchestrator(pipeline_id)
        status = "running"
        message = "Retry started for failed runs."
        if include_hard_cap:
            message = "Retry started for failed and hard-cap runs."
    else:
        config["retry_mode_active"] = False
        training_pipeline_storage.update_pipeline(pipeline_id, {"config": config})
        status = "stopped"
        message = "Failed runs prepared for retry. Start retry from the Retry Failed action."
        if include_hard_cap:
            message = "Failed and hard-cap runs prepared for retry. Start retry from the Retry Failed action."

    return {
        "status": status,
        "message": message,
        "failed_count": len(failed_runs),
        "prepared_fixed_slots": prepared_fixed_slots,
        "deleted_failed_run_dirs": deleted_failed_run_dirs,
        "include_hard_cap": include_hard_cap,
    }


def _retryable_failed_runs(runs: list[Any], *, include_hard_cap: bool = False) -> list[dict[str, Any]]:
    completed_slots: set[str] = set()
    hard_cap_slots: set[str] = set()
    for run in runs:
        if not isinstance(run, dict):
            continue
        slot_key = _retry_slot_key(
            project_name=str(run.get("project_name") or ""),
            phase_num=run.get("phase"),
            run_num=run.get("run"),
            model_id=run.get("test_model_id") or run.get("source_model_id"),
        )
        legacy_slot_key = _retry_slot_key(
            project_name=str(run.get("project_name") or ""),
            phase_num=run.get("phase"),
            run_num=run.get("run"),
            model_id=None,
        )
        status = str(run.get("status") or "").lower()
        if status in {"success", "partial_completed"} and not _is_hard_cap_retry_record(run):
            completed_slots.add(slot_key)
        elif _is_hard_cap_retry_record(run):
            hard_cap_slots.add(slot_key)

    failed_by_slot: dict[str, dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        status = str(run.get("status") or "").lower()
        if status != "failed" and not (include_hard_cap and _is_hard_cap_retry_record(run)):
            continue
        slot_key = _retry_slot_key(
            project_name=str(run.get("project_name") or ""),
            phase_num=run.get("phase"),
            run_num=run.get("run"),
            model_id=run.get("test_model_id") or run.get("source_model_id"),
        )
        legacy_slot_key = _retry_slot_key(
            project_name=str(run.get("project_name") or ""),
            phase_num=run.get("phase"),
            run_num=run.get("run"),
            model_id=None,
        )
        if slot_key in completed_slots:
            continue
        if slot_key in hard_cap_slots and not (
            include_hard_cap and _is_hard_cap_retry_record(run)
        ):
            continue
        current = failed_by_slot.get(slot_key)
        if current is None or str(run.get("completed_at") or run.get("timestamp") or "") > str(
            current.get("completed_at") or current.get("timestamp") or ""
        ):
            failed_by_slot[slot_key] = run
    return list(failed_by_slot.values())


def _is_hard_cap_retry_record(run: dict[str, Any]) -> bool:
    if str(run.get("status") or "").lower() == "hard_cap_reached":
        return True
    if run.get("gaussian_cap_reached") is True:
        return True
    return str(run.get("reason") or run.get("partial_reason") or "").lower() == "gaussian_hard_cap_reached"


def _require_pipeline(pipeline_id: str) -> dict[str, Any]:
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
    if not pipeline:
        raise FileNotFoundError("Pipeline not found")
    return pipeline


def _require_pipeline_folder(config: dict[str, Any]) -> Path:
    raw_folder = str(config.get("pipeline_folder") or "").strip()
    if not raw_folder:
        raise FileNotFoundError("Pipeline folder is not configured")
    pipeline_folder = Path(raw_folder).expanduser()
    if not pipeline_folder.exists():
        raise FileNotFoundError("Pipeline folder not found")
    return pipeline_folder


def _stop_runtime_before_destructive_action(pipeline_id: str) -> None:
    try:
        training_pipeline_orchestrator.stop_pipeline_orchestrator(pipeline_id)
    except Exception as exc:
        logger.warning("Could not stop pipeline orchestrator before restart for %s: %s", pipeline_id, exc)

    try:
        pipeline = training_pipeline_storage.get_pipeline(pipeline_id) or {}
        stop_flags = _write_pipeline_project_stop_flags(pipeline, reason="destructive_action")
        if stop_flags:
            logger.info("Wrote %d project stop flag(s) before destructive pipeline action", stop_flags)
    except Exception:
        logger.debug("Failed to write project stop flags before destructive action", exc_info=True)

    from bimba3d_backend.app.services.colmap import stop_all_local_workers

    killed = stop_all_local_workers()
    if killed:
        logger.info("Killed %d active worker process(es) before restart", killed)
    time.sleep(5 if killed else 2)


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
            logger.warning("Could not write stop flag for project %s before %s: %s", project_name, reason, exc)
    return written


def _clean_pipeline_projects(
    *,
    pipeline: dict[str, Any],
    config: dict[str, Any],
    pipeline_folder: Path,
    keep_baseline: bool,
) -> list[dict[str, Any]]:
    projects = config.get("projects") if isinstance(config.get("projects"), list) else []
    deleted_summary: list[dict[str, Any]] = []
    for project_cfg in projects:
        project_name = str(project_cfg.get("name") or "").strip()
        if not project_name:
            logger.warning("Restart: project has no name, skipping")
            continue

        project_dir = pipeline_folder / project_name.replace(" ", "_")
        if not project_dir.is_dir():
            logger.warning("Restart: project dir not found for '%s' at %s, skipping", project_name, project_dir)
            continue

        project_id = _read_project_id(project_dir)
        baseline_run_id = _find_baseline_run_id(project_dir)
        deleted_runs = _delete_project_runs(project_dir, baseline_run_id=baseline_run_id, keep_baseline=keep_baseline)

        _rmtree(project_dir / "outputs" / "engines")
        _rmtree(project_dir / "models")
        _safe_unlink(project_dir / ".batch_lineage_latest.json")
        _safe_unlink(project_dir / ".project_model_state.json")
        _safe_unlink(project_dir / "exif_features.json")
        _reset_project_status(project_dir, project_id=project_id, baseline_run_id=baseline_run_id)
        _sync_project_config(project_dir, pipeline=pipeline, config=config, pipeline_folder=pipeline_folder)

        deleted_summary.append(
            {
                "project_name": project_name,
                "project_id": project_id,
                "baseline_kept": baseline_run_id if keep_baseline else None,
                "deleted_runs": deleted_runs,
            }
        )
    return deleted_summary


def _read_project_id(project_dir: Path) -> str | None:
    payload = _read_json(project_dir / "config.json")
    return str(payload.get("id")) if isinstance(payload.get("id"), str) else None


def _find_baseline_run_id(project_dir: Path) -> str | None:
    runs_root = project_dir / "runs"
    if not runs_root.is_dir():
        return None
    run_dirs = sorted([path for path in runs_root.iterdir() if path.is_dir()], key=lambda path: path.name)
    return run_dirs[0].name if run_dirs else None


def _delete_project_runs(project_dir: Path, *, baseline_run_id: str | None, keep_baseline: bool) -> list[str]:
    runs_root = project_dir / "runs"
    deleted_runs: list[str] = []
    if not runs_root.is_dir():
        return deleted_runs

    for run_dir in list(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if keep_baseline and run_dir.name == baseline_run_id:
            continue
        _rmtree(run_dir)
        deleted_runs.append(run_dir.name)
    return deleted_runs


def _reset_project_status(project_dir: Path, *, project_id: str | None, baseline_run_id: str | None) -> None:
    status_file = project_dir / "status.json"
    existing_status = _read_json(status_file)
    reset_status = {
        "project_id": project_id,
        "status": "pending",
        "progress": 0,
        "name": existing_status.get("name"),
        "created_at": existing_status.get("created_at"),
        "base_session_id": baseline_run_id,
    }
    _write_json_atomic(status_file, reset_status)


def _sync_project_config(
    project_dir: Path,
    *,
    pipeline: dict[str, Any],
    config: dict[str, Any],
    pipeline_folder: Path,
) -> None:
    config_file = project_dir / "config.json"
    if not config_file.exists():
        return

    project_config = _read_json(config_file)
    shared_config = config.get("shared_config") if isinstance(config.get("shared_config"), dict) else {}
    project_config.update(shared_config)
    project_config["pipeline_id"] = pipeline["id"]
    project_config["pipeline_name"] = config.get("name")
    project_config["pipeline_path"] = str(pipeline_folder)

    if config.get("pipeline_type") == "test":
        project_config["source_model_id"] = config.get("source_model_id")
        project_config["source_model_ids"] = config.get("source_model_ids") or (
            [config.get("source_model_id")] if config.get("source_model_id") else []
        )
    else:
        project_config.pop("source_model_id", None)
        project_config.pop("source_model_ids", None)

    _write_json_atomic(config_file, project_config)


def _sync_project_ids_after_restart(
    projects: list[Any],
    deleted_summary: list[dict[str, Any]],
    *,
    keep_baseline: bool,
) -> list[dict[str, Any]]:
    ids_by_name = {item.get("project_name"): item.get("project_id") for item in deleted_summary}
    updated_projects: list[dict[str, Any]] = []
    for project_cfg in projects:
        if not isinstance(project_cfg, dict):
            continue
        project_name = project_cfg.get("name")
        updated_project = dict(project_cfg)
        if ids_by_name.get(project_name):
            updated_project["project_id"] = ids_by_name[project_name]
        if not keep_baseline:
            updated_project.pop("baseline_run_id", None)
        updated_projects.append(updated_project)
    return updated_projects


def _regenerate_fixed_log_space_schedule(config: dict[str, Any]) -> None:
    config.update(build_fixed_log_space_config(config))


def _baseline_runs_to_keep(runs: Any, *, keep_baseline: bool) -> list[dict[str, Any]]:
    if not keep_baseline or not isinstance(runs, list):
        return []
    return [
        run
        for run in runs
        if isinstance(run, dict)
        and int(run.get("phase") or 0) == 1
        and str(run.get("status") or "").lower() == "success"
    ]


def _read_retry_fixed_params(pipeline_folder: Path, failed_run: dict[str, Any]) -> dict[str, float]:
    project_name = str(failed_run.get("project_name") or "").strip()
    run_id = str(failed_run.get("run_id") or "").strip()
    if not project_name or not run_id:
        return {}

    run_dir = pipeline_folder / project_name.replace(" ", "_") / "runs" / run_id
    fixed = _read_retry_snapshot(run_dir / "retry_snapshot.json")
    if fixed:
        return fixed
    return _read_retry_analytics(run_dir / "analytics" / "run_analytics_v1.json")


def _delete_failed_run_dir(pipeline_folder: Path, failed_run: dict[str, Any]) -> str | None:
    project_name = str(failed_run.get("project_name") or "").strip()
    run_id = str(failed_run.get("run_id") or "").strip()
    if not project_name or not run_id:
        return None

    pipeline_root = pipeline_folder.resolve()
    run_dir = (pipeline_folder / project_name.replace(" ", "_") / "runs" / run_id).resolve()
    if pipeline_root not in run_dir.parents or not run_dir.is_dir():
        return None

    shutil.rmtree(run_dir)
    return str(run_dir)


def _read_retry_snapshot(snapshot_file: Path) -> dict[str, float]:
    payload = _read_json(snapshot_file)
    initial_params = payload.get("initial_params") if isinstance(payload, dict) else {}
    return _selector_owned_float_params(initial_params)


def _read_retry_analytics(analytics_file: Path) -> dict[str, float]:
    payload = _read_json(analytics_file)
    ai_block = payload.get("ai") if isinstance(payload, dict) else {}
    insights = ai_block.get("input_mode_insights") if isinstance(ai_block, dict) else {}
    initial_params = insights.get("initial_params") if isinstance(insights, dict) else {}
    return _selector_owned_float_params(initial_params)


def _selector_owned_float_params(params: Any) -> dict[str, float]:
    if not isinstance(params, dict):
        return {}
    return {
        key: float(value)
        for key, value in params.items()
        if key in SELECTOR_OWNED_KEYS and isinstance(value, (int, float))
    }


def _retry_slot_key(*, project_name: str, phase_num: Any, run_num: Any, model_id: Any = None) -> str:
    model_part = str(model_id or "").strip()
    return f"{project_name}|{phase_num}|{run_num}|{model_part}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    temp_path.replace(path)


def _remove_readonly(func, path, _exc_info) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    func(path)


def _safe_unlink(path: Path, retries: int = 8) -> bool:
    if not path.exists() and not path.is_symlink():
        return True

    for attempt in range(retries):
        try:
            path.unlink(missing_ok=True)
            return True
        except PermissionError as exc:
            if attempt == retries - 1:
                logger.warning("Restart cleanup: could not delete locked file %s (%s)", path, exc)
                return False
            time.sleep(min(2.0, 0.1 * (2**attempt)))
        except FileNotFoundError:
            return True
        except Exception as exc:
            logger.warning("Restart cleanup: failed to delete file %s (%s)", path, exc)
            return False
    return False


def _rmtree(path: Path, retries: int = 8) -> bool:
    if not path.exists() and not path.is_symlink():
        return True

    for attempt in range(retries):
        try:
            if path.is_file() or path.is_symlink():
                return _safe_unlink(path, retries=1)
            shutil.rmtree(path, onerror=_remove_readonly)
            return True
        except PermissionError as exc:
            if attempt == retries - 1:
                logger.warning("Restart cleanup: could not remove locked path %s (%s)", path, exc)
                return False
            time.sleep(min(2.5, 0.15 * (2**attempt)))
        except FileNotFoundError:
            return True
        except Exception as exc:
            logger.warning("Restart cleanup: failed to remove path %s (%s)", path, exc)
            return False
    return False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

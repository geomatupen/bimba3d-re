"""Prediction preview helpers for testing workflow pipelines."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bimba3d_backend.app.services import model_registry
from bimba3d_backend.app.services import training_pipeline_storage
from bimba3d_backend.app.services import workflow_model_seeding
from bimba3d_backend.app.services.training_pipeline_orchestrator import PipelineOrchestrator
from bimba3d_backend.worker.ai_input_modes.resolver import apply_initial_preset

logger = logging.getLogger(__name__)


async def predict_multipliers(pipeline_id: str, request_payload: dict[str, Any]) -> dict[str, Any]:
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
    if not pipeline:
        raise FileNotFoundError("Pipeline not found")

    config = pipeline.get("config", {}) if isinstance(pipeline.get("config"), dict) else {}
    pipeline_type = str(config.get("pipeline_type") or pipeline.get("pipeline_type") or "offline_data").strip().lower()
    if pipeline_type != "test":
        raise ValueError("Prediction preview is available only for test pipelines.")

    projects = config.get("projects") if isinstance(config.get("projects"), list) else []
    if not projects:
        raise ValueError("No projects configured in pipeline.")

    selected_model_ids = _selected_model_ids(config, request_payload)
    if not selected_model_ids:
        raise ValueError("No model selected for this test pipeline.")

    orchestrator = PipelineOrchestrator(pipeline_id)
    shared_config = dict(config.get("shared_config") or {})
    candidate_logs = config.get("test_candidate_log_multipliers") if isinstance(config.get("test_candidate_log_multipliers"), dict) else {}
    if candidate_logs:
        shared_config["candidate_log_multipliers_by_group"] = candidate_logs
        shared_config["test_candidate_seed"] = config.get("test_candidate_seed")
        shared_config["test_candidate_count"] = config.get("test_candidate_count")
    results: list[dict[str, Any]] = []

    for project_cfg in projects:
        if not isinstance(project_cfg, dict):
            continue
        project_name = str(project_cfg.get("name") or "").strip()
        if not project_name:
            continue

        try:
            project_dir = orchestrator._get_or_create_project_dir(pipeline, project_cfg)
        except Exception as exc:
            results.append(
                {
                    "project_name": project_name,
                    "project_id": project_cfg.get("project_id"),
                    "status": "error",
                    "error": f"failed_to_prepare_project_dir: {exc}",
                }
            )
            continue

        image_dir = project_dir / "images_resized"
        if not image_dir.exists():
            image_dir = project_dir / "images"
        colmap_dir = project_dir / "outputs" / "sparse"
        if not colmap_dir.exists():
            colmap_dir = project_dir / "outputs"

        for model_id in selected_model_ids:
            results.append(
                _predict_project_model(
                    project_cfg=project_cfg,
                    project_name=project_name,
                    project_dir=project_dir,
                    image_dir=image_dir,
                    colmap_dir=colmap_dir,
                    model_id=model_id,
                    shared_config=shared_config,
                )
            )

    preview_entry = _build_preview_entry(
        pipeline_id=pipeline_id,
        pipeline=pipeline,
        config=config,
        results=results,
    )
    _save_preview_entry(pipeline_id, preview_entry)

    return {
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline.get("name"),
        "preview_key": preview_entry["preview_key"],
        "generated_at": preview_entry["generated_at"],
        "total": preview_entry["total"],
        "ok": preview_entry["ok"],
        "failed": preview_entry["failed"],
        "results": results,
    }


def _selected_model_ids(config: dict[str, Any], request_payload: dict[str, Any]) -> list[str]:
    raw_ids: list[Any] = []
    if request_payload.get("model_id"):
        raw_ids.append(request_payload.get("model_id"))
    elif isinstance(request_payload.get("model_ids"), list):
        raw_ids.extend(request_payload.get("model_ids") or [])
    elif isinstance(config.get("source_model_ids"), list):
        raw_ids.extend(config.get("source_model_ids") or [])
    elif config.get("source_model_id"):
        raw_ids.append(config.get("source_model_id"))

    selected: list[str] = []
    seen: set[str] = set()
    for model_id in raw_ids:
        clean_id = str(model_id or "").strip()
        if clean_id and clean_id not in seen:
            selected.append(clean_id)
            seen.add(clean_id)
    return selected


def _predict_project_model(
    *,
    project_cfg: dict[str, Any],
    project_name: str,
    project_dir: Path,
    image_dir: Path,
    colmap_dir: Path,
    model_id: str,
    shared_config: dict[str, Any],
) -> dict[str, Any]:
    resolved_mode = str(shared_config.get("ai_input_mode") or "").strip().lower()
    seed_path = None

    try:
        workflow_model = workflow_model_seeding.read_workflow_model(model_id)
        model_record = None if workflow_model is not None else model_registry.resolve_reusable_model(model_id)
        if workflow_model is None and not model_record:
            return _prediction_error(project_cfg, project_name, model_id, "model_not_found")
        model_name = (
            str(workflow_model.model_name or "").strip()
            if workflow_model is not None
            else str((model_record or {}).get("model_name") or "").strip()
        )

        profile = (
            workflow_model_seeding.model_ai_profile(workflow_model)
            if workflow_model is not None
            else model_registry.resolve_model_ai_profile(model_record)
        )
        profile_mode = str(profile.get("ai_input_mode") or "").strip().lower()
        if profile_mode:
            resolved_mode = profile_mode

        seeded = (
            workflow_model_seeding.seed_workflow_model_into_project(workflow_model, project_dir)
            if workflow_model is not None
            else model_registry.seed_learner_weights_into_project(model_record, project_dir)
        )
        seed_path = str(seeded) if seeded else None
        if not resolved_mode:
            resolved_mode = "exif_compact_featurewise"

        params = dict(shared_config)
        params["mode"] = "modified"
        params["run_jitter_only"] = False
        params["ai_input_mode"] = resolved_mode
        selector_strategy = str(profile.get("ai_selector_strategy") or "").strip().lower()
        if selector_strategy:
            params["ai_selector_strategy"] = selector_strategy

        prediction = apply_initial_preset(
            params,
            image_dir=image_dir,
            colmap_dir=colmap_dir,
            logger=logger,
        )
    except Exception as exc:
        row = _prediction_error(project_cfg, project_name, model_id, str(exc))
        if "model_name" in locals() and model_name:
            row["model_name"] = model_name
        row["mode"] = resolved_mode or None
        row["seed_path"] = seed_path
        return row

    return {
        "project_name": project_name,
        "project_id": project_cfg.get("project_id"),
        "model_id": model_id,
        "model_name": model_name or model_id,
        "mode": resolved_mode,
        "status": "ok",
        "selected_multipliers": prediction.get("selected_multipliers") or {},
        "selected_multipliers_raw": prediction.get("selected_multipliers_raw") or {},
        "selected_log_multipliers": prediction.get("selected_log_multipliers") or {},
        "selected_log_multipliers_raw": prediction.get("selected_log_multipliers_raw") or {},
        "group_multipliers": prediction.get("group_multipliers") or {},
        "group_log_multipliers": prediction.get("group_log_multipliers") or {},
        "features": prediction.get("features") or {},
        "cache_used": bool(prediction.get("cache_used")),
        "has_signal": bool(prediction.get("has_signal", True)),
        "n_runs": int(prediction.get("n_runs") or 0),
        "candidate_points": int(prediction.get("candidate_points") or 0),
        "score_spreads": prediction.get("score_spreads") or {},
        "candidate_score_checks": prediction.get("candidate_score_checks") or {},
        "run_jitter_multiplier": float(prediction.get("run_jitter_multiplier") or 1.0),
        "effective_params": prediction.get("effective_params") or {},
        "remarks": prediction.get("remarks"),
        "selected_preset": prediction.get("selected_preset"),
        "seed_path": seed_path,
    }


def _prediction_error(
    project_cfg: dict[str, Any],
    project_name: str,
    model_id: str,
    error: str,
) -> dict[str, Any]:
    return {
        "project_name": project_name,
        "project_id": project_cfg.get("project_id"),
        "model_id": model_id,
        "status": "error",
        "error": error,
    }


def _build_preview_entry(
    *,
    pipeline_id: str,
    pipeline: dict[str, Any],
    config: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    ok_count = sum(1 for row in results if row.get("status") == "ok")
    mode_signature = sorted({str(row.get("mode") or "") for row in results if row.get("mode")})
    model_signature = sorted({str(row.get("model_id") or "") for row in results if row.get("model_id")})
    preview_key = f"{','.join(mode_signature)}|{','.join(model_signature)}"

    return {
        "preview_key": preview_key,
        "generated_at": _utc_now(),
        "restart_version": int(config.get("restart_version") or 0),
        "restart_token": str(config.get("restart_token") or ""),
        "mode_signature": mode_signature,
        "model_signature": model_signature,
        "total": len(results),
        "ok": ok_count,
        "failed": len(results) - ok_count,
        "results": results,
    }


def _save_preview_entry(
    pipeline_id: str,
    preview_entry: dict[str, Any],
) -> None:
    latest_pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
    if not latest_pipeline:
        raise FileNotFoundError("Pipeline not found")
    preview_key = str(preview_entry["preview_key"])
    prediction_previews = dict(latest_pipeline.get("prediction_previews") or {})
    prediction_previews[preview_key] = preview_entry
    training_pipeline_storage.update_pipeline(
        pipeline_id,
        {
            "prediction_previews": prediction_previews,
            "latest_prediction_preview_key": preview_key,
        },
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


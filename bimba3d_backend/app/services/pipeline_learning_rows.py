"""Collect learning rows from completed offline-data pipelines."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bimba3d_backend.app.services import relative_quality_scoring, training_pipeline_storage

logger = logging.getLogger(__name__)

COMPACT_BASE_PARAM_DEFAULTS: dict[str, float] = {
    "feature_lr": 2.5e-3,
    "position_lr_init": 1.6e-4,
    "scaling_lr": 5.0e-3,
    "opacity_lr": 5.0e-2,
    "rotation_lr": 1.0e-3,
    "densify_grad_threshold": 2.0e-4,
    "opacity_threshold": 0.005,
    "lambda_dssim": 0.2,
}


def collect_pipeline_learning_rows(pipeline_id: str, *, include_hard_cap: bool = False) -> dict[str, Any]:
    """Return the aggregated learning table rows for one offline-data pipeline."""
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
    if not pipeline:
        raise FileNotFoundError("Pipeline not found")

    config = pipeline.get("config", {}) if isinstance(pipeline.get("config"), dict) else {}
    pipeline_folder_value = config.get("pipeline_folder") or pipeline.get("pipeline_folder")
    if not pipeline_folder_value:
        raise FileNotFoundError("Pipeline folder not found")

    pipeline_folder = Path(pipeline_folder_value)
    if not pipeline_folder.exists():
        raise FileNotFoundError("Pipeline folder not found")

    known_run_ids = _learning_run_ids_from_pipeline_runs(
        pipeline.get("runs", []),
        include_hard_cap=include_hard_cap,
    )

    rows = _collect_rows_from_folder(pipeline_folder, known_run_ids=known_run_ids or None)
    rows = _dedupe_rows(rows)
    rows.sort(key=lambda row: (row.get("project_name", ""), row.get("run_id", "")))
    _augment_rows_with_visual_scores(rows)

    return {
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline.get("name"),
        "rows": rows,
        "total_rows": len(rows),
        "training_data_config_snapshot": training_data_config_snapshot(pipeline_id),
        "pre_generated_log_multipliers": config.get("pre_generated_log_multipliers", {}),
        "multiplier_current_index": config.get("multiplier_current_index", 0),
    }


def training_data_config_snapshot(pipeline_id: str) -> dict[str, Any]:
    """Return the training settings that must be frozen into a Training Data artifact."""
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
    if not pipeline:
        raise FileNotFoundError("Pipeline not found")
    config = pipeline.get("config", {}) if isinstance(pipeline.get("config"), dict) else {}
    return _training_data_config_snapshot(config)


def _training_data_config_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    bounds, source = _log_multiplier_bounds_snapshot(config)
    base_params, base_params_source = _base_params_snapshot(config)
    return {
        "log_multiplier_bounds": bounds,
        "log_multiplier_bounds_source": source,
        "base_params": base_params,
        "base_params_source": base_params_source,
        "score_key": "relative_quality_score",
    }


def _log_multiplier_bounds_snapshot(config: dict[str, Any]) -> tuple[dict[str, list[float]], str]:
    fixed = config.get("fixed_log_space_bounds") if isinstance(config.get("fixed_log_space_bounds"), dict) else {}
    if fixed:
        return _normalise_bounds(fixed), "source_pipeline.fixed_log_space_bounds_snapshot"

    shared = config.get("shared_config") if isinstance(config.get("shared_config"), dict) else {}
    shared_bounds = {
        "geometry_lr": [shared.get("geometry_log_multiplier_min"), shared.get("geometry_log_multiplier_max")],
        "appearance_lr": [shared.get("appearance_log_multiplier_min"), shared.get("appearance_log_multiplier_max")],
        "scale_lr": [shared.get("densification_log_multiplier_min"), shared.get("densification_log_multiplier_max")],
    }
    if any(any(value is not None for value in pair) for pair in shared_bounds.values()):
        return _normalise_bounds(shared_bounds), "source_pipeline.shared_config_snapshot"

    return _default_multiplier_bounds(), "built_in_bounds_snapshot"


def _normalise_bounds(bounds: dict[str, Any]) -> dict[str, list[float]]:
    aliases = {
        "geometry": "geometry_lr",
        "geometry_lr_mult": "geometry_lr",
        "appearance": "appearance_lr",
        "appearance_lr_mult": "appearance_lr",
        "densification": "scale_lr",
        "densification_lr": "scale_lr",
        "densification_mult": "scale_lr",
    }
    defaults = _default_multiplier_bounds()
    out: dict[str, list[float]] = {}
    for group, default in defaults.items():
        raw = bounds.get(group)
        if raw is None:
            raw = next((value for key, value in bounds.items() if aliases.get(str(key), str(key)) == group), None)
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            lo = _safe_positive_float(raw[0], default[0])
            hi = _safe_positive_float(raw[1], default[1])
        else:
            lo, hi = default
        if hi < lo:
            lo, hi = hi, lo
        out[group] = [lo, hi]
    return out


def _default_multiplier_bounds() -> dict[str, list[float]]:
    return {
        "geometry_lr": [0.5, 2.0],
        "appearance_lr": [0.5, 2.0],
        "scale_lr": [0.7, 1.42],
    }


def _base_params_snapshot(config: dict[str, Any]) -> tuple[dict[str, float], str]:
    """Capture the base values used before multiplier application."""
    shared = config.get("shared_config") if isinstance(config.get("shared_config"), dict) else {}
    out: dict[str, float] = {}
    used_source = False
    for key, default in COMPACT_BASE_PARAM_DEFAULTS.items():
        raw = config.get(key)
        if raw is None:
            raw = shared.get(key)
        parsed = _safe_positive_float(raw, default)
        used_source = used_source or parsed != default
        out[key] = parsed
    source = "source_pipeline.base_params_snapshot" if used_source else "built_in_base_params_snapshot"
    return out, source


def _safe_positive_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if parsed > 0.0 else default
    return default


def _collect_rows_from_folder(pipeline_folder: Path, *, known_run_ids: set[str] | None = None) -> list[dict[str, Any]]:
    from bimba3d_backend.app.api.projects import (
        _build_learning_param_rows,
        _analytics_metrics,
        _read_json_if_exists,
    )

    rows: list[dict[str, Any]] = []
    for project_dir in pipeline_folder.iterdir():
        if not _is_project_folder(project_dir, pipeline_folder):
            continue

        runs_dir = project_dir / "runs"
        if not runs_dir.exists():
            continue

        project_config = _read_json_if_exists(project_dir / "config.json") or {}
        if not isinstance(project_config, dict):
            project_config = {}

        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if known_run_ids is not None and run_dir.name not in known_run_ids:
                continue

            try:
                row = _build_row(
                    project_dir=project_dir,
                    project_config=project_config,
                    run_dir=run_dir,
                    analytics_metrics=_analytics_metrics,
                    build_learning_param_rows=_build_learning_param_rows,
                    read_json_if_exists=_read_json_if_exists,
                )
            except Exception as exc:  # keep one bad run from hiding a completed pipeline
                logger.warning("Failed to load learning data from %s/%s: %s", project_dir.name, run_dir.name, exc)
                continue

            if row is not None:
                rows.append(row)

    return rows


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per source project/run in a pipeline."""
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        project_id = str(row.get("project_id") or row.get("project_name") or "").strip()
        run_id = str(row.get("run_id") or "").strip()
        key = (project_id, run_id)
        if not project_id or not run_id:
            deduped.append(row)
            continue
        if key in seen:
            logger.warning("Skipping duplicate pipeline learning row project=%s run=%s", project_id, run_id)
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _learning_run_ids_from_pipeline_runs(runs: Any, *, include_hard_cap: bool) -> set[str]:
    """Return run ids that should contribute rows to the final Training Data table.

    The pipeline keeps retry attempts as raw history. For supervised Training
    Data, however, one project/phase/run slot should only become one row. A
    later successful retry replaces earlier failed or hard-cap attempts for the
    same slot. If no retry succeeded and hard-cap rows are requested, the latest
    hard-cap attempt is kept as one penalty-row candidate.
    """
    completed_statuses = {"success", "completed", "done", "ok", "partial_completed"}
    completed_slots: set[tuple[Any, int, int, str | None]] = set()
    completed_run_ids: set[str] = set()
    hard_cap_by_slot: dict[tuple[Any, int, int, str | None], dict[str, Any]] = {}

    for run in runs if isinstance(runs, list) else []:
        if not isinstance(run, dict):
            continue
        run_id = str(run.get("run_id") or "").strip()
        if not run_id:
            continue

        slot_key = _pipeline_run_slot_key(run)
        status = str(run.get("status") or "").lower()
        is_hard_cap = _is_hard_cap_pipeline_run(run)

        if status in completed_statuses and not is_hard_cap:
            completed_slots.add(slot_key)
            completed_run_ids.add(run_id)
            continue

        if include_hard_cap and is_hard_cap:
            previous = hard_cap_by_slot.get(slot_key)
            if previous is None or _pipeline_run_order_key(run) >= _pipeline_run_order_key(previous):
                hard_cap_by_slot[slot_key] = run

    selected = set(completed_run_ids)
    if include_hard_cap:
        for slot_key, run in hard_cap_by_slot.items():
            if slot_key in completed_slots:
                continue
            run_id = str(run.get("run_id") or "").strip()
            if run_id:
                selected.add(run_id)
    return selected


def _pipeline_run_slot_key(run: dict[str, Any]) -> tuple[Any, int, int, str | None]:
    project = run.get("project_name") or run.get("project") or run.get("project_id")
    phase = _safe_int(run.get("phase"))
    phase_run = _safe_int(run.get("run") or run.get("phase_run"))
    model_id = run.get("test_model_id") or run.get("source_model_id")
    return project, phase, phase_run, str(model_id) if model_id else None


def _is_hard_cap_pipeline_run(run: dict[str, Any]) -> bool:
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


def _pipeline_run_order_key(run: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(run.get("completed_at") or ""),
        str(run.get("updated_at") or ""),
        str(run.get("created_at") or ""),
        str(run.get("run_id") or ""),
    )


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _is_project_folder(project_dir: Path, pipeline_folder: Path) -> bool:
    if not project_dir.is_dir():
        return False
    if project_dir.name in {"shared_models", "training_pipelines"}:
        return False
    if (project_dir / "pipeline.json").exists() and project_dir == pipeline_folder:
        return False
    return True


def _build_row(
    *,
    project_dir: Path,
    project_config: dict[str, Any],
    run_dir: Path,
    analytics_metrics,
    build_learning_param_rows,
    read_json_if_exists,
) -> dict[str, Any] | None:
    analytics_file = run_dir / "analytics" / "run_analytics_v1.json"
    if not analytics_file.exists():
        return None

    analytics_data = read_json_if_exists(analytics_file)
    if not isinstance(analytics_data, dict) or not analytics_data:
        return None

    run_config = read_json_if_exists(run_dir / "run_config.json")
    retry_snapshot = read_json_if_exists(run_dir / "retry_snapshot.json")
    if not isinstance(retry_snapshot, dict):
        retry_snapshot = {}
    run_jitter_only = _read_run_jitter_only(run_config)

    summary = analytics_data.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    ai_block = analytics_data.get("ai") if isinstance(analytics_data.get("ai"), dict) else {}
    ai_insights = ai_block.get("input_mode_insights") if isinstance(ai_block.get("input_mode_insights"), dict) else {}

    learning_data = _read_learning_data(run_dir, analytics_data, read_json_if_exists)
    baseline_cmp = _read_baseline_comparison(learning_data)
    penalty_row = _is_gaussian_cap_penalty(learning_data, analytics_data)
    eval_summary = analytics_metrics(analytics_data)
    is_baseline = summary.get("mode") == "baseline"
    stored_learning_rows = _normalise_stored_learning_param_rows(learning_data)
    score_reference_step = _score_reference_step(baseline_cmp, run_config, summary, learning_data)
    loss_at_reference_step_run = _baseline_loss_value(baseline_cmp, "run", score_reference_step)
    loss_at_reference_step_base = _baseline_loss_value(baseline_cmp, "base", score_reference_step)

    group_multipliers = ai_insights.get("group_multipliers") if isinstance(ai_insights.get("group_multipliers"), dict) else {}
    selected_multipliers = _first_dict(
        learning_data.get("selected_multipliers"),
        learning_data.get("predicted_input_params"),
        learning_data.get("yhat_scores"),
        retry_snapshot.get("selected_multipliers"),
        _selected_multipliers_from_groups(group_multipliers),
        _selected_multipliers_from_learning_rows(stored_learning_rows, "final_multiplier"),
    )
    selected_log_multipliers = _first_dict(
        learning_data.get("selected_log_multipliers"),
        learning_data.get("selected_log_multipliers_raw"),
        learning_data.get("log_multipliers"),
        retry_snapshot.get("selected_log_multipliers"),
        _selected_log_multipliers_from_groups(group_multipliers),
        _selected_multipliers_from_learning_rows(stored_learning_rows, "log_multiplier"),
    )

    initial_params = ai_insights.get("initial_params", {}) if isinstance(ai_insights, dict) else {}
    if not isinstance(initial_params, dict):
        initial_params = {}
    run_config_model_id = run_config.get("test_model_id") if isinstance(run_config, dict) else None

    row = {
        "project_id": project_config.get("id") or project_config.get("project_id") or project_dir.name,
        "project_name": project_config.get("name") or project_dir.name,
        "run_id": run_dir.name,
        "run_name": summary.get("run_name") or run_dir.name,
        "ai_input_mode": ai_insights.get("ai_input_mode") or learning_data.get("mode"),
        "ai_selector_strategy": project_config.get("ai_selector_strategy"),
        "baseline_run_id": learning_data.get("baseline_run_id") or ai_insights.get("baseline_session_id"),
        "model_id": ai_insights.get("model_id") or learning_data.get("model_id") or run_config_model_id,
        "selected_preset": ai_insights.get("selected_preset") or learning_data.get("selected_preset") or retry_snapshot.get("selected_preset"),
        "phase": learning_data.get("phase") or (run_config.get("phase") if isinstance(run_config, dict) else None),
        "is_baseline_row": is_baseline,
        "is_warmup": learning_data.get("is_warmup", False),
        "best_loss": eval_summary.get("best_loss"),
        "best_loss_step": eval_summary.get("best_loss_step"),
        "final_loss": eval_summary.get("final_loss") or summary.get("metrics", {}).get("final_loss"),
        "final_loss_step": eval_summary.get("final_loss_step") or summary.get("major_params", {}).get("total_steps_completed"),
        "best_psnr": eval_summary.get("best_psnr"),
        "best_psnr_step": eval_summary.get("best_psnr_step"),
        "final_psnr": eval_summary.get("final_psnr"),
        "final_psnr_step": eval_summary.get("final_psnr_step"),
        "best_ssim": eval_summary.get("best_ssim"),
        "best_ssim_step": eval_summary.get("best_ssim_step"),
        "final_ssim": eval_summary.get("final_ssim"),
        "final_ssim_step": eval_summary.get("final_ssim_step"),
        "best_lpips": eval_summary.get("best_lpips"),
        "best_lpips_step": eval_summary.get("best_lpips_step"),
        "final_lpips": eval_summary.get("final_lpips"),
        "final_lpips_step": eval_summary.get("final_lpips_step"),
        "time_seconds": _safe_float_or_none(summary.get("metrics", {}).get("total_time_seconds") if isinstance(summary.get("metrics"), dict) else None),
        "time_diff_seconds": None,
        "t_best": learning_data.get("t_best"),
        "t_eval_best": learning_data.get("t_eval_best"),
        "t_end": learning_data.get("t_end"),
        "s_best": learning_data.get("s_best"),
        "s_end": learning_data.get("s_end"),
        "s_run": learning_data.get("s_run"),
        "s_base_best": baseline_cmp.get("s_base_best") or baseline_cmp.get("s_run_best"),
        "s_base_end": baseline_cmp.get("s_base_end") or baseline_cmp.get("s_run_end"),
        "s_base": baseline_cmp.get("s_base") or learning_data.get("s_base"),
        "score": learning_data.get("score") or learning_data.get("relative_score") or ai_insights.get("score"),
        "run_best_l": baseline_cmp.get("run_best_l"),
        "run_best_q": baseline_cmp.get("run_best_q"),
        "run_best_t": baseline_cmp.get("run_best_t"),
        "run_best_s": baseline_cmp.get("s_run_best") or baseline_cmp.get("run_best_s"),
        "run_best_elapsed": baseline_cmp.get("run_best_elapsed"),
        "run_end_l": baseline_cmp.get("run_end_l"),
        "run_end_q": baseline_cmp.get("run_end_q"),
        "run_end_t": baseline_cmp.get("run_end_t"),
        "run_end_s": baseline_cmp.get("s_run_end") or baseline_cmp.get("run_end_s"),
        "run_end_elapsed": baseline_cmp.get("run_end_elapsed"),
        "base_best_l": baseline_cmp.get("base_best_l"),
        "base_best_q": baseline_cmp.get("base_best_q"),
        "base_best_t": baseline_cmp.get("base_best_t"),
        "base_best_elapsed": baseline_cmp.get("base_best_elapsed"),
        "base_end_l": baseline_cmp.get("base_end_l"),
        "base_end_q": baseline_cmp.get("base_end_q"),
        "base_end_t": baseline_cmp.get("base_end_t"),
        "base_end_elapsed": baseline_cmp.get("base_end_elapsed"),
        "time_ref": baseline_cmp.get("time_ref"),
        "relative_quality_score": baseline_cmp.get("r_quality"),
        "convergence_score": baseline_cmp.get("r_convergence"),
        "auc_loss_run": baseline_cmp.get("auc_loss_run"),
        "auc_loss_base": baseline_cmp.get("auc_loss_base"),
        "score_reference_step": score_reference_step,
        "loss_at_reference_step_run": loss_at_reference_step_run,
        "loss_at_reference_step_base": loss_at_reference_step_base,
        "exploration_mode": ai_insights.get("exploration_mode") or learning_data.get("exploration_mode"),
        "remarks": learning_data.get("remarks"),
        "x_features": _first_dict(
            learning_data.get("x_features"),
            ai_insights.get("x_features"),
            ai_insights.get("features"),
            ai_insights.get("feature_details"),
        ),
        "learned_input_params": initial_params,
        "selected_multipliers": selected_multipliers,
        "selected_log_multipliers": selected_log_multipliers,
        "candidate_score_checks": retry_snapshot.get("candidate_score_checks") if isinstance(retry_snapshot.get("candidate_score_checks"), dict) else learning_data.get("candidate_score_checks"),
        "learned_input_params_source": learning_data.get("learned_input_params_source"),
        "learned_input_params_status": learning_data.get("learned_input_params_status"),
        "action_jitter_multiplier": initial_params.get("run_jitter_multiplier"),
        "run_jitter_only": run_jitter_only,
        "final_multiplier_formula": _final_multiplier_formula(learning_data, run_jitter_only),
        "learning_param_rows": (
            stored_learning_rows
            if stored_learning_rows is not None
            else build_learning_param_rows(
                initial_params,
                selected_multipliers,
                learning_data.get("selected_multipliers_raw") if isinstance(learning_data.get("selected_multipliers_raw"), dict) else None,
                initial_params.get("run_jitter_multiplier"),
                run_jitter_only=run_jitter_only,
                is_baseline_row=is_baseline,
            )
        ),
    }
    if penalty_row:
        row.update(
            {
                "score": -1.0,
                "relative_quality_score": -1.0,
                "convergence_score": None,
                "quality_score_source": "gaussian_cap_penalty",
                "exclude_from_normalization": True,
                "valid_completed_quality": False,
                "gaussian_cap_reached": True,
                "gaussian_cap_step": learning_data.get("gaussian_cap_step") or summary.get("gaussian_cap_step"),
                "gaussian_cap_count": learning_data.get("gaussian_cap_count") or summary.get("gaussian_cap_count"),
                "gaussian_hard_cap": learning_data.get("gaussian_hard_cap") or summary.get("gaussian_hard_cap"),
                "remarks": learning_data.get("remarks")
                or "Gaussian hard cap reached; partial run kept with penalty and excluded from normalization.",
            }
        )
    return row


def _read_learning_data(run_dir: Path, analytics_data: dict[str, Any], read_json_if_exists) -> dict[str, Any]:
    ai_block = analytics_data.get("ai") if isinstance(analytics_data.get("ai"), dict) else {}
    analytics_learning = ai_block.get("input_mode_learning") if isinstance(ai_block.get("input_mode_learning"), dict) else {}
    return analytics_learning if isinstance(analytics_learning, dict) else {}


def _is_gaussian_cap_penalty(learning_data: dict[str, Any], analytics_data: dict[str, Any]) -> bool:
    if str(learning_data.get("quality_score_source") or "") == "gaussian_cap_penalty":
        return True
    if bool(learning_data.get("gaussian_cap_reached")):
        return True
    if str(learning_data.get("reason") or "") == "gaussian_hard_cap_reached":
        return True

    summary = analytics_data.get("summary") if isinstance(analytics_data.get("summary"), dict) else {}
    if str(summary.get("status") or "").lower() == "partial_completed" and bool(summary.get("gaussian_cap_reached")):
        return True
    return False


def _read_baseline_comparison(learning_data: dict[str, Any]) -> dict[str, Any]:
    baseline_cmp = learning_data.get("baseline_comparison") or {}
    if isinstance(baseline_cmp, dict) and baseline_cmp:
        return baseline_cmp

    transition = learning_data.get("transition") or {}
    nested = transition.get("baseline_comparison") if isinstance(transition, dict) else {}
    return nested if isinstance(nested, dict) else {}


def _read_eval_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    return dict(metrics)


def _read_run_jitter_only(run_config: Any) -> bool:
    if not isinstance(run_config, dict):
        return False
    for cfg in (run_config, run_config.get("resolved_params"), run_config.get("requested_params")):
        if isinstance(cfg, dict) and cfg.get("run_jitter_only") is not None:
            return bool(cfg.get("run_jitter_only"))
    return False


def _score_reference_step(
    baseline_cmp: dict[str, Any] | None,
    run_config: Any,
    summary: dict[str, Any] | None,
    learning_data: dict[str, Any] | None,
) -> int | None:
    sources: list[Any] = []
    if isinstance(baseline_cmp, dict):
        sources.extend([baseline_cmp.get("score_reference_step"), baseline_cmp.get("auc_max_step")])
    if isinstance(learning_data, dict):
        sources.extend([learning_data.get("score_reference_step"), learning_data.get("comparison_step")])
    if isinstance(run_config, dict):
        sources.extend([run_config.get("score_reference_step"), run_config.get("comparison_step"), run_config.get("max_steps")])
        for nested_key in ("resolved_params", "requested_params"):
            nested = run_config.get(nested_key)
            if isinstance(nested, dict):
                sources.extend([nested.get("score_reference_step"), nested.get("comparison_step"), nested.get("max_steps")])
    if isinstance(summary, dict):
        major = summary.get("major_params") if isinstance(summary.get("major_params"), dict) else {}
        sources.extend([major.get("score_reference_step"), major.get("max_steps")])

    for value in sources:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and int(value) > 0:
            return int(value)
    return None


def _baseline_loss_value(baseline_cmp: dict[str, Any], side: str, reference_step: int | None) -> float | None:
    if not isinstance(baseline_cmp, dict):
        return None
    if reference_step is not None:
        value = _safe_float_or_none(baseline_cmp.get(f"loss_at_{int(reference_step)}_{side}"))
        if value is not None:
            return value
    for key, value in baseline_cmp.items():
        if isinstance(key, str) and key.startswith("loss_at_") and key.endswith(f"_{side}"):
            parsed = _safe_float_or_none(value)
            if parsed is not None:
                return parsed
    return None


def _normalise_stored_learning_param_rows(learning_data: dict[str, Any]) -> list[Any] | None:
    rows = learning_data.get("learning_param_rows")
    if not isinstance(rows, list):
        return None

    normalised: list[Any] = []
    for entry in rows:
        if not isinstance(entry, dict):
            normalised.append(entry)
            continue

        row = dict(entry)
        if "selected_multiplier" not in row and "predicted_multiplier" in row:
            row["selected_multiplier"] = row.get("predicted_multiplier")
        if "log_multiplier" not in row and "jitter" in row:
            row["log_multiplier"] = row.get("jitter")
        normalised.append(row)

    return normalised


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict) and value:
            return value
    return {}


def _selected_multipliers_from_groups(group_multipliers: Any) -> dict[str, float]:
    if not isinstance(group_multipliers, dict):
        return {}
    mapping = {
        "geometry_lr": "geometry_lr_mult",
        "appearance_lr": "appearance_lr_mult",
        "densification": "densification_mult",
        "scale_lr": "densification_mult",
    }
    out: dict[str, float] = {}
    for source_key, target_key in mapping.items():
        entry = group_multipliers.get(source_key)
        if isinstance(entry, dict):
            value = entry.get("multiplier")
        else:
            value = entry
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[target_key] = float(value)
    return out


def _selected_log_multipliers_from_groups(group_multipliers: Any) -> dict[str, float]:
    if not isinstance(group_multipliers, dict):
        return {}
    mapping = {
        "geometry_lr": "geometry_lr_mult",
        "appearance_lr": "appearance_lr_mult",
        "densification": "densification_mult",
        "scale_lr": "densification_mult",
    }
    out: dict[str, float] = {}
    for source_key, target_key in mapping.items():
        entry = group_multipliers.get(source_key)
        if isinstance(entry, dict):
            value = entry.get("log_action")
        else:
            value = None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[target_key] = float(value)
    return out


def _selected_multipliers_from_learning_rows(rows: Any, field: str) -> dict[str, float]:
    if not isinstance(rows, list):
        return {}

    groups = {
        "geometry_lr_mult": {"position_lr_init", "scaling_lr", "rotation_lr"},
        "appearance_lr_mult": {"feature_lr", "opacity_lr", "lambda_dssim"},
        "densification_mult": {"densify_grad_threshold", "opacity_threshold"},
    }
    values: dict[str, list[float]] = {key: [] for key in groups}

    for entry in rows:
        if not isinstance(entry, dict):
            continue
        param_key = entry.get("key")
        raw_value = entry.get(field)
        if not isinstance(param_key, str) or not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
            continue
        for group_key, param_keys in groups.items():
            if param_key in param_keys:
                values[group_key].append(float(raw_value))

    out: dict[str, float] = {}
    for group_key, group_values in values.items():
        if not group_values:
            continue
        first = group_values[0]
        if all(abs(value - first) <= 1e-9 for value in group_values):
            out[group_key] = first
    return out


def _final_multiplier_formula(learning_data: dict[str, Any], run_jitter_only: bool) -> str | None:
    formula = learning_data.get("final_multiplier_formula")
    if isinstance(formula, str) and formula:
        return formula.replace("jitter", "log_multiplier")
    if run_jitter_only:
        return "params * log_multiplier"
    if any(isinstance(learning_data.get(key), dict) for key in ("selected_multipliers", "predicted_input_params", "yhat_scores")):
        return "params * selected_multiplier * log_multiplier"
    return None


def _safe_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        if out == out and out not in (float("inf"), float("-inf")):
            return out
    return None


def _pair_normalize(run_value: float | None, base_value: float | None, *, invert: bool = False) -> tuple[float | None, float | None]:
    if run_value is None or base_value is None:
        return None, None
    lo = min(run_value, base_value)
    hi = max(run_value, base_value)
    if abs(hi - lo) < 1e-12:
        run_norm = 0.5
        base_norm = 0.5
    else:
        run_norm = (run_value - lo) / (hi - lo)
        base_norm = (base_value - lo) / (hi - lo)
    if invert:
        run_norm = 1.0 - run_norm
        base_norm = 1.0 - base_norm
    return float(run_norm), float(base_norm)


def _quality_score_pair(
    run_psnr: float | None,
    run_ssim: float | None,
    run_lpips: float | None,
    base_psnr: float | None,
    base_ssim: float | None,
    base_lpips: float | None,
) -> tuple[float | None, float | None]:
    run_values = (run_psnr, run_ssim, run_lpips)
    base_values = (base_psnr, base_ssim, base_lpips)
    if not all(isinstance(value, (int, float)) for value in run_values + base_values):
        return None, None
    run_q = (float(run_psnr) + float(run_ssim) + (1.0 - float(run_lpips))) / 3.0
    base_q = (float(base_psnr) + float(base_ssim) + (1.0 - float(base_lpips))) / 3.0
    return float(run_q), float(base_q)


def _compute_visual_score_against_baseline(run_row: dict[str, Any], baseline_row: dict[str, Any]) -> dict[str, float | None]:
    run_best_q, base_best_q = _quality_score_pair(
        _safe_float_or_none(run_row.get("best_psnr")),
        _safe_float_or_none(run_row.get("best_ssim")),
        _safe_float_or_none(run_row.get("best_lpips")),
        _safe_float_or_none(baseline_row.get("best_psnr")),
        _safe_float_or_none(baseline_row.get("best_ssim")),
        _safe_float_or_none(baseline_row.get("best_lpips")),
    )
    run_end_q, base_end_q = _quality_score_pair(
        _safe_float_or_none(run_row.get("final_psnr")),
        _safe_float_or_none(run_row.get("final_ssim")),
        _safe_float_or_none(run_row.get("final_lpips")),
        _safe_float_or_none(baseline_row.get("final_psnr")),
        _safe_float_or_none(baseline_row.get("final_ssim")),
        _safe_float_or_none(baseline_row.get("final_lpips")),
    )

    run_best_l, base_best_l = _pair_normalize(
        _safe_float_or_none(run_row.get("best_loss")),
        _safe_float_or_none(baseline_row.get("best_loss")),
        invert=True,
    )
    run_end_l, base_end_l = _pair_normalize(
        _safe_float_or_none(run_row.get("final_loss")),
        _safe_float_or_none(baseline_row.get("final_loss")),
        invert=True,
    )

    s_run = run_end_q if isinstance(run_end_q, (int, float)) else None
    s_base = base_end_q if isinstance(base_end_q, (int, float)) else None
    relative_quality_score = (s_run - s_base) if isinstance(s_run, (int, float)) and isinstance(s_base, (int, float)) else None

    reference_step = _score_reference_step({}, None, None, run_row)
    loss_at_reference_step_run = _safe_float_or_none(run_row.get("loss_at_reference_step_run"))
    loss_at_reference_step_base = _safe_float_or_none(run_row.get("loss_at_reference_step_base"))
    convergence_score = (
        loss_at_reference_step_base - loss_at_reference_step_run
        if isinstance(loss_at_reference_step_base, (int, float)) and isinstance(loss_at_reference_step_run, (int, float))
        else None
    )

    score = None
    if isinstance(relative_quality_score, (int, float)) and isinstance(convergence_score, (int, float)):
        score = relative_quality_score + convergence_score
    elif isinstance(relative_quality_score, (int, float)):
        score = relative_quality_score
    elif isinstance(convergence_score, (int, float)):
        score = convergence_score

    return {
        "run_best_l": run_best_l,
        "run_best_q": run_best_q,
        "run_best_t": 0.0 if run_best_q is not None else None,
        "run_best_s": run_best_q,
        "run_end_l": run_end_l,
        "run_end_q": run_end_q,
        "run_end_t": 0.0 if run_end_q is not None else None,
        "run_end_s": run_end_q,
        "s_best": run_best_q,
        "s_end": run_end_q,
        "s_run": s_run,
        "base_best_l": base_best_l,
        "base_best_q": base_best_q,
        "base_best_t": 0.0 if base_best_q is not None else None,
        "base_end_l": base_end_l,
        "base_end_q": base_end_q,
        "base_end_t": 0.0 if base_end_q is not None else None,
        "s_base_best": base_best_q,
        "s_base_end": base_end_q,
        "s_base": s_base,
        "relative_quality_score": relative_quality_score,
        "convergence_score": convergence_score,
        "score_reference_step": reference_step,
        "loss_at_reference_step_run": loss_at_reference_step_run,
        "loss_at_reference_step_base": loss_at_reference_step_base,
        "score": score,
    }


def _augment_rows_with_visual_scores(learning_rows: list[dict[str, Any]]) -> None:
    try:
        relative_quality_scoring.apply_pipeline_normalized_quality(learning_rows)
    except ValueError as exc:
        logger.warning("Pipeline quality normalization skipped: %s", exc)

    for row in learning_rows:
        if _is_penalty_row(row):
            _apply_penalty_score(row)
            continue
        if isinstance(row.get("q_quality"), (int, float)):
            row["s_best"] = float(row["q_quality"])
            row["s_end"] = float(row["q_quality"])
            row["s_run"] = float(row["q_quality"])
            row["run_best_q"] = float(row["q_quality"])
            row["run_best_s"] = float(row["q_quality"])
            row["run_end_q"] = float(row["q_quality"])
            row["run_end_s"] = float(row["q_quality"])
            if bool(row.get("is_baseline_row")):
                row["s_base"] = None
                row["relative_quality_score"] = None
                row["score"] = None

    rows_by_project: dict[str, list[dict[str, Any]]] = {}
    for row in learning_rows:
        project_name = str(row.get("project_name") or "").strip()
        if project_name:
            rows_by_project.setdefault(project_name, []).append(row)

    for project_rows in rows_by_project.values():
        baseline_row = next((row for row in project_rows if bool(row.get("is_baseline_row"))), None)
        if not isinstance(baseline_row, dict):
            continue

        baseline_time = _safe_float_or_none(baseline_row.get("time_seconds"))
        for row in project_rows:
            if row is baseline_row or bool(row.get("is_baseline_row")):
                continue
            if _is_penalty_row(row):
                _apply_penalty_score(row)
                if not row.get("baseline_run_id"):
                    row["baseline_run_id"] = baseline_row.get("run_id")
                continue
            computed = _compute_visual_score_against_baseline(row, baseline_row)
            run_time = _safe_float_or_none(row.get("time_seconds"))
            if run_time is not None and baseline_time is not None:
                computed["time_diff_seconds"] = run_time - baseline_time
            if isinstance(row.get("q_quality"), (int, float)) and isinstance(baseline_row.get("q_quality"), (int, float)):
                computed["s_run"] = float(row["q_quality"])
                computed["s_base"] = float(baseline_row["q_quality"])
                computed["s_best"] = float(row["q_quality"])
                computed["s_end"] = float(row["q_quality"])
                computed["run_best_q"] = float(row["q_quality"])
                computed["run_best_s"] = float(row["q_quality"])
                computed["run_end_q"] = float(row["q_quality"])
                computed["run_end_s"] = float(row["q_quality"])
                computed["base_best_q"] = float(baseline_row["q_quality"])
                computed["base_end_q"] = float(baseline_row["q_quality"])
                computed["s_base_best"] = float(baseline_row["q_quality"])
                computed["s_base_end"] = float(baseline_row["q_quality"])
                computed["relative_quality_score"] = float(row["q_quality"]) - float(baseline_row["q_quality"])
                if isinstance(computed.get("convergence_score"), (int, float)):
                    computed["score"] = computed["relative_quality_score"] + float(computed["convergence_score"])
                else:
                    computed["score"] = computed["relative_quality_score"]
            if not isinstance(computed.get("score"), (int, float)):
                continue
            for key, value in computed.items():
                if key in {
                    "relative_quality_score",
                    "convergence_score",
                    "score",
                    "s_best",
                    "s_end",
                    "s_run",
                    "s_base_best",
                    "s_base_end",
                    "s_base",
                    "run_best_q",
                    "run_best_s",
                    "run_end_q",
                    "run_end_s",
                    "base_best_q",
                    "base_end_q",
                    "time_diff_seconds",
                }:
                    row[key] = value
                elif row.get(key) is None:
                    row[key] = value
            if not row.get("baseline_run_id"):
                row["baseline_run_id"] = baseline_row.get("run_id")
            if not row.get("remarks"):
                row["remarks"] = "visual baseline score (display only)"


def _is_penalty_row(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("gaussian_cap_reached"))
        or bool(row.get("exclude_from_normalization"))
        or str(row.get("quality_score_source") or "") == "gaussian_cap_penalty"
    )


def _apply_penalty_score(row: dict[str, Any]) -> None:
    row["score"] = -1.0
    row["relative_quality_score"] = -1.0
    row["convergence_score"] = None
    row["quality_score_source"] = "gaussian_cap_penalty"
    row["exclude_from_normalization"] = True
    row["valid_completed_quality"] = False
    row["gaussian_cap_reached"] = True
    row["remarks"] = row.get("remarks") or "Gaussian hard cap reached; partial run kept with penalty and excluded from normalization."


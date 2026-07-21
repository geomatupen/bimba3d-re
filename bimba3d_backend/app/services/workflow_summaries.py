"""Read-only summaries for workflow overview pages."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from bimba3d_backend.app.services import pipeline_learning_rows
from bimba3d_backend.app.services import training_pipeline_storage
from bimba3d_backend.app.services import workflow_model_registry
from bimba3d_backend.app.services.workflow_paths import DEFAULT_WORKFLOW_PATHS, WorkflowPaths


def build_model_training_summary(paths: WorkflowPaths = DEFAULT_WORKFLOW_PATHS) -> dict[str, Any]:
    models = workflow_model_registry.list_models(paths=paths)
    ridge_models = [
        model
        for model in models
        if model.model_family in {
            "featurewise_ridge_regression",
            "compact_featurewise_ridge_regression",
        }
    ]
    mlp_models = [
        model
        for model in models
        if model.model_family in {"featurewise_mlp", "compact_featurewise_mlp", "compact_descriptor_mlp"}
    ]
    compact_models = [
        model
        for model in models
        if str(model.model_family).startswith("compact_featurewise_")
        or model.model_family in {"compact_descriptor_mlp"}
    ]
    model_rows = []
    for model in models:
        model_rows.append(
            {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "model_family": model.model_family,
                "source_training_data_id": model.source_training_data_id,
                "source_pipeline_id": model.source_pipeline_id,
                "trained_at": model.trained_at,
                "training_samples": model.training_samples,
                "selected_lambda": _first_existing(model.metrics, model.config, keys=("selected_lambda", "lambda_ridge", "ridge_lambda")),
                "best_val_loss": _first_existing(model.metrics, model.config, keys=("best_val_loss", "validation_loss", "val_loss")),
                "final_train_loss": _first_existing(model.metrics, model.config, keys=("final_train_loss", "train_loss")),
                "best_model_step": _first_existing(model.metrics, model.config, keys=("best_model_step", "best_epoch")),
                "metrics": model.metrics,
            }
        )

    return {
        "total_models": len(models),
        "ridge_count": len(ridge_models),
        "mlp_count": len(mlp_models),
        "compact_count": len(compact_models),
        "latest_model": model_rows[0] if model_rows else None,
        "total_training_samples": sum(model.training_samples or 0 for model in models),
        "models": model_rows,
    }


def build_fixed_log_space_schedule(pipeline_id: str) -> dict[str, Any]:
    pipeline = _require_pipeline(pipeline_id)
    config = pipeline.get("config", {}) if isinstance(pipeline.get("config"), dict) else {}
    schedule = config.get("pre_generated_log_multipliers") if isinstance(config.get("pre_generated_log_multipliers"), dict) else {}
    current_index = int(config.get("multiplier_current_index") or 0)

    groups = {}
    for group, values in schedule.items():
        numeric_values = [float(value) for value in values] if isinstance(values, list) else []
        last_index = current_index - 1 if current_index > 0 and numeric_values else None
        next_index = current_index if current_index < len(numeric_values) else None
        groups[group] = {
            "values": numeric_values,
            "current_index": current_index,
            "last_index": last_index,
            "last_value": numeric_values[last_index] if isinstance(last_index, int) else None,
            "next_index": next_index,
            "next_value": numeric_values[next_index] if isinstance(next_index, int) else None,
        }

    return {
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline.get("name"),
        "restart_version": int(config.get("restart_version") or 0),
        "restart_token": str(config.get("restart_token") or ""),
        "last_restart_at": config.get("last_restart_at"),
        "fixed_log_space_seed": config.get("fixed_log_space_seed"),
        "test_candidate_seed": config.get("test_candidate_seed"),
        "test_candidate_count": config.get("test_candidate_count"),
        "test_candidate_generation": config.get("test_candidate_generation"),
        "fixed_log_space_generated_at": config.get("fixed_log_space_generated_at"),
        "fixed_log_space_method": config.get("fixed_log_space_method"),
        "fixed_log_space_interval_count": config.get("fixed_log_space_interval_count"),
        "fixed_log_space_bounds": config.get("fixed_log_space_bounds"),
        "fixed_log_space_bounds_source": config.get("fixed_log_space_bounds_source"),
        "current_index": current_index,
        "groups": groups,
        "test_candidate_log_multipliers": config.get("test_candidate_log_multipliers") or {},
    }


def build_offline_data_preparation_summary(pipeline_id: str) -> dict[str, Any]:
    pipeline = _require_pipeline(pipeline_id)
    config = pipeline.get("config", {}) if isinstance(pipeline.get("config"), dict) else {}
    pipeline_type = str(config.get("pipeline_type") or pipeline.get("pipeline_type") or "offline_data").strip().lower()
    if pipeline_type == "test":
        raise ValueError("Offline Data summary is only available for offline data preparation pipelines.")

    learning_payload = pipeline_learning_rows.collect_pipeline_learning_rows(pipeline_id)
    rows = learning_payload.get("rows") if isinstance(learning_payload, dict) else []
    rows = rows if isinstance(rows, list) else []
    non_baseline_rows = [row for row in rows if isinstance(row, dict) and not bool(row.get("is_baseline_row"))]
    baseline_rows = [row for row in rows if isinstance(row, dict) and bool(row.get("is_baseline_row"))]

    return {
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline.get("name"),
        "status": pipeline.get("status"),
        "total_projects": len(config.get("projects") or []),
        "total_runs": pipeline.get("total_runs", 0),
        "completed_runs": pipeline.get("completed_runs", 0),
        "failed_runs": pipeline.get("failed_runs", 0),
        "hard_cap_runs": pipeline.get("hard_cap_runs", 0),
        "pending_runs": pipeline.get("pending_runs", 0),
        "learning_rows": len(rows),
        "baseline_rows": len(baseline_rows),
        "non_baseline_rows": len(non_baseline_rows),
        "mean_relative_score": _mean([_row_relative_score(row) for row in non_baseline_rows]),
        "best_relative_score": _max_number([_row_relative_score(row) for row in non_baseline_rows]),
        "fixed_log_space_schedule": build_fixed_log_space_schedule(pipeline_id),
        "multiplier_score_distribution": _multiplier_score_distribution(non_baseline_rows),
    }


def build_testing_pipeline_summary(pipeline_id: str) -> dict[str, Any]:
    pipeline = _require_pipeline(pipeline_id)

    config = pipeline.get("config", {}) if isinstance(pipeline.get("config"), dict) else {}
    pipeline_type = str(config.get("pipeline_type") or pipeline.get("pipeline_type") or "offline_data").strip().lower()
    if pipeline_type != "test":
        raise ValueError("Testing summary is only available for testing pipelines.")

    model_ids = _test_model_ids(config)
    runs = pipeline.get("runs", []) if isinstance(pipeline.get("runs"), list) else []
    per_model = _summarise_runs_by_model(runs, model_ids)
    preview_summary = _summarise_prediction_previews(pipeline)

    return {
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline.get("name"),
        "status": pipeline.get("status"),
        "total_test_projects": len(config.get("projects") or []),
        "models_tested": len(model_ids),
        "model_ids": model_ids,
        "completed_runs": pipeline.get("completed_runs", 0),
        "failed_runs": pipeline.get("failed_runs", 0),
        "hard_cap_runs": pipeline.get("hard_cap_runs", 0),
        "pending_runs": pipeline.get("pending_runs", 0),
        "success_rate": pipeline.get("success_rate"),
        "best_relative_score": pipeline.get("best_relative_score"),
        "mean_relative_score": pipeline.get("mean_relative_score"),
        "per_model_status": per_model,
        "prediction_preview": preview_summary,
    }


def build_testing_candidate_curves(pipeline_id: str, *, preview_key: str | None = None) -> dict[str, Any]:
    pipeline = _require_pipeline(pipeline_id)

    config = pipeline.get("config", {}) if isinstance(pipeline.get("config"), dict) else {}
    pipeline_type = str(config.get("pipeline_type") or pipeline.get("pipeline_type") or "offline_data").strip().lower()
    if pipeline_type != "test":
        raise ValueError("Candidate curves are only available for testing pipelines.")

    selected_key, preview = _select_prediction_preview(pipeline, preview_key=preview_key)
    rows = _preview_rows(preview)
    curves = []
    total_points = 0

    for row in rows:
        if not isinstance(row, dict):
            continue
        checks = row.get("candidate_score_checks")
        if not isinstance(checks, dict) or not checks:
            continue

        points = []
        highest_by_group: dict[str, dict[str, Any]] = {}
        for group, raw_points in checks.items():
            if not isinstance(raw_points, list):
                continue
            normalised_group_points = [
                _normalise_candidate_point(group=str(group), index=index, raw=raw)
                for index, raw in enumerate(raw_points)
                if isinstance(raw, dict)
            ]
            if not normalised_group_points:
                continue
            selected_point = _selected_or_highest_point(normalised_group_points)
            highest_by_group[str(group)] = selected_point
            points.extend(normalised_group_points)

        if not points:
            continue

        total_points += len(points)
        curves.append(
            {
                "project_id": row.get("project_id"),
                "project_name": row.get("project_name"),
                "model_id": row.get("model_id"),
                "mode": row.get("mode"),
                "candidate_count": len(points),
                "highest_point_by_group": highest_by_group,
                "points": points,
            }
        )

    return {
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline.get("name"),
        "preview_key": selected_key,
        "curves": curves,
        "total_curves": len(curves),
        "total_points": total_points,
    }


def _summarise_runs_by_model(runs: list[Any], model_ids: list[str]) -> list[dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {
        model_id: {"model_id": model_id, "completed": 0, "failed": 0, "total": 0}
        for model_id in model_ids
    }
    unknown = defaultdict(lambda: {"model_id": "unknown", "completed": 0, "failed": 0, "total": 0})

    for run in runs:
        if not isinstance(run, dict):
            continue
        model_id = str(
            run.get("model_id")
            or run.get("source_model_id")
            or run.get("current_test_model_id")
            or "unknown"
        )
        bucket = summary.get(model_id) or unknown[model_id]
        bucket["total"] += 1
        status = str(run.get("status") or "").strip().lower()
        if status in {"success", "completed", "partial_completed"}:
            bucket["completed"] += 1
        elif status in {"failed", "error"}:
            bucket["failed"] += 1

    return list(summary.values()) + [value for key, value in unknown.items() if key not in summary]


def _multiplier_score_distribution(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    distribution: dict[str, list[dict[str, Any]]] = {
        "geometry": [],
        "appearance": [],
        "densification": [],
    }
    for row in rows:
        log_multipliers = row.get("selected_log_multipliers") if isinstance(row.get("selected_log_multipliers"), dict) else {}
        multipliers = row.get("selected_multipliers") if isinstance(row.get("selected_multipliers"), dict) else {}
        score = _row_relative_score(row)
        for key, log_value in log_multipliers.items():
            group = _multiplier_group(str(key))
            if group is None:
                continue
            distribution[group].append(
                {
                    "project_id": row.get("project_id"),
                    "project_name": row.get("project_name"),
                    "run_id": row.get("run_id"),
                    "multiplier_key": key,
                    "log_multiplier": _safe_number(log_value),
                    "multiplier": _safe_number(multipliers.get(key)),
                    "relative_score": score,
                    "phase": row.get("phase"),
                }
            )
    return distribution


def _multiplier_group(key: str) -> str | None:
    lowered = key.lower()
    if "geometry" in lowered:
        return "geometry"
    if "appearance" in lowered:
        return "appearance"
    if "densification" in lowered or "scale" in lowered:
        return "densification"
    return None


def _row_relative_score(row: dict[str, Any]) -> float | None:
    return _first_number(
        row,
        keys=(
            "relative_quality_score",
            "score",
            "s_run",
        ),
    )


def _summarise_prediction_previews(pipeline: dict[str, Any]) -> dict[str, Any]:
    previews = pipeline.get("prediction_previews") if isinstance(pipeline.get("prediction_previews"), dict) else {}
    latest_key, latest = _select_prediction_preview(pipeline)

    if not isinstance(latest, dict):
        return {
            "latest_key": None,
            "total_previews": len(previews),
            "rows": 0,
            "candidate_curve_rows": 0,
            "candidate_points": 0,
        }

    rows = _preview_rows(latest)
    candidate_rows = 0
    candidate_points = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        checks = row.get("candidate_score_checks")
        if not isinstance(checks, dict) or not checks:
            continue
        candidate_rows += 1
        for value in checks.values():
            if isinstance(value, list):
                candidate_points += len(value)

    return {
        "latest_key": latest_key or None,
        "total_previews": len(previews),
        "rows": len(rows),
        "candidate_curve_rows": candidate_rows,
        "candidate_points": candidate_points,
        "created_at": latest.get("created_at") or latest.get("timestamp"),
    }


def _test_model_ids(config: dict[str, Any]) -> list[str]:
    raw_ids = config.get("source_model_ids") or []
    if not raw_ids and config.get("source_model_id"):
        raw_ids = [config.get("source_model_id")]
    seen: set[str] = set()
    out: list[str] = []
    for item in raw_ids:
        model_id = str(item or "").strip()
        if model_id and model_id not in seen:
            out.append(model_id)
            seen.add(model_id)
    return out


def _select_prediction_preview(pipeline: dict[str, Any], *, preview_key: str | None = None) -> tuple[str | None, dict[str, Any] | None]:
    previews = pipeline.get("prediction_previews") if isinstance(pipeline.get("prediction_previews"), dict) else {}
    selected_key = str(preview_key or pipeline.get("latest_prediction_preview_key") or "").strip()
    selected = previews.get(selected_key) if selected_key else None
    if selected is None and previews:
        selected_key, selected = sorted(
            previews.items(),
            key=lambda item: str((item[1] or {}).get("generated_at") or item[0]),
            reverse=True,
        )[0]
    return (selected_key or None), selected if isinstance(selected, dict) else None


def _preview_rows(preview: dict[str, Any] | None) -> list[Any]:
    if not isinstance(preview, dict):
        return []
    rows = preview.get("results") or preview.get("rows") or []
    return rows if isinstance(rows, list) else []


def _normalise_candidate_point(*, group: str, index: int, raw: dict[str, Any]) -> dict[str, Any]:
    predicted_score = _first_number(
        raw,
        keys=(
            "predicted_relative_score",
            "relative_score",
            "predicted_score",
            "score",
            "y",
        ),
    )
    return {
        "candidate_index": int(raw.get("candidate_index") or raw.get("index") or index),
        "group": group,
        "candidate_log_multiplier": _first_number(raw, keys=("candidate_log_multiplier", "log_multiplier", "log_value", "x")),
        "candidate_multiplier": _first_number(raw, keys=("candidate_multiplier", "multiplier", "value")),
        "predicted_relative_score": predicted_score,
        "predicted_score": _first_number(raw, keys=("predicted_score", "score")),
        "selected": bool(raw.get("selected") or raw.get("is_selected")),
        "raw": raw,
    }


def _selected_or_highest_point(points: list[dict[str, Any]]) -> dict[str, Any]:
    selected = next((point for point in points if point.get("selected")), None)
    if selected is not None:
        return selected
    return max(
        points,
        key=lambda point: (
            point.get("predicted_relative_score")
            if isinstance(point.get("predicted_relative_score"), (int, float))
            else float("-inf")
        ),
    )


def _first_number(source: dict[str, Any], *, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _safe_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mean(values: list[float | None]) -> float | None:
    numbers = [value for value in values if isinstance(value, (int, float))]
    if not numbers:
        return None
    return float(sum(numbers) / len(numbers))


def _max_number(values: list[float | None]) -> float | None:
    numbers = [value for value in values if isinstance(value, (int, float))]
    if not numbers:
        return None
    return float(max(numbers))


def _require_pipeline(pipeline_id: str) -> dict[str, Any]:
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
    if not pipeline:
        raise FileNotFoundError("Pipeline not found")
    return pipeline


def _first_existing(*sources: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            if source.get(key) is not None:
                return source.get(key)
    return None


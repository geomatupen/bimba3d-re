"""Build reusable Training Data rows from existing pipeline learning rows."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from bimba3d_backend.app.schemas.workflow_data import ArtifactReference, TrainingDataRow
from bimba3d_backend.app.services import training_data_registry
from bimba3d_backend.app.services.workflow_paths import DEFAULT_WORKFLOW_PATHS, WorkflowPaths

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


@dataclass
class TrainingDataBuildResult:
    """Outcome of converting source rows into a reusable Training Data artifact."""

    training_data_id: str
    imported_rows: int
    skipped_rows: int = 0
    hard_cap_penalty_rows: int = 0
    hard_cap_penalty: float | None = None
    errors: list[str] = field(default_factory=list)


def build_from_learning_rows(
    *,
    training_data_id: str,
    source_pipeline_id: str,
    rows: list[dict[str, Any]],
    include_hard_cap_penalty_rows: bool = False,
    training_data_config_snapshot: dict[str, Any] | None = None,
    paths: WorkflowPaths = DEFAULT_WORKFLOW_PATHS,
) -> TrainingDataBuildResult:
    """Replace a Training Data artifact with normalized rows from a pipeline.

    This function does not read project folders or pipeline folders by itself.
    Callers provide the already-collected learning rows, which keeps data
    ownership clear and makes the conversion easy to test.
    """
    converted: list[TrainingDataRow] = []
    errors: list[str] = []
    seen_rows: set[tuple[str, str, str]] = set()
    skipped_duplicates = 0
    skipped_reference_rows = 0
    hard_cap_candidates: list[tuple[int, dict[str, Any]]] = []

    for index, row in enumerate(rows):
        if isinstance(row, dict) and bool(row.get("is_baseline_row")):
            skipped_reference_rows += 1
            continue
        if isinstance(row, dict) and _is_excluded_quality_row(row):
            if include_hard_cap_penalty_rows and _is_hard_cap_quality_row(row):
                hard_cap_candidates.append((index, row))
            else:
                skipped_reference_rows += 1
            continue
        try:
            converted_row = convert_learning_row(row, source_pipeline_id=source_pipeline_id)
        except ValueError as exc:
            errors.append(f"row {index}: {exc}")
            continue

        dedupe_key = (converted_row.project_id, converted_row.run_id, converted_row.source_pipeline_id)
        if dedupe_key in seen_rows:
            skipped_duplicates += 1
            continue
        seen_rows.add(dedupe_key)
        converted.append(converted_row)

    hard_cap_penalty: float | None = None
    hard_cap_penalty_rows = 0
    hard_cap_limit_values = _hard_cap_limit_values([row for _, row in hard_cap_candidates])
    hard_cap_limit_value = hard_cap_limit_values[0] if len(hard_cap_limit_values) == 1 else None
    hard_cap_limit_missing_rows = _hard_cap_limit_missing_rows([row for _, row in hard_cap_candidates])
    if include_hard_cap_penalty_rows and hard_cap_candidates:
        if hard_cap_limit_missing_rows:
            errors.append(
                f"hard-cap penalty rows requested, but {hard_cap_limit_missing_rows} hard-cap row(s) "
                "were missing gaussian_hard_cap from the original offline preparation metadata"
            )
        hard_cap_penalty = _minimum_relative_quality_score(converted)
        if hard_cap_penalty is None:
            errors.append("hard-cap penalty rows requested, but no non-hard-cap relative_quality_score was available")
        else:
            for index, row in hard_cap_candidates:
                try:
                    converted_row = convert_hard_cap_penalty_row(
                        row,
                        source_pipeline_id=source_pipeline_id,
                        hard_cap_penalty=hard_cap_penalty,
                    )
                except ValueError as exc:
                    errors.append(f"row {index}: {exc}")
                    continue

                dedupe_key = (converted_row.project_id, converted_row.run_id, converted_row.source_pipeline_id)
                if dedupe_key in seen_rows:
                    skipped_duplicates += 1
                    continue
                seen_rows.add(dedupe_key)
                converted.append(converted_row)
                hard_cap_penalty_rows += 1

    if errors:
        training_data_registry.mark_failed(training_data_id, errors, paths=paths)
        return TrainingDataBuildResult(
            training_data_id=training_data_id,
            imported_rows=0,
            skipped_rows=len(errors),
            hard_cap_penalty_rows=0,
            hard_cap_penalty=hard_cap_penalty,
            errors=errors,
        )

    # Hard-cap penalty rows are dataset-local supervised examples only. They are
    # not written back to offline analytics and do not affect score normalization.
    try:
        config_snapshot = _required_training_data_config_snapshot(training_data_config_snapshot)
    except ValueError as exc:
        training_data_registry.mark_failed(training_data_id, [str(exc)], paths=paths)
        return TrainingDataBuildResult(
            training_data_id=training_data_id,
            imported_rows=0,
            skipped_rows=0,
            hard_cap_penalty_rows=0,
            hard_cap_penalty=hard_cap_penalty,
            errors=[str(exc)],
        )
    training_data_registry.replace_rows(
        training_data_id,
        converted,
        build_options={
            "include_hard_cap_penalty_rows": bool(include_hard_cap_penalty_rows),
        },
        build_summary={
            "source_rows": len(rows),
            "hard_cap_candidates": len(hard_cap_candidates),
            "hard_cap_penalty_rows": hard_cap_penalty_rows,
            "hard_cap_penalty": hard_cap_penalty,
            "hard_cap_penalty_source": "minimum_non_hard_cap_relative_quality_score" if hard_cap_penalty is not None else None,
            "gaussian_hard_cap": hard_cap_limit_value,
            "gaussian_hard_cap_values": hard_cap_limit_values,
            "gaussian_hard_cap_missing_rows": hard_cap_limit_missing_rows,
            "score_key": config_snapshot.get("score_key") or "relative_quality_score",
            "log_multiplier_bounds": config_snapshot["log_multiplier_bounds"],
            "log_multiplier_bounds_source": config_snapshot["log_multiplier_bounds_source"],
            "base_params": config_snapshot["base_params"],
            "base_params_source": config_snapshot["base_params_source"],
            "training_data_config_snapshot": config_snapshot,
        },
        paths=paths,
    )
    return TrainingDataBuildResult(
        training_data_id=training_data_id,
        imported_rows=len(converted),
        skipped_rows=skipped_duplicates + skipped_reference_rows,
        hard_cap_penalty_rows=hard_cap_penalty_rows,
        hard_cap_penalty=hard_cap_penalty,
        errors=[],
    )


def _is_excluded_quality_row(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("exclude_from_normalization"))
        or bool(row.get("gaussian_cap_reached"))
        or str(row.get("quality_score_source") or "") == "gaussian_cap_penalty"
    )


def _is_hard_cap_quality_row(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("gaussian_cap_reached"))
        or str(row.get("quality_score_source") or "") == "gaussian_cap_penalty"
        or str(row.get("reason") or row.get("partial_reason") or "").lower() == "gaussian_hard_cap_reached"
    )


def convert_learning_row(row: dict[str, Any], *, source_pipeline_id: str) -> TrainingDataRow:
    """Convert one pipeline learning row into the report-aligned row shape."""
    if not isinstance(row, dict):
        raise ValueError("expected object row")

    project_id = _required_text(row, "project_id")
    run_id = _required_text(row, "run_id")
    features = _clean_x_features(_required_dict(row, "x_features"))
    selected_multipliers = _optional_number_dict(row, "selected_multipliers")
    if not selected_multipliers:
        selected_multipliers = _group_multipliers_from_learning_rows(row.get("learning_param_rows"), "final_multiplier")
    selected_log_multipliers = _optional_number_dict(row, "selected_log_multipliers")
    if not selected_log_multipliers:
        selected_log_multipliers = _group_multipliers_from_learning_rows(row.get("learning_param_rows"), "log_multiplier")

    audit_metadata_keys = (
        "run_name",
        "baseline_run_id",
        "model_id",
        "selected_preset",
        "best_loss",
        "best_loss_step",
        "final_loss",
        "final_loss_step",
        "best_psnr",
        "best_psnr_step",
        "final_psnr",
        "final_psnr_step",
        "best_ssim",
        "best_ssim_step",
        "final_ssim",
        "final_ssim_step",
        "best_lpips",
        "best_lpips_step",
        "final_lpips",
        "final_lpips_step",
        "s_best",
        "s_end",
        "s_run",
        "s_base_best",
        "s_base_end",
        "s_base",
        "psnr_norm",
        "ssim_norm",
        "lpips_norm",
        "q_quality",
        "normalization_ranges",
        "quality_score_source",
        "exclude_from_normalization",
        "valid_completed_quality",
        "gaussian_cap_reached",
        "gaussian_cap_step",
        "gaussian_cap_count",
        "gaussian_hard_cap",
        "auc_loss_run",
        "auc_loss_base",
        "exploration_mode",
        "remarks",
        "final_multiplier_formula",
        "learned_input_params",
        "learning_param_rows",
    )
    metadata = {key: row.get(key) for key in audit_metadata_keys if row.get(key) is not None}

    return TrainingDataRow(
        project_id=project_id,
        project_name=_optional_text(row, "project_name"),
        run_id=run_id,
        source_pipeline_id=source_pipeline_id,
        phase=_optional_int(row, "phase"),
        is_baseline_row=bool(row.get("is_baseline_row", False)),
        x_features=features,
        selected_multipliers=selected_multipliers,
        selected_log_multipliers=selected_log_multipliers,
        relative_quality_score=_optional_number(row, "relative_quality_score"),
        convergence_score=_optional_number(row, "convergence_score"),
        score_reference_step=_optional_int(row, "score_reference_step"),
        loss_at_reference_step_run=_optional_number(row, "loss_at_reference_step_run"),
        loss_at_reference_step_base=_optional_number(row, "loss_at_reference_step_base"),
        source=ArtifactReference(
            project_id=project_id,
            run_id=run_id,
            pipeline_id=source_pipeline_id,
            stage="offline_data_preparation",
            artifact_version=_optional_text(row, "artifact_version"),
        ),
        metadata=metadata,
    )


def convert_hard_cap_penalty_row(
    row: dict[str, Any],
    *,
    source_pipeline_id: str,
    hard_cap_penalty: float,
) -> TrainingDataRow:
    """Convert a hard-cap run using the dataset-local hard-cap penalty."""
    adjusted = dict(row)
    original_relative_quality_score = row.get("relative_quality_score")
    adjusted["relative_quality_score"] = float(hard_cap_penalty)
    adjusted["score"] = float(hard_cap_penalty)
    adjusted["convergence_score"] = None
    adjusted["quality_score_source"] = "training_data_hard_cap_penalty"
    adjusted["exclude_from_normalization"] = False
    adjusted["remarks"] = (
        "Hard cap penalty row; relative_quality_score set to the minimum non-hard-cap "
        "relative_quality_score during Training Data build."
    )
    converted = convert_learning_row(adjusted, source_pipeline_id=source_pipeline_id)
    converted.metadata.update(
        {
            "is_hard_cap_penalty_row": True,
            "hard_cap_penalty": float(hard_cap_penalty),
            "hard_cap_penalty_source": "minimum_non_hard_cap_relative_quality_score",
            "gaussian_hard_cap_used": row.get("gaussian_hard_cap"),
            "original_relative_quality_score": original_relative_quality_score,
            "original_quality_score_source": row.get("quality_score_source"),
            "original_exclude_from_normalization": row.get("exclude_from_normalization"),
            "original_gaussian_cap_reached": row.get("gaussian_cap_reached"),
        }
    )
    return converted


def _minimum_relative_quality_score(rows: list[TrainingDataRow]) -> float | None:
    values = [
        float(row.relative_quality_score)
        for row in rows
        if isinstance(row.relative_quality_score, (int, float)) and math.isfinite(float(row.relative_quality_score))
    ]
    return min(values) if values else None


def _hard_cap_limit_values(rows: list[dict[str, Any]]) -> list[float]:
    values: set[float] = set()
    for row in rows:
        value = row.get("gaussian_hard_cap")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            values.add(numeric)
    return sorted(values)


def _hard_cap_limit_missing_rows(rows: list[dict[str, Any]]) -> int:
    missing = 0
    for row in rows:
        value = row.get("gaussian_hard_cap")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            missing += 1
    return missing


def _required_training_data_config_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(snapshot, dict) or not snapshot:
        raise ValueError("Training Data build requires a source-pipeline training settings snapshot")

    score_key = snapshot.get("score_key")
    if not isinstance(score_key, str) or not score_key.strip():
        raise ValueError("Training Data build snapshot is missing score_key")

    bounds = snapshot.get("log_multiplier_bounds")
    if not isinstance(bounds, dict) or not bounds:
        raise ValueError("Training Data build snapshot is missing log_multiplier_bounds")
    required_bounds = {"geometry_lr", "appearance_lr", "scale_lr"}
    missing_bounds = sorted(required_bounds.difference(bounds.keys()))
    if missing_bounds:
        raise ValueError(f"Training Data build snapshot is missing log_multiplier_bounds for: {missing_bounds}")
    for key in required_bounds:
        value = bounds.get(key)
        if (
            not isinstance(value, (list, tuple))
            or len(value) < 2
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in value[:2])
        ):
            raise ValueError(f"Training Data build snapshot has invalid log_multiplier_bounds for: {key}")
    bounds_source = snapshot.get("log_multiplier_bounds_source")
    if not isinstance(bounds_source, str) or not bounds_source.strip():
        raise ValueError("Training Data build snapshot is missing log_multiplier_bounds_source")

    base_params = snapshot.get("base_params")
    if not isinstance(base_params, dict) or not base_params:
        raise ValueError("Training Data build snapshot is missing base_params")
    missing_base_params = sorted(COMPACT_BASE_PARAM_DEFAULTS.keys() - base_params.keys())
    if missing_base_params:
        raise ValueError(f"Training Data build snapshot is missing base_params for: {missing_base_params}")
    for key in COMPACT_BASE_PARAM_DEFAULTS:
        value = base_params.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Training Data build snapshot has invalid base_params value for: {key}")
    base_params_source = snapshot.get("base_params_source")
    if not isinstance(base_params_source, str) or not base_params_source.strip():
        raise ValueError("Training Data build snapshot is missing base_params_source")

    return {
        "score_key": score_key.strip(),
        "log_multiplier_bounds": bounds,
        "log_multiplier_bounds_source": bounds_source.strip(),
        "base_params": base_params,
        "base_params_source": base_params_source.strip(),
    }


def _required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing required text field '{key}'")
    return value.strip()


def _optional_text(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_int(row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _optional_number(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _required_dict(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"missing required object field '{key}'")
    return value


def _clean_x_features(features: dict[str, Any]) -> dict[str, Any]:
    """Drop diagnostic missing flags from final supervised Training Data rows."""
    return {
        str(key): value
        for key, value in features.items()
        if not str(key).lower().endswith("_missing")
    }


def _optional_number_dict(row: dict[str, Any], key: str) -> dict[str, float]:
    value = row.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"field '{key}' must be an object")

    result: dict[str, float] = {}
    for item_key, item_value in value.items():
        if isinstance(item_value, bool) or not isinstance(item_value, (int, float)):
            raise ValueError(f"field '{key}.{item_key}' must be numeric")
        result[str(item_key)] = float(item_value)
    return result


def _group_multipliers_from_learning_rows(rows: Any, field: str) -> dict[str, float]:
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

    result: dict[str, float] = {}
    for group_key, group_values in values.items():
        if not group_values:
            continue
        first = group_values[0]
        if all(abs(value - first) <= 1e-9 for value in group_values):
            result[group_key] = first
    return result


"""Append completed project test results to reusable Training Data artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from bimba3d_backend.app.schemas.workflow_data import ArtifactReference, TrainingDataRow
from bimba3d_backend.app.services import training_data_registry
from bimba3d_backend.app.services.pipeline_learning_rows import _build_row
from bimba3d_backend.app.services.project_json import read_json_if_exists


def append_project_run_to_training_data(
    *,
    project_dir: Path,
    project_id: str,
    run_id: str,
    training_data_id: str,
    params: dict[str, Any],
) -> training_data_registry.TrainingDataManifest:
    """Add one completed project test run to a ready Training Data artifact."""
    manifest = training_data_registry.read_manifest(training_data_id)
    if manifest is None:
        raise FileNotFoundError(f"Training Data target not found: {training_data_id}")
    if not training_data_registry.is_usable_manifest(manifest):
        raise ValueError(
            f"Training Data target '{manifest.name}' is not usable. Build it before project updates."
        )

    run_dir = project_dir / "runs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run folder not found for Training Data update: {run_id}")

    project_config = read_json_if_exists(project_dir / "config.json")
    if not isinstance(project_config, dict):
        project_config = {"id": project_id, "name": project_dir.name}

    from bimba3d_backend.app.api.projects import _analytics_metrics  # pylint: disable=import-outside-toplevel

    row_payload = _build_row(
        project_dir=project_dir,
        project_config=project_config,
        run_dir=run_dir,
        analytics_metrics=_analytics_metrics,
        build_learning_param_rows=_build_learning_param_rows,
        read_json_if_exists=read_json_if_exists,
    )
    if not isinstance(row_payload, dict):
        raise ValueError(f"Run '{run_id}' does not contain normalized learning data.")

    x_features = _clean_x_features(_require_dict(row_payload.get("x_features"), "x_features"))
    selected_multipliers = _require_numeric_dict(row_payload.get("selected_multipliers"), "selected_multipliers")
    selected_log_multipliers = _numeric_dict(row_payload.get("selected_log_multipliers"))
    relative_quality_score = _optional_float(row_payload.get("relative_quality_score"))
    if relative_quality_score is None:
        raise ValueError(f"Run '{run_id}' does not contain a relative quality score.")

    metadata = {
        "added_from": "project_test",
        "source_workflow_model_id": _clean(params.get("source_workflow_model_id") or params.get("source_model_id")),
        "source_model_name": _clean(params.get("source_model_name")),
        "source_model_family": _clean(params.get("source_model_family") or params.get("ai_selector_strategy")),
        "source_training_data_id": _clean(params.get("source_training_data_id")),
        "ai_input_mode": _clean(params.get("ai_input_mode")),
        "ai_selector_strategy": _clean(params.get("ai_selector_strategy")),
        "model_evaluation_step": _evaluation_step(row_payload, params),
    }
    audit_metadata_keys = (
        "run_name",
        "baseline_run_id",
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
        "score",
        "s_best",
        "s_end",
        "s_run",
        "s_base_best",
        "s_base_end",
        "s_base",
        "auc_loss_run",
        "auc_loss_base",
        "exploration_mode",
        "remarks",
        "final_multiplier_formula",
        "learned_input_params",
        "learning_param_rows",
    )
    metadata.update({key: row_payload.get(key) for key in audit_metadata_keys if row_payload.get(key) is not None})
    metadata = {key: value for key, value in metadata.items() if value not in (None, "")}

    new_row = TrainingDataRow(
        project_id=str(row_payload.get("project_id") or project_id),
        project_name=row_payload.get("project_name"),
        run_id=run_id,
        source_pipeline_id=manifest.source_pipeline_id,
        phase=_optional_int(row_payload.get("phase")),
        is_baseline_row=False,
        x_features=x_features,
        selected_multipliers=selected_multipliers,
        selected_log_multipliers=selected_log_multipliers,
        relative_quality_score=relative_quality_score,
        convergence_score=_optional_float(row_payload.get("convergence_score")),
        score_reference_step=_optional_int(row_payload.get("score_reference_step")),
        loss_at_reference_step_run=_optional_float(row_payload.get("loss_at_reference_step_run")),
        loss_at_reference_step_base=_optional_float(row_payload.get("loss_at_reference_step_base")),
        source=ArtifactReference(
            project_id=project_id,
            run_id=run_id,
            pipeline_id=manifest.source_pipeline_id,
            stage="testing",
            path=str(run_dir),
        ),
        metadata=metadata,
    )

    rows = training_data_registry.read_rows(training_data_id)
    rows = _replace_matching_row(rows, new_row)
    return training_data_registry.replace_rows(training_data_id, rows)


def _replace_matching_row(rows: list[TrainingDataRow], new_row: TrainingDataRow) -> list[TrainingDataRow]:
    model_id = str(new_row.metadata.get("source_workflow_model_id") or "")
    eval_step = str(new_row.metadata.get("model_evaluation_step") or "")
    out: list[TrainingDataRow] = []
    replaced = False
    for row in rows:
        same_source = row.project_id == new_row.project_id and row.run_id == new_row.run_id
        same_model = str(row.metadata.get("source_workflow_model_id") or "") == model_id
        same_step = str(row.metadata.get("model_evaluation_step") or "") == eval_step
        if same_source and same_model and same_step:
            if not replaced:
                out.append(new_row)
                replaced = True
            continue
        out.append(row)
    if not replaced:
        out.append(new_row)
    return out


def _evaluation_step(row: dict[str, Any], params: dict[str, Any]) -> int | None:
    for source in (
        params.get("model_evaluation_step"),
        row.get("score_reference_step"),
        params.get("score_reference_step"),
        params.get("comparison_step"),
        params.get("max_steps"),
    ):
        value = _optional_int(source)
        if value is not None:
            return value
    return None


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Run learning data is missing {field_name}.")
    return dict(value)


def _clean_x_features(features: dict[str, Any]) -> dict[str, Any]:
    """Keep final Training Data descriptors free of diagnostic missing flags."""
    return {
        str(key): value
        for key, value in features.items()
        if not str(key).lower().endswith("_missing")
    }


def _require_numeric_dict(value: Any, field_name: str) -> dict[str, float]:
    data = _numeric_dict(value)
    if not data:
        raise ValueError(f"Run learning data is missing {field_name}.")
    return data


def _numeric_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        number = _optional_float(raw)
        if number is not None:
            out[str(key)] = number
    return out


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _build_learning_param_rows(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    from bimba3d_backend.app.api.projects import _build_learning_param_rows as build_rows

    return build_rows(*args, **kwargs)

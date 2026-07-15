"""Model-training service for report-aligned Ridge and MLP models."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bimba3d_backend.app.schemas.workflow_data import ModelFamily, TrainingDataRow, WorkflowModelManifest
from bimba3d_backend.app.services import training_data_registry, workflow_model_registry
from bimba3d_backend.app.services.workflow_paths import DEFAULT_WORKFLOW_PATHS, WorkflowPaths
from bimba3d_backend.worker.ai_input_modes.compact_featurewise_mlp import train_compact_featurewise_mlp_model
from bimba3d_backend.worker.ai_input_modes.compact_featurewise_ridge_regression import (
    train_compact_featurewise_ridge_model,
)
from bimba3d_backend.worker.ai_input_modes.feature_schema import GROUP_KEYS
from bimba3d_backend.worker.ai_input_modes.featurewise_mlp import train_featurewise_mlp_model
from bimba3d_backend.worker.ai_input_modes.featurewise_ridge_regression import (
    train_featurewise_ridge_model,
)

RIDGE_LAMBDA_CANDIDATES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


@dataclass(frozen=True)
class ModelTrainingOptions:
    """Configuration for one offline model-training run."""

    model_family: ModelFamily
    model_name: str
    source_training_data_id: str
    source_pipeline_id: str | None = None
    lambda_ridge: float | None = None
    # Prediction fallback grid size stored in the artifact; explicit testing grids take precedence.
    candidate_points: int = 30
    include_phases: list[int] | None = None
    include_run_ids: list[str] | None = None


def train_model_from_training_data(
    options: ModelTrainingOptions,
    *,
    paths: WorkflowPaths = DEFAULT_WORKFLOW_PATHS,
) -> WorkflowModelManifest:
    """Train a report model from a reusable Training Data artifact."""
    rows = training_data_registry.read_rows(options.source_training_data_id, paths=paths)
    training_rows = _filter_rows(rows, options)
    if not training_rows:
        raise ValueError("No valid Training Data rows are available for model training.")

    training_log = [
        _training_log(f"Loaded {len(rows)} row{'' if len(rows) == 1 else 's'} from Training Data."),
        _training_log(
            f"Using {len(training_rows)} non-baseline row{'' if len(training_rows) == 1 else 's'} after phase/run filters."
        ),
    ]
    model_id = _build_model_id(options.model_name, options.model_family)
    model_dir = paths.model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    training_log.append(_training_log(f"Prepared model artifact folder: {model_id}."))

    if options.model_family == "featurewise_ridge_regression":
        return _train_ridge(options, training_rows, model_id=model_id, model_dir=model_dir, paths=paths, training_log=training_log)
    if options.model_family == "featurewise_mlp":
        return _train_mlp(options, training_rows, model_id=model_id, model_dir=model_dir, paths=paths, training_log=training_log)
    if options.model_family == "compact_featurewise_ridge_regression":
        return _train_compact_ridge(options, training_rows, model_id=model_id, model_dir=model_dir, paths=paths, training_log=training_log)
    if options.model_family == "compact_featurewise_mlp":
        return _train_compact_mlp(options, training_rows, model_id=model_id, model_dir=model_dir, paths=paths, training_log=training_log)

    raise ValueError(f"Unsupported model family: {options.model_family}")


def _train_ridge(
    options: ModelTrainingOptions,
    rows: list[TrainingDataRow],
    *,
    model_id: str,
    model_dir: Path,
    paths: WorkflowPaths,
    training_log: list[dict[str, str]],
) -> WorkflowModelManifest:
    score_key = "relative_quality_score"
    model_evaluation_step = _model_evaluation_step(rows)
    multiplier_bounds, multiplier_bounds_source = _source_multiplier_bounds(options.source_training_data_id)
    train_rows = _ridge_rows(rows)
    if not train_rows:
        raise ValueError("No rows contain valid multipliers and quality scores for Ridge training.")
    training_log.extend(
        [
            _training_log(f"Validated one model evaluation step: {model_evaluation_step}."),
            _training_log(f"Loaded multiplier bounds from {multiplier_bounds_source}."),
            _training_log(f"Prepared {len(train_rows)} Ridge score row{'' if len(train_rows) == 1 else 's'}."),
        ]
    )

    lambda_candidates = (
        [float(options.lambda_ridge)]
        if isinstance(options.lambda_ridge, (int, float))
        else [float(value) for value in RIDGE_LAMBDA_CANDIDATES]
    )
    candidate_points = int(max(5, options.candidate_points))
    training_log.append(
        _training_log(
            f"Searching {len(lambda_candidates)} Ridge lambda candidate{'' if len(lambda_candidates) == 1 else 's'} "
            f"with {candidate_points} multiplier candidate points."
        )
    )

    best_model: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_lambda: float | None = None
    best_theta_norms: dict[str, float] | None = None
    lambda_report: list[dict[str, float]] = []

    for candidate_lambda in lambda_candidates:
        model, metrics, theta_norms = train_featurewise_ridge_model(
            rows=train_rows,
            score_key=score_key,
            lambda_ridge=float(candidate_lambda),
            candidate_points=candidate_points,
            group_bounds=multiplier_bounds,
        )
        avg_mse = float(metrics.get("avg_val_mse", float("inf")))
        lambda_report.append({"lambda_ridge": float(candidate_lambda), "avg_val_mse": avg_mse})
        training_log.append(_training_log(f"Lambda {candidate_lambda:g}: average validation MSE {avg_mse:.6f}."))

        if best_metrics is None or avg_mse < float(best_metrics.get("avg_val_mse", float("inf"))):
            best_model = model
            best_metrics = metrics
            best_lambda = float(candidate_lambda)
            best_theta_norms = theta_norms

    if best_model is None or best_metrics is None or best_lambda is None or best_theta_norms is None:
        raise RuntimeError("Failed to train Featurewise Ridge Regression.")
    training_log.append(_training_log(f"Selected lambda {best_lambda:g} with lowest validation MSE.", "success"))

    artifact_path = model_dir / "featurewise_ridge_model.json"
    metadata_path = model_dir / "featurewise_ridge_metadata.json"
    payload = {
        "schema": "featurewise_ridge_regression_v1",
        "model_family": "featurewise_ridge_regression",
        "score_key": score_key,
        "generated_at": _utc_now(),
        "lambda_ridge": best_lambda,
        "metrics": {
            "lambda_selected": best_lambda,
            "selected_lambda": best_lambda,
            "train_fit_metrics": best_metrics,
            "theta_norms": best_theta_norms,
            "lambda_search": lambda_report,
            "candidate_points": candidate_points,
            "training_log": training_log + [_training_log("Saved Ridge model artifact and metadata.")],
        },
        "model": best_model,
    }
    _write_json(artifact_path, payload)
    _write_json(
        metadata_path,
        {
            "model_family": "featurewise_ridge_regression",
            "source_training_data_id": options.source_training_data_id,
            "source_pipeline_id": options.source_pipeline_id,
            "training_samples": int(best_model.get("runs", 0)),
            "model_evaluation_step": model_evaluation_step,
            "lambda_selected": best_lambda,
            "candidate_points": candidate_points,
        },
    )

    return workflow_model_registry.register_model(
        model_id=model_id,
        model_name=options.model_name,
        model_family="featurewise_ridge_regression",
        source_training_data_id=options.source_training_data_id,
        source_pipeline_id=options.source_pipeline_id,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        training_samples=int(best_model.get("runs", 0)),
        model_evaluation_step=model_evaluation_step,
        metrics=payload["metrics"],
        config={
            "candidate_points": candidate_points,
            "lambda_selected": best_lambda,
            "lambda_candidates": lambda_candidates,
            "lambda_search_count": len(lambda_report),
            "model_evaluation_step": model_evaluation_step,
            "score_reference_step": model_evaluation_step,
            "log_multiplier_bounds": multiplier_bounds,
            "log_multiplier_bounds_source": multiplier_bounds_source,
            "include_phases": options.include_phases,
            "include_run_ids": options.include_run_ids,
        },
        paths=paths,
    )


def _train_compact_ridge(
    options: ModelTrainingOptions,
    rows: list[TrainingDataRow],
    *,
    model_id: str,
    model_dir: Path,
    paths: WorkflowPaths,
    training_log: list[dict[str, str]],
) -> WorkflowModelManifest:
    score_key = "relative_quality_score"
    model_evaluation_step = _model_evaluation_step(rows)
    multiplier_bounds, multiplier_bounds_source = _source_multiplier_bounds(options.source_training_data_id)
    train_rows = _ridge_rows(rows)
    if not train_rows:
        raise ValueError("No rows contain valid multipliers and quality scores for Compact Ridge training.")
    training_log.extend(
        [
            _training_log(f"Validated one model evaluation step: {model_evaluation_step}."),
            _training_log(f"Loaded multiplier bounds from {multiplier_bounds_source}."),
            _training_log(f"Prepared {len(train_rows)} compact Ridge score row{'' if len(train_rows) == 1 else 's'}."),
        ]
    )

    lambda_candidates = (
        [float(options.lambda_ridge)]
        if isinstance(options.lambda_ridge, (int, float))
        else [float(value) for value in RIDGE_LAMBDA_CANDIDATES]
    )
    candidate_points = int(max(5, options.candidate_points))
    training_log.append(
        _training_log(
            f"Searching {len(lambda_candidates)} compact Ridge lambda candidate{'' if len(lambda_candidates) == 1 else 's'} "
            f"with {candidate_points} multiplier candidate points."
        )
    )
    best_model: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_lambda: float | None = None
    best_theta_norm: float | None = None
    lambda_report: list[dict[str, float]] = []

    for candidate_lambda in lambda_candidates:
        model, metrics, theta_norm = train_compact_featurewise_ridge_model(
            rows=train_rows,
            score_key=score_key,
            lambda_ridge=float(candidate_lambda),
            candidate_points=candidate_points,
            group_bounds=multiplier_bounds,
        )
        avg_mse = float(metrics.get("avg_val_mse", metrics.get("mse", float("inf"))))
        lambda_report.append({"lambda_ridge": float(candidate_lambda), "avg_val_mse": avg_mse})
        training_log.append(_training_log(f"Lambda {candidate_lambda:g}: average validation MSE {avg_mse:.6f}."))
        if best_metrics is None or avg_mse < float(best_metrics.get("avg_val_mse", best_metrics.get("mse", float("inf")))):
            best_model = model
            best_metrics = metrics
            best_lambda = float(candidate_lambda)
            best_theta_norm = float(theta_norm)

    if best_model is None or best_metrics is None or best_lambda is None or best_theta_norm is None:
        raise RuntimeError("Failed to train Compact Featurewise Ridge Regression.")
    training_log.append(_training_log(f"Selected lambda {best_lambda:g} with lowest validation MSE.", "success"))

    artifact_path = model_dir / "compact_featurewise_ridge_model.json"
    metadata_path = model_dir / "compact_featurewise_ridge_metadata.json"
    payload = {
        "schema": "compact_featurewise_ridge_regression_v1",
        "model_family": "compact_featurewise_ridge_regression",
        "score_key": score_key,
        "generated_at": _utc_now(),
        "lambda_ridge": best_lambda,
        "metrics": {
            "lambda_selected": best_lambda,
            "selected_lambda": best_lambda,
            "train_fit_metrics": best_metrics,
            "theta_norm": best_theta_norm,
            "lambda_search": lambda_report,
            "candidate_points": candidate_points,
            "training_log": training_log + [_training_log("Saved compact Ridge model artifact and metadata.")],
        },
        "model": best_model,
    }
    _write_json(artifact_path, payload)
    _write_json(
        metadata_path,
        {
            "model_family": "compact_featurewise_ridge_regression",
            "source_training_data_id": options.source_training_data_id,
            "source_pipeline_id": options.source_pipeline_id,
            "training_samples": int(best_model.get("runs", 0)),
            "model_evaluation_step": model_evaluation_step,
            "lambda_selected": best_lambda,
            "candidate_points": candidate_points,
        },
    )

    return workflow_model_registry.register_model(
        model_id=model_id,
        model_name=options.model_name,
        model_family="compact_featurewise_ridge_regression",
        source_training_data_id=options.source_training_data_id,
        source_pipeline_id=options.source_pipeline_id,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        training_samples=int(best_model.get("runs", 0)),
        model_evaluation_step=model_evaluation_step,
        metrics=payload["metrics"],
        config={
            "candidate_points": candidate_points,
            "lambda_selected": best_lambda,
            "lambda_candidates": lambda_candidates,
            "lambda_search_count": len(lambda_report),
            "model_evaluation_step": model_evaluation_step,
            "score_reference_step": model_evaluation_step,
            "log_multiplier_bounds": multiplier_bounds,
            "log_multiplier_bounds_source": multiplier_bounds_source,
            "include_phases": options.include_phases,
            "include_run_ids": options.include_run_ids,
        },
        paths=paths,
    )


def _train_compact_mlp(
    options: ModelTrainingOptions,
    rows: list[TrainingDataRow],
    *,
    model_id: str,
    model_dir: Path,
    paths: WorkflowPaths,
    training_log: list[dict[str, str]],
) -> WorkflowModelManifest:
    model_evaluation_step = _model_evaluation_step(rows)
    multiplier_bounds, multiplier_bounds_source = _source_multiplier_bounds(options.source_training_data_id)
    training_data = [_mlp_entry(row) for row in rows if row.relative_quality_score is not None and row.selected_multipliers]
    if not training_data:
        raise ValueError("No rows contain valid multipliers and quality scores for Compact MLP training.")
    training_log.extend(
        [
            _training_log(f"Validated one model evaluation step: {model_evaluation_step}."),
            _training_log(f"Loaded multiplier bounds from {multiplier_bounds_source}."),
            _training_log(f"Prepared {len(training_data)} compact MLP score row{'' if len(training_data) == 1 else 's'}."),
            _training_log("Started compact MLP optimizer with train/validation split and early stopping."),
        ]
    )

    result = train_compact_featurewise_mlp_model(
        training_data=training_data,
        save_dir=model_dir,
        group_bounds=multiplier_bounds,
    )
    if not result.get("trained"):
        raise RuntimeError(f"Compact Featurewise MLP training failed: {result.get('error', 'unknown error')}")
    training_log.extend(_mlp_training_result_logs(result, compact=True))

    artifact_path = Path(str(result.get("model_path") or "")).expanduser()
    metadata_path = Path(str(result.get("metadata_path") or "")).expanduser() if result.get("metadata_path") else None
    return workflow_model_registry.register_model(
        model_id=model_id,
        model_name=options.model_name,
        model_family="compact_featurewise_mlp",
        source_training_data_id=options.source_training_data_id,
        source_pipeline_id=options.source_pipeline_id,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        training_samples=int(result.get("training_samples") or 0),
        model_evaluation_step=model_evaluation_step,
        metrics={
            "best_val_loss": result.get("best_val_loss"),
            "final_train_loss": result.get("final_train_loss"),
            "final_val_loss": result.get("final_val_loss"),
            "epochs_trained": result.get("epochs_trained"),
            "max_epochs": result.get("max_epochs"),
            "best_epoch": result.get("best_epoch"),
            "early_stopping_patience": result.get("early_stopping_patience"),
            "total_parameters": result.get("total_parameters"),
            "learning_rate": result.get("learning_rate"),
            "weight_decay": result.get("weight_decay"),
            "hidden": result.get("hidden"),
            "dropout": result.get("dropout"),
            "seed": result.get("seed"),
            "training_log": training_log + [_training_log("Saved compact MLP checkpoint and metadata.")],
        },
        config={
            "candidate_points": result.get("candidate_points"),
            "hidden": result.get("hidden"),
            "dropout": result.get("dropout"),
            "learning_rate": result.get("learning_rate"),
            "weight_decay": result.get("weight_decay"),
            "max_epochs": result.get("max_epochs"),
            "early_stopping_patience": result.get("early_stopping_patience"),
            "seed": result.get("seed"),
            "model_evaluation_step": model_evaluation_step,
            "score_reference_step": model_evaluation_step,
            "log_multiplier_bounds": multiplier_bounds,
            "log_multiplier_bounds_source": multiplier_bounds_source,
            "include_phases": options.include_phases,
            "include_run_ids": options.include_run_ids,
        },
        paths=paths,
    )


def _train_mlp(
    options: ModelTrainingOptions,
    rows: list[TrainingDataRow],
    *,
    model_id: str,
    model_dir: Path,
    paths: WorkflowPaths,
    training_log: list[dict[str, str]],
) -> WorkflowModelManifest:
    model_evaluation_step = _model_evaluation_step(rows)
    multiplier_bounds, multiplier_bounds_source = _source_multiplier_bounds(options.source_training_data_id)
    training_data = [_mlp_entry(row) for row in rows if row.relative_quality_score is not None and row.selected_multipliers]
    if not training_data:
        raise ValueError("No rows contain valid multipliers and quality scores for MLP training.")
    training_log.extend(
        [
            _training_log(f"Validated one model evaluation step: {model_evaluation_step}."),
            _training_log(f"Loaded multiplier bounds from {multiplier_bounds_source}."),
            _training_log(f"Prepared {len(training_data)} featurewise MLP score row{'' if len(training_data) == 1 else 's'}."),
            _training_log("Started featurewise MLP optimizer with train/validation split and early stopping."),
        ]
    )

    result = train_featurewise_mlp_model(
        training_data=training_data,
        save_dir=model_dir,
        group_bounds=multiplier_bounds,
    )
    if not result.get("trained"):
        raise RuntimeError(f"Featurewise MLP training failed: {result.get('error', 'unknown error')}")
    training_log.extend(_mlp_training_result_logs(result, compact=False))

    artifact_path = Path(str(result.get("model_path") or "")).expanduser()
    metadata_path = Path(str(result.get("metadata_path") or "")).expanduser() if result.get("metadata_path") else None
    return workflow_model_registry.register_model(
        model_id=model_id,
        model_name=options.model_name,
        model_family="featurewise_mlp",
        source_training_data_id=options.source_training_data_id,
        source_pipeline_id=options.source_pipeline_id,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        training_samples=int(result.get("training_samples") or 0),
        model_evaluation_step=model_evaluation_step,
        metrics={
            "best_val_loss": result.get("best_val_loss"),
            "final_train_loss": result.get("final_train_loss"),
            "final_val_loss": result.get("final_val_loss"),
            "epochs_trained": result.get("epochs_trained"),
            "max_epochs": result.get("max_epochs"),
            "best_epoch": result.get("best_epoch"),
            "early_stopping_patience": result.get("early_stopping_patience"),
            "total_parameters": result.get("total_parameters"),
            "learning_rate": result.get("learning_rate"),
            "weight_decay": result.get("weight_decay"),
            "hidden": result.get("hidden"),
            "dropout": result.get("dropout"),
            "training_log": training_log + [_training_log("Saved featurewise MLP checkpoint and metadata.")],
        },
        config={
            "candidate_points": result.get("candidate_points"),
            "hidden": result.get("hidden"),
            "dropout": result.get("dropout"),
            "learning_rate": result.get("learning_rate"),
            "weight_decay": result.get("weight_decay"),
            "max_epochs": result.get("max_epochs"),
            "early_stopping_patience": result.get("early_stopping_patience"),
            "model_evaluation_step": model_evaluation_step,
            "score_reference_step": model_evaluation_step,
            "log_multiplier_bounds": multiplier_bounds,
            "log_multiplier_bounds_source": multiplier_bounds_source,
            "include_phases": options.include_phases,
            "include_run_ids": options.include_run_ids,
        },
        paths=paths,
    )


def _filter_rows(rows: list[TrainingDataRow], options: ModelTrainingOptions) -> list[TrainingDataRow]:
    out: list[TrainingDataRow] = []
    include_run_ids = set(options.include_run_ids or [])
    include_phases = set(options.include_phases or [])
    for row in rows:
        if row.is_baseline_row:
            continue
        if include_run_ids and row.run_id not in include_run_ids:
            continue
        if include_phases and row.phase not in include_phases:
            continue
        out.append(row)
    return out


def _model_evaluation_step(rows: list[TrainingDataRow]) -> int:
    steps: set[int] = set()
    missing = 0
    for row in rows:
        if row.score_reference_step is None:
            missing += 1
            continue
        steps.add(int(row.score_reference_step))
    if missing:
        raise ValueError("Training Data rows must include score_reference_step before model training.")
    if not steps:
        raise ValueError("Training Data rows do not contain a model evaluation step.")
    if len(steps) != 1:
        values = ", ".join(str(value) for value in sorted(steps))
        raise ValueError(f"Training Data rows mix multiple model evaluation steps: {values}")
    return next(iter(steps))


def _source_multiplier_bounds(training_data_id: str) -> tuple[dict[str, list[float]], str]:
    manifest = training_data_registry.read_manifest(training_data_id)
    if manifest is not None and isinstance(manifest.build_summary, dict):
        bounds = manifest.build_summary.get("log_multiplier_bounds")
        source = manifest.build_summary.get("log_multiplier_bounds_source")
        if isinstance(bounds, dict) and bounds:
            if not isinstance(source, str) or not source.strip():
                raise ValueError("Training Data manifest is missing log_multiplier_bounds_source")
            return _normalise_bounds(bounds), source.strip()
    raise ValueError("Training Data manifest is missing log_multiplier_bounds; rebuild the dataset before model training.")


def _normalise_bounds(bounds: dict[str, Any]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for group in ("geometry_lr", "appearance_lr", "scale_lr"):
        raw = bounds.get(group)
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            lo = _safe_positive_float(raw[0], group)
            hi = _safe_positive_float(raw[1], group)
        else:
            raise ValueError(f"Training Data manifest is missing log_multiplier_bounds for {group}")
        if hi < lo:
            lo, hi = hi, lo
        out[group] = [lo, hi]
    return out


def _safe_positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Training Data manifest has invalid multiplier bound for {label}")
    if isinstance(value, (int, float)):
        parsed = float(value)
        if parsed > 0:
            return parsed
    raise ValueError(f"Training Data manifest has invalid multiplier bound for {label}")


def _ridge_rows(rows: list[TrainingDataRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.relative_quality_score is None or not row.selected_multipliers:
            continue
        out.append(
            {
                "project_name": row.project_name,
                "x_features": row.x_features,
                "selected_multipliers": row.selected_multipliers,
                "relative_quality_score": float(row.relative_quality_score),
            }
        )
    return out


def _mlp_entry(row: TrainingDataRow) -> dict[str, Any]:
    return {
        "features": row.x_features,
        "selected_multipliers": row.selected_multipliers,
        "selected_log_multipliers": row.selected_log_multipliers,
        "relative_quality_score": float(row.relative_quality_score or 0.0),
        "convergence_score": float(row.convergence_score or 0.0),
        "run_id": row.run_id,
        "project_name": row.project_name,
        "phase": row.phase,
        "yhat_scores": {},
    }


def _training_log(message: str, level: str = "info") -> dict[str, str]:
    return {"level": level, "message": message}


def _mlp_training_result_logs(result: dict[str, Any], *, compact: bool) -> list[dict[str, str]]:
    logs: list[dict[str, str]] = []
    samples = int(result.get("training_samples") or 0)
    split = max(1, int(0.8 * samples)) if samples else 0
    val_count = max(0, samples - split)
    if samples:
        logs.append(_training_log(f"Split data into {split} training row{'' if split == 1 else 's'} and {val_count} validation row{'' if val_count == 1 else 's'}."))

    if compact:
        input_dim = result.get("input_dim")
        hidden = result.get("hidden")
        if input_dim is not None:
            logs.append(_training_log(f"Built compact MLP network: input_dim={input_dim}, hidden={hidden}."))
    else:
        geo_dim = result.get("geo_dim")
        app_dim = result.get("app_dim")
        den_dim = result.get("den_dim")
        hidden = result.get("hidden")
        if geo_dim is not None or app_dim is not None or den_dim is not None:
            logs.append(_training_log(f"Built featurewise MLP heads: geo_dim={geo_dim}, app_dim={app_dim}, den_dim={den_dim}, hidden={hidden}."))

    epochs = result.get("epochs_trained")
    max_epochs = result.get("max_epochs")
    best_epoch = result.get("best_epoch")
    best_val = result.get("best_val_loss")
    final_train = result.get("final_train_loss")
    final_val = result.get("final_val_loss")
    if epochs is not None:
        logs.append(
            _training_log(
                f"Finished {epochs}/{max_epochs or '-'} epochs; best epoch {best_epoch or '-'}; best validation loss {_format_metric(best_val)}.",
                "success",
            )
        )
    if final_train is not None or final_val is not None:
        logs.append(_training_log(f"Final train loss {_format_metric(final_train)}; final validation loss {_format_metric(final_val)}."))

    lr = result.get("learning_rate")
    weight_decay = result.get("weight_decay")
    dropout = result.get("dropout")
    params = result.get("total_parameters")
    logs.append(_training_log(f"Optimizer config: lr={lr}, weight_decay={weight_decay}, dropout={dropout}, parameters={params}."))
    return logs


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return "-"


def _build_model_id(model_name: str, model_family: ModelFamily) -> str:
    token = re.sub(r"[^a-zA-Z0-9_-]+", "-", model_name.strip().lower())
    token = re.sub(r"-+", "-", token).strip("-_") or "model"
    family = {
        "featurewise_ridge_regression": "ridge",
        "featurewise_mlp": "mlp",
        "compact_featurewise_ridge_regression": "compact-ridge",
        "compact_featurewise_mlp": "compact-mlp",
    }.get(model_family, "model")
    return f"model_{family}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{token}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


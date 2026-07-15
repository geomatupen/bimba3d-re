"""Featurewise Ridge Regression helpers for thesis model training.

NON_COMPACT_FEATUREWISE: legacy group-wise path kept for comparison.

This module is the report-named home for Ridge model operations. It currently
delegates to the existing working implementation while the backend is being
restructured, so behavior stays stable during the rename.
"""
from __future__ import annotations

from typing import Any

from bimba3d_backend.scripts.train_offline_models import (
    _compute_metrics,
    _theta_norms,
    _train_model_on_rows,
    compute_feature_scaler,
)
from bimba3d_backend.worker.ai_input_modes.feature_schema import GROUP_KEYS
from bimba3d_backend.worker.ai_input_modes.featurewise_ridge_regression_runtime import (
    _build_updates as build_featurewise_ridge_updates,
    record_run_penalty_featurewise_ridge,
    select_featurewise_ridge_multipliers,
    update_from_run_featurewise_ridge,
)


def train_featurewise_ridge_model(
    *,
    rows: list[dict[str, Any]],
    score_key: str,
    lambda_ridge: float,
    candidate_points: int,
    group_bounds: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    """Train one Ridge candidate and return model, metrics, and theta norms."""
    scalers = {group: compute_feature_scaler(rows, group) for group in GROUP_KEYS}
    model = _train_model_on_rows(
        rows=rows,
        score_key=score_key,
        lambda_ridge=float(lambda_ridge),
        scalers=scalers,
        candidate_points=int(max(5, candidate_points)),
        group_bounds=group_bounds,
    )
    metrics = _compute_metrics(model, rows, score_key, group_bounds=group_bounds)
    theta_norms = _theta_norms(model)
    return model, metrics, theta_norms


__all__ = [
    "GROUP_KEYS",
    "build_featurewise_ridge_updates",
    "compute_feature_scaler",
    "record_run_penalty_featurewise_ridge",
    "select_featurewise_ridge_multipliers",
    "train_featurewise_ridge_model",
    "update_from_run_featurewise_ridge",
]


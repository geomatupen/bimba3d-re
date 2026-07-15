"""Featurewise MLP helpers for thesis model training and prediction.

NON_COMPACT_FEATUREWISE: legacy group-wise path kept for comparison.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from bimba3d_backend.worker.ai_input_modes.featurewise_mlp_runtime import (
    predict_featurewise_neural_multipliers,
    train_featurewise_neural_model,
)


def train_featurewise_mlp_model(
    *,
    training_data: list[dict[str, Any]],
    save_dir: Path,
    group_bounds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the thesis Featurewise MLP using the existing working trainer."""
    return train_featurewise_neural_model(
        training_data=training_data,
        save_dir=save_dir,
        topk_per_project=0,
        group_bounds=group_bounds,
    )


def predict_featurewise_mlp_multipliers(
    *,
    shared_models_dir: Path,
    features: dict[str, Any],
    mode_name: str = "featurewise",
    params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Predict multipliers with the latest Featurewise MLP checkpoint."""
    return predict_featurewise_neural_multipliers(
        shared_models_dir=shared_models_dir,
        mode_name=mode_name,
        features=features,
        params=params,
    )


__all__ = ["predict_featurewise_mlp_multipliers", "train_featurewise_mlp_model"]

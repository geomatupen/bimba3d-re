from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModeContext:
    metadata_image_dir: Path
    processing_image_dir: Path
    colmap_dir: Path
    params: dict[str, Any]


@dataclass
class PresetResult:
    mode: str
    updates: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) < 1e-12:
        return 0.0
    return float(numerator) / float(denominator)


PRESET_UPDATES: dict[str, dict[str, Any]] = {
    "conservative": {
        "feature_lr_mult": 0.80,
        "position_lr_init_mult": 0.80,
        "scaling_lr_mult": 0.84,
        "opacity_lr_mult": 0.84,
        "rotation_lr_mult": 0.90,
        "densify_grad_threshold_mult": 1.40,
        "opacity_threshold_mult": 1.30,
        "lambda_dssim_mult": 1.10,
    },
    "balanced": {
        "feature_lr_mult": 1.00,
        "position_lr_init_mult": 1.00,
        "scaling_lr_mult": 1.00,
        "opacity_lr_mult": 1.00,
        "rotation_lr_mult": 1.00,
        "densify_grad_threshold_mult": 1.00,
        "opacity_threshold_mult": 1.00,
        "lambda_dssim_mult": 1.00,
    },
    "geometry_fast": {
        "feature_lr_mult": 0.90,
        "position_lr_init_mult": 1.16,
        "scaling_lr_mult": 1.16,
        "opacity_lr_mult": 0.96,
        "rotation_lr_mult": 1.12,
        "densify_grad_threshold_mult": 0.84,
        "opacity_threshold_mult": 0.84,
        "lambda_dssim_mult": 0.96,
    },
    "appearance_fast": {
        "feature_lr_mult": 1.20,
        "position_lr_init_mult": 0.96,
        "scaling_lr_mult": 0.96,
        "opacity_lr_mult": 1.16,
        "rotation_lr_mult": 0.98,
        "densify_grad_threshold_mult": 1.10,
        "opacity_threshold_mult": 1.04,
        "lambda_dssim_mult": 1.16,
    },
}


def apply_preset_updates(params: dict[str, Any], preset_name: str) -> dict[str, Any]:
    preset = PRESET_UPDATES.get(preset_name, PRESET_UPDATES["balanced"])

    feature_lr = float(params.get("feature_lr", 2.5e-3))
    position_lr_init = float(params.get("position_lr_init", 1.6e-4))
    scaling_lr = float(params.get("scaling_lr", 5.0e-3))
    opacity_lr = float(params.get("opacity_lr", 5.0e-2))
    rotation_lr = float(params.get("rotation_lr", 1.0e-3))
    densify_grad_threshold = float(params.get("densify_grad_threshold", 2.0e-4))
    opacity_threshold = float(params.get("opacity_threshold", 0.005))
    lambda_dssim = float(params.get("lambda_dssim", 0.2))

    updates = {
        "preset_name": preset_name,
        "feature_lr": clamp_float(feature_lr * float(preset["feature_lr_mult"]), 5e-4, 8e-3),
        "position_lr_init": clamp_float(position_lr_init * float(preset["position_lr_init_mult"]), 5e-5, 5e-4),
        "scaling_lr": clamp_float(scaling_lr * float(preset["scaling_lr_mult"]), 1e-4, 2e-2),
        "opacity_lr": clamp_float(opacity_lr * float(preset["opacity_lr_mult"]), 1e-3, 1e-1),
        "rotation_lr": clamp_float(rotation_lr * float(preset["rotation_lr_mult"]), 1e-4, 1e-2),
        "densify_grad_threshold": clamp_float(
            densify_grad_threshold * float(preset["densify_grad_threshold_mult"]),
            5e-5,
            5e-4,
        ),
        "opacity_threshold": clamp_float(
            opacity_threshold * float(preset["opacity_threshold_mult"]),
            0.001,
            0.02,
        ),
        "lambda_dssim": clamp_float(lambda_dssim * float(preset["lambda_dssim_mult"]), 0.05, 0.5),
    }
    return updates


def keep_only_feature_keys(features: dict[str, Any], allowed_keys: set[str]) -> dict[str, Any]:
    return {k: features[k] for k in features.keys() if k in allowed_keys}

"""Compact feature schema shared by compact Ridge and compact MLP models."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

COMPACT_FEATUREWISE_SCHEMA_NAME = "compact_scene_descriptors_v1"
COMPACT_MODEL_GROUP_KEYS = ["geometry_lr_mult", "appearance_lr_mult", "densification_mult"]
COMPACT_FEATURE_KEYS = [
    "intercept",
    "gsd_median",
    "overlap_proxy",
    "coverage_spread",
    "camera_angle_bucket",
    "heading_consistency",
    "texture_density",
    "blur_motion_risk",
    "terrain_roughness_proxy",
    "vegetation_complexity_score",
    "vegetation_cover_percentage",
]
COMPACT_LOG_TRANSFORM_FEATURES = {"gsd_median"}
COMPACT_SCALER_STD_FLOOR = 0.01

COMPACT_GROUP_BOUNDS: dict[str, tuple[float, float]] = {
    "geometry_lr_mult": (0.5, 2.0),
    "appearance_lr_mult": (0.5, 2.0),
    "densification_mult": (0.7, 1.4285714286),
}

COMPACT_GROUP_BOUND_ALIASES: dict[str, str] = {
    "geometry_lr": "geometry_lr_mult",
    "geometry": "geometry_lr_mult",
    "appearance_lr": "appearance_lr_mult",
    "appearance": "appearance_lr_mult",
    "scale_lr": "densification_mult",
    "densification": "densification_mult",
    "densification_lr": "densification_mult",
}

COMPACT_PARAMETER_GROUPS: dict[str, list[str]] = {
    "geometry_lr_mult": ["position_lr_init_mult", "scaling_lr_mult", "rotation_lr_mult"],
    "appearance_lr_mult": ["feature_lr_mult", "opacity_lr_mult", "lambda_dssim_mult"],
    "densification_mult": ["densify_grad_threshold_mult", "opacity_threshold_mult"],
}


def normalise_compact_group_bounds(bounds: dict[str, Any] | None = None) -> dict[str, tuple[float, float]]:
    out = dict(COMPACT_GROUP_BOUNDS)
    if not isinstance(bounds, dict):
        return out
    for raw_key, raw_value in bounds.items():
        group_key = COMPACT_GROUP_BOUND_ALIASES.get(str(raw_key), str(raw_key))
        if group_key not in out:
            continue
        if not isinstance(raw_value, (list, tuple)) or len(raw_value) < 2:
            continue
        try:
            lo = float(raw_value[0])
            hi = float(raw_value[1])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(lo) or not math.isfinite(hi) or lo <= 0.0 or hi <= 0.0:
            continue
        if hi < lo:
            lo, hi = hi, lo
        out[group_key] = (lo, hi)
    return out


def _feature_float(features: dict[str, Any], key: str, default: float, *, positive: bool = False) -> float:
    value = features.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        parsed = float(value)
        if math.isfinite(parsed) and (parsed > 0.0 if positive else True):
            return parsed
    return default


def compact_raw_feature(features: dict[str, Any], name: str) -> float:
    if name == "gsd_median":
        return _feature_float(features, "gsd_median", 0.05, positive=True)
    if name == "overlap_proxy":
        return _feature_float(features, "overlap_proxy", 0.5)
    if name == "coverage_spread":
        return _feature_float(features, "coverage_spread", 0.5)
    if name == "camera_angle_bucket":
        return _feature_float(features, "camera_angle_bucket", 0.0)
    if name == "heading_consistency":
        return _feature_float(features, "heading_consistency", 0.5)
    if name == "texture_density":
        return _feature_float(features, "texture_density", 0.5)
    if name == "blur_motion_risk":
        return _feature_float(features, "blur_motion_risk", 0.5)
    if name == "terrain_roughness_proxy":
        if "terrain_roughness_proxy" in features:
            return _feature_float(features, "terrain_roughness_proxy", 0.5)
        return _feature_float(features, "terrain_roughness", 0.5)
    if name == "vegetation_complexity_score":
        if "vegetation_complexity_score" in features:
            return _feature_float(features, "vegetation_complexity_score", 0.5)
        return _feature_float(features, "vegetation_complexity", 0.5)
    if name == "vegetation_cover_percentage":
        if "vegetation_cover_percentage" in features:
            return _feature_float(features, "vegetation_cover_percentage", 0.5)
        return _feature_float(features, "vegetation_cover", 0.5)
    return 0.0


def compact_transform_feature(name: str, value: float) -> float:
    return math.log(max(float(value), 1e-9)) if name in COMPACT_LOG_TRANSFORM_FEATURES else float(value)


def build_compact_feature_scaler(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    values_by_feature: dict[str, list[float]] = {name: [] for name in COMPACT_FEATURE_KEYS if name != "intercept"}
    for row in rows:
        features = row.get("x_features")
        if not isinstance(features, dict) or not features:
            continue
        for name in values_by_feature:
            values_by_feature[name].append(compact_transform_feature(name, compact_raw_feature(features, name)))

    scaler: dict[str, dict[str, float]] = {}
    for name, values in values_by_feature.items():
        if not values:
            scaler[name] = {"mean": 0.0, "std": 1.0}
            continue
        arr = np.array(values, dtype=np.float64)
        scaler[name] = {
            "mean": float(np.mean(arr)),
            "std": max(float(np.std(arr)), COMPACT_SCALER_STD_FLOOR),
        }
    return scaler


def build_compact_vector(features: dict[str, Any], scaler: dict[str, dict[str, float]] | None = None) -> np.ndarray:
    values: list[float] = []
    scaler = scaler or {}
    for name in COMPACT_FEATURE_KEYS:
        if name == "intercept":
            values.append(1.0)
            continue
        transformed = compact_transform_feature(name, compact_raw_feature(features, name))
        stats = scaler.get(name)
        if isinstance(stats, dict):
            std = max(float(stats.get("std", 1.0)), COMPACT_SCALER_STD_FLOOR)
            values.append((transformed - float(stats.get("mean", 0.0))) / std)
        else:
            values.append(transformed)
    return np.array(values, dtype=np.float64)


def compact_action_logs_from_multipliers(
    multipliers: dict[str, Any],
    *,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray | None:
    bounds = normalise_compact_group_bounds(bounds)
    logs: list[float] = []
    for group_key in COMPACT_MODEL_GROUP_KEYS:
        direct = multipliers.get(group_key)
        value: float | None = None
        if isinstance(direct, (int, float)) and not isinstance(direct, bool) and float(direct) > 0:
            value = float(direct)
        else:
            members = [
                float(multipliers[key])
                for key in COMPACT_PARAMETER_GROUPS[group_key]
                if isinstance(multipliers.get(key), (int, float)) and not isinstance(multipliers.get(key), bool) and float(multipliers[key]) > 0
            ]
            if members:
                value = float(np.mean(members))
        if value is None:
            return None
        lo, hi = bounds[group_key]
        value = min(max(value, lo), hi)
        logs.append(float(math.log(max(value, 1e-9))))
    return np.array(logs, dtype=np.float64)


def build_compact_score_design_vector(x_context: np.ndarray, action_logs: np.ndarray) -> np.ndarray:
    """One compact model feature map: [x, a_geo/app/dens, a^2, x_without_intercept * each a]."""
    x = np.asarray(x_context, dtype=np.float64)
    a = np.asarray(action_logs, dtype=np.float64)
    interactions = np.concatenate([x[1:] * float(value) for value in a], axis=0)
    return np.concatenate([x, a, a * a, interactions], axis=0)


def expand_compact_group_multipliers(group_multipliers: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    multipliers: dict[str, float] = {}
    log_multipliers: dict[str, float] = {}
    for group_key, member_keys in COMPACT_PARAMETER_GROUPS.items():
        group_mult = float(group_multipliers[group_key])
        group_log = float(math.log(max(group_mult, 1e-9)))
        for member_key in member_keys:
            multipliers[member_key] = group_mult
            log_multipliers[member_key] = group_log
    return multipliers, log_multipliers


__all__ = [
    "COMPACT_FEATUREWISE_SCHEMA_NAME",
    "COMPACT_FEATURE_KEYS",
    "COMPACT_GROUP_BOUNDS",
    "COMPACT_MODEL_GROUP_KEYS",
    "COMPACT_PARAMETER_GROUPS",
    "build_compact_feature_scaler",
    "build_compact_score_design_vector",
    "build_compact_vector",
    "compact_action_logs_from_multipliers",
    "expand_compact_group_multipliers",
    "normalise_compact_group_bounds",
]

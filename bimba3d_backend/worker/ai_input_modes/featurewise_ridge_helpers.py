"""Shared helpers for the legacy three-model Featurewise Ridge path."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

FEATUREWISE_GROUP_FEATURES: dict[str, list[str]] = {
    "geometry_lr_mult": [
        "intercept",
        "focal_norm",
        "gsd_norm",
        "overlap_proxy",
        "coverage_spread",
        "camera_angle_bucket",
        "heading_consistency",
        "blur_motion_risk",
        "terrain_roughness_proxy",
        "vegetation_cover",
    ],
    "appearance_lr_mult": [
        "intercept",
        "iso_norm",
        "image_resolution_norm",
        "blur_motion_risk",
        "texture_density",
        "vegetation_cover",
        "vegetation_complexity",
    ],
    "densification_mult": [
        "intercept",
        "gsd_norm",
        "overlap_proxy",
        "coverage_spread",
        "camera_angle_bucket",
        "texture_density",
        "blur_motion_risk",
        "terrain_roughness_proxy",
        "vegetation_complexity",
    ],
}

GROUP_KEYS = ["geometry_lr_mult", "appearance_lr_mult", "densification_mult"]

GROUP_BOUNDS: dict[str, tuple[float, float]] = {
    "geometry_lr_mult": (0.5, 2.0),
    "appearance_lr_mult": (0.5, 2.0),
    "densification_mult": (0.7, 1.4285714286),
}

GROUP_BOUND_ALIASES: dict[str, str] = {
    "geometry_lr": "geometry_lr_mult",
    "geometry": "geometry_lr_mult",
    "appearance_lr": "appearance_lr_mult",
    "appearance": "appearance_lr_mult",
    "scale_lr": "densification_mult",
    "densification": "densification_mult",
    "densification_lr": "densification_mult",
}

GROUP_MULTIPLIERS_MAP: dict[str, list[str]] = {
    "geometry_lr_mult": ["position_lr_init_mult", "scaling_lr_mult", "rotation_lr_mult"],
    "appearance_lr_mult": ["feature_lr_mult", "opacity_lr_mult", "lambda_dssim_mult"],
    "densification_mult": ["densify_grad_threshold_mult", "opacity_threshold_mult"],
}

SCALER_STD_FLOOR = 0.01
LOG_TRANSFORM_FEATURES = {"focal_norm", "gsd_norm", "iso_norm"}


def normalise_group_bounds(bounds: dict[str, Any] | None = None) -> dict[str, tuple[float, float]]:
    """Return positive multiplier bounds keyed by group model names."""
    out = dict(GROUP_BOUNDS)
    if not isinstance(bounds, dict):
        return out

    for raw_key, raw_value in bounds.items():
        group_key = GROUP_BOUND_ALIASES.get(str(raw_key), str(raw_key))
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


def _extract_raw_feature(features: dict[str, Any], name: str) -> float:
    if name == "focal_norm":
        return float(features.get("focal_length_mm", 24.0) or 24.0)
    if name == "iso_norm":
        return float(features.get("iso", 400.0) or 400.0)
    if name == "image_resolution_norm":
        w = float(features.get("img_width_median", 4000.0) or 4000.0)
        h = float(features.get("img_height_median", 3000.0) or 3000.0)
        return (w * h) / 1e6
    if name == "gsd_norm":
        return float(features.get("gsd_median", 0.05) or 0.05)
    if name == "overlap_proxy":
        return float(features.get("overlap_proxy", 0.5) or 0.5)
    if name == "coverage_spread":
        return float(features.get("coverage_spread", 0.5) or 0.5)
    if name == "camera_angle_bucket":
        return float(features.get("camera_angle_bucket", 0) or 0)
    if name == "heading_consistency":
        return float(features.get("heading_consistency", 0.5) or 0.5)
    if name == "texture_density":
        return float(features.get("texture_density", 0.5) or 0.5)
    if name == "blur_motion_risk":
        return float(features.get("blur_motion_risk", 0.5) or 0.5)
    if name == "terrain_roughness_proxy":
        return float(features.get("terrain_roughness_proxy", features.get("terrain_roughness", 0.5)) or 0.5)
    if name == "vegetation_cover":
        value = features.get("vegetation_cover_percentage", features.get("vegetation_cover", 0.5))
        return float(value or 0.5)
    if name == "vegetation_complexity":
        value = features.get("vegetation_complexity_score", features.get("vegetation_complexity", 0.5))
        return float(value or 0.5)
    return 0.0


def _apply_log_transform(name: str, value: float) -> float:
    if name in LOG_TRANSFORM_FEATURES:
        return math.log(max(value, 1e-9))
    return value


def compute_feature_scaler(rows: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, float]]:
    feature_names = FEATUREWISE_GROUP_FEATURES[group_key]
    values_by_feature: dict[str, list[float]] = {name: [] for name in feature_names if name != "intercept"}

    for row in rows:
        x_features = row.get("x_features")
        if not isinstance(x_features, dict) or not x_features:
            continue
        for name in feature_names:
            if name == "intercept":
                continue
            raw = _extract_raw_feature(x_features, name)
            values_by_feature[name].append(_apply_log_transform(name, raw))

    scaler: dict[str, dict[str, float]] = {}
    for name, values in values_by_feature.items():
        if not values:
            scaler[name] = {"mean": 0.0, "std": 1.0}
            continue
        arr = np.array(values, dtype=np.float64)
        std = max(float(np.std(arr)), SCALER_STD_FLOOR)
        scaler[name] = {"mean": float(np.mean(arr)), "std": std}
    return scaler


def apply_scaler_to_vector(features: dict[str, Any], group_key: str, scaler: dict[str, dict[str, float]]) -> np.ndarray:
    feature_names = FEATUREWISE_GROUP_FEATURES[group_key]
    values: list[float] = []
    for name in feature_names:
        if name == "intercept":
            values.append(1.0)
            continue
        raw = _extract_raw_feature(features, name)
        transformed = _apply_log_transform(name, raw)
        stats = scaler.get(name, {"mean": 0.0, "std": 1.0})
        std = max(float(stats.get("std", 1.0)), SCALER_STD_FLOOR)
        values.append((transformed - float(stats.get("mean", 0.0))) / std)
    return np.array(values, dtype=np.float64)


def _extract_group_log_action(
    row: dict[str, Any],
    group_key: str,
    group_bounds: dict[str, tuple[float, float]] | None = None,
) -> float | None:
    bounds = normalise_group_bounds(group_bounds)
    selected = row.get("selected_multipliers")
    if not isinstance(selected, dict):
        return None
    direct_val = selected.get(group_key)
    if isinstance(direct_val, (int, float)):
        value = float(direct_val)
        if math.isfinite(value) and value > 0.0:
            lo, hi = bounds[group_key]
            value = min(max(value, lo), hi)
            return float(math.log(value))

    keys = GROUP_MULTIPLIERS_MAP.get(group_key, [])
    if not keys:
        return None
    raw_value = selected.get(keys[0])
    if not isinstance(raw_value, (int, float)):
        return None
    value = float(raw_value)
    if not math.isfinite(value) or value <= 0.0:
        return None
    lo, hi = bounds[group_key]
    value = min(max(value, lo), hi)
    return float(math.log(value))


def _build_score_design_vector(x_context: np.ndarray, action_log: float) -> np.ndarray:
    interactions = x_context[1:] * float(action_log)
    return np.concatenate(
        [
            x_context,
            np.array([float(action_log), float(action_log) * float(action_log)], dtype=np.float64),
            interactions.astype(np.float64),
        ]
    )


def _build_empty_model(
    lambda_ridge: float,
    scalers: dict[str, dict[str, dict[str, float]]],
    candidate_points: int,
    group_bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    bounds = normalise_group_bounds(group_bounds)
    models: dict[str, Any] = {}
    for key in GROUP_KEYS:
        d_ctx = len(FEATUREWISE_GROUP_FEATURES[key])
        d_phi = d_ctx + 2 + (d_ctx - 1)
        models[key] = {
            "A": (np.eye(d_phi, dtype=np.float64) * float(lambda_ridge)).tolist(),
            "b": np.zeros(d_phi, dtype=np.float64).tolist(),
            "n": 0,
            "design_dim": d_phi,
            "context_features": FEATUREWISE_GROUP_FEATURES[key],
            "design": "x_plus_a_plus_a2_plus_xa",
            "action_space": "log_multiplier",
            "action_bounds": [float(math.log(bounds[key][0])), float(math.log(bounds[key][1]))],
        }

    return {
        "version": 6,
        "model_family": "featurewise_ridge_regression",
        "mode": "exif_compact_featurewise",
        "lambda_ridge": float(lambda_ridge),
        "runs": 0,
        "score_mean": 0.0,
        "feature_scalers": scalers,
        "candidate_points": int(max(5, candidate_points)),
        "log_multiplier_bounds": {key: [float(bounds[key][0]), float(bounds[key][1])] for key in GROUP_KEYS},
        "models": models,
    }


def _train_model_on_rows(
    rows: list[dict[str, Any]],
    score_key: str,
    lambda_ridge: float,
    scalers: dict[str, dict[str, dict[str, float]]],
    candidate_points: int,
    group_bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    bounds = normalise_group_bounds(group_bounds)
    model = _build_empty_model(lambda_ridge, scalers, candidate_points, bounds)

    runs = 0
    score_sum = 0.0
    for row in rows:
        x_features = row.get("x_features")
        score = row.get(score_key)
        if not isinstance(x_features, dict) or not isinstance(score, (int, float)):
            continue
        score_f = float(score)
        if not math.isfinite(score_f):
            continue

        used_row = False
        for group_key in GROUP_KEYS:
            action_log = _extract_group_log_action(row, group_key, bounds)
            if action_log is None:
                continue
            x = apply_scaler_to_vector(x_features, group_key, scalers.get(group_key, {}))
            phi = _build_score_design_vector(x, action_log)

            md = model["models"][group_key]
            A = np.array(md["A"], dtype=np.float64)
            b = np.array(md["b"], dtype=np.float64)
            A += np.outer(phi, phi)
            b += score_f * phi
            md["A"] = A.tolist()
            md["b"] = b.tolist()
            md["n"] = int(md["n"]) + 1
            used_row = True

        if used_row:
            runs += 1
            score_sum += score_f

    model["runs"] = runs
    model["score_mean"] = (score_sum / runs) if runs > 0 else 0.0
    return model


def _predict_score_for_action(
    model: dict[str, Any],
    group_key: str,
    x_features: dict[str, Any],
    action_log: float,
) -> float | None:
    md = model.get("models", {}).get(group_key)
    if not isinstance(md, dict):
        return None
    x = apply_scaler_to_vector(x_features, group_key, model.get("feature_scalers", {}).get(group_key, {}))
    phi = _build_score_design_vector(x, action_log)

    A = np.array(md.get("A"), dtype=np.float64)
    b = np.array(md.get("b"), dtype=np.float64)
    try:
        theta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        try:
            theta = np.linalg.pinv(A) @ b
        except Exception:
            return None
    return float(phi @ theta)


def _compute_metrics(
    model: dict[str, Any],
    data_rows: list[dict[str, Any]],
    score_key: str,
    group_bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    bounds = normalise_group_bounds(group_bounds)
    out: dict[str, Any] = {}
    group_mses: list[float] = []

    for group_key in GROUP_KEYS:
        targets: list[float] = []
        abs_err: list[float] = []
        sq_err: list[float] = []

        for row in data_rows:
            x_features = row.get("x_features")
            score = row.get(score_key)
            if not isinstance(x_features, dict) or not isinstance(score, (int, float)):
                continue
            action_log = _extract_group_log_action(row, group_key, bounds)
            if action_log is None:
                continue
            pred = _predict_score_for_action(model, group_key, x_features, action_log)
            if pred is None or not math.isfinite(pred):
                continue
            target = float(score)
            targets.append(target)
            residual = pred - target
            sq_err.append(residual * residual)
            abs_err.append(abs(residual))

        if not sq_err:
            out[group_key] = {
                "mse": float("inf"),
                "rmse": float("inf"),
                "mae": float("inf"),
                "r_squared": 0.0,
                "samples": 0,
            }
            continue

        mse = float(np.mean(sq_err))
        y_mean = float(np.mean(targets))
        ss_tot = float(np.sum((np.array(targets) - y_mean) ** 2))
        ss_res = float(np.sum(sq_err))

        out[group_key] = {
            "mse": mse,
            "rmse": float(math.sqrt(mse)),
            "mae": float(np.mean(abs_err)),
            "r_squared": 1.0 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0,
            "samples": len(sq_err),
        }
        group_mses.append(mse)

    out["avg_val_mse"] = float(np.mean(group_mses)) if group_mses else float("inf")
    return out


def _theta_norms(model: dict[str, Any]) -> dict[str, float]:
    norms: dict[str, float] = {}
    for group_key in GROUP_KEYS:
        md = model.get("models", {}).get(group_key, {})
        A = np.array(md.get("A", []), dtype=np.float64)
        b = np.array(md.get("b", []), dtype=np.float64)
        if A.size == 0 or b.size == 0:
            norms[group_key] = 0.0
            continue
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(A) @ b
        norms[group_key] = float(np.linalg.norm(theta))
    return norms


__all__ = [
    "FEATUREWISE_GROUP_FEATURES",
    "GROUP_BOUNDS",
    "GROUP_KEYS",
    "GROUP_MULTIPLIERS_MAP",
    "apply_scaler_to_vector",
    "compute_feature_scaler",
    "normalise_group_bounds",
    "_build_score_design_vector",
    "_compute_metrics",
    "_theta_norms",
    "_train_model_on_rows",
]

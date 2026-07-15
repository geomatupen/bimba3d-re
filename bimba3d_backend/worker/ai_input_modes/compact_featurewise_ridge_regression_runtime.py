"""Runtime and training helpers for one-model compact Featurewise Ridge."""
from __future__ import annotations

import itertools
import json
import math
import numpy as np
from pathlib import Path
from typing import Any

from .common import clamp_float
from .compact_featurewise_schema import (
    COMPACT_MODEL_GROUP_KEYS,
    build_compact_feature_scaler,
    build_compact_score_design_vector,
    build_compact_vector,
    compact_action_logs_from_multipliers,
    expand_compact_group_multipliers,
    normalise_compact_group_bounds,
)


def train_compact_featurewise_ridge_model(
    *,
    rows: list[dict[str, Any]],
    score_key: str,
    lambda_ridge: float,
    candidate_points: int,
    group_bounds: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    bounds = normalise_compact_group_bounds(group_bounds)
    scaler = build_compact_feature_scaler(rows)
    model = _empty_model(lambda_ridge=lambda_ridge, scaler=scaler, candidate_points=candidate_points, bounds=bounds)
    score_sum = 0.0
    runs = 0

    for row in rows:
        features = row.get("x_features")
        score = row.get(score_key)
        selected = row.get("selected_multipliers")
        if not isinstance(features, dict) or not isinstance(selected, dict) or not isinstance(score, (int, float)):
            continue
        score_f = float(score)
        if not math.isfinite(score_f):
            continue
        action_logs = compact_action_logs_from_multipliers(selected, bounds=bounds)
        if action_logs is None:
            continue

        x = build_compact_vector(features, scaler)
        phi = build_compact_score_design_vector(x, action_logs)
        model["A"] = (np.array(model["A"], dtype=np.float64) + np.outer(phi, phi)).tolist()
        model["b"] = (np.array(model["b"], dtype=np.float64) + score_f * phi).tolist()
        model["n"] = int(model.get("n", 0)) + 1
        runs += 1
        score_sum += score_f

    model["runs"] = runs
    model["score_mean"] = score_sum / runs if runs else 0.0
    metrics = compute_compact_ridge_metrics(model, rows, score_key=score_key, group_bounds=bounds)
    theta_norm = compact_ridge_theta_norm(model)
    return model, metrics, theta_norm


def select_compact_featurewise_ridge_multipliers(
    *,
    project_dir: Path,
    mode: str,
    x_features: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    model = load_compact_ridge_model(project_dir)
    if model is None:
        raise FileNotFoundError(f"Compact Featurewise Ridge model not found for project {project_dir}.")
    if str(model.get("model_family") or "") != "compact_featurewise_ridge_regression":
        raise RuntimeError("Only Compact Featurewise Ridge Regression models are supported.")

    selection = select_compact_ridge_from_model(
        model=model,
        x_features=x_features,
        candidate_log_multipliers_by_group=params.get("candidate_log_multipliers_by_group"),
    )
    updates = _build_updates(params, selection["selected_multipliers"])
    return {
        **selection,
        "updates": updates,
        "selected_preset": "compact_featurewise_ridge_regression",
        "exploration_mode": "greedy",
    }


def select_compact_ridge_from_model(
    *,
    model: dict[str, Any],
    x_features: dict[str, Any],
    candidate_log_multipliers_by_group: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bounds = _bounds_from_model(model)
    candidates_by_group = _candidate_logs_by_group(model, bounds, candidate_log_multipliers_by_group)
    scaler = model.get("feature_scaler") if isinstance(model.get("feature_scaler"), dict) else {}
    x = build_compact_vector(x_features, scaler)
    theta = _solve_theta(model)

    combos = list(itertools.product(*(candidates_by_group[group] for group in COMPACT_MODEL_GROUP_KEYS)))
    scores = []
    for combo in combos:
        phi = build_compact_score_design_vector(x, np.array(combo, dtype=np.float64))
        scores.append(float(phi @ theta))

    spread = float(max(scores) - min(scores)) if scores else 0.0
    if spread < 1e-6 or not scores:
        selected_logs = np.zeros(len(COMPACT_MODEL_GROUP_KEYS), dtype=np.float64)
        selected_score = scores[len(scores) // 2] if scores else 0.0
        has_signal = False
    else:
        best_idx = int(np.argmax(scores))
        selected_logs = np.array(combos[best_idx], dtype=np.float64)
        selected_score = float(scores[best_idx])
        has_signal = True

    group_multipliers: dict[str, float] = {}
    group_log_multipliers: dict[str, float] = {}
    for index, group in enumerate(COMPACT_MODEL_GROUP_KEYS):
        lo, hi = bounds[group]
        mult = clamp_float(float(math.exp(float(selected_logs[index]))), lo, hi)
        group_multipliers[group] = mult
        group_log_multipliers[group] = float(math.log(max(mult, 1e-9)))

    selected_multipliers, selected_log_multipliers = expand_compact_group_multipliers(group_multipliers)
    candidate_score_checks = _candidate_checks_by_group(candidates_by_group, combos, scores, group_log_multipliers)

    return {
        "selected_preset": "compact_featurewise_ridge_regression",
        "yhat_scores": selected_multipliers,
        "selected_multipliers": selected_multipliers,
        "selected_multipliers_raw": dict(selected_multipliers),
        "selected_log_multipliers": selected_log_multipliers,
        "selected_log_multipliers_raw": dict(selected_log_multipliers),
        "group_multipliers": group_multipliers,
        "group_log_multipliers": group_log_multipliers,
        "selected_score": selected_score,
        "score_spreads": {group: spread for group in COMPACT_MODEL_GROUP_KEYS},
        "candidate_score_checks": candidate_score_checks,
        "candidate_points": int(model.get("candidate_points") or 0),
        "has_signal": has_signal,
        "n_runs": int(model.get("runs") or model.get("n") or 0),
        "model_type": "compact_featurewise_ridge_regression",
    }


def compute_compact_ridge_metrics(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    score_key: str,
    group_bounds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bounds = normalise_compact_group_bounds(group_bounds or model.get("log_multiplier_bounds"))
    scaler = model.get("feature_scaler") if isinstance(model.get("feature_scaler"), dict) else {}
    theta = _solve_theta(model)
    residuals: list[float] = []
    targets: list[float] = []
    for row in rows:
        features = row.get("x_features")
        selected = row.get("selected_multipliers")
        score = row.get(score_key)
        if not isinstance(features, dict) or not isinstance(selected, dict) or not isinstance(score, (int, float)):
            continue
        action_logs = compact_action_logs_from_multipliers(selected, bounds=bounds)
        if action_logs is None:
            continue
        pred = float(build_compact_score_design_vector(build_compact_vector(features, scaler), action_logs) @ theta)
        target = float(score)
        if math.isfinite(target):
            residuals.append(pred - target)
            targets.append(target)
    if not residuals:
        return {"mse": float("inf"), "rmse": float("inf"), "mae": float("inf"), "r_squared": 0.0, "samples": 0, "avg_val_mse": float("inf")}
    arr = np.array(residuals, dtype=np.float64)
    target_arr = np.array(targets, dtype=np.float64)
    mse = float(np.mean(arr * arr))
    ss_tot = float(np.sum((target_arr - float(np.mean(target_arr))) ** 2))
    ss_res = float(np.sum(arr * arr))
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(np.mean(np.abs(arr))),
        "r_squared": 1.0 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0,
        "samples": len(residuals),
        "avg_val_mse": mse,
    }


def compact_ridge_theta_norm(model: dict[str, Any]) -> float:
    theta = _solve_theta(model)
    return float(np.linalg.norm(theta))


def load_compact_ridge_model(project_dir: Path) -> dict[str, Any] | None:
    seeded_dir = project_dir / "models" / "compact_featurewise_ridge_regression"
    for path in sorted(seeded_dir.glob("*.json"), key=lambda p: p.stem, reverse=True):
        payload = _load_model_payload(path)
        if payload is not None:
            return payload
    return None


def _empty_model(
    *,
    lambda_ridge: float,
    scaler: dict[str, dict[str, float]],
    candidate_points: int,
    bounds: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    d_x = len(build_compact_vector({}, scaler))
    d_phi = d_x + len(COMPACT_MODEL_GROUP_KEYS) + len(COMPACT_MODEL_GROUP_KEYS) + (d_x - 1) * len(COMPACT_MODEL_GROUP_KEYS)
    return {
        "version": 1,
        "model_family": "compact_featurewise_ridge_regression",
        "mode": "exif_compact_featurewise",
        "lambda_ridge": float(lambda_ridge),
        "runs": 0,
        "score_mean": 0.0,
        "feature_scaler": scaler,
        "candidate_points": int(max(5, candidate_points)),
        "log_multiplier_bounds": {key: [float(bounds[key][0]), float(bounds[key][1])] for key in COMPACT_MODEL_GROUP_KEYS},
        "action_space": "joint_log_multiplier",
        "design": "x_plus_three_actions_plus_action2_plus_xa",
        "A": (np.eye(d_phi, dtype=np.float64) * float(lambda_ridge)).tolist(),
        "b": np.zeros(d_phi, dtype=np.float64).tolist(),
        "n": 0,
        "design_dim": d_phi,
    }


def _solve_theta(model: dict[str, Any]) -> np.ndarray:
    A = np.array(model.get("A", []), dtype=np.float64)
    b = np.array(model.get("b", []), dtype=np.float64)
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


def _bounds_from_model(model: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return normalise_compact_group_bounds(model.get("log_multiplier_bounds"))


def _candidate_logs_by_group(
    model: dict[str, Any],
    bounds: dict[str, tuple[float, float]],
    source: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for group in COMPACT_MODEL_GROUP_KEYS:
        lo, hi = bounds[group]
        raw = source.get(group) if isinstance(source, dict) else None
        if isinstance(raw, list) and raw:
            values = [clamp_float(float(value), math.log(lo), math.log(hi)) for value in raw if isinstance(value, (int, float))]
            out[group] = np.array(values or [0.0], dtype=np.float64)
        else:
            # Use the stored fallback only when testing did not pass an explicit candidate grid.
            out[group] = np.linspace(math.log(lo), math.log(hi), int(max(5, model.get("candidate_points", 30))), dtype=np.float64)
    return out


def _candidate_checks_by_group(
    candidates_by_group: dict[str, np.ndarray],
    combos: list[tuple[float, ...]],
    scores: list[float],
    selected_logs: dict[str, float],
) -> dict[str, list[dict[str, Any]]]:
    checks: dict[str, list[dict[str, Any]]] = {}
    for group_index, group in enumerate(COMPACT_MODEL_GROUP_KEYS):
        rows: list[dict[str, Any]] = []
        selected_log = float(selected_logs[group])
        selected_index = int(np.argmin(np.abs(candidates_by_group[group] - selected_log))) if len(candidates_by_group[group]) else -1
        for candidate_index, candidate_log in enumerate(candidates_by_group[group]):
            matching_scores = [scores[index] for index, combo in enumerate(combos) if abs(float(combo[group_index]) - float(candidate_log)) < 1e-12]
            score = float(max(matching_scores)) if matching_scores else 0.0
            rows.append(
                {
                    "candidate_log_multiplier": float(candidate_log),
                    "candidate_multiplier": float(math.exp(float(candidate_log))),
                    "predicted_score": score,
                    "selected": candidate_index == selected_index,
                }
            )
        checks[group] = rows
    return checks


def _build_updates(params: dict[str, Any], multipliers: dict[str, float]) -> dict[str, Any]:
    feature_lr = float(params.get("feature_lr", 2.5e-3))
    position_lr_init = float(params.get("position_lr_init", 1.6e-4))
    scaling_lr = float(params.get("scaling_lr", 5.0e-3))
    opacity_lr = float(params.get("opacity_lr", 5.0e-2))
    rotation_lr = float(params.get("rotation_lr", 1.0e-3))
    densify_grad_threshold = float(params.get("densify_grad_threshold", 2.0e-4))
    opacity_threshold = float(params.get("opacity_threshold", 0.005))
    lambda_dssim = float(params.get("lambda_dssim", 0.2))
    return {
        "preset_name": "compact_featurewise_ridge_regression",
        "feature_lr": feature_lr * multipliers["feature_lr_mult"],
        "position_lr_init": position_lr_init * multipliers["position_lr_init_mult"],
        "scaling_lr": scaling_lr * multipliers["scaling_lr_mult"],
        "opacity_lr": opacity_lr * multipliers["opacity_lr_mult"],
        "rotation_lr": rotation_lr * multipliers["rotation_lr_mult"],
        "densify_grad_threshold": densify_grad_threshold * multipliers["densify_grad_threshold_mult"],
        "opacity_threshold": opacity_threshold * multipliers["opacity_threshold_mult"],
        "lambda_dssim": lambda_dssim * multipliers["lambda_dssim_mult"],
    }


def _load_model_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict) and payload.get("schema") == "compact_featurewise_ridge_regression_v1":
        inner = payload.get("model")
        if isinstance(inner, dict):
            if "metrics" not in inner and isinstance(payload.get("metrics"), dict):
                inner["metrics"] = payload["metrics"]
            if "model_family" not in inner:
                inner["model_family"] = "compact_featurewise_ridge_regression"
            return inner
    if isinstance(payload, dict) and payload.get("model_family") == "compact_featurewise_ridge_regression":
        nested = payload.get("model")
        return nested if isinstance(nested, dict) else payload
    return None


__all__ = [
    "compact_ridge_theta_norm",
    "compute_compact_ridge_metrics",
    "select_compact_featurewise_ridge_multipliers",
    "select_compact_ridge_from_model",
    "train_compact_featurewise_ridge_model",
]

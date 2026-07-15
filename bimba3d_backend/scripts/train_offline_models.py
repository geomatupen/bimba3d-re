#!/usr/bin/env python3
"""Offline training for contextual-continuous score-optimizer ridge models.

Revised approach:
  - Train score models, not direct multiplier models.
  - Learn f(x, a) -> score where:
      x: context features
      a: action (group log-multiplier)
  - Select action at inference by bounded search over candidate actions.

The script performs project-level CV for lambda selection, then retrains on all
available development rows and writes a quality model.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
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

logger = logging.getLogger(__name__)


def normalise_group_bounds(bounds: dict[str, Any] | None = None) -> dict[str, tuple[float, float]]:
    """Return positive multiplier bounds keyed by report group model names."""
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
        v = features.get("vegetation_cover_percentage", features.get("vegetation_cover", 0.5))
        return float(v or 0.5)
    if name == "vegetation_complexity":
        v = features.get("vegetation_complexity_score", features.get("vegetation_complexity", 0.5))
        return float(v or 0.5)
    return 0.0


def _apply_log_transform(name: str, value: float) -> float:
    if name in LOG_TRANSFORM_FEATURES:
        return math.log(max(value, 1e-9))
    return value


def compute_feature_scaler(rows: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, float]]:
    feature_names = FEATUREWISE_GROUP_FEATURES[group_key]
    values_by_feature: dict[str, list[float]] = {n: [] for n in feature_names if n != "intercept"}

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
    for name, vals in values_by_feature.items():
        if not vals:
            scaler[name] = {"mean": 0.0, "std": 1.0}
            continue
        arr = np.array(vals, dtype=np.float64)
        std = max(float(np.std(arr)), SCALER_STD_FLOOR)
        scaler[name] = {"mean": float(np.mean(arr)), "std": std}
    return scaler


def apply_scaler_to_vector(features: dict[str, Any], group_key: str, scaler: dict[str, dict[str, float]]) -> np.ndarray:
    feature_names = FEATUREWISE_GROUP_FEATURES[group_key]
    x: list[float] = []
    for name in feature_names:
        if name == "intercept":
            x.append(1.0)
            continue
        raw = _extract_raw_feature(features, name)
        transformed = _apply_log_transform(name, raw)
        stats = scaler.get(name, {"mean": 0.0, "std": 1.0})
        std = max(float(stats.get("std", 1.0)), SCALER_STD_FLOOR)
        x.append((transformed - float(stats.get("mean", 0.0))) / std)
    return np.array(x, dtype=np.float64)


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
        fval = float(direct_val)
        if math.isfinite(fval) and fval > 0.0:
            lo, hi = bounds[group_key]
            fval = min(max(fval, lo), hi)
            return float(math.log(fval))

    keys = GROUP_MULTIPLIERS_MAP.get(group_key, [])
    if not keys:
        return None
    val = selected.get(keys[0])
    if not isinstance(val, (int, float)):
        return None
    fval = float(val)
    if not math.isfinite(fval) or fval <= 0.0:
        return None
    lo, hi = bounds[group_key]
    fval = min(max(fval, lo), hi)
    return float(math.log(fval))


def _build_score_design_vector(x_context: np.ndarray, action_log: float) -> np.ndarray:
    # phi = [x, a, a^2, x_no_intercept * a]
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
        preds: list[float] = []
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
            preds.append(pred)
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
        rmse = float(math.sqrt(mse))
        mae = float(np.mean(abs_err))
        y_mean = float(np.mean(targets))
        ss_tot = float(np.sum((np.array(targets) - y_mean) ** 2))
        ss_res = float(np.sum(sq_err))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

        out[group_key] = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r_squared": r2,
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


def _project_names(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(r.get("project_name")) for r in rows if r.get("project_name")})


def _build_group_cv_splits(projects: list[str], folds: int, repeats: int, seed: int) -> list[tuple[list[str], list[str], int, int]]:
    folds = max(2, int(folds))
    repeats = max(1, int(repeats))
    projects_arr = np.array(projects)
    splits: list[tuple[list[str], list[str], int, int]] = []
    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        shuffled = projects_arr.copy()
        rng.shuffle(shuffled)
        fold_parts = np.array_split(shuffled, folds)
        for fold_idx in range(folds):
            val = set(str(v) for v in fold_parts[fold_idx].tolist())
            train = [str(p) for p in shuffled.tolist() if str(p) not in val]
            splits.append((train, sorted(val), rep, fold_idx))
    return splits


def _select_lambda_via_group_cv(
    rows: list[dict[str, Any]],
    score_key: str,
    candidates: list[float],
    folds: int,
    repeats: int,
    seed: int,
    candidate_points: int,
) -> tuple[float, dict[str, Any]]:
    projects = _project_names(rows)
    splits = _build_group_cv_splits(projects, folds, repeats, seed)
    rows_by_project: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        pn = str(r.get("project_name"))
        rows_by_project.setdefault(pn, []).append(r)

    score_book: dict[str, Any] = {str(l): {"split_scores": [], "split_metrics": []} for l in candidates}

    for train_projects, val_projects, rep, fold_idx in splits:
        train_rows = [rr for p in train_projects for rr in rows_by_project.get(p, [])]
        val_rows = [rr for p in val_projects for rr in rows_by_project.get(p, [])]
        if not train_rows or not val_rows:
            continue

        scalers = {g: compute_feature_scaler(train_rows, g) for g in GROUP_KEYS}

        for lam in candidates:
            model = _train_model_on_rows(train_rows, score_key, lam, scalers, candidate_points)
            metrics = _compute_metrics(model, val_rows, score_key)
            score = float(metrics.get("avg_val_mse", float("inf")))
            entry = score_book[str(lam)]
            entry["split_scores"].append(score)
            entry["split_metrics"].append(
                {
                    "repeat": rep,
                    "fold": fold_idx,
                    "train_projects": len(train_projects),
                    "val_projects": len(val_projects),
                    "avg_val_mse": score,
                    "per_group": {k: v for k, v in metrics.items() if k in GROUP_KEYS},
                }
            )

    best_lambda = candidates[0]
    best_mean = float("inf")
    for lam in candidates:
        scores = score_book[str(lam)]["split_scores"]
        mean_score = float(np.mean(scores)) if scores else float("inf")
        std_score = float(np.std(scores)) if scores else float("inf")
        score_book[str(lam)]["cv_mean_mse"] = mean_score
        score_book[str(lam)]["cv_std_mse"] = std_score
        score_book[str(lam)]["cv_splits"] = len(scores)
        if mean_score < best_mean:
            best_mean = mean_score
            best_lambda = lam

    cv_report = {
        "folds": int(folds),
        "repeats": int(repeats),
        "total_splits": int(folds * repeats),
        "projects": len(projects),
        "lambda_scores": score_book,
    }
    return best_lambda, cv_report


def _write_model_payload(
    out_path: Path,
    score_key: str,
    lambda_ridge: float,
    model: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    payload = {
        "schema": "offline_model_v3",
        "version": 3,
        "model_family": "featurewise_ridge_regression",
        "score_key": score_key,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lambda_ridge": float(lambda_ridge),
        "metrics": metadata,
        "model": model,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _quality_rows(all_rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in all_rows:
        val = r.get(score_key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            out.append(r)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Offline training for Featurewise Ridge Regression quality model")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("bimba3d_backend/data/_offline_training/offline_dataset_v1.json"),
        help="Path to offline_dataset_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("bimba3d_backend/data/_offline_training/models"),
        help="Directory to write the quality model and reports",
    )
    parser.add_argument("--lambda-candidates", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    parser.add_argument("--cv-folds", type=int, default=5, help="Project-level CV folds")
    parser.add_argument("--cv-repeats", type=int, default=3, help="How many independent CV reshuffles")
    parser.add_argument("--cv-seed", type=int, default=11, help="Seed base for repeated CV")
    parser.add_argument("--candidate-points", type=int, default=30, help="Fallback prediction grid size per group when no explicit test grid is provided")
    parser.add_argument("--exclude-baseline", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.dataset.exists():
        logger.error("Dataset not found: %s", args.dataset)
        return 1

    raw = json.loads(args.dataset.read_text(encoding="utf-8"))
    all_rows: list[dict[str, Any]] = raw.get("rows", []) if isinstance(raw, dict) else []
    if not all_rows:
        logger.error("Dataset is empty")
        return 1

    if args.exclude_baseline:
        all_rows = [r for r in all_rows if not r.get("is_baseline_run", False)]

    if not all_rows:
        logger.error("No rows remain after filtering")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    lambda_candidates = sorted(set(float(v) for v in args.lambda_candidates))

    training_report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(args.dataset),
        "total_rows": len(all_rows),
        "total_projects": len(_project_names(all_rows)),
        "cv": {
            "folds": int(max(2, args.cv_folds)),
            "repeats": int(max(1, args.cv_repeats)),
            "seed": int(args.cv_seed),
        },
        "lambda_candidates": lambda_candidates,
        "quality_model": {},
    }

    score_key = "relative_quality_score"
    out_path = args.out_dir / "quality_model.json"
    rows_quality = _quality_rows(all_rows, score_key)
    logger.info("[quality] rows=%d projects=%d", len(rows_quality), len(_project_names(rows_quality)))

    if len(_project_names(rows_quality)) < 3:
        logger.warning("[quality] not enough projects to perform CV; writing empty model")
        scalers_empty = {g: compute_feature_scaler(rows_quality, g) for g in GROUP_KEYS}
        empty = _build_empty_model(2.0, scalers_empty, args.candidate_points)
        empty_meta = {"warning": "insufficient_projects_for_cv", "rows": len(rows_quality)}
        _write_model_payload(out_path, score_key, 2.0, empty, empty_meta)
        training_report["quality_model"] = empty_meta
    else:
        best_lambda, cv_report = _select_lambda_via_group_cv(
            rows_quality,
            score_key,
            lambda_candidates,
            args.cv_folds,
            args.cv_repeats,
            args.cv_seed,
            args.candidate_points,
        )

        final_scalers = {g: compute_feature_scaler(rows_quality, g) for g in GROUP_KEYS}
        final_model = _train_model_on_rows(rows_quality, score_key, best_lambda, final_scalers, args.candidate_points)
        final_metrics = _compute_metrics(final_model, rows_quality, score_key)
        norms = _theta_norms(final_model)

        model_meta = {
            "score_key": score_key,
            "lambda_selected": float(best_lambda),
            "rows_used": len(rows_quality),
            "projects_used": len(_project_names(rows_quality)),
            "candidate_points": int(max(5, args.candidate_points)),
            "cv_report": cv_report,
            "train_fit_metrics": final_metrics,
            "theta_norms": norms,
        }

        final_model["metrics"] = {
            "lambda_selected": float(best_lambda),
            "train_fit_metrics": final_metrics,
            "theta_norms": norms,
        }

        _write_model_payload(out_path, score_key, best_lambda, final_model, model_meta)
        training_report["quality_model"] = model_meta
        logger.info("[quality] lambda=%.4f wrote model=%s", best_lambda, out_path)

    (args.out_dir / "training_report.json").write_text(json.dumps(training_report, indent=2), encoding="utf-8")
    (args.out_dir / "split_report.json").write_text(
        json.dumps(
            {
                "note": "Revised trainer uses repeated project-level CV; no single fixed train/val split.",
                "cv": training_report.get("cv", {}),
                "total_projects": training_report.get("total_projects", 0),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Done. Reports written to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


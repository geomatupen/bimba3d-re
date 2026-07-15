from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

from .common import ModeContext, apply_preset_updates
from .compact_scene_descriptors import build_preset as build_compact_scene_descriptor_preset
from .featurewise_ridge_regression import (
    build_featurewise_ridge_updates,
    select_featurewise_ridge_multipliers,
)
from .legacy_support import normalize_ai_input_mode_token, with_legacy_groupwise_features

logger = logging.getLogger(__name__)

# DISABLED: Context jitter removed - only using run jitter (LR multiplier) now
# Import jitter functions for multi-run learning
# try:
#     from bimba3d_backend.app.services.context_jitter import apply_context_jitter
# except ImportError:
#     # Fallback if import path changes
#     apply_context_jitter = None

VALID_AI_INPUT_MODES = {
    "exif_compact_featurewise",
}

GSPLAT_PARAMETER_DEFAULTS = {
    "feature_lr": 0.0025,
    "position_lr_init": 0.00016,
    "position_lr_final": 1.6e-06,
    "scaling_lr": 0.005,
    "opacity_lr": 0.05,
    "rotation_lr": 0.001,
    "densify_grad_threshold": 0.0002,
    "opacity_threshold": 0.005,
    "lambda_dssim": 0.2,
}

def _multiply_param(params: dict, key: str, multiplier: float) -> None:
    default = GSPLAT_PARAMETER_DEFAULTS.get(key)
    raw = params.get(key, default)
    if not isinstance(raw, (int, float)):
        return
    params[key] = float(raw) * float(multiplier)


def _apply_group_specific_multipliers(params: dict) -> None:
    """Apply pre-generated group-specific log space multipliers to parameters.
    
    Uses different multiplier factors for the thesis parameter groups:
    - geometry_lr: position, scale, and rotation learning rates
    - appearance_lr: feature, opacity, and DSSIM loss balance
    - densification: gradient and opacity thresholds
    """
    # Check if pre-generated multipliers are provided
    geom_mult = params.get("geometry_lr_multiplier")
    app_mult = params.get("appearance_lr_multiplier")
    scale_mult = params.get("densification_multiplier", params.get("scale_lr_multiplier"))
    
    if not (isinstance(geom_mult, (int, float)) or isinstance(app_mult, (int, float)) or isinstance(scale_mult, (int, float))):
        # No pre-generated multipliers provided, skip this step
        return
    
    # Geometry group: position, scale, and rotation LRs
    if isinstance(geom_mult, (int, float)) and geom_mult > 0:
        geom_mult = float(geom_mult)
        for key in ("position_lr_init", "position_lr_final", "scaling_lr", "rotation_lr"):
            _multiply_param(params, key, geom_mult)
    
    # Appearance group: feature and opacity LRs
    if isinstance(app_mult, (int, float)) and app_mult > 0:
        app_mult = float(app_mult)
        for key in ("feature_lr", "opacity_lr", "lambda_dssim"):
            _multiply_param(params, key, app_mult)
    
    # Densification group: creation/pruning thresholds
    if isinstance(scale_mult, (int, float)) and scale_mult > 0:
        scale_mult = float(scale_mult)
        for key in ("densify_grad_threshold", "opacity_threshold"):
            _multiply_param(params, key, scale_mult)
    
    # Store which multiplier was used (reference only, not the actual combined multiplier)
    # Use geometry as primary reference for legacy single-value consumers.
    params["run_jitter_multiplier"] = float(geom_mult) if isinstance(geom_mult, (int, float)) else 1.0
    if isinstance(geom_mult, (int, float)) and geom_mult > 0:
        params["geometry_lr_log_action"] = math.log(float(geom_mult))
    if isinstance(app_mult, (int, float)) and app_mult > 0:
        params["appearance_lr_log_action"] = math.log(float(app_mult))
    if isinstance(scale_mult, (int, float)) and scale_mult > 0:
        params["densification_multiplier"] = float(scale_mult)
        params["densification_log_action"] = math.log(float(scale_mult))
    params["_fixed_group_multipliers_applied"] = True


def _apply_run_jitter_inline(params: dict) -> None:
    """Apply independent per-group log-space jitter for data collection.

    Samples 3 independent multipliers â€” one per parameter group:
      geometry_lr_mult    â†’ position_lr_init, position_lr_final, scaling_lr, rotation_lr
      appearance_lr_mult  â†’ feature_lr, opacity_lr, lambda_dssim
      densification_mult  â†’ densify_grad_threshold, opacity_threshold

    Each multiplier is sampled independently:
      a_g ~ U(log(m_min), log(m_max))   [log-space uniform]
      m_g = exp(a_g)                     [convert back to multiplier]

    The sampled log-value a_g is stored in params for the training row:
      (x, a, r) where a = log(m), not the raw multiplier â€” per Â§4.X.2.

    Document bounds: m_min=0.5, m_max=2.0 (per-group overrides supported).
    """
    jitter_mode = str(params.get("run_jitter_mode", "")).strip().lower()
    if jitter_mode != "random":
        has_fixed_schedule = any(
            isinstance(params.get(key), (int, float)) and float(params.get(key)) > 0
            for key in ("geometry_lr_multiplier", "appearance_lr_multiplier", "scale_lr_multiplier", "densification_multiplier")
        )
        if has_fixed_schedule:
            geom = float(params.get("geometry_lr_multiplier") or 1.0)
            app = float(params.get("appearance_lr_multiplier") or 1.0)
            dens = float(params.get("densification_multiplier") or params.get("scale_lr_multiplier") or 1.0)
            if not bool(params.get("_fixed_group_multipliers_applied")):
                for key in ("position_lr_init", "position_lr_final", "scaling_lr", "rotation_lr"):
                    _multiply_param(params, key, geom)
                for key in ("feature_lr", "opacity_lr", "lambda_dssim"):
                    _multiply_param(params, key, app)
                for key in ("densify_grad_threshold", "opacity_threshold"):
                    _multiply_param(params, key, dens)
                params["_fixed_group_multipliers_applied"] = True
            params["geometry_lr_multiplier"] = geom
            params["appearance_lr_multiplier"] = app
            params["densification_multiplier"] = dens
            params["run_jitter_multiplier"] = geom
            if geom > 0:
                params["geometry_lr_log_action"] = float(params.get("geometry_lr_log_action") or math.log(geom))
            if app > 0:
                params["appearance_lr_log_action"] = float(params.get("appearance_lr_log_action") or math.log(app))
            if dens > 0:
                params["densification_log_action"] = float(params.get("densification_log_action") or math.log(dens))
            return
        params["run_jitter_multiplier"] = 1.0
        params["geometry_lr_multiplier"] = 1.0
        params["appearance_lr_multiplier"] = 1.0
        params["densification_multiplier"] = 1.0
        return

    # Read shared bounds (per-group overrides use _mult_{group} suffix if present)
    # Default bounds per document Â§4.X.6:
    #   Geometry / Appearance: [0.5, 2.0]
    #   Densification: [0.7, 1.42]  (tighter â€” density control affects reconstruction stability)
    def _get_bounds(group_suffix: str) -> tuple[float, float]:
        default_lo = 0.7 if group_suffix == "densification" else 0.5
        default_hi = 1.42 if group_suffix == "densification" else 2.0
        lo = params.get(f"run_jitter_min_{group_suffix}", params.get("run_jitter_min", default_lo))
        hi = params.get(f"run_jitter_max_{group_suffix}", params.get("run_jitter_max", default_hi))
        lo, hi = float(lo), float(hi)
        if lo > hi:
            lo, hi = hi, lo
        # Hard limits
        lo = max(lo, 0.1)
        hi = min(hi, 10.0)
        return lo, hi

    def _sample_log_multiplier(lo: float, hi: float) -> tuple[float, float]:
        """Returns (log_value, multiplier). Ensures |mult - 1| > 0.05."""
        log_lo = math.log(max(lo, 1e-9))
        log_hi = math.log(max(hi, 1e-9))
        if abs(log_hi - log_lo) < 0.01:
            return 0.0, 1.0
        max_tries = 20
        for _ in range(max_tries):
            a = random.uniform(log_lo, log_hi)
            m = math.exp(a)
            if abs(m - 1.0) > 0.05:
                return a, m
        # Fallback: pick endpoint further from 1.0
        a = log_lo if abs(math.exp(log_lo) - 1.0) > abs(math.exp(log_hi) - 1.0) else log_hi
        return a, math.exp(a)

    # â”€â”€ Sample 3 independent multipliers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    geom_lo, geom_hi = _get_bounds("geometry")
    app_lo, app_hi = _get_bounds("appearance")
    dens_lo, dens_hi = _get_bounds("densification")

    a_geom, m_geom = _sample_log_multiplier(geom_lo, geom_hi)
    a_app, m_app = _sample_log_multiplier(app_lo, app_hi)
    a_dens, m_dens = _sample_log_multiplier(dens_lo, dens_hi)

    # â”€â”€ Geometry group â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for k in ("position_lr_init", "position_lr_final", "scaling_lr", "rotation_lr"):
        _multiply_param(params, k, m_geom)

    # â”€â”€ Appearance group â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for k in ("feature_lr", "opacity_lr", "lambda_dssim"):
        _multiply_param(params, k, m_app)

    # â”€â”€ Densification group â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for k in ("densify_grad_threshold", "opacity_threshold"):
        _multiply_param(params, k, m_dens)

    # â”€â”€ Store log-space actions a = log(m) and multipliers for training row â”€â”€â”€
    # Per Â§4.X.2: training rows store a = log(m), not the raw multiplier
    params["geometry_lr_multiplier"] = float(m_geom)
    params["appearance_lr_multiplier"] = float(m_app)
    params["densification_multiplier"] = float(m_dens)
    params["geometry_lr_log_action"] = float(a_geom)
    params["appearance_lr_log_action"] = float(a_app)
    params["densification_log_action"] = float(a_dens)
    # Legacy single-scalar field: use geometry as representative
    params["run_jitter_multiplier"] = float(m_geom)
    logger.info(
        "Applied group jitter: geometry=%.4f (log=%.4f) appearance=%.4f (log=%.4f) densification=%.4f (log=%.4f)",
        m_geom, a_geom, m_app, a_app, m_dens, a_dens,
    )

CACHE_VERSION = 1
VALID_PRESET_OVERRIDES = {"conservative", "balanced", "geometry_fast", "appearance_fast"}


def _normalize_preset_override(value: Any) -> str:
    token = str(value or "").strip().lower()
    return token if token in VALID_PRESET_OVERRIDES else ""


def _project_exif_file_path(project_dir: Path) -> Path:
    """Get path to project-level EXIF features file"""
    return project_dir / "exif_features.json"


def _load_project_exif(project_dir: Path, mode: str, fingerprint: str) -> dict[str, Any] | None:
    """Load project-level EXIF features if they exist and match fingerprint"""
    exif_path = _project_exif_file_path(project_dir)
    if not exif_path.exists():
        return None
    try:
        payload = json.loads(exif_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version", 0) or 0) != CACHE_VERSION:
        return None
    if str(payload.get("mode") or "") != mode:
        return None
    if str(payload.get("fingerprint") or "") != fingerprint:
        return None
    return payload


def _save_project_exif(project_dir: Path, mode: str, fingerprint: str, payload: dict[str, Any]) -> Path:
    """Save project-level EXIF features"""
    exif_path = _project_exif_file_path(project_dir)
    tmp_path = exif_path.with_suffix(exif_path.suffix + ".tmp")
    tmp_payload = {
        "version": CACHE_VERSION,
        "mode": mode,
        "fingerprint": fingerprint,
        "features": dict(payload.get("features") or {}),
        "notes": list(payload.get("notes") or []),
        "heuristic_preset": str(payload.get("heuristic_preset") or "balanced"),
        "extracted_at": payload.get("extracted_at"),
    }
    tmp_path.write_text(json.dumps(tmp_payload, indent=2), encoding="utf-8")
    tmp_path.replace(exif_path)
    logger.info(f"Saved project-level EXIF features to {exif_path}")
    return exif_path


def _image_fingerprint(image_dir: Path) -> str:
    digest = hashlib.sha256()
    files = [p for p in Path(image_dir).glob("*") if p.is_file()]
    files.sort()
    for path in files:
        try:
            stat = path.stat()
        except Exception:
            continue
        rel_name = path.name.lower().encode("utf-8", errors="ignore")
        digest.update(rel_name)
        digest.update(str(int(stat.st_size)).encode("utf-8"))
        digest.update(str(int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))).encode("utf-8"))
    return digest.hexdigest()


def _combined_image_fingerprint(metadata_image_dir: Path, processing_image_dir: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_image_fingerprint(metadata_image_dir).encode("utf-8"))
    digest.update(_image_fingerprint(processing_image_dir).encode("utf-8"))
    return digest.hexdigest()


def normalize_ai_input_mode(value: Any) -> str:
    mode = normalize_ai_input_mode_token(value)
    if mode in VALID_AI_INPUT_MODES:
        return mode
    return ""


def _count_supported_images(image_dir: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    try:
        return sum(1 for p in Path(image_dir).glob("*") if p.is_file() and p.suffix.lower() in exts)
    except Exception:
        return 0


def _build_feature_log_details(mode: str, features: dict[str, Any], image_count: int) -> dict[str, Any]:
    details: dict[str, Any] = {
        "image_count": int(image_count),
    }

    # Emit the extracted descriptor set used by model selection/training.
    for key in sorted(features.keys()):
        details[key] = features.get(key)

    return details


def _build_initial_params_log(params: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "tune_start_step",
        "tune_end_step",
        "trend_scope",
        "feature_lr",
        "opacity_lr",
        "scaling_lr",
        "rotation_lr",
        "position_lr_init",
        "position_lr_final",
        "densification_interval",
        "densify_grad_threshold",
        "opacity_threshold",
        "lambda_dssim",
        "run_jitter_only",
        "run_jitter_multiplier",
        "geometry_lr_multiplier",
        "appearance_lr_multiplier",
        "scale_lr_multiplier",
        "densification_multiplier",
        "geometry_lr_log_action",
        "appearance_lr_log_action",
        "densification_log_action",
    ]
    details: dict[str, Any] = {}
    for key in keys:
        value = params.get(key)
        if value is not None:
            details[key] = value
    return details


def _persist_retry_snapshot(
    *,
    project_dir: Path,
    params: dict[str, Any],
    mode: str,
    selected_preset: str,
    selected_multipliers: dict[str, Any] | None,
    selected_multipliers_raw: dict[str, Any] | None,
    candidate_score_checks: dict[str, Any] | None = None,
    selected_log_multipliers: dict[str, Any] | None = None,
) -> None:
    """Persist minimal run-start snapshot for retrying interrupted runs.

    Stores the effective initial LR params and selection details in:
      {project_dir}/runs/{run_id}/retry_snapshot.json
    """
    run_id = str(params.get("run_id") or "").strip()
    if not run_id:
        return

    run_dir = project_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = run_dir / "retry_snapshot.json"

    payload = {
        "run_id": run_id,
        "mode": mode,
        "selected_preset": selected_preset,
        "phase": params.get("phase"),
        "run": params.get("run") or params.get("phase_run"),
        "phase_run": params.get("phase_run"),
        "initial_params": _build_initial_params_log(params),
        "selected_multipliers": dict(selected_multipliers or {}),
        "selected_multipliers_raw": dict(selected_multipliers_raw or {}),
        "selected_log_multipliers": dict(selected_log_multipliers or {}),
        "candidate_score_checks": dict(candidate_score_checks or {}),
    }

    tmp = snapshot_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(snapshot_path)


def apply_initial_preset(
    params: dict[str, Any],
    *,
    image_dir: Path,
    colmap_dir: Path,
    logger,
) -> dict[str, Any]:
    """Apply the initial parameter preset for the selected AI input mode."""
    mode = normalize_ai_input_mode(params.get("ai_input_mode"))
    selector_strategy = str(params.get("ai_selector_strategy") or "").strip().lower()
    valid_selector_strategies = {
        "featurewise_ridge_regression",
        "featurewise_mlp",
        "compact_featurewise_ridge_regression",
        "compact_featurewise_mlp",
    }
    if selector_strategy not in valid_selector_strategies:
        selector_strategy = "featurewise_ridge_regression"
    params["ai_selector_strategy"] = selector_strategy
    if not mode:
        return {
            "mode": "not_configured",
            "applied": False,
            "updates": {},
            "features": {},
            "notes": ["No ai_input_mode selected."],
            "cache_used": False,
            "selector_strategy": selector_strategy,
        }

    image_dir_path = Path(image_dir)
    project_dir = image_dir_path.resolve().parent

    run_mode = str(params.get("mode") or "").strip().lower()
    if run_mode == "baseline":
        forced_preset = _normalize_preset_override(params.get("preset_override")) or _normalize_preset_override(
            params.get("ai_preset_override")
        )
        selected_preset = forced_preset or "balanced"
        selected_updates = apply_preset_updates(params, selected_preset)
        for key, value in selected_updates.items():
            if key == "preset_name":
                continue
            params[key] = value

        try:
            _persist_retry_snapshot(
                project_dir=project_dir,
                params=params,
                mode="baseline",
                selected_preset=selected_preset,
                selected_multipliers={},
                selected_multipliers_raw={},
            )
        except Exception:
            logger.debug("Failed to persist retry snapshot for baseline run", exc_info=True)

        return {
            "mode": "baseline",
            "applied": True,
            "updates": selected_updates,
            "features": {},
            "notes": ["Baseline mode: applied preset override without selector predictions."],
            "heuristic_preset": selected_preset,
            "selected_preset": selected_preset,
            "preset_forced": bool(forced_preset),
            "selector_strategy": "featurewise_ridge_regression",
            "yhat_scores": {},
            "project_dir": str(project_dir),
            "cache_used": False,
        }

    # Keep metadata extraction tied to original uploads when resized copies exist,
    # because some EXIF/XMP fields may be dropped during resize.
    metadata_image_dir = image_dir_path
    if image_dir_path.name == "images_resized":
        original_dir = project_dir / "images"
        if original_dir.exists() and original_dir.is_dir():
            metadata_image_dir = original_dir

    processing_image_dir = image_dir_path

    ctx = ModeContext(
        metadata_image_dir=metadata_image_dir,
        processing_image_dir=processing_image_dir,
        colmap_dir=Path(colmap_dir),
        params=params,
    )
    fingerprint = _combined_image_fingerprint(metadata_image_dir, processing_image_dir)

    # Load from project-level EXIF file ONLY (single source of truth)
    project_exif = _load_project_exif(project_dir, mode, fingerprint)
    if project_exif is not None:
        result_features = dict(project_exif.get("features") or {})
        result_notes = list(project_exif.get("notes") or [])
        heuristic_preset = str(project_exif.get("heuristic_preset") or "balanced")
        cache_used = True
        logger.info(f"Loaded EXIF from project-level file: {_project_exif_file_path(project_dir)}")
    else:
        # Extract fresh compact scene descriptors
        from datetime import datetime

        result = build_compact_scene_descriptor_preset(ctx)

        result_features = dict(result.features)
        result_notes = list(result.notes)
        heuristic_preset = str(result.updates.get("preset_name") or "balanced")

        # Save to project-level EXIF file (ONLY location)
        _save_project_exif(
            project_dir,
            mode,
            fingerprint,
            {
                "features": result_features,
                "notes": result_notes,
                "heuristic_preset": heuristic_preset,
                "extracted_at": datetime.utcnow().isoformat() + "Z",
            },
        )
        cache_used = False
        logger.info(f"Extracted and saved fresh EXIF features to project-level file")



    selector_choice = str(params.get("ai_selector_strategy") or "").strip().lower()
    model_features = with_legacy_groupwise_features(result_features, ctx, selector_choice)

    # For jitter-only runs (Stage 1 exploration): skip model prediction entirely.
    # Featurewise Ridge with an untrained model produces noise; no meaning.
    # Jitter is applied directly to base parameters; no "selected multiplier" exists.
    run_jitter_only = bool(params.get("run_jitter_only", False))

    if run_jitter_only:
        logger.info(
            "RUN_JITTER_ONLY enabled: skipping model prediction â€” run explores base parameter space via jitter only."
        )
        selection: dict[str, Any] = {
            "yhat_scores": {},
            "selected_multipliers": {},
            "selected_multipliers_raw": {},
            "selected_log_multipliers": {},
            "selected_log_multipliers_raw": {},
            "updates": {},
        }
        selected_updates: dict[str, Any] = {}
        selected_preset = "jitter_only"
        preset_forced = False
    else:
        if selector_choice == "featurewise_mlp":
            from .featurewise_mlp import predict_featurewise_mlp_multipliers
            shared_models_dir = project_dir / "models"

            neural_result = predict_featurewise_mlp_multipliers(
                shared_models_dir=shared_models_dir,
                mode_name="featurewise",
                features=model_features,
                params=params,
            )

            if neural_result is not None:
                selection = neural_result
                selected_updates = build_featurewise_ridge_updates(params, neural_result["yhat_scores"])
                selected_preset = str(neural_result.get("selected_preset", "featurewise_mlp"))
                preset_forced = False
                logger.info(
                    "FEATUREWISE_MLP_PREDICTION mode=%s type=%s multipliers=%s",
                    mode,
                    neural_result.get("model_type", "unknown"),
                    json.dumps(neural_result.get("group_multipliers", {})),
                )
            else:
                raise RuntimeError(
                    f"Neural featurewise prediction failed for project {project_dir}; no fallback to ridge is allowed."
                )
        elif selector_choice == "compact_featurewise_mlp":
            from .compact_featurewise_mlp import predict_compact_featurewise_mlp_multipliers
            shared_models_dir = project_dir / "models"

            compact_neural_result = predict_compact_featurewise_mlp_multipliers(
                shared_models_dir=shared_models_dir,
                features=model_features,
                params=params,
            )
            if compact_neural_result is not None:
                selection = compact_neural_result
                selected_updates = build_featurewise_ridge_updates(params, compact_neural_result["yhat_scores"])
                selected_preset = str(compact_neural_result.get("selected_preset", "compact_featurewise_mlp"))
                preset_forced = False
                logger.info(
                    "COMPACT_FEATUREWISE_MLP_PREDICTION mode=%s multipliers=%s",
                    mode,
                    json.dumps(compact_neural_result.get("group_multipliers", {})),
                )
            else:
                raise RuntimeError(
                    f"Compact neural featurewise prediction failed for project {project_dir}; no fallback to ridge is allowed."
                )
        elif selector_choice == "compact_featurewise_ridge_regression":
            from .compact_featurewise_ridge_regression import select_compact_featurewise_ridge_multipliers

            selection = select_compact_featurewise_ridge_multipliers(
                project_dir=project_dir,
                mode=mode,
                x_features=model_features,
                params=params,
            )
            selected_updates = dict(selection.get("updates") or {})
            selected_preset = str(selection.get("selected_preset") or "compact_featurewise_ridge_regression")
            preset_forced = False
        else:
            # Featurewise Ridge Regression uses the offline-trained quality model.
            selection = select_featurewise_ridge_multipliers(
                project_dir=project_dir,
                mode=mode,
                x_features=model_features,
                params=params,
            )
            selected_updates = dict(selection.get("updates") or {})
            selected_preset = str(selection.get("selected_preset") or "featurewise_ridge_regression")
            preset_forced = False

    # Apply selected parameter updates (empty for jitter-only runs since prediction was skipped)
    for key, value in selected_updates.items():
        if key == "preset_name":
            continue
        params[key] = value

    # Apply run jitter if configured (for pipeline exploration)
    # Apply pre-generated group-specific multipliers first (if available)
    _apply_group_specific_multipliers(params)

    # Apply run jitter if configured (for pipeline exploration)
    _apply_run_jitter_inline(params)

    logger.info(
        "FIXED_MULTIPLIER_EFFECTIVE_PARAMS params=%s",
        json.dumps(
            {key: params.get(key) for key in sorted(GSPLAT_PARAMETER_DEFAULTS.keys()) if params.get(key) is not None},
            sort_keys=True,
        ),
    )

    feature_details = _build_feature_log_details(mode, model_features, _count_supported_images(processing_image_dir))
    logger.info(
        "AI_INPUT_MODE_FEATURES mode=%s details=%s",
        mode,
        json.dumps(feature_details, sort_keys=True),
    )
    selector_strategy = str(params.get("ai_selector_strategy") or "").strip().lower()
    if selector_strategy not in valid_selector_strategies:
        selector_strategy = "featurewise_ridge_regression"
    logger.info(
        "AI_INPUT_MODE_PRESET mode=%s heuristic=%s selected=%s cache_used=%s forced=%s strategy=%s",
        mode,
        heuristic_preset,
        selected_preset,
        str(bool(cache_used)).lower(),
        str(preset_forced).lower(),
        selector_strategy,
    )
    logger.info(
        "AI_INPUT_MODE_INITIAL_PARAMS mode=%s params=%s",
        mode,
        json.dumps(_build_initial_params_log(params), sort_keys=True),
    )

    logger.info(
        "AI input preset applied mode=%s selected_preset=%s cache_used=%s updates=%s features=%s",
        mode,
        selected_preset,
        cache_used,
        selected_updates,
        model_features,
    )

    try:
        _persist_retry_snapshot(
            project_dir=project_dir,
            params=params,
            mode=mode,
            selected_preset=selected_preset,
            selected_multipliers=dict(selection.get("selected_multipliers") or selection.get("yhat_scores") or {}),
            selected_multipliers_raw=dict(selection.get("selected_multipliers_raw") or {}),
            selected_log_multipliers=dict(selection.get("selected_log_multipliers") or {}),
            candidate_score_checks=dict(selection.get("candidate_score_checks") or {}),
        )
    except Exception:
        logger.debug("Failed to persist retry snapshot for run", exc_info=True)

    return {
        "mode": mode,
        "applied": True,
        "updates": selected_updates,
        "features": model_features,
        "notes": result_notes,
        "heuristic_preset": heuristic_preset,
        "selected_preset": selected_preset,
        "preset_forced": preset_forced,
        "selector_strategy": selector_strategy,
        "yhat_scores": dict(selection.get("yhat_scores") or {}),
        "selected_multipliers": dict(selection.get("selected_multipliers") or selection.get("yhat_scores") or {}),
        "selected_multipliers_raw": dict(selection.get("selected_multipliers_raw") or {}),
        "selected_log_multipliers": dict(selection.get("selected_log_multipliers") or {}),
        "selected_log_multipliers_raw": dict(selection.get("selected_log_multipliers_raw") or {}),
        "score_spreads": dict(selection.get("score_spreads") or {}),
        "candidate_score_checks": dict(selection.get("candidate_score_checks") or {}),
        "candidate_points": int(selection.get("candidate_points") or 0),
        "has_signal": bool(selection.get("has_signal", True)),
        "n_runs": int(selection.get("n_runs") or 0),
        "run_jitter_multiplier": float(params.get("run_jitter_multiplier") or 1.0),
        # Per-group log-space multipliers (Â§4.X.2) â€” stored as (multiplier, log_action) pairs
        "group_multipliers": {
            "geometry_lr": {
                "multiplier": float(params.get("geometry_lr_multiplier") or 1.0),
                "log_action": float(params.get("geometry_lr_log_action") or 0.0),
                "bounds": [0.5, 2.0],
            },
            "appearance_lr": {
                "multiplier": float(params.get("appearance_lr_multiplier") or 1.0),
                "log_action": float(params.get("appearance_lr_log_action") or 0.0),
                "bounds": [0.5, 2.0],
            },
            "densification": {
                "multiplier": float(params.get("densification_multiplier") or 1.0),
                "log_action": float(params.get("densification_log_action") or 0.0),
                "bounds": [0.7, 1.42],
            },
        },
        "effective_params": _build_initial_params_log(params),
        "remarks": params.get("remarks"),
        "project_dir": str(project_dir),
        "cache_used": cache_used,
    }


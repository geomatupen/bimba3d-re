from __future__ import annotations

import json
import math
import time
import numpy as np
from pathlib import Path
from typing import Any

from .common import clamp_float, clamp_int, safe_ratio
from .legacy_support import normalize_ai_input_mode_token
from .relative_quality_score import compute_relative_quality_summary

# NON_COMPACT_FEATUREWISE: legacy group-wise path kept for comparison.
# Three parameter groups for better sample efficiency
# Each group learns a single multiplier for related parameters
PARAMETER_GROUPS = {
    "geometry_lr_mult": ["position_lr_init_mult", "scaling_lr_mult", "rotation_lr_mult"],
    "appearance_lr_mult": ["feature_lr_mult", "opacity_lr_mult", "lambda_dssim_mult"],
    "densification_mult": ["densify_grad_threshold_mult", "opacity_threshold_mult"],
}

# Group model keys (what we actually learn)
GROUP_KEYS = ["geometry_lr_mult", "appearance_lr_mult", "densification_mult"]

# Individual multiplier keys expanded from the learned group multipliers.
MULT_KEYS = [
    "feature_lr_mult",
    "position_lr_init_mult",
    "scaling_lr_mult",
    "opacity_lr_mult",
    "rotation_lr_mult",
    "densify_grad_threshold_mult",
    "opacity_threshold_mult",
    "lambda_dssim_mult",
]

# Safe bounds for group multipliers
GROUP_BOUNDS = {
    "geometry_lr_mult": (0.5, 2.0),
    "appearance_lr_mult": (0.5, 2.0),
    "densification_mult": (0.7, 1.4285714286),
}

# Safe bounds for individual multipliers.
SAFE_BOUNDS = {
    "feature_lr_mult": (0.5, 2.0),
    "position_lr_init_mult": (0.5, 2.0),
    "scaling_lr_mult": (0.5, 2.0),
    "opacity_lr_mult": (0.5, 2.0),
    "rotation_lr_mult": (0.5, 2.0),
    "densify_grad_threshold_mult": (0.7, 1.4285714286),
    "opacity_threshold_mult": (0.7, 1.4285714286),
    "lambda_dssim_mult": (0.5, 2.0),
}

# Mode-specific context dimensions
MODE_CONTEXT_DIMS = {
    "exif_compact_featurewise": 16,   # 1 intercept + compact scene descriptors
}

def normalize_context_mode(mode: str) -> str:
    return normalize_ai_input_mode_token(mode)

# ========== FEATUREWISE MODE ==========
# Each parameter group uses a different subset of features, tailored to
# what physically affects that group.  This reduces per-group dimensionality
# and improves sample efficiency.
#
# Feature key â†’ normalised value mapping (all include intercept=1.0 at [0]):
#   focal_norm:              (focal_length_mm - 50) / 150
#   iso_norm:                (log10(iso) - 2) / 3
#   image_resolution_norm:   (megapixels - 12) / 12
#   gsd_norm:                gsd_median / 0.5
#   overlap_proxy:           [0-1] as-is
#   coverage_spread:         [0-1] as-is
#   camera_angle_bucket:     bucket / 3.0
#   heading_consistency:     [0-1] as-is
#   texture_density:         [0-1] as-is
#   blur_motion_risk:        [0-1] as-is
#   terrain_roughness_proxy: [0-1] as-is
#   vegetation_cover:        [0-1] as-is
#   vegetation_complexity:   [0-1] as-is

FEATUREWISE_GROUP_FEATURES = {
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
    ],  # dim = 10
    "appearance_lr_mult": [
        "intercept",
        "iso_norm",
        "image_resolution_norm",
        "blur_motion_risk",
        "texture_density",
        "vegetation_cover",
        "vegetation_complexity",
    ],  # dim = 7
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
    ],  # dim = 9
}

FEATUREWISE_GROUP_DIMS = {k: len(v) for k, v in FEATUREWISE_GROUP_FEATURES.items()}


def _get_shared_model_dir(project_dir: Path) -> Path | None:
    """Get shared model directory from project config if available (pipeline context)."""
    config_path = project_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            shared_dir = config.get("shared_model_dir")
            if shared_dir:
                return Path(shared_dir)
    except Exception:
        pass
    return None


def _default_score_optimizer_model(candidate_points: int = 30) -> dict[str, Any]:
    """Default v6 score-optimizer ridge model used when no offline model exists."""
    models: dict[str, Any] = {}
    for key in GROUP_KEYS:
        d_ctx = len(FEATUREWISE_GROUP_FEATURES[key])
        d_phi = d_ctx + 2 + (d_ctx - 1)
        models[key] = {
            "A": np.eye(d_phi, dtype=np.float64).tolist(),
            "b": np.zeros(d_phi, dtype=np.float64).tolist(),
            "n": 0,
            "design_dim": d_phi,
            "context_features": FEATUREWISE_GROUP_FEATURES[key],
            "design": "x_plus_a_plus_a2_plus_xa",
            "action_space": "log_multiplier",
            "action_bounds": [
                float(math.log(GROUP_BOUNDS[key][0])),
                float(math.log(GROUP_BOUNDS[key][1])),
            ],
        }

    return {
        "version": 6,
        "model_family": "featurewise_ridge_regression",
        "mode": "exif_compact_featurewise",
        "lambda_ridge": 2.0,
        "runs": 0,
        "score_mean": 0.0,
        "feature_scalers": {
            g: {name: {"mean": 0.0, "std": 1.0} for name in FEATUREWISE_GROUP_FEATURES[g] if name != "intercept"}
            for g in GROUP_KEYS
        },
        "candidate_points": int(max(5, candidate_points)),
        "models": models,
    }


# â”€â”€ offline model loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _load_offline_model(model_path: Path) -> dict[str, Any] | None:
    """Load an offline-trained Featurewise Ridge quality model from disk.

        Returns None if the file is missing or invalid.

        Supported payload shapes:
            1) offline_model_v3 wrapper: {"schema": "offline_model_v3", "model": {...}}
            2) direct model payload used by seeded learner files in test pipelines.
    """
    if not model_path.exists():
        return None
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("schema") == "offline_model_v3":
            inner = payload.get("model")
            if isinstance(inner, dict):
                # Preserve metrics from offline training
                if "metrics" not in inner and "metrics" in payload:
                    inner["metrics"] = payload.get("metrics")
                if "model_family" not in inner and "model_family" in payload:
                    inner["model_family"] = payload.get("model_family")
                return inner
        # Test pipelines may seed a direct learner JSON without wrapper schema.
        if isinstance(payload, dict) and payload.get("model_family") == "featurewise_ridge_regression":
            nested = payload.get("model")
            # Some artifacts wrap the trained structure under payload["model"].
            if isinstance(nested, dict):
                if "metrics" not in nested and "metrics" in payload:
                    nested["metrics"] = payload.get("metrics")
                if "model_family" not in nested:
                    nested["model_family"] = payload.get("model_family")
                return nested
            return payload
    except Exception:
        pass
    return None


def _offline_model_dir(project_dir: Path) -> Path:
    """Return the directory where offline quality models are stored.

    Checks project config for a shared_model_dir, then falls back to a
    sibling _offline_training/models directory relative to the project.
    """
    # 1. Check project config for an explicit override
    config_path = project_dir / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            od = cfg.get("offline_model_dir")
            if od:
                return Path(od)
        except Exception:
            pass

    # 2. Shared model dir (pipeline context) â†’ sibling _offline_training folder
    shared_dir = _get_shared_model_dir(project_dir)
    if shared_dir:
        return shared_dir.parent / "_offline_training" / "models"

    # 3. Global training-data model store: bimba3d_backend/data/_offline_training/models
    workspace_root = Path(__file__).resolve().parents[3]
    return workspace_root / "bimba3d_backend" / "data" / "_offline_training" / "models"


def load_offline_quality_model(project_dir: Path) -> dict[str, Any] | None:
    """Load the most recently trained offline quality model.

    Scans for quality_model_*.json files (newest first) since no file is
    overwritten â€” every training creates a new versioned file.
    """
    model_dir = _offline_model_dir(project_dir)
    # Try versioned files newest-first
    versioned = sorted(model_dir.glob("quality_model_*.json"), key=lambda p: p.stem, reverse=True)
    for path in versioned:
        m = _load_offline_model(path)
        if m is not None:
            return m
    # Test-model seed copied into the project-local model directory.
    seeded_dir = project_dir / "models" / "featurewise_ridge_regression"
    seeded = sorted(seeded_dir.glob("*.json"), key=lambda p: p.stem, reverse=True)
    for path in seeded:
        m = _load_offline_model(path)
        if m is not None:
            return m
    return None


def build_context_vector(features: dict[str, Any], mode: str) -> np.ndarray:
    """Build normalized 16-D context vector for compact scene descriptors.

    Expected dimensions:
        1 intercept + compact EXIF, flight-geometry, image, vegetation, and terrain descriptors = 16
    """
    mode = normalize_context_mode(mode)
    if mode != "exif_compact_featurewise":
        raise ValueError(
            f"Unsupported mode '{mode}' for contextual ridge. Expected 'exif_compact_featurewise'."
        )

    x = [1.0]  # Intercept term

    # â”€â”€ EXIF features (all modes) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    focal = features.get("focal_length_mm", 50.0)
    x.append((focal - 50.0) / 150.0)           # [8-300] â†’ ~[-0.28, 1.67]

    shutter = features.get("shutter_s", 0.001)
    x.append(np.log10(shutter + 1e-6) / 3.0)   # log scale

    iso = features.get("iso", 400.0)
    x.append((np.log10(iso) - 2.0) / 3.0)      # log scale

    img_w = features.get("img_width_median", 4000.0)
    x.append((img_w - 4000.0) / 2000.0)

    img_h = features.get("img_height_median", 3000.0)
    x.append((img_h - 3000.0) / 1500.0)

    # â”€â”€ Flight plan features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    gsd = features.get("gsd_median", 0.05)
    x.append(gsd / 0.5)                     # [0.001-0.5] â†’ [0.002, 1.0]

    x.append(features.get("overlap_proxy", 0.5))
    x.append(features.get("coverage_spread", 0.5))

    angle_bucket = features.get("camera_angle_bucket", 0)
    x.append(float(angle_bucket) / 3.0)     # {0,1,2,3} â†’ [0, 0.33, 0.67, 1.0]

    x.append(features.get("heading_consistency", 0.5))

    # â”€â”€ External / image-analysis features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    x.append(features.get("vegetation_cover_percentage", 0.5))
    x.append(features.get("vegetation_complexity_score", 0.5))
    x.append(features.get("terrain_roughness_proxy", 0.5))
    x.append(features.get("texture_density", 0.5))
    x.append(features.get("blur_motion_risk", 0.5))

    return np.array(x, dtype=np.float32)


def _build_compact_vector(features: dict[str, Any]) -> np.ndarray:
    """Build context vector for the exif_compact mode.

    10 dimensions:
      [0]  intercept (1.0)
      [1]  focal_norm          â€” focal length normalised
      [2]  image_resolution_norm â€” combined image size proxy
      [3]  gsd_norm            â€” ground sampling distance
      [4]  overlap_proxy       â€” frame overlap [0-1]
      [5]  coverage_spread     â€” geographic extent [0-1]
      [6]  camera_angle_bucket â€” {0,1,2,3} normalised to [0,1]
      [7]  heading_consistency â€” flight path regularity [0-1]
      [8]  texture_density     â€” image texture richness [0-1]
      [9]  blur_motion_risk    â€” motion blur risk [0-1]

    Rationale: these 9 features cover the most discriminative scene
    properties while keeping the context dimension small (10), which
    improves sample efficiency and reduces the ridge regularisation
    needed to keep the model stable.
    """
    x = [1.0]  # intercept

    focal = features.get("focal_length_mm", 50.0)
    x.append((focal - 50.0) / 150.0)

    # Image resolution: combine width and height into a single proxy
    img_w = features.get("img_width_median", 4000.0)
    img_h = features.get("img_height_median", 3000.0)
    # Normalise total pixel count relative to a 12 MP reference
    megapixels = (img_w * img_h) / 1e6
    x.append((megapixels - 12.0) / 12.0)       # centred at 12 MP

    gsd = features.get("gsd_median", 0.05)
    x.append(gsd / 0.5)

    x.append(features.get("overlap_proxy", 0.5))
    x.append(features.get("coverage_spread", 0.5))

    angle_bucket = features.get("camera_angle_bucket", 0)
    x.append(float(angle_bucket) / 3.0)

    x.append(features.get("heading_consistency", 0.5))
    x.append(features.get("texture_density", 0.5))
    x.append(features.get("blur_motion_risk", 0.5))

    return np.array(x, dtype=np.float32)


def _build_featurewise_vector(features: dict[str, Any], group_key: str) -> np.ndarray:
    """Build a per-group feature vector using deterministic normalization."""
    feature_names = FEATUREWISE_GROUP_FEATURES[group_key]
    x: list[float] = []

    for name in feature_names:
        if name == "intercept":
            x.append(1.0)
        elif name == "focal_norm":
            focal = features.get("focal_length_mm", 24.0)
            x.append(math.log(max(float(focal or 24.0), 1e-9)))
        elif name == "iso_norm":
            iso = features.get("iso", 400.0)
            x.append(math.log(max(float(iso or 400.0), 1e-9)))
        elif name == "image_resolution_norm":
            img_w = features.get("img_width_median", 4000.0)
            img_h = features.get("img_height_median", 3000.0)
            megapixels = (float(img_w or 4000.0) * float(img_h or 3000.0)) / 1e6
            x.append(megapixels)
        elif name == "gsd_norm":
            gsd = features.get("gsd_median", 0.05)
            x.append(math.log(max(float(gsd or 0.05), 1e-9)))
        elif name == "overlap_proxy":
            x.append(float(features.get("overlap_proxy", 0.5) or 0.5))
        elif name == "coverage_spread":
            x.append(float(features.get("coverage_spread", 0.5) or 0.5))
        elif name == "camera_angle_bucket":
            angle_bucket = features.get("camera_angle_bucket", 0)
            x.append(float(angle_bucket or 0))
        elif name == "heading_consistency":
            x.append(float(features.get("heading_consistency", 0.5) or 0.5))
        elif name == "texture_density":
            x.append(float(features.get("texture_density", 0.5) or 0.5))
        elif name == "blur_motion_risk":
            x.append(float(features.get("blur_motion_risk", 0.5) or 0.5))
        elif name == "terrain_roughness_proxy":
            x.append(float(features.get("terrain_roughness_proxy", features.get("terrain_roughness", 0.5)) or 0.5))
        elif name == "vegetation_cover":
            val = features.get("vegetation_cover_percentage", features.get("vegetation_cover", 0.5))
            x.append(float(val or 0.5))
        elif name == "vegetation_complexity":
            val = features.get("vegetation_complexity_score", features.get("vegetation_complexity", 0.5))
            x.append(float(val or 0.5))
        else:
            x.append(0.0)

    return np.array(x, dtype=np.float32)


# Features requiring log transform before z-score
_LOG_TRANSFORM_FEATURES = {"focal_norm", "gsd_norm", "iso_norm"}
_SCALER_STD_FLOOR = 0.01


def _build_featurewise_vector_scaled(
    features: dict[str, Any],
    group_key: str,
    scaler: dict[str, dict[str, float]],
) -> np.ndarray:
    """Build a z-score standardized feature vector using saved training-set scaler.

    Intercept = 1.0 always.
    For each feature: apply log transform if needed, then (value - mean) / std.
    Uses the scaler saved alongside the offline model â€” never recomputes from data.
    """
    feature_names = FEATUREWISE_GROUP_FEATURES[group_key]
    x: list[float] = []

    for name in feature_names:
        if name == "intercept":
            x.append(1.0)
            continue

        # Extract raw value
        if name == "focal_norm":
            raw = float(features.get("focal_length_mm", 24.0) or 24.0)
        elif name == "iso_norm":
            raw = float(features.get("iso", 400.0) or 400.0)
        elif name == "image_resolution_norm":
            w = float(features.get("img_width_median", 4000.0) or 4000.0)
            h = float(features.get("img_height_median", 3000.0) or 3000.0)
            raw = (w * h) / 1e6
        elif name == "gsd_norm":
            raw = float(features.get("gsd_median", 0.05) or 0.05)
        elif name == "overlap_proxy":
            raw = float(features.get("overlap_proxy", 0.5) or 0.5)
        elif name == "coverage_spread":
            raw = float(features.get("coverage_spread", 0.5) or 0.5)
        elif name == "camera_angle_bucket":
            raw = float(features.get("camera_angle_bucket", 0) or 0)
        elif name == "heading_consistency":
            raw = float(features.get("heading_consistency", 0.5) or 0.5)
        elif name == "texture_density":
            raw = float(features.get("texture_density", 0.5) or 0.5)
        elif name == "blur_motion_risk":
            raw = float(features.get("blur_motion_risk", 0.5) or 0.5)
        elif name == "terrain_roughness_proxy":
            raw = float(features.get("terrain_roughness_proxy", features.get("terrain_roughness", 0.5)) or 0.5)
        elif name == "vegetation_cover":
            val = features.get("vegetation_cover_percentage", features.get("vegetation_cover", 0.5))
            raw = float(val or 0.5)
        elif name == "vegetation_complexity":
            val = features.get("vegetation_complexity_score", features.get("vegetation_complexity", 0.5))
            raw = float(val or 0.5)
        else:
            raw = 0.0

        # Log transform for multiplicative features
        if name in _LOG_TRANSFORM_FEATURES:
            transformed = math.log(max(raw, 1e-9))
        else:
            transformed = raw

        # Z-score using saved training-set statistics
        stats = scaler.get(name, {"mean": 0.0, "std": 1.0})
        mean = float(stats.get("mean", 0.0))
        std = max(float(stats.get("std", 1.0)), _SCALER_STD_FLOOR)
        x.append((transformed - mean) / std)

    return np.array(x, dtype=np.float32)


def _build_score_design_vector(x_context: np.ndarray, action_log: float) -> np.ndarray:
    """Build score-model feature map phi(x,a) = [x, a, a^2, x_no_intercept*a]."""
    interactions = x_context[1:] * float(action_log)
    return np.concatenate(
        [
            x_context.astype(np.float64),
            np.array([float(action_log), float(action_log) * float(action_log)], dtype=np.float64),
            interactions.astype(np.float64),
        ]
    )


def _solve_theta(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the mean theta by solving the ridge normal equations."""
    A_inv = np.linalg.inv(A)
    return A_inv @ b


def _select_group_action_from_score_model(
    *,
    model: dict[str, Any],
    group_key: str,
    x_features: dict[str, Any],
    candidate_logs: list[float] | None = None,
) -> tuple[float, float, float, float, dict[str, Any]]:
    """Select bounded log-action by maximizing predicted score for one group.

    Returns:
      (best_log_action, best_multiplier, selected_score, mean_selected_score, model_state)
    """
    md = model["models"][group_key]
    A = np.array(md["A"], dtype=np.float64)
    b = np.array(md["b"], dtype=np.float64)
    n = int(md.get("n", 0))

    scaler = model.get("feature_scalers", {}).get(group_key, {})
    x = _build_featurewise_vector_scaled(x_features, group_key, scaler)

    action_bounds = md.get("action_bounds")
    if isinstance(action_bounds, list) and len(action_bounds) >= 2:
        lo = float(action_bounds[0])
        hi = float(action_bounds[1])
        if not math.isfinite(lo) or not math.isfinite(hi):
            lo = float(math.log(GROUP_BOUNDS[group_key][0]))
            hi = float(math.log(GROUP_BOUNDS[group_key][1]))
    else:
        lo = float(math.log(GROUP_BOUNDS[group_key][0]))
        hi = float(math.log(GROUP_BOUNDS[group_key][1]))
    if hi < lo:
        lo, hi = hi, lo
    lo_mult, hi_mult = float(math.exp(lo)), float(math.exp(hi))
    if candidate_logs:
        candidates = np.array([clamp_float(float(value), lo, hi) for value in candidate_logs], dtype=np.float64)
    else:
        # Use the stored fallback only when testing did not pass an explicit candidate grid.
        points = int(max(5, model.get("candidate_points", 30)))
        candidates = np.linspace(lo, hi, points, dtype=np.float64)

    theta = _solve_theta(A, b)

    # Score every candidate action.
    all_scores: list[float] = []
    for a_log in candidates:
        phi = _build_score_design_vector(x, float(a_log))
        all_scores.append(float(phi @ theta))
    all_mean_scores = all_scores

    score_min = min(all_scores)
    score_max = max(all_scores)
    score_spread = score_max - score_min

    # If the score surface is flat (no meaningful signal), return neutral (1.0).
    # This prevents the lower-bound candidate from winning by default when
    # theta â‰ˆ 0 (cold-start or under-trained model).
    SIGNAL_THRESHOLD = 1e-6
    if score_spread < SIGNAL_THRESHOLD:
        best_log = 0.0          # log(1.0) = 0 â†’ neutral multiplier
        best_score = all_scores[len(all_scores) // 2]
        best_mean_score = all_mean_scores[len(all_mean_scores) // 2]
    else:
        best_log = 0.0
        best_score = float("-inf")
        best_mean_score = float("-inf")
        for i, a_log in enumerate(candidates):
            if all_scores[i] > best_score:
                best_score = all_scores[i]
                best_mean_score = all_mean_scores[i]
                best_log = float(a_log)

    best_mult = float(math.exp(best_log))
    best_mult = clamp_float(best_mult, lo_mult, hi_mult)
    best_log = float(math.log(best_mult))

    theta_norm = float(np.linalg.norm(theta))
    state = {
        "selected_log_raw": best_log,
        "selected_log_clamped": best_log,
        "selected_mult_raw": best_mult,
        "selected_mult_clamped": best_mult,
        "selected_score": best_score,
        "selected_score_mean": best_mean_score,
        "score_spread": score_spread,
        "has_signal": score_spread >= SIGNAL_THRESHOLD,
        "theta_norm": theta_norm,
        "n": n,
        "dim": int(len(x)),
        "candidate_curve": [
            {
                "candidate_log_multiplier": float(a_log),
                "candidate_multiplier": float(math.exp(float(a_log))),
                "predicted_score": float(all_scores[i]),
                "predicted_score_mean": float(all_mean_scores[i]),
                "selected": bool(abs(float(a_log) - best_log) < 1e-12),
            }
            for i, a_log in enumerate(candidates)
        ],
    }
    return best_log, best_mult, best_score, best_mean_score, state


def _build_updates(params: dict[str, Any], multipliers: dict[str, float]) -> dict[str, Any]:
    """Build parameter updates from multipliers."""
    feature_lr = float(params.get("feature_lr", 2.5e-3))
    position_lr_init = float(params.get("position_lr_init", 1.6e-4))
    scaling_lr = float(params.get("scaling_lr", 5.0e-3))
    opacity_lr = float(params.get("opacity_lr", 5.0e-2))
    rotation_lr = float(params.get("rotation_lr", 1.0e-3))
    densify_grad_threshold = float(params.get("densify_grad_threshold", 2.0e-4))
    opacity_threshold = float(params.get("opacity_threshold", 0.005))
    lambda_dssim = float(params.get("lambda_dssim", 0.2))

    return {
        "preset_name": "featurewise_ridge_regression",
        "feature_lr": feature_lr * multipliers["feature_lr_mult"],
        "position_lr_init": position_lr_init * multipliers["position_lr_init_mult"],
        "scaling_lr": scaling_lr * multipliers["scaling_lr_mult"],
        "opacity_lr": opacity_lr * multipliers["opacity_lr_mult"],
        "rotation_lr": rotation_lr * multipliers["rotation_lr_mult"],
        "densify_grad_threshold": densify_grad_threshold * multipliers["densify_grad_threshold_mult"],
        "opacity_threshold": opacity_threshold * multipliers["opacity_threshold_mult"],
        "lambda_dssim": lambda_dssim * multipliers["lambda_dssim_mult"],
    }


def select_featurewise_ridge_multipliers(
    *,
    project_dir: Path,
    mode: str,
    x_features: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Select bounded multipliers by maximizing the Featurewise Ridge quality model."""
    model = load_offline_quality_model(project_dir)

    if model is None:
        raise FileNotFoundError(f"Featurewise Ridge quality model not found for project {project_dir}.")

    if str(model.get("model_family") or "") != "featurewise_ridge_regression":
        raise RuntimeError(
            "Only Featurewise Ridge Regression quality models are supported. Retrain the model from Training Data."
        )

    group_multipliers: dict[str, float] = {}
    group_multipliers_raw: dict[str, float] = {}
    group_log_multipliers: dict[str, float] = {}
    group_log_multipliers_raw: dict[str, float] = {}
    model_states: dict[str, dict[str, float]] = {}

    for group_key in GROUP_KEYS:
        candidate_logs = _candidate_logs_for_group(params, group_key)
        best_log, best_mult, _, _, state = _select_group_action_from_score_model(
            model=model,
            group_key=group_key,
            x_features=x_features,
            candidate_logs=candidate_logs,
        )
        group_multipliers[group_key] = best_mult
        group_multipliers_raw[group_key] = best_mult
        group_log_multipliers[group_key] = best_log
        group_log_multipliers_raw[group_key] = best_log
        model_states[group_key] = state

    # Expand 3 group multipliers to 8 individual multipliers
    multipliers: dict[str, float] = {}
    multipliers_raw: dict[str, float] = {}
    log_multipliers: dict[str, float] = {}
    log_multipliers_raw: dict[str, float] = {}
    for group_key, member_keys in PARAMETER_GROUPS.items():
        group_mult = group_multipliers[group_key]
        group_mult_raw = group_multipliers_raw[group_key]
        group_log = group_log_multipliers[group_key]
        group_log_raw = group_log_multipliers_raw[group_key]
        for member_key in member_keys:
            multipliers[member_key] = group_mult
            multipliers_raw[member_key] = group_mult_raw
            log_multipliers[member_key] = group_log
            log_multipliers_raw[member_key] = group_log_raw

    updates = _build_updates(params, multipliers)

    representative_x = _build_featurewise_vector(x_features, "geometry_lr_mult")

    has_signal = all(model_states[g].get("has_signal", False) for g in GROUP_KEYS)
    score_spreads = {g: model_states[g].get("score_spread", 0.0) for g in GROUP_KEYS}
    candidate_score_checks = {
        g: list(model_states[g].get("candidate_curve") or [])
        for g in GROUP_KEYS
    }

    return {
        "selected_preset": "featurewise_ridge_regression",
        "yhat_scores": multipliers,
        "selected_multipliers": multipliers,
        "selected_multipliers_raw": multipliers_raw,
        "selected_log_multipliers": log_multipliers,
        "selected_log_multipliers_raw": log_multipliers_raw,
        "updates": updates,
        "context_vector": representative_x.tolist(),
        "context_norm": float(np.linalg.norm(representative_x)),
        "model_states": model_states,
        "exploration_mode": "greedy",
        "has_signal": has_signal,
        "score_spreads": score_spreads,
        "candidate_score_checks": candidate_score_checks,
        "n_runs": int(model.get("runs", 0)),
    }


def _candidate_logs_for_group(params: dict[str, Any], group_key: str) -> list[float] | None:
    source = params.get("candidate_log_multipliers_by_group")
    if not isinstance(source, dict):
        return None
    raw_values = source.get(group_key)
    if not isinstance(raw_values, list):
        return None
    values = [float(value) for value in raw_values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return values or None


def _normalize_series(values: list[float], invert: bool = False) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        out = [0.5 for _ in values]
    else:
        out = [(v - lo) / (hi - lo) for v in values]
    if invert:
        out = [1.0 - v for v in out]
    return out


def _step_value_with_neighbors(values: dict[int, float], step: int) -> float | None:
    for candidate in (step, step + 1, step - 1):
        value = values.get(int(candidate))
        if isinstance(value, (int, float)):
            return float(value)
    return None


def update_from_run_featurewise_ridge(
    *,
    project_dir: Path,
    mode: str,
    selected_preset: str,
    yhat_scores: dict[str, float],
    eval_history: list[dict[str, Any]],
    baseline_eval_history: list[dict[str, Any]] | None,
    loss_by_step: dict[int, float],
    elapsed_by_step: dict[int, float],
    x_features: dict[str, Any] | None,
    run_id: str,
    logger,
    apply_update: bool = True,
    baseline_loss_by_step_override: dict[int, float] | None = None,
    score_reference_step: int | None = None,
) -> dict[str, Any]:
    """Update contextual models with observed score.

    This follows the same score calculation as the existing learners
    but updates all 8 multiplier models with the context vector.
    """
    if not eval_history:
        return {"updated": False, "reason": "no_eval_history"}

    eval_rows = [row for row in eval_history if isinstance(row, dict) and isinstance(row.get("step"), (int, float))]
    if not eval_rows:
        return {"updated": False, "reason": "no_eval_steps"}

    eval_rows.sort(key=lambda r: int(r.get("step", 0)))
    eval_steps = [int(r["step"]) for r in eval_rows]

    # Find best step by quality (PSNR + SSIM + LPIPS), not by loss
    # Quality metrics available only at eval steps
    quality_scores_by_step: dict[int, float] = {}
    for row in eval_rows:
        step = int(row["step"])
        psnr = float(row.get("convergence_speed", 0.0) or 0.0)
        ssim = float(row.get("sharpness_mean", 0.0) or 0.0)
        lpips = float(row.get("lpips_mean", 0.0) or 0.0)
        # Composite quality: 40% PSNR, 30% SSIM, 30% LPIPS (lower is better, so invert)
        # Note: using raw values here, normalization happens later
        quality_scores_by_step[step] = psnr + ssim + (1.0 - lpips) if lpips > 0 else psnr + ssim

    # Best step = highest quality score
    if quality_scores_by_step:
        t_best = int(max(quality_scores_by_step.keys(), key=lambda s: quality_scores_by_step[s]))
    else:
        t_best = int(eval_steps[-1])

    t_eval_best = t_best  # Already an evaluated step, no need to find nearest
    t_end = int(max(eval_steps))

    psnr_vals = [float(r.get("convergence_speed", 0.0) or 0.0) for r in eval_rows]
    ssim_vals = [float(r.get("sharpness_mean", 0.0) or 0.0) for r in eval_rows]
    lpips_vals = [float(r.get("lpips_mean", 0.0) or 0.0) for r in eval_rows]

    score_summary = compute_relative_quality_summary(
        eval_rows=eval_rows,
        baseline_eval_history=baseline_eval_history,
        loss_by_step=loss_by_step,
        elapsed_by_step=elapsed_by_step,
        t_eval_best=t_eval_best,
        t_end=t_end,
        prefer_quality_best=False,
        include_breakdown=True,
        baseline_loss_by_step_override=baseline_loss_by_step_override,
        score_reference_step=score_reference_step,
    )
    s_best = float(score_summary["s_best"])
    s_end = float(score_summary["s_end"])
    s_run = float(score_summary["s_run"])
    relative_score = float(score_summary["relative_score"])
    baseline_comparison = score_summary.get("baseline_comparison")

    if apply_update and x_features is not None:
        # Online model updates are permanently disabled (offline-only training framework).
        # apply_update is always False in production (allow_input_mode_learning_updates=False).
        # This branch is kept as a no-op stub so callers don't need changing.
        logger.info(
            "CONTEXTUAL_CONTINUOUS_COMPARE_ONLY mode=%s s_best=%.4f s_end=%.4f s_run=%.4f score=%.4f",
            mode, s_best, s_end, s_run, relative_score,
        )
    else:
        logger.info(
            "CONTEXTUAL_CONTINUOUS_COMPARE_ONLY mode=%s s_best=%.4f s_end=%.4f s_run=%.4f score=%.4f",
            mode, s_best, s_end, s_run, relative_score,
        )

    logger.info(
        "CONTEXTUAL_CONTINUOUS_SCORE mode=%s preset=%s score=%.4f score_positive=%s",
        mode,
        selected_preset,
        relative_score,
        str(relative_score > 0.0).lower(),
    )

    transition = {
        "x": dict(x_features or {}),
        "yhat": dict(yhat_scores),
        "k_star": selected_preset,
        "s_run": s_run,
        "baseline_comparison": baseline_comparison,
        "relative_score": relative_score,
    }

    return {
        "updated": bool(apply_update),
        "mode": mode,
        "selected_preset": selected_preset,
        "t_best": t_best,
        "t_eval_best": t_eval_best,
        "t_end": t_end,
        "s_best": s_best,
        "s_end": s_end,
        "s_run": s_run,
        "yhat_scores": yhat_scores,
        "transition": transition,
        "baseline_comparison": baseline_comparison,
        "relative_score": relative_score,
        "compare_only": not bool(apply_update),
    }


def record_run_penalty_featurewise_ridge(
    *,
    project_dir: Path,
    mode: str,
    selected_preset: str,
    yhat_scores: dict[str, float],
    penalty_score: float,
    x_features: dict[str, Any],
    reason: str,
    run_id: str,
    logger,
) -> dict[str, Any]:
    """No-op stub â€” online penalty updates are permanently disabled.

    Online model writes are disabled in the offline-only training framework.
    The project argument keeps the call site explicit even though updates use
    shared model files.
    """
    logger.info(
        "CONTEXTUAL_CONTINUOUS_PENALTY_SKIPPED mode=%s reason=%s score=%.4f (online updates disabled)",
        mode, reason, float(penalty_score),
    )
    return {
        "updated": False,
        "mode": mode,
        "selected_preset": selected_preset,
        "yhat_scores": yhat_scores,
        "relative_score": float(penalty_score),
        "reason": reason,
    }

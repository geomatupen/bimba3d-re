"""Legacy ai_input_mode aliases kept only for old saved metadata.

Delete this file and its imports when old run/model records no longer need to
load mode names that existed before Compact Scene Descriptors.
"""
from __future__ import annotations

from statistics import median
from typing import Any

from .common import ModeContext
from .compact_scene_descriptors import _as_float, _collect_processing_sizes, _iter_images
from .exif_extractors import extract_camera_exif


LEGACY_AI_INPUT_MODE_ALIASES = {
    "exif_only": "exif_compact_featurewise",
    "exif_plus_flight_plan": "exif_compact_featurewise",
    "exif_plus_flight_plan_plus_external": "exif_compact_featurewise",
}

LEGACY_GROUPWISE_SELECTOR_STRATEGIES = {
    "featurewise_ridge_regression",
    "featurewise_mlp",
}

LEGACY_GROUPWISE_FEATURE_KEYS = {
    "focal_length_mm",
    "shutter_s",
    "iso",
    "img_width_median",
    "img_height_median",
}


def normalize_ai_input_mode_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    return LEGACY_AI_INPUT_MODE_ALIASES.get(token, token)


def selector_needs_legacy_groupwise_features(value: Any) -> bool:
    return str(value or "").strip().lower() in LEGACY_GROUPWISE_SELECTOR_STRATEGIES


def build_legacy_groupwise_features(ctx: ModeContext) -> dict[str, float | int]:
    """Build extra context values required by legacy non-compact featurewise models."""
    focal_lengths: list[float] = []
    exposure_times: list[float] = []
    iso_values: list[float] = []

    for path in _iter_images(ctx.metadata_image_dir, limit=24):
        try:
            exif, _, _ = extract_camera_exif(path)
        except Exception:
            continue
        focal = _as_float(exif.get("FocalLength"))
        exposure = _as_float(exif.get("ExposureTime"))
        iso = _as_float(exif.get("ISOSpeedRatings"))
        if focal is not None:
            focal_lengths.append(focal)
        if exposure is not None:
            exposure_times.append(exposure)
        if iso is not None:
            iso_values.append(iso)

    widths, heights = _collect_processing_sizes(ctx.processing_image_dir, limit=24)
    return {
        "focal_length_mm": max(2.0, min(300.0, float(median(focal_lengths)) if focal_lengths else 24.0)),
        "shutter_s": max(0.0001, min(1.0, float(median(exposure_times)) if exposure_times else 0.004)),
        "iso": max(50.0, min(102400.0, float(median(iso_values)) if iso_values else 100.0)),
        "img_width_median": max(640, min(8000, int(median(widths)) if widths else 4000)),
        "img_height_median": max(480, min(6000, int(median(heights)) if heights else 3000)),
    }


def with_legacy_groupwise_features(features: dict[str, Any], ctx: ModeContext, selector_strategy: Any) -> dict[str, Any]:
    if not selector_needs_legacy_groupwise_features(selector_strategy):
        return dict(features)
    return {**features, **build_legacy_groupwise_features(ctx)}


__all__ = [
    "LEGACY_AI_INPUT_MODE_ALIASES",
    "LEGACY_GROUPWISE_FEATURE_KEYS",
    "LEGACY_GROUPWISE_SELECTOR_STRATEGIES",
    "build_legacy_groupwise_features",
    "normalize_ai_input_mode_token",
    "selector_needs_legacy_groupwise_features",
    "with_legacy_groupwise_features",
]

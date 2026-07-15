"""Context jitter utility for multi-pass training with feature diversity."""
from __future__ import annotations

import random
from typing import Any


# Feature bounds from the report-aligned featurewise model code and feature extraction.
FEATURE_BOUNDS = {
    # EXIF features
    "focal_length_mm": (8.0, 300.0),
    "shutter_s": (0.0001, 1.0),
    "iso": (50.0, 102400.0),
    "img_width_median": (640.0, 8000.0),
    "img_height_median": (480.0, 6000.0),

    # Flight plan features
    "gsd_median": (0.001, 0.5),
    "overlap_proxy": (0.0, 1.0),
    "coverage_spread": (0.0, 1.0),
    "camera_angle_bucket": (0, 3),  # Discrete: {0, 1, 2, 3}
    "heading_consistency": (0.0, 1.0),

    # External features
    "vegetation_cover_percentage": (0.0, 1.0),
    "vegetation_complexity_score": (0.0, 1.0),
    "terrain_roughness_proxy": (0.0, 1.0),
    "texture_density": (0.0, 1.0),
    "blur_motion_risk": (0.0, 1.0),
}


def apply_context_jitter(features: dict[str, Any], jitter_mode: str = "uniform") -> dict[str, Any]:
    """Apply jitter to context features for multi-pass learning diversity.

    Instead of ±X% around actual value, samples uniformly from valid bounds.
    This provides true exploration of the feature space.

    Args:
        features: Original features extracted from dataset
        jitter_mode: "uniform" (sample from bounds) or "gaussian" (±σ around value)

    Returns:
        Jittered features (new dict, original unchanged)

    Example:
        Original: focal_length_mm = 24.0

        Old approach (±5%):
          Pass 2: 24.0 * 1.03 = 24.72
          Pass 3: 24.0 * 0.97 = 23.28
          Limited exploration around actual value

        New approach (uniform from bounds [8, 300]):
          Pass 2: 156.3
          Pass 3: 42.8
          Pass 4: 201.5
          Full feature space exploration
    """
    jittered = {}

    for key, value in features.items():
        # Skip missing flags (binary indicators)
        if key.endswith("_missing"):
            jittered[key] = value
            continue

        # Skip if no bounds defined (pass through)
        if key not in FEATURE_BOUNDS:
            jittered[key] = value
            continue

        bounds = FEATURE_BOUNDS[key]
        min_val, max_val = bounds

        if jitter_mode == "uniform":
            # Sample uniformly from valid bounds
            if key == "camera_angle_bucket":
                # Discrete values: randomly choose from {0, 1, 2, 3}
                jittered[key] = random.randint(int(min_val), int(max_val))
            else:
                # Continuous values: uniform sampling
                jittered[key] = random.uniform(min_val, max_val)

        elif jitter_mode == "gaussian":
            # Sample from Gaussian centered at actual value, clipped to bounds
            # σ = 15% of range
            range_size = max_val - min_val
            sigma = 0.15 * range_size
            jittered_value = random.gauss(value, sigma)

            # Clip to bounds
            jittered[key] = max(min_val, min(max_val, jittered_value))

        else:
            # No jitter (pass through)
            jittered[key] = value

    return jittered


def apply_mild_jitter(features: dict[str, Any]) -> dict[str, Any]:
    """Apply mild jitter (±10% around actual value, clipped to bounds).

    This is a middle ground: explores near the actual value but with wider range.
    """
    jittered = {}

    for key, value in features.items():
        # Skip missing flags
        if key.endswith("_missing"):
            jittered[key] = value
            continue

        # Skip if no bounds defined
        if key not in FEATURE_BOUNDS:
            jittered[key] = value
            continue

        bounds = FEATURE_BOUNDS[key]
        min_val, max_val = bounds

        if key == "camera_angle_bucket":
            # Discrete: small chance to change
            if random.random() < 0.3:  # 30% chance to change
                jittered[key] = random.randint(int(min_val), int(max_val))
            else:
                jittered[key] = value
        else:
            # Continuous: ±10% of actual value, clipped to bounds
            jitter_factor = random.uniform(0.9, 1.1)
            jittered_value = value * jitter_factor
            jittered[key] = max(min_val, min(max_val, jittered_value))

    return jittered


def get_jitter_stats(original: dict[str, Any], jittered: dict[str, Any]) -> dict[str, float]:
    """Calculate jitter statistics for logging/debugging.

    Returns:
        Dict with mean absolute change, max change, changed feature count
    """
    changes = []
    changed_count = 0

    for key in original:
        if key.endswith("_missing"):
            continue

        if key not in jittered:
            continue

        orig_val = original[key]
        jitt_val = jittered[key]

        if orig_val != jitt_val:
            changed_count += 1

            # Calculate relative change (skip if original is 0)
            if orig_val != 0:
                rel_change = abs((jitt_val - orig_val) / orig_val)
                changes.append(rel_change)

    return {
        "mean_relative_change": sum(changes) / len(changes) if changes else 0.0,
        "max_relative_change": max(changes) if changes else 0.0,
        "changed_feature_count": changed_count,
        "total_features": len([k for k in original if not k.endswith("_missing")]),
    }

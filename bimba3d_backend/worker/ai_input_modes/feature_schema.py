"""Shared feature and multiplier schema for thesis models."""
from __future__ import annotations

from bimba3d_backend.worker.ai_input_modes.featurewise_ridge_helpers import (
    FEATUREWISE_GROUP_FEATURES,
    GROUP_BOUNDS,
    GROUP_KEYS,
    GROUP_MULTIPLIERS_MAP,
)

FEATURE_SCHEMA_NAME = "mode3_exif_flight_scene_v1"
MULTIPLIER_SCHEMA_NAME = "geometry_appearance_densification_v1"

__all__ = [
    "FEATUREWISE_GROUP_FEATURES",
    "FEATURE_SCHEMA_NAME",
    "GROUP_BOUNDS",
    "GROUP_KEYS",
    "GROUP_MULTIPLIERS_MAP",
    "MULTIPLIER_SCHEMA_NAME",
]

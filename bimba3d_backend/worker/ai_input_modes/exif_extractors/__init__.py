"""
EXIF Extraction System - Router-based architecture for camera-specific extraction

This module provides automatic routing to appropriate extractors based on camera manufacturer.

Architecture:
    1. Quick manufacturer detection (PIL reads Make field)
    2. Route to manufacturer-specific extractor
    3. Fall back to generic extraction + ExifTool if needed

Usage:
    from exif_extractors import extract_camera_exif

    exif_dict, width, height = extract_camera_exif(image_path)

The returned exif_dict contains all standard and manufacturer-specific EXIF fields.
"""

from .router import extract_camera_exif

__all__ = ["extract_camera_exif"]

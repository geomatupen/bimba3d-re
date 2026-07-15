"""
Router - Detects camera manufacturer and dispatches to appropriate extractor
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

from PIL import Image, ExifTags

logger = logging.getLogger(__name__)


def detect_camera_make(image_path: Path) -> str:
    """
    Quick detection of camera manufacturer from EXIF Make field.

    Args:
        image_path: Path to image file

    Returns:
        Camera manufacturer (e.g., "DJI", "Sony", "Canon", "Unknown")
    """
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if not exif:
                return "Unknown"

            for key, value in exif.items():
                tag_name = ExifTags.TAGS.get(key, key)
                if tag_name == "Make":
                    make = str(value).strip()
                    return make if make else "Unknown"

            return "Unknown"

    except Exception as e:
        logger.warning(f"Failed to detect camera make from {image_path.name}: {e}")
        return "Unknown"


def extract_camera_exif(image_path: Path) -> Tuple[Dict[str, Any], int, int]:
    """
    Main entry point - routes to appropriate extractor based on camera manufacturer.

    Process:
        1. Detect manufacturer (quick PIL read of Make field)
        2. Route to manufacturer-specific extractor
        3. Fall back to generic + ExifTool if unknown

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (exif_dict, width, height)

    Example:
        >>> exif, w, h = extract_camera_exif(Path("image.jpg"))
        >>> print(f"Camera: {exif.get('Make')} {exif.get('Model')}")
        >>> print(f"Focal: {exif.get('FocalLength')}mm")
    """
    # Step 1: Detect manufacturer
    make = detect_camera_make(image_path)

    # Step 2: Route to appropriate extractor
    if make == "DJI":
        from .dji_extractor import DJIExtractor
        extractor = DJIExtractor()
        logger.debug(f"Using DJI extractor for {image_path.name}")

    elif make in ["Sony", "SONY"]:
        # TODO: Implement Sony extractor
        # For now, fall back to generic
        from .fallback_extractor import FallbackExtractor
        extractor = FallbackExtractor()
        logger.debug(f"Using fallback extractor for Sony camera: {image_path.name}")

    elif make in ["Canon"]:
        # TODO: Implement Canon extractor
        from .fallback_extractor import FallbackExtractor
        extractor = FallbackExtractor()
        logger.debug(f"Using fallback extractor for Canon camera: {image_path.name}")

    else:
        # Unknown manufacturer - use fallback (ExifTool or generic PIL)
        from .fallback_extractor import FallbackExtractor
        extractor = FallbackExtractor()
        logger.debug(f"Using fallback extractor for {make} camera: {image_path.name}")

    # Step 3: Extract EXIF
    try:
        exif_dict, width, height = extractor.extract(image_path)
        logger.debug(
            f"Successfully extracted EXIF from {image_path.name} "
            f"using {extractor.get_name()} ({len(exif_dict)} fields)"
        )
        return exif_dict, width, height

    except Exception as e:
        logger.error(f"EXIF extraction failed for {image_path.name}: {e}")
        # Last resort: return empty dict with image dimensions
        try:
            with Image.open(image_path) as img:
                return {}, img.width, img.height
        except Exception:
            return {}, 0, 0

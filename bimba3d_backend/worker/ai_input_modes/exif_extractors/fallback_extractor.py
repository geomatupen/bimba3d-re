"""
Fallback extractor for unknown camera manufacturers

Tries ExifTool first (comprehensive), falls back to generic PIL if unavailable.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple

from PIL import Image, ExifTags

from .base import BaseExtractor

logger = logging.getLogger(__name__)


class FallbackExtractor(BaseExtractor):
    """
    Fallback extractor for unknown or unsupported cameras.

    Strategy:
    1. Try ExifTool (if available) - gets 100% of EXIF/XMP/MakerNote
    2. Fall back to generic PIL extraction - gets standard EXIF only

    This ensures we always extract something, even for unknown cameras.
    """

    def __init__(self):
        self._exiftool_available = self._check_exiftool()

    def extract(self, image_path: Path) -> Tuple[Dict[str, Any], int, int]:
        """
        Extract EXIF using best available method.

        Args:
            image_path: Path to image

        Returns:
            (exif_dict, width, height)
        """
        # Try ExifTool first
        if self._exiftool_available:
            try:
                return self._extract_via_exiftool(image_path)
            except Exception as e:
                logger.warning(f"ExifTool extraction failed, falling back to PIL: {e}")

        # Fall back to generic PIL extraction
        return self._extract_via_pil(image_path)

    @staticmethod
    def _check_exiftool() -> bool:
        """Check if exiftool is available"""
        try:
            result = subprocess.run(
                ["exiftool", "-ver"],
                capture_output=True,
                timeout=2,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"ExifTool available: version {result.stdout.strip()}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        logger.debug("ExifTool not available, will use PIL fallback")
        return False

    def _extract_via_exiftool(self, image_path: Path) -> Tuple[Dict[str, Any], int, int]:
        """
        Extract EXIF via ExifTool (comprehensive extraction).

        ExifTool reads ALL metadata formats:
        - Standard EXIF
        - XMP (all namespaces)
        - MakerNote (manufacturer-specific)
        - IPTC
        - More...
        """
        result = subprocess.run(
            ["exiftool", "-j", "-a", "-G1", str(image_path)],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            raise RuntimeError(f"ExifTool failed: {result.stderr}")

        data = json.loads(result.stdout)[0]

        # Normalize ExifTool output to our standard format
        exif = self._normalize_exiftool_output(data)

        # Get image dimensions
        width = int(data.get("ImageWidth", 0))
        height = int(data.get("ImageHeight", 0))

        if width == 0 or height == 0:
            # Fall back to PIL for dimensions
            with Image.open(image_path) as img:
                width, height = img.size

        logger.info(f"Extracted {len(exif)} EXIF fields via ExifTool from {image_path.name}")
        return exif, width, height

    @staticmethod
    def _normalize_exiftool_output(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize ExifTool output to match PIL field names.

        ExifTool uses group prefixes like "EXIF:FocalLength".
        We strip the prefix to match PIL's naming.
        """
        exif: Dict[str, Any] = {}

        for key, value in data.items():
            # Remove group prefix (e.g., "EXIF:FocalLength" -> "FocalLength")
            if ":" in key:
                _, field_name = key.split(":", 1)
            else:
                field_name = key

            # Store value
            exif[field_name] = value

            # Special handling for GPS
            if "GPS" in key and "GPSInfo" not in exif:
                exif["GPSInfo"] = {}

        # Build GPSInfo dict from individual GPS fields
        gps_fields = {k: v for k, v in exif.items() if k.startswith("GPS")}
        if gps_fields:
            gps_info = {}
            # Convert GPS to decimal degrees
            if "GPSLatitude" in exif:
                gps_info["lat"] = FallbackExtractor._parse_gps_coord(exif["GPSLatitude"])
            if "GPSLongitude" in exif:
                gps_info["lon"] = FallbackExtractor._parse_gps_coord(exif["GPSLongitude"])
            if "GPSAltitude" in exif:
                try:
                    gps_info["alt"] = float(str(exif["GPSAltitude"]).replace(" m", ""))
                except:
                    pass
            if gps_info:
                exif["GPSInfo"] = gps_info

        return exif

    @staticmethod
    def _parse_gps_coord(coord_str: str) -> float | None:
        """Parse GPS coordinate from ExifTool format"""
        try:
            # ExifTool format: "28 deg 34' 37.58\" N" or decimal
            if "deg" in coord_str:
                # Parse DMS format
                import re
                match = re.match(r"(\d+) deg (\d+)' ([\d.]+)\" ([NSEW])", coord_str)
                if match:
                    deg, min, sec, direction = match.groups()
                    decimal = float(deg) + float(min) / 60 + float(sec) / 3600
                    if direction in ["S", "W"]:
                        decimal = -decimal
                    return decimal
            else:
                # Already decimal
                return float(coord_str)
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_via_pil(image_path: Path) -> Tuple[Dict[str, Any], int, int]:
        """
        Generic PIL extraction (standard EXIF only).

        This is the bare minimum fallback when ExifTool is not available.
        """
        with Image.open(image_path) as img:
            width, height = img.size
            raw = img.getexif() or {}
            exif: Dict[str, Any] = {}

            # Read main EXIF tags
            for key, value in raw.items():
                tag_name = ExifTags.TAGS.get(key, key)
                exif[str(tag_name)] = value

            # Read ExifIFD
            if hasattr(raw, "get_ifd"):
                try:
                    exif_ifd = raw.get_ifd(0x8769)
                    if isinstance(exif_ifd, dict):
                        for k, v in exif_ifd.items():
                            exif_name = ExifTags.TAGS.get(k, k)
                            if exif_name not in exif or exif[exif_name] is None:
                                exif[exif_name] = v
                except Exception:
                    pass

            # Read GPS IFD
            if not isinstance(exif.get("GPSInfo"), dict) and hasattr(raw, "get_ifd"):
                try:
                    gps_ifd = raw.get_ifd(0x8825)
                    if isinstance(gps_ifd, dict):
                        gps: Dict[Any, Any] = {}
                        for k, v in gps_ifd.items():
                            gps_name = ExifTags.GPSTAGS.get(k, k)
                            gps[gps_name] = v
                            gps[k] = v
                        exif["GPSInfo"] = gps
                except Exception:
                    pass

            logger.info(f"Extracted {len(exif)} EXIF fields via PIL from {image_path.name}")
            return exif, width, height
